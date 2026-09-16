from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


UNKNOWN_LABEL = -1


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    labels: np.ndarray
    fold: str


def _first_present(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    raise KeyError(f"None of the required fields is present: {list(keys)}")


def _decode_labels(raw: Any) -> np.ndarray:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            raw = json.loads(text)
        elif any(separator in text for separator in (",", " ", "\t")):
            raw = [token for token in text.replace(",", " ").split()]
        else:
            mapping = {"0": 0, "1": 1, "-": UNKNOWN_LABEL, "?": UNKNOWN_LABEL, "X": UNKNOWN_LABEL}
            raw = [mapping.get(character.upper(), UNKNOWN_LABEL) for character in text]
    labels = np.asarray(raw, dtype=np.int64)
    labels = np.where(np.isin(labels, [0, 1]), labels, UNKNOWN_LABEL).astype(np.int64)
    return labels


def load_manifest(path: str | Path) -> list[ProteinRecord]:
    records: list[ProteinRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            protein_id = str(_first_present(row, ("protein_id", "id", "name", "accession")))
            sequence = str(_first_present(row, ("sequence", "seq"))).replace(" ", "").upper()
            labels = _decode_labels(
                _first_present(row, ("labels", "residue_labels", "label", "targets"))
            )
            fold = str(_first_present(row, ("fold", "fold_id", "split_fold")))
            if not sequence:
                raise ValueError(f"Empty sequence on manifest line {line_number}")
            if len(sequence) != len(labels):
                raise ValueError(
                    f"Sequence/label length mismatch on line {line_number}: "
                    f"{len(sequence)} != {len(labels)}"
                )
            records.append(ProteinRecord(protein_id, sequence, labels, fold))
    if not records:
        raise ValueError(f"Manifest contains no records: {path}")
    if len({record.protein_id for record in records}) != len(records):
        raise ValueError("Manifest protein IDs are not unique")
    return records


def window_starts(length: int, max_residues: int, stride: int) -> list[int]:
    if length <= max_residues:
        return [0]
    starts = list(range(0, length - max_residues + 1, stride))
    last = length - max_residues
    if starts[-1] != last:
        starts.append(last)
    return starts


class ProteinWindowDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ProteinRecord],
        *,
        max_residues: int,
        stride: int,
    ) -> None:
        if max_residues <= 0 or stride <= 0 or stride > max_residues:
            raise ValueError("Require 0 < stride <= max_residues")
        self.records = list(records)
        self.max_residues = int(max_residues)
        self.stride = int(stride)
        self.windows: list[tuple[int, int, int]] = []
        self.coverage: list[np.ndarray] = []
        for record_index, record in enumerate(self.records):
            starts = window_starts(len(record.sequence), self.max_residues, self.stride)
            coverage = np.zeros(len(record.sequence), dtype=np.int64)
            for start in starts:
                end = min(start + self.max_residues, len(record.sequence))
                coverage[start:end] += 1
                self.windows.append((record_index, start, end))
            if np.any(coverage == 0):
                raise RuntimeError(f"Windowing left uncovered residues in {record.protein_id}")
            self.coverage.append(coverage)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, start, end = self.windows[index]
        record = self.records[record_index]
        return {
            "record_index": record_index,
            "protein_id": record.protein_id,
            "start": start,
            "end": end,
            "sequence": record.sequence[start:end],
            "labels": record.labels[start:end].copy(),
            "loss_weight": (1.0 / self.coverage[record_index][start:end]).astype(np.float32),
        }


class EsmWindowCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        sequences = [sample["sequence"] for sample in samples]
        encoded = self.tokenizer(
            sequences,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].long()
        attention_mask = encoded["attention_mask"].long()
        if "special_tokens_mask" in encoded:
            special_tokens_mask = encoded["special_tokens_mask"].bool()
        else:
            special_tokens_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            for row in range(input_ids.shape[0]):
                special_tokens_mask[row] = torch.tensor(
                    self.tokenizer.get_special_tokens_mask(
                        input_ids[row].tolist(), already_has_special_tokens=True
                    ),
                    dtype=torch.bool,
                )
        residue_mask = attention_mask.bool() & ~special_tokens_mask
        labels = torch.full_like(input_ids, UNKNOWN_LABEL, dtype=torch.long)
        loss_weight = torch.zeros_like(input_ids, dtype=torch.float32)
        residue_position = torch.full_like(input_ids, -1, dtype=torch.long)

        for row, sample in enumerate(samples):
            token_positions = torch.nonzero(residue_mask[row], as_tuple=False).flatten()
            residue_count = len(sample["sequence"])
            if token_positions.numel() != residue_count:
                raise ValueError(
                    f"Tokenizer did not preserve one-token-per-residue for {sample['protein_id']}: "
                    f"{token_positions.numel()} tokens for {residue_count} residues"
                )
            labels[row, token_positions] = torch.from_numpy(sample["labels"]).long()
            loss_weight[row, token_positions] = torch.from_numpy(sample["loss_weight"]).float()
            residue_position[row, token_positions] = torch.arange(residue_count, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "residue_mask": residue_mask,
            "labels": labels,
            "loss_weight": loss_weight,
            "residue_position": residue_position,
            "record_index": torch.tensor([sample["record_index"] for sample in samples]),
            "start": torch.tensor([sample["start"] for sample in samples]),
            "protein_id": [sample["protein_id"] for sample in samples],
        }


def label_counts(records: Iterable[ProteinRecord]) -> tuple[int, int]:
    positive = 0
    negative = 0
    for record in records:
        positive += int(np.sum(record.labels == 1))
        negative += int(np.sum(record.labels == 0))
    return positive, negative
