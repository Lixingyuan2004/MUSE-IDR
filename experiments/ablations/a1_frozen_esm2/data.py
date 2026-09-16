from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    labels: np.ndarray
    fold: str


def _normalize_labels(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    labels = np.asarray(value, dtype=np.int8)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    invalid = set(np.unique(labels).tolist()) - {-1, 0, 1}
    if invalid:
        raise ValueError(f"labels contain values outside -1/0/1: {sorted(invalid)}")
    return labels


def load_manifest(path: str | Path) -> list[ProteinRecord]:
    """Load one JSON object per line.

    Required fields are ``id``, ``sequence``, ``labels`` and ``fold``.
    Unknown residues must have label -1.
    """

    manifest_path = Path(path)
    records: list[ProteinRecord] = []
    seen_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            protein_id = str(row["id"])
            sequence = str(row["sequence"]).upper().replace(" ", "")
            labels = _normalize_labels(row["labels"])
            fold = str(row["fold"])

            if protein_id in seen_ids:
                raise ValueError(f"duplicate protein id: {protein_id}")
            if len(sequence) != len(labels):
                raise ValueError(
                    f"{protein_id}: sequence length {len(sequence)} != labels length {len(labels)}"
                )
            if not sequence:
                raise ValueError(f"{protein_id}: empty sequence")

            seen_ids.add(protein_id)
            records.append(ProteinRecord(protein_id, sequence, labels, fold))

    if not records:
        raise ValueError(f"manifest is empty: {manifest_path}")
    if len({record.fold for record in records}) < 2:
        raise ValueError("at least two folds are required")
    return records


def window_starts(length: int, max_residues: int, stride: int) -> list[int]:
    if max_residues <= 0 or stride <= 0 or stride > max_residues:
        raise ValueError("require 0 < stride <= max_residues")
    if length <= max_residues:
        return [0]
    starts = list(range(0, length - max_residues + 1, stride))
    final_start = length - max_residues
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def center_merge_weights(length: int) -> np.ndarray:
    """Triangular center weighting with non-zero edge weights."""

    if length == 1:
        return np.ones(1, dtype=np.float32)
    position = np.linspace(-1.0, 1.0, length, dtype=np.float32)
    return 0.1 + 0.9 * (1.0 - np.abs(position))


class ProteinWindowDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ProteinRecord],
        max_residues: int,
        stride: int,
    ) -> None:
        self.records = list(records)
        self.max_residues = int(max_residues)
        self.stride = int(stride)
        self.windows: list[tuple[int, int, int, np.ndarray]] = []

        for record_index, record in enumerate(self.records):
            starts = window_starts(len(record.sequence), self.max_residues, self.stride)
            coverage = np.zeros(len(record.sequence), dtype=np.int16)
            spans: list[tuple[int, int]] = []
            for start in starts:
                end = min(start + self.max_residues, len(record.sequence))
                coverage[start:end] += 1
                spans.append((start, end))
            if np.any(coverage == 0):
                raise RuntimeError(f"window coverage failure for {record.protein_id}")
            for start, end in spans:
                # Each original residue contributes total loss weight one even
                # when it occurs in two overlapping windows.
                loss_weight = 1.0 / coverage[start:end].astype(np.float32)
                self.windows.append((record_index, start, end, loss_weight))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, start, end, loss_weight = self.windows[index]
        record = self.records[record_index]
        return {
            "record_index": record_index,
            "protein_id": record.protein_id,
            "start": start,
            "end": end,
            "sequence": record.sequence[start:end],
            "labels": record.labels[start:end],
            "loss_weight": loss_weight,
            "merge_weight": center_merge_weights(end - start),
        }


class EsmWindowCollator:
    def __init__(self, tokenizer, unknown_label: int = -1) -> None:
        self.tokenizer = tokenizer
        self.unknown_label = int(unknown_label)

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [example["sequence"] for example in examples],
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        attention_mask = encoded["attention_mask"].bool()
        special_mask = encoded["special_tokens_mask"].bool()
        residue_mask = attention_mask & ~special_mask

        token_labels = torch.full(
            encoded["input_ids"].shape,
            fill_value=self.unknown_label,
            dtype=torch.float32,
        )
        token_loss_weight = torch.zeros_like(token_labels)

        for batch_index, example in enumerate(examples):
            expected = len(example["sequence"])
            positions = torch.where(residue_mask[batch_index])[0]
            if len(positions) != expected:
                raise ValueError(
                    f"{example['protein_id']}[{example['start']}:{example['end']}]: "
                    f"tokenizer produced {len(positions)} residue tokens for {expected} residues"
                )
            token_labels[batch_index, positions] = torch.from_numpy(
                example["labels"].astype(np.float32, copy=False)
            )
            token_loss_weight[batch_index, positions] = torch.from_numpy(
                example["loss_weight"].astype(np.float32, copy=False)
            )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "residue_mask": residue_mask,
            "labels": token_labels,
            "loss_weight": token_loss_weight,
            "metadata": list(examples),
        }


def count_known_labels(records: Iterable[ProteinRecord]) -> tuple[int, int]:
    positives = 0
    negatives = 0
    for record in records:
        positives += int(np.sum(record.labels == 1))
        negatives += int(np.sum(record.labels == 0))
    return positives, negatives
