"""Verified sequence-preserving access to A8 multi-layer ESM2 cache files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.ablations.a1_frozen_esm2.data import ProteinRecord
from experiments.ablations.a1_frozen_esm2.train_cached_a1 import (
    file_sha256,
    sequence_sha256,
)


@dataclass(frozen=True)
class MultiLayerCacheBundle:
    cache_dir: Path
    entries: dict[str, dict[str, Any]]
    hidden_size: int
    layer_indices: tuple[int, ...]
    model_revision: str
    index_sha256: str
    verification: dict[str, Any]


def _embedding_path(cache_dir: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe cache embedding path: {relative_path}")
    return cache_dir / relative


def load_verified_multilayer_cache(
    cache_dir: Path,
    records: Sequence[ProteinRecord],
    expected_revision: str | None = None,
    expected_layers: tuple[int, ...] | None = None,
) -> MultiLayerCacheBundle:
    verification_path = cache_dir / "cache_verification_report.json"
    index_path = cache_dir / "cache_index.jsonl"
    if not verification_path.is_file():
        raise FileNotFoundError(
            f"missing independent cache verification report: {verification_path}"
        )
    if not index_path.is_file():
        raise FileNotFoundError(f"missing cache index: {index_path}")

    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    required_checks = {
        "status": "pass",
        "all_finite": True,
        "all_file_hashes_match": True,
        "input_contains_residue_labels": False,
        "caid2_caid3_labels_accessed": False,
    }
    for key, expected in required_checks.items():
        if verification.get(key) != expected:
            raise ValueError(
                f"cache verification failed requirement {key}: "
                f"{verification.get(key)!r} != {expected!r}"
            )

    observed_index_sha256 = file_sha256(index_path)
    if verification.get("cache_index_sha256") != observed_index_sha256:
        raise ValueError("cache index changed after independent verification")
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    entries = {str(row["protein_id"]): row for row in rows}
    if len(entries) != len(rows):
        raise ValueError("cache index contains duplicate protein IDs")
    manifest_ids = {record.protein_id for record in records}
    if set(entries) != manifest_ids:
        missing = sorted(manifest_ids - set(entries))
        extra = sorted(set(entries) - manifest_ids)
        raise ValueError(
            f"labeled manifest/cache ID mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    hidden_sizes: set[int] = set()
    revisions: set[str] = set()
    selected_layers: set[tuple[int, ...]] = set()
    for record in records:
        entry = entries[record.protein_id]
        if entry.get("sequence_sha256") != sequence_sha256(record.sequence):
            raise ValueError(f"sequence/cache mismatch for {record.protein_id}")
        if int(entry.get("sequence_length", -1)) != len(record.sequence):
            raise ValueError(f"sequence length/cache mismatch for {record.protein_id}")
        if str(entry.get("fold")) != record.fold:
            raise ValueError(f"fold/cache mismatch for {record.protein_id}")
        hidden_size = int(entry["hidden_size"])
        layers = tuple(int(value) for value in entry["layer_indices"])
        embedding = np.load(
            _embedding_path(cache_dir, str(entry["embedding_file"])),
            mmap_mode="r",
            allow_pickle=False,
        )
        expected_shape = (len(record.sequence), len(layers), hidden_size)
        if embedding.shape != expected_shape:
            raise ValueError(f"embedding shape/cache mismatch for {record.protein_id}")
        if embedding.dtype != np.float16:
            raise ValueError(f"embedding dtype must be float16 for {record.protein_id}")
        hidden_sizes.add(hidden_size)
        revisions.add(str(entry["model_revision"]))
        selected_layers.add(layers)

    if len(hidden_sizes) != 1 or len(revisions) != 1 or len(selected_layers) != 1:
        raise ValueError("cache mixes hidden sizes, revisions, or selected layers")
    model_revision = next(iter(revisions))
    layer_indices = next(iter(selected_layers))
    if expected_revision is not None and model_revision != expected_revision:
        raise ValueError(
            f"cache model revision {model_revision} != configured revision {expected_revision}"
        )
    if expected_layers is not None and layer_indices != expected_layers:
        raise ValueError(
            f"cache layers {layer_indices} != configured layers {expected_layers}"
        )
    return MultiLayerCacheBundle(
        cache_dir=cache_dir,
        entries=entries,
        hidden_size=next(iter(hidden_sizes)),
        layer_indices=layer_indices,
        model_revision=model_revision,
        index_sha256=observed_index_sha256,
        verification=verification,
    )


class MultiLayerCachedProteinDataset(Dataset):
    def __init__(
        self, records: Sequence[ProteinRecord], cache: MultiLayerCacheBundle
    ) -> None:
        self.records = list(records)
        self.cache = cache

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        entry = self.cache.entries[record.protein_id]
        embedding = np.load(
            _embedding_path(self.cache.cache_dir, str(entry["embedding_file"])),
            mmap_mode="r",
            allow_pickle=False,
        )
        features = torch.from_numpy(np.array(embedding, dtype=np.float16, copy=True))
        labels = torch.from_numpy(record.labels.astype(np.float32, copy=True))
        return {
            "protein_id": record.protein_id,
            "features": features,
            "labels": labels,
            "length": len(record.sequence),
        }


def pad_multilayer_cached_proteins(
    examples: Sequence[dict[str, Any]], unknown_label: int = -1
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(examples)
    max_length = max(int(example["length"]) for example in examples)
    num_layers = int(examples[0]["features"].shape[1])
    hidden_size = int(examples[0]["features"].shape[2])
    features = torch.zeros(
        (batch_size, max_length, num_layers, hidden_size), dtype=torch.float16
    )
    labels = torch.full(
        (batch_size, max_length), float(unknown_label), dtype=torch.float32
    )
    residue_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    protein_ids: list[str] = []
    lengths: list[int] = []
    for index, example in enumerate(examples):
        length = int(example["length"])
        expected_shape = (length, num_layers, hidden_size)
        if example["features"].shape != expected_shape:
            raise ValueError(f"cached feature shape mismatch for {example['protein_id']}")
        if example["labels"].shape != (length,):
            raise ValueError(f"label shape mismatch for {example['protein_id']}")
        features[index, :length] = example["features"]
        labels[index, :length] = example["labels"]
        residue_mask[index, :length] = True
        protein_ids.append(str(example["protein_id"]))
        lengths.append(length)
    return {
        "features": features,
        "labels": labels,
        "residue_mask": residue_mask,
        "protein_ids": protein_ids,
        "lengths": lengths,
    }
