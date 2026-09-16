"""Sequence-preserving access to verified A1 frozen ESM2 cache files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.ablations.a1_frozen_esm2.data import ProteinRecord
from experiments.ablations.a1_frozen_esm2.train_cached_a1 import CacheBundle


class CachedProteinDataset(Dataset):
    def __init__(self, records: Sequence[ProteinRecord], cache: CacheBundle) -> None:
        self.records = list(records)
        self.cache = cache

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        entry = self.cache.entries[record.protein_id]
        relative = Path(str(entry["embedding_file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe cache path for {record.protein_id}")
        embedding = np.load(
            self.cache.cache_dir / relative,
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


def pad_cached_proteins(
    examples: Sequence[dict[str, Any]], unknown_label: int = -1
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(examples)
    max_length = max(int(example["length"]) for example in examples)
    hidden_size = int(examples[0]["features"].shape[1])
    features = torch.zeros(
        (batch_size, max_length, hidden_size), dtype=torch.float16
    )
    labels = torch.full(
        (batch_size, max_length), float(unknown_label), dtype=torch.float32
    )
    residue_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    protein_ids: list[str] = []
    lengths: list[int] = []

    for index, example in enumerate(examples):
        length = int(example["length"])
        if example["features"].shape != (length, hidden_size):
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
