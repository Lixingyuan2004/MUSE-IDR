from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.ablations.a1_frozen_esm2.data import ProteinRecord
from experiments.ablations.a1_frozen_esm2.train_cached_a1 import (
    CachedIdrHead,
    file_sha256,
    load_known_residue_tensors,
    load_verified_cache,
)


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def write_cache(root: Path, records: list[ProteinRecord], hidden_size: int = 3) -> None:
    embedding_dir = root / "embeddings"
    embedding_dir.mkdir(parents=True)
    rows = []
    for index, record in enumerate(records):
        relative = f"embeddings/{index}.npy"
        values = np.arange(
            len(record.sequence) * hidden_size, dtype=np.float16
        ).reshape(len(record.sequence), hidden_size)
        np.save(root / relative, values, allow_pickle=False)
        rows.append(
            {
                "protein_id": record.protein_id,
                "sequence_sha256": sequence_hash(record.sequence),
                "sequence_length": len(record.sequence),
                "fold": record.fold,
                "hidden_size": hidden_size,
                "model_revision": "locked-revision",
                "embedding_file": relative,
            }
        )
    index_path = root / "cache_index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    verification = {
        "status": "pass",
        "all_finite": True,
        "all_file_hashes_match": True,
        "input_contains_residue_labels": False,
        "caid2_caid3_labels_accessed": False,
        "cache_index_sha256": file_sha256(index_path),
    }
    (root / "cache_verification_report.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )


def example_records() -> list[ProteinRecord]:
    return [
        ProteinRecord("p1", "ACD", np.array([1, -1, 0], dtype=np.int8), "0"),
        ProteinRecord("p2", "EF", np.array([0, 1], dtype=np.int8), "1"),
    ]


def test_verified_cache_joins_labels_without_storing_them() -> None:
    records = example_records()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_cache(root, records)
        cache = load_verified_cache(root, records, "locked-revision")
        features, labels = load_known_residue_tensors(records, cache)

    assert features.shape == (4, 3)
    assert features.dtype == torch.float16
    assert labels.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert cache.hidden_size == 3


def test_verified_cache_rejects_sequence_mismatch() -> None:
    records = example_records()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_cache(root, records)
        changed = [
            ProteinRecord("p1", "AAA", records[0].labels, "0"),
            records[1],
        ]
        with pytest.raises(ValueError, match="sequence/cache mismatch"):
            load_verified_cache(root, changed, "locked-revision")


def test_cached_head_outputs_one_logit_per_residue() -> None:
    model = CachedIdrHead(hidden_size=3, dropout=0.0)
    logits = model(torch.zeros((7, 3), dtype=torch.float32))
    assert logits.shape == (7,)
    assert model.classifier.out_features == 1
