"""Dependency-light smoke test for the verified-cache A1 training path."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from data import ProteinRecord
from train_cached_a1 import (
    CachedIdrHead,
    file_sha256,
    load_known_residue_tensors,
    load_verified_cache,
)


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def main() -> None:
    records = [
        ProteinRecord("p1", "ACD", np.array([1, -1, 0], dtype=np.int8), "0"),
        ProteinRecord("p2", "EF", np.array([0, 1], dtype=np.int8), "1"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        cache_dir = Path(directory)
        embedding_dir = cache_dir / "embeddings"
        embedding_dir.mkdir()
        rows = []
        for index, record in enumerate(records):
            relative = f"embeddings/{index}.npy"
            np.save(
                cache_dir / relative,
                np.ones((len(record.sequence), 3), dtype=np.float16),
                allow_pickle=False,
            )
            rows.append(
                {
                    "protein_id": record.protein_id,
                    "sequence_sha256": sequence_hash(record.sequence),
                    "sequence_length": len(record.sequence),
                    "fold": record.fold,
                    "hidden_size": 3,
                    "model_revision": "locked-revision",
                    "embedding_file": relative,
                }
            )
        index_path = cache_dir / "cache_index.jsonl"
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
        (cache_dir / "cache_verification_report.json").write_text(
            json.dumps(verification), encoding="utf-8"
        )

        cache = load_verified_cache(cache_dir, records, "locked-revision")
        features, labels = load_known_residue_tensors(records, cache)
        if features.shape != (4, 3) or labels.tolist() != [1.0, 0.0, 0.0, 1.0]:
            raise RuntimeError("known-residue cache join failed")
        logits = CachedIdrHead(3, 0.0)(features.float())
        if logits.shape != (4,):
            raise RuntimeError("single-output head shape failed")

    print(
        json.dumps(
            {
                "status": "pass",
                "verified_cache_join": True,
                "unknown_labels_excluded": True,
                "output_heads": 1,
                "output_scores": int(logits.numel()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
