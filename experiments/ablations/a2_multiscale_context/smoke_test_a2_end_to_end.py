"""Run a tiny end-to-end A2 fold without downloading a protein language model."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a1_frozen_esm2.data import ProteinRecord
from experiments.ablations.a1_frozen_esm2.train_cached_a1 import (
    file_sha256,
    load_verified_cache,
)
from experiments.ablations.a2_multiscale_context.train_a2 import run_fold


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def make_records() -> list[ProteinRecord]:
    rows = [
        ("train_a", "ACDEFGH", [0, 0, 1, 1, -1, 0, 1], "0"),
        ("train_b", "IKLMNPQR", [1, 1, 0, 0, 1, -1, 0, 1], "0"),
        ("train_c", "STVWYACDE", [0, 1, 0, 1, 0, 1, -1, 0, 1], "0"),
        ("valid_a", "FGHIKLM", [0, 1, 1, 0, -1, 0, 1], "1"),
        ("valid_b", "NPQRSTVW", [1, 0, 1, 0, 1, -1, 0, 1], "1"),
    ]
    return [
        ProteinRecord(
            protein_id,
            sequence,
            np.asarray(labels, dtype=np.int8),
            fold,
        )
        for protein_id, sequence, labels, fold in rows
    ]


def write_verified_cache(
    cache_dir: Path, records: list[ProteinRecord], hidden_size: int
) -> None:
    embedding_dir = cache_dir / "embeddings"
    embedding_dir.mkdir(parents=True)
    rows = []
    for index, record in enumerate(records):
        relative = f"embeddings/{index}.npy"
        generator = np.random.default_rng(index + 1)
        embedding = generator.normal(
            size=(len(record.sequence), hidden_size)
        ).astype(np.float16)
        np.save(cache_dir / relative, embedding, allow_pickle=False)
        rows.append(
            {
                "protein_id": record.protein_id,
                "sequence_sha256": sequence_hash(record.sequence),
                "sequence_length": len(record.sequence),
                "fold": record.fold,
                "hidden_size": hidden_size,
                "model_revision": "synthetic-locked-revision",
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


def main() -> None:
    records = make_records()
    config = {
        "experiment": {"name": "a2_synthetic_smoke"},
        "model": {
            "input_size": 16,
            "hidden_size": 8,
            "kernels": [3, 5],
            "dropout": 0.0,
        },
        "training": {
            "epochs": 2,
            "patience": 2,
            "protein_batch_size": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "num_workers": 0,
            "class_balance": True,
        },
        "evaluation": {"unknown_label": -1},
    }

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cache_dir = root / "cache"
        write_verified_cache(cache_dir, records, hidden_size=16)
        cache = load_verified_cache(
            cache_dir, records, expected_revision="synthetic-locked-revision"
        )
        report, validation_records, predictions = run_fold(
            "1",
            records,
            cache,
            config,
            seed=17,
            output_root=root / "outputs",
            device=torch.device("cpu"),
        )
        if report["output_heads"] != 1:
            raise RuntimeError("A2 end-to-end smoke created more than one output head")
        if report["metrics"]["micro_roc_auc"] is None:
            raise RuntimeError("A2 end-to-end smoke did not produce ROC-AUC")
        for record in validation_records:
            if predictions[record.protein_id].shape != record.labels.shape:
                raise RuntimeError(f"prediction shape mismatch for {record.protein_id}")
        required = [
            root / "outputs" / "fold_1" / "best_head.pt",
            root / "outputs" / "fold_1" / "metrics.json",
            root / "outputs" / "fold_1" / "predictions.csv",
        ]
        if not all(path.is_file() for path in required):
            raise RuntimeError("A2 end-to-end smoke did not write all fold artifacts")

    print(
        json.dumps(
            {
                "status": "pass",
                "fold_training": True,
                "checkpoint_written": True,
                "prediction_alignment": True,
                "micro_roc_auc_defined": True,
                "output_heads": 1,
                "caid2_caid3_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
