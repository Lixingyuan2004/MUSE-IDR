"""Run a tiny end-to-end A6 fold using a verified synthetic cache."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a1_frozen_esm2.train_cached_a1 import load_verified_cache
from experiments.ablations.a2_multiscale_context.smoke_test_a2_end_to_end import (
    make_records,
    write_verified_cache,
)
from experiments.ablations.a6_long_range_dilated_context.train_a6 import run_fold


def main() -> None:
    records = make_records()
    config = {
        "experiment": {"name": "a6_synthetic_smoke"},
        "model": {
            "input_size": 16,
            "hidden_size": 8,
            "kernels": [3, 5],
            "dilations": [1, 2, 4],
            "dropout": 0.0,
            "dilated_dropout": 0.0,
            "residual_gate_init": -2.0,
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
            raise RuntimeError("A6 end-to-end smoke created more than one output head")
        if report["metrics"]["micro_roc_auc"] is None:
            raise RuntimeError("A6 end-to-end smoke did not produce ROC-AUC")
        if report["maximum_input_receptive_field"] != 19:
            raise RuntimeError("A6 synthetic receptive-field metadata is incorrect")
        for record in validation_records:
            if predictions[record.protein_id].shape != record.labels.shape:
                raise RuntimeError(f"prediction shape mismatch for {record.protein_id}")
        required = [
            root / "outputs" / "fold_1" / "best_head.pt",
            root / "outputs" / "fold_1" / "metrics.json",
            root / "outputs" / "fold_1" / "predictions.csv",
        ]
        if not all(path.is_file() for path in required):
            raise RuntimeError("A6 end-to-end smoke did not write all fold artifacts")

    print(
        json.dumps(
            {
                "status": "pass",
                "fold_training": True,
                "long_range_stack_trained": True,
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
