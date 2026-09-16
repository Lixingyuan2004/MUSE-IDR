"""Run a tiny end-to-end A3 fold using a synthetic verified cache."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a1_frozen_esm2.train_cached_a1 import (  # noqa: E402
    load_verified_cache,
)
from experiments.ablations.a2_multiscale_context.smoke_test_a2_end_to_end import (  # noqa: E402
    make_records,
    write_verified_cache,
)
from experiments.ablations.a3_auc_ranking_loss.train_a3 import run_fold  # noqa: E402


def main() -> None:
    records = make_records()
    config = {
        "experiment": {"name": "a3_synthetic_smoke"},
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
        "loss": {
            "name": "weighted_bce_plus_sampled_pairwise_auc",
            "rank_weight": 0.2,
            "rank_margin": 0.0,
            "max_pairs_per_batch": 64,
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
            raise RuntimeError("A3 end-to-end smoke created multiple output heads")
        if report["metrics"]["micro_roc_auc"] is None:
            raise RuntimeError("A3 end-to-end smoke did not produce ROC-AUC")
        if not all(row["train_rank_pairs"] > 0 for row in report["history"]):
            raise RuntimeError("A3 training did not use any ranking pairs")
        if report["caid2_caid3_labels_accessed"]:
            raise RuntimeError("A3 smoke unexpectedly accessed CAID labels")
        for record in validation_records:
            if predictions[record.protein_id].shape != record.labels.shape:
                raise RuntimeError(f"prediction shape mismatch for {record.protein_id}")
        required = [
            root / "outputs" / "fold_1" / "best_head.pt",
            root / "outputs" / "fold_1" / "metrics.json",
            root / "outputs" / "fold_1" / "predictions.csv",
        ]
        if not all(path.is_file() for path in required):
            raise RuntimeError("A3 end-to-end smoke did not write all fold artifacts")

    print(
        json.dumps(
            {
                "status": "pass",
                "fold_training": True,
                "ranking_pairs_used": True,
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
