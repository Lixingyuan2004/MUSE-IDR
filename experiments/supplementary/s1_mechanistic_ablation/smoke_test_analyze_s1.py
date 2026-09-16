"""Numerical smoke tests for the locked S1 statistical analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.supplementary.s1_mechanistic_ablation.analyze_s1 import (  # noqa: E402
    ProteinBlock,
    _bootstrap_chunk,
    _init_worker,
    bootstrap_summary,
    equal_logit_mean,
    holm_adjust,
    roc_auc_full_precision,
)


def main() -> None:
    seed_probabilities = [
        np.asarray([0.2, 0.8], dtype=np.float64),
        np.asarray([0.4, 0.6], dtype=np.float64),
        np.asarray([0.5, 0.5], dtype=np.float64),
    ]
    expected_logit_mean = 1.0 / (
        1.0
        + np.exp(
            -np.mean(
                np.stack(
                    [
                        np.log(values / (1.0 - values))
                        for values in seed_probabilities
                    ],
                    axis=0,
                ),
                axis=0,
            )
        )
    )
    observed_logit_mean = equal_logit_mean(seed_probabilities)
    if not np.allclose(observed_logit_mean, expected_logit_mean, atol=1e-15):
        raise AssertionError("equal-logit ensemble calculation is incorrect")

    labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.35, 0.8, 0.4, 0.7], dtype=np.float64)
    observed_auc = roc_auc_full_precision(labels, scores)
    expected_auc = float(roc_auc_score(labels, scores))
    if not np.isclose(observed_auc, expected_auc, atol=1e-15):
        raise AssertionError("full-precision ROC-AUC does not match sklearn")

    adjusted = holm_adjust([0.01, 0.04, 0.03])
    if not np.allclose(adjusted, [0.03, 0.06, 0.06], atol=1e-15):
        raise AssertionError(f"unexpected Holm adjustment: {adjusted}")

    model_count = 9
    blocks: list[ProteinBlock] = []
    for fold in range(5):
        for protein_index in range(3):
            block_labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
            base = np.asarray([0.1, 0.9, 0.3, 0.7], dtype=np.float64)
            columns = []
            for model_index in range(model_count):
                offset = (model_index - 4) * 0.002 + protein_index * 0.001
                columns.append(np.clip(base + offset, 1e-6, 1.0 - 1e-6))
            blocks.append(
                ProteinBlock(
                    protein_id=f"fold{fold}_protein{protein_index}",
                    fold=str(fold),
                    labels=block_labels,
                    scores=np.stack(columns, axis=1),
                )
            )

    fold_indices = tuple(
        np.asarray(
            [index for index, block in enumerate(blocks) if block.fold == str(fold)],
            dtype=np.int64,
        )
        for fold in range(5)
    )
    _init_worker(blocks, fold_indices, 20260904)
    first_start, first = _bootstrap_chunk(0, 12)
    second_start, second = _bootstrap_chunk(0, 12)
    if first_start != 0 or second_start != 0 or not np.array_equal(first, second):
        raise AssertionError("bootstrap is not reproducible for a fixed seed")
    if first.shape != (12, model_count):
        raise AssertionError(f"unexpected bootstrap shape: {first.shape}")

    summary = bootstrap_summary(0.1, np.asarray([0.05, 0.1, 0.15, 0.2]))
    required_summary_keys = {
        "mean_delta",
        "percentile_95_ci",
        "probability_delta_positive",
        "two_sided_empirical_p_plus_one_correction",
    }
    if not required_summary_keys.issubset(summary):
        raise AssertionError("bootstrap summary is incomplete")

    print(
        json.dumps(
            {
                "status": "pass",
                "equal_logit_mean_verified": True,
                "full_precision_auc_matches_sklearn": True,
                "holm_adjustment_verified": True,
                "fold_stratified_bootstrap_reproducible": True,
                "bootstrap_models": model_count,
                "caid_labels_accessed": False,
                "locked_a10_modified": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
