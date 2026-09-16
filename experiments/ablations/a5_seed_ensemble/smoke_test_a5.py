from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a5_seed_ensemble.ensemble_a2 import (
    build_untuned_methods,
    compute_metrics,
    logits,
    search_weights,
)


def main() -> None:
    protein_ids = ["p1"] * 6 + ["p2"] * 6
    labels = np.asarray([0, 0, 1, 1, -1, 1, 0, 1, 0, 1, 1, -1], dtype=np.int8)
    probabilities = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.8, 0.7, 0.6],
            [0.9, 0.8, 0.7],
            [0.7, 0.2, 0.9],
            [0.6, 0.9, 0.8],
            [0.1, 0.3, 0.2],
            [0.7, 0.8, 0.6],
            [0.4, 0.2, 0.3],
            [0.8, 0.6, 0.9],
            [0.9, 0.7, 0.8],
            [0.3, 0.9, 0.2],
        ],
        dtype=np.float64,
    )
    methods = build_untuned_methods(probabilities)
    metrics = compute_metrics(
        protein_ids, labels, methods["uniform_probability_mean"]
    )
    best_auc, weights, candidates = search_weights(
        labels, logits(probabilities), step=0.25
    )
    result = {
        "status": "pass",
        "three_aligned_inputs": probabilities.shape[1] == 3,
        "untuned_methods": sorted(methods),
        "unknown_labels_excluded": metrics["known_residues"] == 10,
        "macro_auc_proteins": metrics["macro_auc_proteins"],
        "weights_sum_to_one": bool(np.isclose(weights.sum(), 1.0)),
        "weight_candidates_bounded": candidates == 15,
        "finite_auc": bool(np.isfinite(best_auc)),
        "single_output_score": all(score.ndim == 1 for score in methods.values()),
        "caid2_caid3_labels_accessed": False,
    }
    if not (
        result["three_aligned_inputs"]
        and result["unknown_labels_excluded"]
        and result["macro_auc_proteins"] == 2
        and result["weights_sum_to_one"]
        and result["weight_candidates_bounded"]
        and result["finite_auc"]
        and result["single_output_score"]
    ):
        result["status"] = "fail"
        print(json.dumps(result, indent=2))
        raise AssertionError("A5 smoke test failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
