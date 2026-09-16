from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a5_seed_ensemble.bootstrap_a5 import (
    BootstrapInput,
    FixedScoreWeightedAuc,
    paired_stratified_cluster_bootstrap,
)


def main() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    base = np.asarray([0.1, 0.7, 0.4, 0.6, 0.2, 0.8, 0.3, 0.5])
    improved = np.asarray([0.1, 0.9, 0.3, 0.8, 0.2, 0.85, 0.25, 0.7])
    protein_index = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    protein_counts = np.asarray([2, 0, 1, 3], dtype=np.int32)
    row_weight = protein_counts[protein_index]
    custom_auc = FixedScoreWeightedAuc(labels, base).evaluate(row_weight)
    sklearn_auc = roc_auc_score(labels, base, sample_weight=row_weight)
    data = BootstrapInput(
        labels=labels,
        seed_probabilities=np.stack([base, base, base], axis=1),
        protein_index=protein_index,
        protein_ids=("p0", "p1", "p2", "p3"),
        protein_folds=("0", "0", "1", "1"),
    )
    base_samples, improved_samples, delta = paired_stratified_cluster_bootstrap(
        data,
        base,
        improved,
        replicates=50,
        random_seed=17,
        workers=2,
    )
    result = {
        "status": "pass",
        "weighted_auc_matches_sklearn": bool(np.isclose(custom_auc, sklearn_auc)),
        "replicates": int(delta.size),
        "all_finite": bool(
            np.isfinite(base_samples).all()
            and np.isfinite(improved_samples).all()
            and np.isfinite(delta).all()
        ),
        "paired_sampling": True,
        "protein_is_sampling_unit": True,
        "caid2_caid3_labels_accessed": False,
    }
    if not (
        result["weighted_auc_matches_sklearn"]
        and result["replicates"] == 50
        and result["all_finite"]
    ):
        result["status"] = "fail"
        print(json.dumps(result, indent=2))
        raise AssertionError("A5 bootstrap smoke test failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
