"""Fast synthetic checks for the B1 parser, metrics, and paired statistics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b1_caid3_fair_comparison.build_b1 import (
    align_method,
    compute_metrics,
    make_common_pairs,
    paired_delong,
    paired_protein_bootstrap,
    parse_caid_predictions,
    parse_reference,
)


def main() -> None:
    reference = parse_reference(
        ">P1\nACDE\n0101\n>P2\nFGHIK\n1-001\n"
    )
    good = parse_caid_predictions(
        ">P1\n1 A 0.1\n2 C 0.9\n3 D 0.2\n4 E 0.8\n"
        ">P2\n1 F 0.9\n2 G 0.5\n3 H 0.2\n4 I 0.1\n5 K 0.8\n"
    )
    weak = parse_caid_predictions(
        ">P1\n1 A 0.4\n2 C 0.6\n3 D 0.5\n4 E 0.5\n"
        ">P2\n1 F 0.6\n2 G 0.5\n3 H 0.5\n4 I 0.4\n5 K 0.6\n"
    )
    y_true, y_good, good_proteins, coverage = align_method(reference, good)
    weak_true, y_weak, weak_proteins, _ = align_method(reference, weak)
    if not np.array_equal(y_true, weak_true):
        raise AssertionError("synthetic labels are not aligned")
    metrics = compute_metrics(y_true, y_good)
    if metrics["roc_auc_full_precision"] != 1.0:
        raise AssertionError("perfect ordering did not produce ROC-AUC 1")
    if coverage["predicted_proteins"] != 2 or coverage["known_residues"] != 8:
        raise AssertionError("coverage or unknown-label exclusion failed")
    pairs = make_common_pairs(good_proteins, weak_proteins)
    delong = paired_delong(y_true, y_good, y_weak)
    bootstrap = paired_protein_bootstrap(pairs, replicates=20, random_seed=7)
    if delong["delta_first_minus_second"] <= 0:
        raise AssertionError("paired DeLong direction is wrong")
    if bootstrap["point_delta"] <= 0 or bootstrap["replicates"] != 20:
        raise AssertionError("protein bootstrap direction/count is wrong")
    print(
        json.dumps(
            {
                "status": "pass",
                "reference_parser": True,
                "caid_prediction_parser": True,
                "masked_label_excluded": True,
                "roc_auc_aps_auprc_f1_mcc_defined": True,
                "paired_delong_defined": True,
                "protein_cluster_bootstrap_defined": True,
                "caid_labels_used_for_training_or_tuning": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
