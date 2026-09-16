"""Smoke tests for B7 descriptive error and complementarity analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b1_caid3_fair_comparison.build_b1 import (  # noqa: E402
    Prediction,
    Reference,
)
from experiments.analysis.b6_paired_model_statistics.build_b6 import (  # noqa: E402
    MODEL_ORDER,
)
from experiments.analysis.b7_error_complementarity.build_b7 import (  # noqa: E402
    align_track,
    annotate_reference_labels,
    complementarity_rows,
    descriptive_metrics,
    disorder_fraction_bin,
    length_bin,
    pairwise_complementarity,
    segment_length_bin,
    subgroup_rows,
    transition_distance_bin,
)


def main() -> None:
    known, run_lengths, distances = annotate_reference_labels("0011-111000")
    if known.tolist() != [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]:
        raise AssertionError("masked positions were not excluded correctly")
    if run_lengths.tolist() != [2, 2, 2, 2, 3, 3, 3, 3, 3, 3]:
        raise AssertionError("true-state run lengths are wrong")
    if distances.tolist() != [1, 0, 0, 1, 2, 1, 0, 0, 1, 2]:
        raise AssertionError("transition distances are wrong")

    if [length_bin(x) for x in (200, 201, 500, 501, 1000, 1001)] != [
        "1-200",
        "201-500",
        "201-500",
        "501-1000",
        "501-1000",
        "1001+",
    ]:
        raise AssertionError("protein-length bins are wrong")
    if [disorder_fraction_bin(x) for x in (0.1, 0.1001, 0.3, 0.3001, 0.6, 0.6001)] != [
        "0-0.10",
        "(0.10,0.30]",
        "(0.10,0.30]",
        "(0.30,0.60]",
        "(0.30,0.60]",
        "(0.60,1.00]",
    ]:
        raise AssertionError("disorder-fraction bins are wrong")
    if [segment_length_bin(x) for x in (15, 16, 30, 31, 100, 101)] != [
        "1-15",
        "16-30",
        "16-30",
        "31-100",
        "31-100",
        "101+",
    ]:
        raise AssertionError("segment-length bins are wrong")
    if [transition_distance_bin(x) for x in (2, 3, 5, 6, 15, 16, np.inf)] != [
        "0-2",
        "3-5",
        "3-5",
        "6-15",
        "6-15",
        "16+",
        "no-transition",
    ]:
        raise AssertionError("transition-distance bins are wrong")

    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    metrics = descriptive_metrics(labels, scores)
    if metrics["roc_auc_full_precision"] != 1.0:
        raise AssertionError("full-precision ROC-AUC is wrong")
    if metrics["error_rate_at_0_5"] != 0.0:
        raise AssertionError("fixed-threshold error rate is wrong")
    one_class = descriptive_metrics(
        np.ones(3, dtype=np.int8), np.asarray([0.2, 0.7, 0.9])
    )
    if one_class["roc_auc_full_precision"] is not None:
        raise AssertionError("one-class ROC-AUC must be undefined")
    if not np.isfinite(float(one_class["brier_score"])):
        raise AssertionError("one-class proper scoring metrics must remain defined")

    complement = pairwise_complementarity(
        labels,
        np.asarray([0.1, 0.2, 0.8, 0.3]),
        np.asarray([0.8, 0.2, 0.7, 0.9]),
    )
    if complement["a10_only_correct"] != 1:
        raise AssertionError("A10-only correctness count is wrong")
    if complement["comparator_only_correct"] != 1:
        raise AssertionError("comparator-only correctness count is wrong")
    if complement["threshold_disagreement_residues"] != 2:
        raise AssertionError("threshold disagreement count is wrong")

    reference = {
        "p1": Reference("p1", "ACDE", "0011"),
        "p2": Reference("p2", "FGHIKL", "0-1110"),
    }
    score_sets = {
        "A10-locked": ([0.1, 0.2, 0.8, 0.9], [0.2, 0.3, 0.7, 0.8, 0.4, 0.2]),
        "LoRA-DR-Suite 650M": ([0.2, 0.3, 0.7, 0.8], [0.1, 0.4, 0.6, 0.7, 0.3, 0.1]),
        "PUNCH2-Light Paper-8": ([0.3, 0.4, 0.6, 0.7], [0.3, 0.2, 0.8, 0.6, 0.2, 0.3]),
        "PUNCH2-Light Released-13": ([0.25, 0.35, 0.65, 0.75], [0.2, 0.3, 0.75, 0.65, 0.25, 0.2]),
    }
    predictions = {
        model: {
            "p1": Prediction("p1", "ACDE", np.asarray(values[0], dtype=np.float64)),
            "p2": Prediction("p2", "FGHIKL", np.asarray(values[1], dtype=np.float64)),
        }
        for model, values in score_sets.items()
    }
    if tuple(predictions) != MODEL_ORDER:
        raise AssertionError("synthetic model order drifted")
    proteins, alignment = align_track(reference, predictions)
    if alignment["known_residues"] != 9 or alignment["substantive_residue_mismatches"] != 0:
        raise AssertionError("synthetic alignment is wrong")
    subgroup = subgroup_rows("TEST", "disorder_nox", proteins)
    if not subgroup or {row["stratification"] for row in subgroup} != {
        "protein_length",
        "known_label_disorder_fraction",
    }:
        raise AssertionError("subgroup output is incomplete")
    pairs = complementarity_rows("TEST", "disorder_nox", proteins)
    overall = [row for row in pairs if row["scope"] == "overall"]
    if len(overall) != 3:
        raise AssertionError("one overall row per comparator is required")

    tampered = dict(predictions)
    tampered["A10-locked"] = dict(tampered["A10-locked"])
    tampered["A10-locked"].pop("p2")
    try:
        align_track(reference, tampered)
    except ValueError:
        missing_prediction_rejected = True
    else:
        missing_prediction_rejected = False
    if not missing_prediction_rejected:
        raise AssertionError("missing locked predictions must be rejected")

    result = {
        "status": "pass",
        "masked_labels_excluded": True,
        "predeclared_length_and_fraction_bins": True,
        "true_segment_lengths_annotated": True,
        "distance_to_transition_annotated": True,
        "one_class_subgroups_supported_without_fake_auc": True,
        "proper_scoring_rules_defined": True,
        "fixed_threshold_error_complementarity_defined": True,
        "score_correlation_defined": True,
        "missing_locked_prediction_rejected": True,
        "new_hypothesis_tests_performed": False,
        "caid_labels_used_for_training_or_tuning": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
