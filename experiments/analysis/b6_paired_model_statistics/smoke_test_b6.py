"""Fast synthetic checks for the B6 paired-statistics pipeline."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b1_caid3_fair_comparison.build_b1 import (
    Prediction,
    align_method,
    compute_metrics,
    parse_reference,
)
from experiments.analysis.b6_paired_model_statistics.build_b6 import (
    MODEL_ORDER,
    ProteinBlock,
    bootstrap_metric_samples,
    bootstrap_summary,
    full_precision_metric_vector,
    holm_adjust,
    parse_prediction_tsv,
    verify_locked_file,
)


def write_prediction(path: Path, position_name: str) -> None:
    text = (
        f"protein_id\t{position_name}\tresidue\tidr_probability\n"
        "P1\t1\tA\t0.1\n"
        "P1\t2\tC\t0.9\n"
        "P1\t3\tD\t0.5\n"
        "P1\t4\tE\t0.8\n"
    )
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="b6_smoke_") as temp_dir:
        temp = Path(temp_dir)
        position_path = temp / "position.tsv"
        residue_index_path = temp / "residue_index.tsv.gz"
        write_prediction(position_path, "position")
        write_prediction(residue_index_path, "residue_index")
        first = parse_prediction_tsv(position_path)
        second = parse_prediction_tsv(residue_index_path)
        assert first["P1"].sequence == second["P1"].sequence == "ACDE"
        assert np.array_equal(first["P1"].scores, second["P1"].scores)

        reference = parse_reference(">P1\nACDE\n01-1\n")
        labels, scores, _, coverage = align_method(reference, first)
        assert labels.tolist() == [0, 1, 1]
        assert scores.tolist() == [0.1, 0.9, 0.8]
        assert coverage["known_residues"] == 3

        expected = compute_metrics(labels, scores)
        observed = full_precision_metric_vector(labels, scores)
        expected_vector = np.asarray(
            [
                expected["roc_auc_full_precision"],
                expected["aucpr_trapezoid_full_precision"],
                expected["aps_full_precision"],
                expected["f1_at_0_5_full_precision"],
                expected["mcc_at_0_5_full_precision"],
                expected["fmax_full_precision"],
            ]
        )
        assert np.allclose(observed, expected_vector, atol=1e-12)

        content_hash = hashlib.sha256(position_path.read_bytes()).hexdigest()
        assert verify_locked_file(position_path, content_hash)["match"] is True
        position_path.write_text(position_path.read_text() + "\n", encoding="utf-8")
        try:
            verify_locked_file(position_path, content_hash)
        except ValueError:
            tampered_rejected = True
        else:
            tampered_rejected = False
        assert tampered_rejected

    blocks = []
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    model_scores = (
        np.asarray([0.05, 0.95, 0.10, 0.90]),
        np.asarray([0.20, 0.80, 0.70, 0.60]),
        np.asarray([0.35, 0.65, 0.45, 0.55]),
        np.asarray([0.25, 0.75, 0.35, 0.65]),
    )
    for index in range(6):
        blocks.append(
            ProteinBlock(
                protein_id=f"P{index}",
                labels=labels.copy(),
                scores=tuple(score.copy() for score in model_scores),
            )
        )
    samples = bootstrap_metric_samples(
        blocks, replicates=32, workers=1, random_seed=20260826, progress_every=0
    )
    rerun = bootstrap_metric_samples(
        blocks, replicates=32, workers=1, random_seed=20260826, progress_every=0
    )
    assert samples.shape == (32, len(MODEL_ORDER), 6)
    assert np.array_equal(samples, rerun)
    roc_delta = samples[:, 0, 0] - samples[:, 1, 0]
    summary = bootstrap_summary(float(roc_delta[0]), roc_delta)
    assert summary["probability_delta_positive"] == 1.0
    assert summary["percentile_95_ci"][0] > 0.0

    raw_p = [0.01, 0.04, 0.03]
    adjusted = holm_adjust(raw_p)
    assert adjusted == [0.03, 0.06, 0.06]

    print(
        json.dumps(
            {
                "status": "pass",
                "position_and_residue_index_tsv_supported": True,
                "unknown_labels_excluded": True,
                "full_precision_metrics_match_b1": True,
                "whole_protein_paired_bootstrap": True,
                "bootstrap_reproducible": True,
                "holm_correction_defined": True,
                "tampered_input_rejected": True,
                "caid_labels_used_for_training_or_tuning": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
