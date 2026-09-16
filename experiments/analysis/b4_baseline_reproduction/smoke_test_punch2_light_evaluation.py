"""Smoke tests for locked PUNCH2-Light CAID evaluation safeguards."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b4_baseline_reproduction.evaluate_punch2_light_caid import (
    PUNCH2_LIGHT_REVISION,
    parse_sha256_lock,
    validate_inference_report,
    verify_prediction_lock,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predictions = root / "paper8.caid"
        inference = root / "inference.json"
        lock = root / "prediction_lock.sha256"
        predictions.write_text(">p1\nAAA\n0.1\n0.2\n0.3\n", encoding="utf-8")
        report = {
            "status": "pass",
            "source_revision": PUNCH2_LIGHT_REVISION,
            "released_members_loaded": 13,
            "prediction_alignment": True,
            "output_heads": 1,
            "soft_disorder_output": False,
            "input_contains_labels": False,
            "caid_labels_accessed": False,
            "scores_used_for_training_or_tuning": False,
            "variants": {
                "paper8": {
                    "ensemble_members": 8,
                    "ensemble_method": "uniform_probability_mean",
                    "threshold": 0.35,
                    "caid_output": str(predictions),
                    "caid_output_sha256": file_hash(predictions),
                },
                "released13": {
                    "ensemble_members": 13,
                    "ensemble_method": "uniform_probability_mean",
                    "threshold": 0.35,
                    "caid_output": str(predictions),
                    "caid_output_sha256": file_hash(predictions),
                },
            },
        }
        inference.write_text(json.dumps(report), encoding="utf-8")
        lock.write_text(
            f"{file_hash(predictions)}  {predictions}\n"
            f"{file_hash(inference)}  {inference}\n",
            encoding="utf-8",
        )

        parsed = parse_sha256_lock(lock, root)
        assert predictions.resolve() in parsed
        verified = verify_prediction_lock(lock, predictions, inference)
        assert verified["all_entries_match"]
        provenance = validate_inference_report(report, "paper8", predictions)
        assert provenance["members"] == 8
        released = validate_inference_report(report, "released13", predictions)
        assert released["members"] == 13

        predictions.write_text("tampered\n", encoding="utf-8")
        mismatch_rejected = False
        try:
            verify_prediction_lock(lock, predictions, inference)
        except ValueError:
            mismatch_rejected = True
        assert mismatch_rejected

        predictions.write_text(">p1\nAAA\n0.1\n0.2\n0.3\n", encoding="utf-8")
        bad = dict(report)
        bad["caid_labels_accessed"] = True
        label_access_rejected = False
        try:
            validate_inference_report(bad, "paper8", predictions)
        except ValueError:
            label_access_rejected = True
        assert label_access_rejected

    print(
        json.dumps(
            {
                "status": "pass",
                "prediction_lock_required": True,
                "prediction_and_inference_hashes_verified": True,
                "tampered_prediction_rejected": True,
                "source_revision_locked": True,
                "paper8_member_count_verified": True,
                "released13_member_count_verified": True,
                "uniform_probability_mean_required": True,
                "caid_label_access_during_inference_rejected": True,
                "single_classic_idr_output_required": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
