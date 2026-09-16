"""Evaluate locked PUNCH2-Light predictions on CAID2 or CAID3.

This script is intentionally separate from the LoRA evaluator.  It verifies the
pre-label-access prediction lock and the PUNCH2-Light inference provenance
before reading either CAID reference file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from experiments.analysis.b4_baseline_reproduction.evaluate_lora_caid import (
    evaluate_track as evaluate_caid_track,
)


PUNCH2_LIGHT_REVISION = "6c7935b3597c056d2e6b3845bb54fc101c5bc574"
VARIANTS = {
    "paper8": {
        "members": 8,
        "definition": "paper_faithful_8_member_uniform_probability_mean",
    },
    "released13": {
        "members": 13,
        "definition": "released_entry_point_13_member_uniform_probability_mean",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("caid2", "caid3"), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inference-report", type=Path, required=True)
    parser.add_argument("--prediction-lock", type=Path, required=True)
    parser.add_argument("--nox-reference", type=Path, required=True)
    parser.add_argument("--pdb-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_sha256_lock(path: Path, workspace: Path) -> dict[Path, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries: dict[Path, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid SHA256 lock line {line_number}: {raw_line!r}")
        expected, name = parts
        name = name.lstrip(" *")
        candidate = Path(name)
        resolved = candidate if candidate.is_absolute() else workspace / candidate
        resolved = resolved.resolve()
        if resolved in entries:
            raise ValueError(f"duplicate SHA256 lock entry: {name}")
        entries[resolved] = expected.lower()
    if not entries:
        raise ValueError("prediction lock is empty")
    return entries


def verify_prediction_lock(
    lock_path: Path,
    predictions: Path,
    inference_report: Path,
) -> dict[str, Any]:
    workspace = Path.cwd().resolve()
    entries = parse_sha256_lock(lock_path, workspace)
    required = (predictions.resolve(), inference_report.resolve())
    missing = [str(path) for path in required if path not in entries]
    if missing:
        raise ValueError(f"prediction lock does not cover required files: {missing}")

    mismatches: list[str] = []
    for path, expected in entries.items():
        if not path.is_file():
            mismatches.append(f"missing:{path}")
            continue
        observed = sha256(path)
        if observed != expected:
            mismatches.append(f"sha256:{path}")
    if mismatches:
        raise ValueError(f"prediction lock verification failed: {mismatches}")
    return {
        "path": str(lock_path),
        "sha256": sha256(lock_path),
        "entries": len(entries),
        "all_entries_match": True,
        "predictions_covered": True,
        "inference_report_covered": True,
    }


def validate_inference_report(
    report: dict[str, Any],
    variant: str,
    predictions: Path,
) -> dict[str, Any]:
    expected = VARIANTS[variant]
    if report.get("status") != "pass":
        raise ValueError("inference report status is not pass")
    if report.get("source_revision") != PUNCH2_LIGHT_REVISION:
        raise ValueError("unexpected PUNCH2-Light source revision")
    if int(report.get("released_members_loaded", -1)) != 13:
        raise ValueError("all 13 released checkpoints were not loaded")
    if report.get("prediction_alignment") is not True:
        raise ValueError("inference report does not confirm prediction alignment")
    if int(report.get("output_heads", -1)) != 1:
        raise ValueError("expected one classic-IDR output head")
    if report.get("soft_disorder_output") is not False:
        raise ValueError("soft-disorder output is not allowed")
    if report.get("input_contains_labels") is not False:
        raise ValueError("inference input must not contain labels")
    if report.get("caid_labels_accessed") is not False:
        raise ValueError("CAID labels were accessed during inference")
    if report.get("scores_used_for_training_or_tuning") is not False:
        raise ValueError("scores were used for training or tuning")

    variants = report.get("variants")
    if not isinstance(variants, dict) or variant not in variants:
        raise ValueError(f"inference report is missing variant {variant}")
    row = variants[variant]
    if int(row.get("ensemble_members", -1)) != expected["members"]:
        raise ValueError(f"unexpected member count for {variant}")
    if row.get("ensemble_method") != "uniform_probability_mean":
        raise ValueError(f"unexpected ensemble method for {variant}")
    if abs(float(row.get("threshold", -1.0)) - 0.35) > 1e-12:
        raise ValueError(f"unexpected auxiliary threshold for {variant}")

    reported_prediction = row.get("caid_output")
    if reported_prediction is None:
        raise ValueError(f"missing CAID output path for {variant}")
    reported_path = Path(reported_prediction)
    if reported_path.resolve() != predictions.resolve():
        raise ValueError("prediction path does not match the inference report")
    reported_hash = row.get("caid_output_sha256")
    if reported_hash != sha256(predictions):
        raise ValueError("prediction hash does not match the inference report")

    return {
        "source_revision": report["source_revision"],
        "released_members_loaded": report["released_members_loaded"],
        "variant": variant,
        "variant_definition": expected["definition"],
        "members": expected["members"],
        "ensemble_method": "uniform_probability_mean",
        "auxiliary_threshold": 0.35,
        "prediction_alignment": True,
        "input_contains_labels": False,
        "caid_labels_accessed_during_inference": False,
        "scores_used_for_training_or_tuning": False,
    }


def main() -> None:
    args = parse_args()
    lock = verify_prediction_lock(
        args.prediction_lock,
        args.predictions,
        args.inference_report,
    )
    inference_report = json.loads(
        args.inference_report.read_text(encoding="utf-8-sig")
    )
    provenance = validate_inference_report(
        inference_report,
        args.variant,
        args.predictions,
    )

    # References are deliberately read only after both lock and provenance pass.
    prediction_text = args.predictions.read_text(encoding="utf-8-sig")
    tracks = {
        "disorder_nox": evaluate_caid_track(args.nox_reference, prediction_text),
        "disorder_pdb": evaluate_caid_track(args.pdb_reference, prediction_text),
    }
    result = {
        "schema_version": 1,
        "experiment": "b4_punch2_light_locked_caid_evaluation",
        "status": "pass",
        "dataset": args.dataset,
        "model": "PUNCH2-Light",
        "variant": args.variant,
        "prediction_lock": lock,
        "inference_provenance": provenance,
        "prediction": str(args.predictions),
        "prediction_sha256": sha256(args.predictions),
        "inference_report": str(args.inference_report),
        "inference_report_sha256": sha256(args.inference_report),
        "tracks": tracks,
        "model_selection_completed_before_label_access": True,
        "single_output_score": True,
        "soft_disorder_output": False,
        "caid_labels_used_for_training_or_tuning": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
