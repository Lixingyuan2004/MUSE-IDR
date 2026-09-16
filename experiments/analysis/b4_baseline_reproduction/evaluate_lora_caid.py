"""Evaluate locked LoRA-DR-Suite predictions on CAID NOX and PDB tracks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b1_caid3_fair_comparison.build_b1 import (  # noqa: E402
    align_method,
    compute_metrics,
    parse_caid_predictions,
    parse_reference,
    sha256,
)
from experiments.analysis.b4_baseline_reproduction.run_lora_official_smoke import (  # noqa: E402
    ALLOWED_MODEL_IDS,
)


def evaluate_track(reference_path: Path, prediction_text: str) -> dict[str, Any]:
    reference = parse_reference(reference_path.read_text(encoding="utf-8-sig"))
    predictions = parse_caid_predictions(prediction_text)
    missing = sorted(set(reference) - set(predictions))
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} reference proteins")
    y_true, y_score, _, alignment = align_method(reference, predictions)
    if (
        alignment["protein_coverage"] != 1.0
        or alignment["residue_coverage"] != 1.0
    ):
        raise ValueError("LoRA evaluation requires complete protein/residue coverage")
    if alignment["substantive_residue_mismatches"] != 0:
        raise ValueError("prediction/reference residue mismatch detected")
    if np.unique(y_true).tolist() != [0, 1]:
        raise ValueError("track does not contain both known classes")
    return {
        **alignment,
        **compute_metrics(y_true, y_score),
        "reference": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "masked_labels_excluded": True,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("caid2", "caid3"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inference-report", type=Path, required=True)
    parser.add_argument("--nox-reference", type=Path, required=True)
    parser.add_argument("--pdb-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference = json.loads(args.inference_report.read_text(encoding="utf-8-sig"))
    prediction_sha256 = sha256(args.predictions)
    if inference.get("status") != "pass":
        raise ValueError("LoRA inference report did not pass")
    if inference.get("model_id") not in ALLOWED_MODEL_IDS:
        raise ValueError("inference used a non-DisProt7 LoRA checkpoint")
    if inference.get("training_regime") != "DisProt_7_only":
        raise ValueError("inference report is not the DisProt7-only baseline")
    if inference.get("caid_output_sha256") != prediction_sha256:
        raise ValueError("prediction hash does not match inference report")
    if inference.get("prediction_alignment") is not True:
        raise ValueError("inference report does not prove residue alignment")
    if inference.get("lora_and_saved_head_loaded") is not True:
        raise ValueError("inference report does not prove official adapter/head loading")
    if inference.get("caid_labels_accessed") is not False:
        raise ValueError("CAID labels were accessed during prediction")
    if inference.get("scores_used_for_training_or_tuning") is not False:
        raise ValueError("external scores were used for model tuning")

    prediction_text = args.predictions.read_text(encoding="utf-8-sig")
    nox = evaluate_track(args.nox_reference, prediction_text)
    pdb = evaluate_track(args.pdb_reference, prediction_text)
    report = {
        "schema_version": 1,
        "experiment": f"b4_lora_dr_suite_official_{args.dataset}_evaluation",
        "status": "pass",
        "dataset": args.dataset,
        "model_id": inference["model_id"],
        "model_revision": inference["model_revision"],
        "base_model_id": inference["base_model_id"],
        "base_model_revision": inference["base_model_revision"],
        "training_regime": "DisProt_7_only",
        "prediction_file": str(args.predictions),
        "prediction_file_sha256": prediction_sha256,
        "inference_report": str(args.inference_report),
        "inference_report_sha256": sha256(args.inference_report),
        "tracks": {"disorder_nox": nox, "disorder_pdb": pdb},
        "model_locked_before_label_access": True,
        "single_output_score": True,
        "soft_disorder_output": False,
        "caid_labels_used_for_training_or_tuning": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
