"""Strictly assemble five LightCNN validation folds into one OOF evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from muse_idr.data.caid import file_sha256
from muse_idr.evaluation import (
    ResiduePrediction,
    evaluate_predictions,
    load_aligned_jsonl,
    load_fold_assignments,
    validate_and_order_oof_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_oof_files(output_dir: Path, records: list[ResiduePrediction]) -> None:
    with (output_dir / "oof_reference.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as reference_handle, (output_dir / "oof_predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as prediction_handle:
        for record in records:
            reference_handle.write(
                json.dumps(
                    {
                        "protein_id": record.protein_id,
                        "sequence": record.sequence,
                        "labels": list(record.labels),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            prediction_handle.write(
                json.dumps(
                    {"protein_id": record.protein_id, "scores": list(record.scores)},
                    separators=(",", ":"),
                )
                + "\n"
            )


def _protocol_signature(manifest: dict[str, object]) -> dict[str, object]:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("Run manifest is missing config")
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("Run config is missing data")
    common_data = {key: value for key, value in data.items() if key != "validation_fold"}
    return {
        "data": common_data,
        "model": config.get("model"),
        "training": config.get("training"),
        "data_sha256": manifest.get("data_sha256"),
        "fold_assignments_sha256": manifest.get("fold_assignments_sha256"),
        "readiness_manifest_sha256": manifest.get("readiness_manifest_sha256"),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = args.assignments.resolve()
    assignments = load_fold_assignments(assignments_path)
    assignments_sha256 = file_sha256(assignments_path)
    records_by_fold: dict[int, list[ResiduePrediction]] = {}
    fold_summaries: list[dict[str, object]] = []
    signatures: list[dict[str, object]] = []
    for run_dir_value in args.run_dirs:
        run_dir = run_dir_value.resolve()
        manifest = _load_json(run_dir / "run_manifest.json")
        if manifest.get("output_heads") != 1:
            raise ValueError(f"Run is not single-output: {run_dir}")
        if manifest.get("caid2_caid3_labels_accessed") is not False:
            raise ValueError(f"Run does not certify CAID2/3 isolation: {run_dir}")
        partitions = manifest.get("partitions")
        optimization = manifest.get("optimization")
        if not isinstance(partitions, dict) or not isinstance(optimization, dict):
            raise ValueError(f"Incomplete run manifest: {run_dir}")
        fold = partitions.get("validation_fold")
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError(f"Invalid validation fold in {run_dir}")
        if fold in records_by_fold:
            raise ValueError(f"Duplicate validation fold {fold}")
        records_by_fold[fold] = load_aligned_jsonl(
            run_dir / "validation_reference.jsonl",
            run_dir / "validation_predictions.jsonl",
        )
        signatures.append(_protocol_signature(manifest))
        fold_summaries.append(
            {
                "fold": fold,
                "run_dir": run_dir.as_posix(),
                "validation_proteins": partitions.get("validation_proteins"),
                "best_epoch": optimization.get("best_epoch"),
                "epochs_executed": optimization.get("epochs_executed"),
                "residue_micro_roc_auc": manifest.get(
                    "best_validation_residue_micro_roc_auc"
                ),
            }
        )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Fold runs do not share one frozen data/model/training protocol")
    if signatures[0]["fold_assignments_sha256"] != assignments_sha256:
        raise ValueError("Provided fold assignments do not match the frozen run protocol")

    records = validate_and_order_oof_records(records_by_fold, assignments)
    _write_oof_files(output_dir, records)
    metrics = evaluate_predictions(
        records,
        threshold=0.5,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    fold_summaries.sort(key=lambda row: int(row["fold"]))
    fold_aucs = [float(row["residue_micro_roc_auc"]) for row in fold_summaries]
    metrics["cross_validation"] = {
        "folds": fold_summaries,
        "fold_auc_mean": statistics.mean(fold_aucs),
        "fold_auc_sample_standard_deviation": statistics.stdev(fold_aucs),
        "pooled_oof_auc_is_primary": True,
    }
    _write_json(output_dir / "oof_metrics.json", metrics)
    training_signature = signatures[0]["training"]
    if not isinstance(training_signature, dict):
        raise ValueError("Protocol signature is missing training settings")
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_stage": "formal_internal_fivefold_single_seed",
        "task": "single_output_classic_idr",
        "output_heads": 1,
        "seed": training_signature.get("seed"),
        "folds": sorted(records_by_fold),
        "proteins": len(records),
        "assignments_path": assignments_path.as_posix(),
        "assignments_sha256": assignments_sha256,
        "protocol_signature": signatures[0],
        "fold_summaries": fold_summaries,
        "metrics_sha256": file_sha256(output_dir / "oof_metrics.json"),
        "reference_sha256": file_sha256(output_dir / "oof_reference.jsonl"),
        "predictions_sha256": file_sha256(output_dir / "oof_predictions.jsonl"),
        "caid2_caid3_labels_accessed": False,
    }
    _write_json(output_dir / "oof_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "folds": sorted(records_by_fold),
                "proteins": len(records),
                "pooled_oof_auc": metrics["metrics"]["residue_micro_roc_auc"],
                "fold_auc_mean": metrics["cross_validation"]["fold_auc_mean"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
