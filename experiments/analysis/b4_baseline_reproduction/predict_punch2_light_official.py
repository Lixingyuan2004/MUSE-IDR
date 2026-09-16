from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from punch2_light_common import (
    DEFAULT_PUNCH2_LIGHT_REPO,
    EXPECTED_PUNCH2_LIGHT_COMMIT,
    ensemble_probabilities,
    load_official_members,
    paper8_member_ids,
    predict_member_probabilities,
    read_sequence_only_fasta,
    sha256,
    validate_feature_array,
)


VARIANT_THRESHOLDS = {"paper8": 0.35, "released13": 0.35}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_tsv(path: Path, records, predictions: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("protein_id", "residue_index", "residue", "idr_probability"))
        for record in records:
            values = predictions[record.protein_id]
            for index, (residue, probability) in enumerate(
                zip(record.sequence, values, strict=True),
                start=1,
            ):
                writer.writerow((record.protein_id, index, residue, f"{probability:.10f}"))


def write_caid(path: Path, records, predictions: dict[str, np.ndarray], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f">{record.protein_id}\n")
            for index, (residue, probability) in enumerate(
                zip(record.sequence, predictions[record.protein_id], strict=True),
                start=1,
            ):
                rounded = round(float(probability), 3)
                label = 1 if rounded > threshold else 0
                handle.write(f"{index}\t{residue}\t{rounded:.3f}\t{label}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_PUNCH2_LIGHT_REPO)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("paper8", "released13"),
        default=("paper8", "released13"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    records = read_sequence_only_fasta(args.fasta)
    feature_report = json.loads(args.feature_report.read_text(encoding="utf-8-sig"))
    if feature_report.get("status") != "pass":
        raise ValueError("feature cache report did not pass")
    if feature_report.get("input_fasta_sha256") != sha256(args.fasta):
        raise ValueError("feature cache FASTA hash mismatch")
    if feature_report.get("input_contains_labels") is not False:
        raise ValueError("feature cache does not prove sequence-only input")
    if feature_report.get("caid_labels_accessed") is not False:
        raise ValueError("feature cache accessed CAID labels")

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    specs, models, checkpoint_hashes = load_official_members(args.repo, device)
    outputs: dict[str, dict[str, np.ndarray]] = {
        variant: {} for variant in args.variants
    }
    residue_count = 0

    for index, record in enumerate(records, start=1):
        onehot = validate_feature_array(
            np.load(args.feature_dir / "onehot" / f"{record.protein_id}.npy", allow_pickle=False),
            len(record.sequence),
            21,
            "onehot",
        )
        prottrans = validate_feature_array(
            np.load(args.feature_dir / "protTrans" / f"{record.protein_id}.npy", allow_pickle=False),
            len(record.sequence),
            1024,
            "protTrans",
        )
        member_predictions = predict_member_probabilities(
            specs,
            models,
            onehot,
            prottrans,
            device,
        )
        for variant in args.variants:
            outputs[variant][record.protein_id] = ensemble_probabilities(
                member_predictions,
                variant,
            )
        residue_count += len(record.sequence)
        print(
            json.dumps(
                {
                    "protein": index,
                    "total": len(records),
                    "protein_id": record.protein_id,
                    "length": len(record.sequence),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    variant_reports: dict[str, object] = {}
    for variant in args.variants:
        tsv_path = args.output_dir / f"punch2_light_{variant}_predictions.tsv.gz"
        caid_path = args.output_dir / f"punch2_light_{variant}.caid"
        write_tsv(tsv_path, records, outputs[variant])
        write_caid(caid_path, records, outputs[variant], VARIANT_THRESHOLDS[variant])
        all_values = np.concatenate(list(outputs[variant].values()))
        variant_reports[variant] = {
            "ensemble_members": 8 if variant == "paper8" else 13,
            "ensemble_method": "uniform_probability_mean",
            "threshold": VARIANT_THRESHOLDS[variant],
            "output": str(tsv_path),
            "output_sha256": sha256(tsv_path),
            "caid_output": str(caid_path),
            "caid_output_sha256": sha256(caid_path),
            "output_scores": int(all_values.size),
            "score_min": float(all_values.min()),
            "score_max": float(all_values.max()),
        }

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "experiment": "b4_punch2_light_official_external_inference",
        "status": "pass",
        "source_repository": "https://github.com/deemeng/punch2_light",
        "source_revision": EXPECTED_PUNCH2_LIGHT_COMMIT,
        "input_fasta": str(args.fasta),
        "input_fasta_sha256": sha256(args.fasta),
        "input_contains_labels": False,
        "feature_report": str(args.feature_report),
        "feature_report_sha256": sha256(args.feature_report),
        "proteins": len(records),
        "residues": residue_count,
        "released_members_loaded": len(specs),
        "paper8_member_ids": sorted(paper8_member_ids()),
        "checkpoint_sha256": checkpoint_hashes,
        "variants": variant_reports,
        "prediction_alignment": all(
            sum(len(values) for values in outputs[variant].values()) == residue_count
            for variant in args.variants
        ),
        "output_heads": 1,
        "output_definition": "classic_idr_probability_per_residue",
        "soft_disorder_output": False,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "caid_labels_accessed": False,
        "caid_labels_used_for_training_or_tuning": False,
        "scores_used_for_training_or_tuning": False,
    }
    if not report["prediction_alignment"]:
        raise RuntimeError("PUNCH2-Light prediction alignment failed")
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
