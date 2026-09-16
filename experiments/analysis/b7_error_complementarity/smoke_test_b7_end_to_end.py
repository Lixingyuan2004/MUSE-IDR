"""End-to-end synthetic smoke test for the B7 command-line program."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).with_name("build_b7.py")
MODELS = (
    "A10-locked",
    "LoRA-DR-Suite 650M",
    "PUNCH2-Light Paper-8",
    "PUNCH2-Light Released-13",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_prediction(path: Path, score_shift: float) -> None:
    records = {
        "p1": ("ACDE", [0.10, 0.30, 0.70, 0.90]),
        "p2": ("FGHIKL", [0.20, 0.40, 0.65, 0.75, 0.35, 0.15]),
    }
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("protein_id", "position", "residue", "idr_probability"),
            delimiter="\t",
        )
        writer.writeheader()
        for protein_id, (sequence, scores) in records.items():
            for position, (residue, score) in enumerate(zip(sequence, scores), start=1):
                writer.writerow(
                    {
                        "protein_id": protein_id,
                        "position": position,
                        "residue": residue,
                        "idr_probability": min(0.99, max(0.01, score + score_shift)),
                    }
                )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="b7_e2e_") as temporary:
        root = Path(temporary)
        reference = root / "reference.fasta"
        reference.write_text(">p1\nACDE\n0011\n>p2\nFGHIKL\n0-1110\n", encoding="utf-8")
        predictions: dict[str, dict[str, str]] = {}
        for index, model in enumerate(MODELS):
            path = root / f"model_{index}.tsv.gz"
            write_prediction(path, 0.01 * index)
            predictions[model] = {"path": str(path), "sha256": sha256(path)}
        reference_entry = {"path": str(reference), "sha256": sha256(reference)}
        manifest = {
            "schema_version": 1,
            "status": "locked",
            "purpose": "synthetic_b7_smoke",
            "models": list(MODELS),
            "comparators": list(MODELS[1:]),
            "datasets": {
                dataset: {
                    "references": {
                        "disorder_nox": reference_entry,
                        "disorder_pdb": reference_entry,
                    },
                    "predictions": predictions,
                }
                for dataset in ("CAID2", "CAID3")
            },
            "protocol": {
                "analysis_type": "descriptive_post_lock",
                "decision_threshold": 0.5,
                "protein_length_bins": [200, 500, 1000],
                "known_label_disorder_fraction_bins": [0.1, 0.3, 0.6],
                "transition_distance_bins": [2, 5, 15],
                "true_segment_length_bins": [15, 30, 100],
                "confirmatory_inference_source": "B6",
                "new_hypothesis_tests": False,
            },
            "model_selection_completed_before_caid_label_access": True,
            "caid_labels_used_for_training_or_tuning": False,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        output_dir = root / "output"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads((output_dir / "b7_report.json").read_text(encoding="utf-8"))
        if report["status"] != "pass" or not report["all_input_hashes_match"]:
            raise AssertionError("end-to-end report did not pass")
        if report["row_counts"]["point_metrics"] != 16:
            raise AssertionError("end-to-end point-metric row count is wrong")
        if report["row_counts"]["per_protein_metrics"] != 32:
            raise AssertionError("end-to-end per-protein row count is wrong")
        if report["new_hypothesis_tests_performed"] is not False:
            raise AssertionError("B7 unexpectedly performed hypothesis tests")
        expected = {
            "b7_point_metrics.csv",
            "b7_per_protein_metrics.csv.gz",
            "b7_subgroup_metrics.csv",
            "b7_boundary_metrics.csv",
            "b7_segment_metrics.csv",
            "b7_pairwise_complementarity.csv",
            "b7_cross_dataset_robustness.csv",
            "b7_summary.md",
            "b7_report.json",
        }
        if {path.name for path in output_dir.iterdir()} != expected:
            raise AssertionError("end-to-end output set is incomplete")
        result = {
            "status": "pass",
            "four_tracks_processed": True,
            "four_locked_models_processed": True,
            "all_input_hashes_verified_before_analysis": True,
            "point_metric_rows": report["row_counts"]["point_metrics"],
            "per_protein_metric_rows": report["row_counts"]["per_protein_metrics"],
            "all_nine_output_files_written": True,
            "new_hypothesis_tests_performed": False,
            "caid_labels_used_for_training_or_tuning": False,
            "child_stdout_lines": len(completed.stdout.splitlines()),
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
