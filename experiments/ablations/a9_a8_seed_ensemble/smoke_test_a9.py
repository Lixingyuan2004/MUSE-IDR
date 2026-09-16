from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a9_a8_seed_ensemble.common import SEEDS, compute_metrics
from experiments.ablations.a9_a8_seed_ensemble.ensemble_a8 import build_ensemble


def write_seed(root: Path, seed: int, rows: list[tuple[str, int, int, float]]) -> None:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    prediction_path = seed_dir / "oof_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "position", "label", "probability"])
        writer.writerows(rows)
    metrics = compute_metrics(
        [row[0] for row in rows],
        np.asarray([row[2] for row in rows], dtype=np.int8),
        np.asarray([row[3] for row in rows], dtype=np.float64),
    )
    with (seed_dir / "oof_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"oof_metrics": metrics}, handle)


def main() -> None:
    base_rows = [
        ("p1", 0, 0, 0.10),
        ("p1", 1, 1, 0.80),
        ("p1", 2, -1, 0.99),
        ("p2", 0, 0, 0.20),
        ("p2", 1, 1, 0.75),
        ("p2", 2, 0, 0.30),
        ("p2", 3, 1, 0.90),
    ]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        input_root = root / "inputs"
        for column, seed in enumerate(SEEDS):
            shift = (column - 1) * 0.02
            rows = [
                (protein, position, label, min(0.999, max(0.001, score + shift)))
                for protein, position, label, score in base_rows
            ]
            write_seed(input_root, seed, rows)
        manifest = root / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"protein_id": "p1", "fold": 0}) + "\n")
            handle.write(json.dumps({"protein_id": "p2", "fold": 1}) + "\n")
        output_dir = root / "output"
        report = build_ensemble(
            input_root=input_root,
            manifest=manifest,
            output_dir=output_dir,
            coarse_step=0.25,
            fine_step=0.25,
            fine_radius=1.0,
        )
        result = {
            "status": "pass",
            "three_inputs_aligned": report["alignment_verified"],
            "known_labels_only_in_metrics": report["known_residues"] == 6,
            "formal_method": report["selected_for_future_external_test"]["method"],
            "equal_weights": bool(
                np.allclose(
                    report["selected_for_future_external_test"]["weights"],
                    [1.0 / 3.0] * 3,
                )
            ),
            "single_output_score": report["single_output_score"],
            "soft_disorder_output": report["soft_disorder_output"],
            "prediction_file_written": (output_dir / "a9_oof_predictions.csv.gz").is_file(),
            "caid2_caid3_labels_accessed": False,
        }
        passed = (
            result["three_inputs_aligned"]
            and result["known_labels_only_in_metrics"]
            and result["formal_method"] == "uniform_logit_mean"
            and result["equal_weights"]
            and result["single_output_score"]
            and not result["soft_disorder_output"]
            and result["prediction_file_written"]
        )
        if not passed:
            result["status"] = "fail"
            print(json.dumps(result, indent=2))
            raise AssertionError("A9 smoke test failed")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
