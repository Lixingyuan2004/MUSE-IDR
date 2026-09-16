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

from experiments.ablations.a10_cross_architecture_ensemble.ensemble_a2_a8 import (
    build_ensemble,
)
from experiments.ablations.a9_a8_seed_ensemble.common import SEEDS, compute_metrics


def write_family(root: Path, family_shift: float) -> None:
    rows = [
        ("p1", 0, 0, 0.10), ("p1", 1, 1, 0.80), ("p1", 2, -1, 0.99),
        ("p2", 0, 0, 0.20), ("p2", 1, 1, 0.75),
        ("p2", 2, 0, 0.30), ("p2", 3, 1, 0.90),
    ]
    for index, seed in enumerate(SEEDS):
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        shifted = [
            (p, pos, label, min(0.999, max(0.001, score + family_shift + (index - 1) * 0.01)))
            for p, pos, label, score in rows
        ]
        with (seed_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["protein_id", "position", "label", "probability"])
            writer.writerows(shifted)
        metrics = compute_metrics(
            [row[0] for row in shifted],
            np.asarray([row[2] for row in shifted], dtype=np.int8),
            np.asarray([row[3] for row in shifted], dtype=np.float64),
        )
        with (seed_dir / "oof_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump({"oof_metrics": metrics}, handle)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        a2_root = root / "a2"
        a8_root = root / "a8"
        write_family(a2_root, 0.0)
        write_family(a8_root, 0.015)
        manifest = root / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"protein_id": "p1", "fold": 0}) + "\n")
            handle.write(json.dumps({"protein_id": "p2", "fold": 1}) + "\n")
        output = root / "output"
        report = build_ensemble(
            a2_root=a2_root,
            a8_root=a8_root,
            manifest=manifest,
            output_dir=output,
        )
        result = {
            "status": "pass",
            "six_models_aligned": report["alignment_verified_across_all_six_models"],
            "known_labels_only_in_metrics": report["known_residues"] == 6,
            "formal_weights_are_label_free": not report["formal_candidate"]["weights_fitted_to_labels"],
            "six_equal_weights": bool(np.allclose(report["formal_candidate"]["model_weights"], [1.0 / 6.0] * 6)),
            "single_output_score": report["single_output_score"],
            "soft_disorder_output": report["soft_disorder_output"],
            "prediction_file_written": (output / "a10_oof_predictions.csv.gz").is_file(),
            "caid2_caid3_labels_accessed": False,
        }
        passed = (
            result["six_models_aligned"]
            and result["known_labels_only_in_metrics"]
            and result["formal_weights_are_label_free"]
            and result["six_equal_weights"]
            and result["single_output_score"]
            and not result["soft_disorder_output"]
            and result["prediction_file_written"]
        )
        if not passed:
            result["status"] = "fail"
            print(json.dumps(result, indent=2))
            raise AssertionError("A10 smoke test failed")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
