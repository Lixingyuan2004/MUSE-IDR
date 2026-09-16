from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.final_models.a10_locked_inference.evaluate_caid3 import (  # noqa: E402
    align_reference,
    auc_roc_caid_compatible,
    auc_roc_full_precision,
    read_predictions,
)
from experiments.final_models.a10_locked_inference.prepare_caid_references import (  # noqa: E402
    parse_reference,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prediction_path = root / "predictions.tsv.gz"
        with gzip.open(prediction_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["protein_id", "position", "residue", "idr_probability"])
            writer.writerows(
                [
                    ["p1", 1, "A", 0.05],
                    ["p1", 2, "C", 0.95],
                    ["p1", 3, "D", 0.30],
                    ["p2", 1, "M", 0.85],
                    ["p2", 2, "J", 0.15],
                ]
            )
        reference_path = root / "reference.fasta"
        reference_path.write_text(
            ">p1\nACD\n01-\n>p2\nMJ\n10\n", encoding="utf-8", newline="\n"
        )
        predictions = read_predictions(prediction_path)
        records = parse_reference(reference_path)
        y_true, y_score, summary = align_reference(records, predictions)
        result = {
            "status": "pass",
            "prediction_alignment": summary["prediction_alignment"],
            "masked_label_excluded": summary["excluded_labels"] == 1 and y_true.size == 4,
            "both_classes_present": np.unique(y_true).tolist() == [0, 1],
            "full_precision_auc_correct": auc_roc_full_precision(y_true, y_score) == 1.0,
            "caid_compatible_auc_correct": auc_roc_caid_compatible(y_true, y_score) == 1.0,
        }
        if not all(value for key, value in result.items() if key != "status"):
            result["status"] = "fail"
            print(json.dumps(result, indent=2))
            raise AssertionError("CAID3 evaluator smoke test failed")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
