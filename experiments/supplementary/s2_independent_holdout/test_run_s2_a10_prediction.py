"""Unit tests for the label-blind S2 A10 prediction driver."""
from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from run_s2_a10_prediction import sha256, validate_inference_report, validate_predictions


class S2A10PredictionTests(unittest.TestCase):
    def test_prediction_alignment_and_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.tsv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["protein_id", "position", "residue", "idr_probability"])
                writer.writerow(["P1", 1, "A", 0.25])
                writer.writerow(["P1", 2, "C", 0.75])
            result = validate_predictions(path, [("P1", "AC")])
            self.assertEqual(result["prediction_rows"], 2)
            self.assertEqual(result["minimum_probability"], 0.25)
            self.assertEqual(result["maximum_probability"], 0.75)

    def test_alignment_error_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.tsv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["protein_id", "position", "residue", "idr_probability"])
                writer.writerow(["P1", 1, "G", 0.5])
            with self.assertRaises(ValueError):
                validate_predictions(path, [("P1", "A")])

    def test_inference_report_requires_locked_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "input.fasta"
            output = root / "output.tsv.gz"
            fasta.write_text(">P1\nA\n", encoding="utf-8")
            output.write_bytes(b"prediction")
            report = {
                "status": "pass",
                "locked_members": 29,
                "a2_members": 15,
                "a8_members": 15,
                "proteins": 87,
                "residues": 56394,
                "input_contains_labels": False,
                "all_checkpoint_hashes_match": True,
                "development_checkpoint_excluded": True,
                "caid1_caid2_caid3_labels_accessed": False,
                "input_fasta_sha256": sha256(fasta),
                "output_sha256": sha256(output),
            }
            with self.assertRaises(ValueError):
                validate_inference_report(report, fasta, output)


if __name__ == "__main__":
    unittest.main()
