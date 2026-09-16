from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_a1_manifest import build_manifest


class A1ManifestTests(unittest.TestCase):
    def test_frozen_dataset_and_fold_assignment_are_joined_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            assignments = root / "assignments.jsonl"
            output = root / "a1.jsonl"
            dataset.write_text(
                json.dumps(
                    {"protein_id": "p1", "sequence": "ACD", "labels": [1, -1, 0]}
                )
                + "\n",
                encoding="utf-8",
            )
            assignments.write_text(
                json.dumps(
                    {
                        "protein_id": "p1",
                        "fold": 4,
                        "sequence_length": 3,
                        "positive_residues": 1,
                        "negative_residues": 1,
                        "unknown_residues": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_manifest(dataset, assignments, output)
            row = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(row["id"], "p1")
        self.assertEqual(row["fold"], "4")
        self.assertEqual(row["labels"], [1, -1, 0])
        self.assertEqual(report["proteins"], 1)
        self.assertEqual(report["positive_residues"], 1)
        self.assertEqual(report["negative_residues"], 1)
        self.assertEqual(report["unknown_residues"], 1)
        self.assertFalse(report["caid2_caid3_labels_accessed"])


if __name__ == "__main__":
    unittest.main()
