"""Tests for the locked S2 label-side evaluator."""

from __future__ import annotations

import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.supplementary.s2_independent_holdout.evaluate_s2_a10 import (
    ProteinBlock,
    align_blocks,
    bootstrap_samples,
    extended_metrics,
    read_labels,
    read_predictions,
    verify_lock_manifest,
)


class S2EvaluationTests(unittest.TestCase):
    def test_tsv_readers_and_unknown_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            label_path = root / "labels.tsv.gz"
            prediction_path = root / "predictions.tsv.gz"
            with gzip.open(label_path, "wt", encoding="utf-8", newline="") as handle:
                handle.write(
                    "protein_id\tposition\tresidue\tlabel\n"
                    "P1\t1\tA\t0\nP1\t2\tC\t-1\nP1\t3\tD\t1\n"
                )
            with gzip.open(
                prediction_path, "wt", encoding="utf-8", newline=""
            ) as handle:
                handle.write(
                    "protein_id\tposition\tresidue\tidr_probability\n"
                    "P1\t1\tA\t0.1\nP1\t2\tC\t0.5\nP1\t3\tD\t0.9\n"
                )
            labels = read_labels(label_path)
            predictions = read_predictions(prediction_path)
            blocks = align_blocks(
                labels,
                predictions,
                {"P1": {"length": "3", "source_entry_release": "2026_06"}},
            )
            known = blocks[0].labels != -1
            metric = extended_metrics(
                blocks[0].labels[known], blocks[0].scores[known]
            )
            self.assertEqual(int(known.sum()), 2)
            self.assertAlmostEqual(metric["roc_auc_full_precision"], 1.0)
            self.assertAlmostEqual(metric["f1_at_0_5_full_precision"], 1.0)

    def test_residue_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sequence mismatch"):
            align_blocks(
                {"P1": ("A", np.asarray([0], dtype=np.int8))},
                {"P1": ("C", np.asarray([0.2], dtype=np.float64))},
                {"P1": {"length": "1", "source_entry_release": "2026_06"}},
            )

    def test_tampered_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.txt"
            target.write_text("locked\n", encoding="utf-8")
            lock = root / "lock.sha256"
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            lock.write_text(f"{digest}  target.txt\n", encoding="utf-8")
            verify_lock_manifest(lock, root)
            target.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_lock_manifest(lock, root)

    def test_protein_bootstrap_is_reproducible(self) -> None:
        blocks = [
            ProteinBlock(
                "P1",
                "ACDE",
                np.asarray([0, 0, 1, 1], dtype=np.int8),
                np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
                "2026_06",
            ),
            ProteinBlock(
                "P2",
                "FGHI",
                np.asarray([0, 1, 0, 1], dtype=np.int8),
                np.asarray([0.3, 0.7, 0.4, 0.6], dtype=np.float64),
                "2026_06",
            ),
        ]
        first = bootstrap_samples(blocks, 8, 20260907, 0)
        second = bootstrap_samples(blocks, 8, 20260907, 0)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
