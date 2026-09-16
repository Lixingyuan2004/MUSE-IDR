"""Tests for the locked three-model S2 evaluation."""
from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.supplementary.s2_independent_holdout.evaluate_s2_three_models import (
    MODEL_ORDER,
    PRIMARY_METRIC,
    SECONDARY_METRIC,
    MultiProteinBlock,
    bootstrap_samples,
    pairwise_rows,
    point_metric_rows,
    read_prediction,
)


class PredictionReaderTests(unittest.TestCase):
    def write_prediction(self, position_name: str, positions: list[int]) -> Path:
        root = Path(self.tempdir.name)
        path = root / f"{position_name}.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["protein_id", position_name, "residue", "idr_probability"]
            )
            for position, residue, score in zip(
                positions, "ACD", (0.1, 0.7, 0.2), strict=True
            ):
                writer.writerow(["P1", position, residue, score])
        return path

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_accepts_position_and_residue_index(self) -> None:
        one_based = read_prediction(self.write_prediction("position", [1, 2, 3]))
        zero_based = read_prediction(
            self.write_prediction("residue_index", [0, 1, 2])
        )
        self.assertEqual(one_based["P1"][0], "ACD")
        np.testing.assert_array_equal(one_based["P1"][1], zero_based["P1"][1])

    def test_rejects_non_contiguous_positions(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            read_prediction(self.write_prediction("position", [1, 3, 4]))


class PairedBootstrapTests(unittest.TestCase):
    @staticmethod
    def blocks() -> list[MultiProteinBlock]:
        return [
            MultiProteinBlock(
                "P1",
                4,
                np.asarray([0, 1, 0, 1], dtype=np.int8),
                (
                    np.asarray([0.05, 0.95, 0.15, 0.85]),
                    np.asarray([0.15, 0.80, 0.30, 0.70]),
                    np.asarray([0.25, 0.75, 0.20, 0.65]),
                ),
                "2025_12",
            ),
            MultiProteinBlock(
                "P2",
                4,
                np.asarray([1, 0, 1, 0], dtype=np.int8),
                (
                    np.asarray([0.85, 0.10, 0.90, 0.20]),
                    np.asarray([0.70, 0.20, 0.75, 0.35]),
                    np.asarray([0.65, 0.30, 0.80, 0.25]),
                ),
                "2026_03",
            ),
            MultiProteinBlock(
                "P3",
                4,
                np.asarray([0, 1, 1, 0], dtype=np.int8),
                (
                    np.asarray([0.20, 0.80, 0.75, 0.10]),
                    np.asarray([0.30, 0.70, 0.65, 0.25]),
                    np.asarray([0.35, 0.60, 0.70, 0.20]),
                ),
                "2026_06",
            ),
        ]

    def test_bootstrap_is_reproducible(self) -> None:
        first = bootstrap_samples(self.blocks(), 25, 1, 20260907, 0)
        second = bootstrap_samples(self.blocks(), 25, 1, 20260907, 0)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape[1], len(MODEL_ORDER))

    def test_only_primary_roc_contrasts_receive_holm_p(self) -> None:
        blocks = self.blocks()
        points = point_metric_rows(blocks)
        samples = bootstrap_samples(blocks, 25, 1, 20260907, 0)
        rows = pairwise_rows(points, samples)
        primary = [row for row in rows if row["metric"] == PRIMARY_METRIC]
        secondary = [row for row in rows if row["metric"] == SECONDARY_METRIC]
        self.assertEqual(len(primary), 2)
        self.assertEqual(len(secondary), 2)
        self.assertTrue(all(row["holm_two_sided_p"] is not None for row in primary))
        self.assertTrue(all(row["holm_two_sided_p"] is None for row in secondary))
        self.assertTrue(all(row["two_sided_empirical_p"] is None for row in secondary))


if __name__ == "__main__":
    unittest.main()
