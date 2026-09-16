"""Tests for prediction-blind S2 pre-freeze selection helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.supplementary.s2_independent_holdout.prepare_prefreeze_cohort import (  # noqa: E402
    choose_representative,
    connected_components,
    local_similarity,
)


class PrefreezeTests(unittest.TestCase):
    def test_connected_components_are_undirected(self):
        self.assertEqual(
            connected_components(["A", "B", "C"], [("B", "A")]),
            [["A", "B"], ["C"]],
        )

    def test_representative_uses_annotation_completeness(self):
        records = {
            "A": {"labels": [1, -1]},
            "B": {"labels": [1, 0]},
        }
        self.assertEqual(choose_representative(["A", "B"], records), "B")

    def test_local_rule_detects_short_high_identity_alignment(self):
        row = {"fident": "0.55", "alnlen": "35", "qcov": "0.1", "tcov": "0.2"}
        self.assertTrue(local_similarity(row))

    def test_weak_short_alignment_is_not_flagged(self):
        row = {"fident": "0.25", "alnlen": "20", "qcov": "0.1", "tcov": "0.1"}
        self.assertFalse(local_similarity(row))


if __name__ == "__main__":
    unittest.main()
