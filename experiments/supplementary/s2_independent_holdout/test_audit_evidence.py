"""Unit tests for conservative S2 evidence classification."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.supplementary.s2_independent_holdout.audit_evidence import (  # noqa: E402
    classify_disorder_region,
    union_positions,
)


def region(**updates):
    value = {
        "start": 3,
        "end": 7,
        "ec_id": "ECO:0006165",
        "released": "2026_06",
        "validated": {"curator": "reviewed"},
        "unpublished": False,
    }
    value.update(updates)
    return value


class EvidenceAuditTests(unittest.TestCase):
    def test_direct_validated_evidence_is_eligible(self):
        decision = classify_disorder_region(region(), 10)
        self.assertTrue(decision["provisional_direct_evidence"])
        self.assertTrue(decision["formal_eligible"])

    def test_prediction_is_never_positive(self):
        decision = classify_disorder_region(region(ec_id="ECO:0008035"), 10)
        self.assertFalse(decision["provisional_direct_evidence"])
        self.assertIn("prediction_based_author_inference", decision["reasons"])

    def test_missing_coordinates_are_never_direct_positive(self):
        decision = classify_disorder_region(region(ec_id="ECO:0006220"), 10)
        self.assertFalse(decision["provisional_direct_evidence"])

    def test_obsolete_and_unvalidated_are_rejected(self):
        decision = classify_disorder_region(region(obsolete={"reason": "wrong"}, validated=None), 10)
        self.assertFalse(decision["provisional_direct_evidence"])
        self.assertIn("obsolete_annotation", decision["reasons"])
        self.assertIn("curator_validation_not_recorded", decision["reasons"])

    def test_construct_alteration_requires_review(self):
        decision = classify_disorder_region(region(construct_alterations=[{"type": "mutation"}]), 10)
        self.assertFalse(decision["provisional_direct_evidence"])

    def test_unpublished_flag_blocks_formal_not_provisional(self):
        decision = classify_disorder_region(region(unpublished=True), 10)
        self.assertTrue(decision["provisional_direct_evidence"])
        self.assertFalse(decision["formal_eligible"])

    def test_union_is_residue_inclusive_and_deduplicated(self):
        self.assertEqual(union_positions([(2, 4), (4, 6)], 10), {2, 3, 4, 5, 6})

    def test_invalid_interval_is_rejected(self):
        decision = classify_disorder_region(region(start=0), 10)
        self.assertFalse(decision["provisional_direct_evidence"])


if __name__ == "__main__":
    unittest.main()
