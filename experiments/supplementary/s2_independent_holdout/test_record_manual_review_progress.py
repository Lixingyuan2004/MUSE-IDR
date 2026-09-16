"""Unit tests for recording prediction-blind S2 manual review progress."""
from __future__ import annotations

import unittest

from record_manual_review_progress import apply_decisions, validate_decision


def decision(outcome: str = "accept") -> dict[str, str]:
    values = {
        "protein_id": "DPTEST",
        "region_id": "DPTESTr001",
        "review_decision": outcome,
        "coordinate_scope_confirmed": "yes",
        "direct_disorder_support_confirmed": "yes",
        "reference_sequence_construct_confirmed": "yes",
        "public_annotation_status_confirmed": "yes",
        "review_notes": "checked",
        "source_evidence_sha256": "abc",
    }
    if outcome == "reject":
        values["direct_disorder_support_confirmed"] = "no"
    return values


class RecordManualReviewProgressTests(unittest.TestCase):
    def test_accept_requires_four_yes_values(self) -> None:
        row = decision()
        row["coordinate_scope_confirmed"] = "no"
        with self.assertRaises(ValueError):
            validate_decision(row)

    def test_reject_requires_a_failed_confirmation(self) -> None:
        row = decision("reject")
        row["direct_disorder_support_confirmed"] = "yes"
        with self.assertRaises(ValueError):
            validate_decision(row)

    def test_apply_decision_preserves_unreviewed_rows(self) -> None:
        source = {
            "protein_id": "DPTEST",
            "region_id": "DPTESTr001",
            "evidence_sha256": "abc",
            "review_decision": "",
            "coordinate_scope_confirmed": "",
            "direct_disorder_support_confirmed": "",
            "reference_sequence_construct_confirmed": "",
            "public_annotation_status_confirmed": "",
            "reviewer": "",
            "review_date": "",
            "review_notes": "",
        }
        untouched = dict(source, region_id="DPTESTr002", evidence_sha256="def")
        merged, count = apply_decisions(
            [source, untouched], [decision()], "Xingyuan Li", "2026-09-06"
        )
        self.assertEqual(count, 1)
        self.assertEqual(merged[0]["review_decision"], "accept")
        self.assertEqual(merged[0]["reviewer"], "Xingyuan Li")
        self.assertEqual(merged[1]["review_decision"], "")

    def test_source_hash_mismatch_is_rejected(self) -> None:
        source = {
            "protein_id": "DPTEST",
            "region_id": "DPTESTr001",
            "evidence_sha256": "wrong",
        }
        with self.assertRaises(ValueError):
            apply_decisions([source], [decision()], "Reviewer", "2026-09-06")


if __name__ == "__main__":
    unittest.main()
