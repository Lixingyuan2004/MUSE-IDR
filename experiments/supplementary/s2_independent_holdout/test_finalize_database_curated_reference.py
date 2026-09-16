"""Unit tests for the S2 database-curated reference freeze."""
from __future__ import annotations

import unittest

from finalize_database_curated_reference import finalize_record, label_counts, summarize


def record(labels: list[int]) -> dict:
    return {
        "protein_id": "DPTEST",
        "sequence": "A" * len(labels),
        "labels": labels,
        "label_counts": label_counts(labels),
        "reference_status": "provisional_not_for_model_selection_or_formal_reporting",
        "source_entry_release": "2026_06",
    }


class FinalizeDatabaseCuratedReferenceTests(unittest.TestCase):
    def test_finalization_changes_status_without_changing_labels(self) -> None:
        source = record([1, 0, -1])
        result = finalize_record(source, "policy")
        self.assertEqual(result["labels"], source["labels"])
        self.assertEqual(result["reference_status"], "final_database_curated_s2_reference")
        self.assertFalse(result["manual_article_review_applied"])

    def test_invalid_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            finalize_record(record([2]), "policy")

    def test_unlabeled_record_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            finalize_record(record([-1, -1]), "policy")

    def test_stored_counts_must_match(self) -> None:
        source = record([1, 0])
        source["label_counts"]["positive"] = 0
        with self.assertRaises(ValueError):
            finalize_record(source, "policy")

    def test_summary_counts_proteins_and_residues(self) -> None:
        rows = [finalize_record(record([1, 0, -1]), "policy")]
        observed = summarize(rows)
        self.assertEqual(observed["proteins"], 1)
        self.assertEqual(observed["residues"], 3)
        self.assertEqual(observed["proteins_with_both"], 1)


if __name__ == "__main__":
    unittest.main()
