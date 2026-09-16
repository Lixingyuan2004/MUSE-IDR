"""Tests for S2 exclusion boundaries and metadata-only parsing."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_sources import accession_parent, forbidden_hit, index, match_reasons, metadata, screen


class AuditTests(unittest.TestCase):
    def row(self, identifier="DP90001", accession="P12345", sequence="ACDE", release="2026_06"):
        return metadata({"disprot_id": identifier, "acc": accession, "sequence": sequence,
                         "length": len(sequence), "released": release,
                         "regions": object(), "disprot_consensus": object()})

    def test_metadata_ignores_annotations(self):
        row = self.row()
        self.assertEqual(row["issues"], [])
        self.assertNotIn("regions", row)
        self.assertNotIn("disprot_consensus", row)

    def test_renamed_identical_sequence_is_excluded(self):
        earlier = index([self.row()])
        current = self.row(identifier="DP90002", accession="Q98765")
        self.assertEqual(match_reasons(current, earlier), ["exact_sequence"])

    def test_isoform_parent_is_excluded(self):
        earlier = index([self.row()])
        current = self.row(identifier="DP90002", accession="P12345-2", sequence="FGHI")
        self.assertIn("accession_or_isoform_parent", match_reasons(current, earlier))
        self.assertEqual(accession_parent("P12345-2"), "P12345")

    def test_release_boundary_and_candidate_duplicates(self):
        empty = index([])
        rows = [self.row(), self.row(identifier="DP90002", accession="Q98765"),
                self.row(identifier="DP90003", sequence="FGHI", release="2025_06")]
        records, count = screen(rows, empty, empty, empty, "2025_06")
        self.assertEqual(len(records), 2)
        self.assertEqual(count["unique_sequence_candidates"], 1)

    def test_new_release_but_old_identifier_is_not_new(self):
        earlier = index([self.row(release="2023_06")])
        records, count = screen([self.row(sequence="FGHI")], earlier, index([]), index([]), "2025_06")
        self.assertEqual(count["unique_sequence_candidates"], 0)
        self.assertIn("older_snapshot:identifier", records[0]["exclusion_reasons"])

    def test_homology_thresholds(self):
        self.assertFalse(forbidden_hit(.30, .99, .99))
        self.assertTrue(forbidden_hit(.301, .80, .80))
        self.assertFalse(forbidden_hit(.90, .79, .99))
        self.assertFalse(forbidden_hit(.90, .99, .79))
        with self.assertRaises(ValueError):
            forbidden_hit(30, 80, 80)


if __name__ == "__main__":
    unittest.main()
