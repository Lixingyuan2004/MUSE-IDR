from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from muse_idr.data.caid import SequenceRecord, sequence_sha256
from muse_idr.data.leakage import (
    HomologyHit,
    audit_records,
    canonical_isoform,
    is_forbidden_homolog,
    parse_homology_tsv,
)


class LeakageTests(unittest.TestCase):
    def test_isoform_parent(self) -> None:
        self.assertEqual(canonical_isoform("p12345-2"), "P12345")

    def test_exact_and_identifier_audit(self) -> None:
        sequence = "ACDEFG"
        manifest = {
            "labels_included": False,
            "targets": [
                {
                    "sequence_sha256": sequence_sha256(sequence),
                    "reference_ids": ["DP01234"],
                    "uniprot_accessions": ["P12345"],
                }
            ],
        }
        records = [
            SequenceRecord("candidate1", sequence, "candidate1"),
            SequenceRecord("P12345-2", "AAAA", "sp|P12345-2|protein"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            findings = audit_records(records, path)
        self.assertEqual(findings[0]["reasons"], ["exact_sequence_sha256"])
        self.assertIn("identifier_or_isoform", findings[1]["reasons"])

    def test_homology_requires_strict_identity_and_both_coverages(self) -> None:
        forbidden = HomologyHit("q", "t", 0.31, 80, 100, 100)
        identity_boundary = HomologyHit("q", "t", 0.30, 80, 100, 100)
        short_target_coverage = HomologyHit("q", "t", 0.90, 80, 100, 200)
        self.assertTrue(is_forbidden_homolog(forbidden))
        self.assertFalse(is_forbidden_homolog(identity_boundary))
        self.assertFalse(is_forbidden_homolog(short_target_coverage))

    def test_mmseqs_explicit_coverages_handle_gapped_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hits.tsv"
            path.write_text(
                "query\ttarget\t0.372\t1792\t1770\t1799\t0.992\t0.976\n",
                encoding="utf-8",
            )
            hit = parse_homology_tsv(path)[0]
        self.assertEqual(hit.alignment_length, 1792)
        self.assertEqual(hit.query_coverage, 0.992)
        self.assertEqual(hit.target_coverage, 0.976)
        self.assertTrue(is_forbidden_homolog(hit))


if __name__ == "__main__":
    unittest.main()
