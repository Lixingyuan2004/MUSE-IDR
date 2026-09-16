from __future__ import annotations

import unittest

from muse_idr.data.caid import SequenceRecord, sequence_sha256
from muse_idr.data.homology_filter import filter_candidate_records, label_summary
from muse_idr.data.leakage import HomologyHit


def candidate(identifier: str, sequence: str, labels: list[int]) -> dict[str, object]:
    return {
        "protein_id": identifier,
        "sequence": sequence,
        "sequence_sha256": sequence_sha256(sequence),
        "labels": labels,
    }


class HomologyFilterTests(unittest.TestCase):
    def test_removes_entire_forbidden_query_and_summarizes_labels(self) -> None:
        records = [
            candidate("q1", "AAAA", [1, 1, -1, 0]),
            candidate("q2", "CCCC", [0, 0, -1, -1]),
        ]
        fasta = [
            SequenceRecord("q1", "AAAA", "q1"),
            SequenceRecord("q2", "CCCC", "q2"),
        ]
        hits = [HomologyHit("q1", "caid", 0.31, 4, 4, 4, 1.0, 1.0)]
        result = filter_candidate_records(records, fasta, hits)
        self.assertEqual([row["protein_id"] for row in result.kept], ["q2"])
        self.assertEqual(result.removed_ids, {"q1"})
        self.assertEqual(label_summary(result.kept)["negative_residues"], 2)

    def test_rejects_jsonl_fasta_sequence_mismatch(self) -> None:
        records = [candidate("q1", "AAAA", [1, 1, 1, 1])]
        fasta = [SequenceRecord("q1", "CCCC", "q1")]
        with self.assertRaisesRegex(ValueError, "sequence mismatch"):
            filter_candidate_records(records, fasta, [])


if __name__ == "__main__":
    unittest.main()
