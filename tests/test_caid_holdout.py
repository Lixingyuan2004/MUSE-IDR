from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from muse_idr.data.caid import (
    SequenceRecord,
    parse_caid_reference,
    parse_fasta,
    reconstruct_caid1_targets,
    sequence_sha256,
    subset_sentinel,
    write_fasta,
)


class CaidReferenceTests(unittest.TestCase):
    def test_parse_valid_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.fasta"
            path.write_text(">DP00001\nACDXU\n01-10\n", encoding="utf-8")
            records = list(parse_caid_reference(path))
        self.assertEqual(records[0].identifier, "DP00001")
        self.assertEqual(records[0].sequence, "ACDXU")
        self.assertEqual(len(records[0].sequence), len(records[0].labels))

    def test_reject_html_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-fasta.fasta"
            path.write_text("<!doctype html>\n<html></html>\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "possibly HTML"):
                list(parse_caid_reference(path))

    def test_reject_sequence_label_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.fasta"
            path.write_text(">DP00001\nACDE\n010\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                list(parse_caid_reference(path))

    def test_sequence_hash_is_normalized(self) -> None:
        self.assertEqual(sequence_sha256("ac d\n"), sequence_sha256("ACD"))

    def test_reconstruct_caid1_delta_and_remove_polyprotein(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current.fasta"
            old = root / "old.fasta"
            metadata = root / "release.json"
            current.write_text(
                ">disprot|DP00001|x\nDDD\n"
                ">disprot|DP00002|x\nDDD\n"
                ">disprot|DP00003|x\nDDD\n",
                encoding="utf-8",
            )
            old.write_text(">disprot|DP00001|x\nDDD\n", encoding="utf-8")
            metadata.write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "disprot_id": "DP00001",
                                "name": "old protein",
                                "sequence": "AAA",
                                "acc": "P00001",
                            },
                            {
                                "disprot_id": "DP00002",
                                "name": "kept protein",
                                "sequence": "ACD",
                                "acc": "P00002",
                            },
                            {
                                "disprot_id": "DP00003",
                                "name": "viral polyprotein",
                                "sequence": "EFG",
                                "acc": "P00003",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            targets = reconstruct_caid1_targets(current, old, metadata)
        self.assertEqual(
            [(record.identifier, acc) for record, acc in targets],
            [("DP00002", "P00002")],
        )

    def test_subset_sentinel_keeps_only_selected_editions_without_labels(self) -> None:
        manifest = {
            "schema_version": 1,
            "generated_utc": "2026-01-01T00:00:00+00:00",
            "labels_included": False,
            "unmapped_reference_id_values": ["DP00003"],
            "sources": [
                {"edition": "caid1", "records": 1},
                {"edition": "caid2", "records": 1},
                {"role": "disprot_to_uniprot_identifier_mapping", "records": 2},
            ],
            "targets": [
                {
                    "sequence_sha256": "a",
                    "reference_ids": ["DP00001"],
                    "uniprot_accessions": ["P00001"],
                    "sources": ["caid1:reconstructed"],
                },
                {
                    "sequence_sha256": "b",
                    "reference_ids": ["DP00003"],
                    "uniprot_accessions": [],
                    "sources": ["caid2:disorder_pdb"],
                },
            ],
        }
        subset = subset_sentinel(manifest, {"caid2"})
        self.assertFalse(subset["labels_included"])
        self.assertEqual(subset["protected_editions"], ["caid2"])
        self.assertEqual(
            [target["sequence_sha256"] for target in subset["targets"]],
            ["b"],
        )
        self.assertEqual(subset["summary"]["unmapped_reference_ids"], 1)

    def test_write_fasta_preserves_aliases_in_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aliases.fasta"
            write_fasta(
                [SequenceRecord("DP00001", "ACD", "DP00001 P12345")],
                path,
            )
            text = path.read_text(encoding="utf-8")
            parsed = list(parse_fasta(path))
        self.assertTrue(text.startswith(">DP00001 P12345\n"))
        self.assertEqual(parsed[0].identifier, "DP00001")
        self.assertEqual(parsed[0].header, "DP00001 P12345")


if __name__ == "__main__":
    unittest.main()
