from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

from muse_idr.data.pdb_negatives import (
    ObservedSegment,
    exact_sequence_accession_map,
    merge_observed_negatives,
    parse_historical_pdb_cross_reference,
    protected_masks,
    qualified_pdb_entries,
    read_observed_segments,
)


class PdbNegativeTests(unittest.TestCase):
    def test_legacy_pdb_cross_reference_normalization(self) -> None:
        self.assertEqual(parse_historical_pdb_cross_reference("1HRP_B"), ["1hrp"])
        self.assertEqual(
            parse_historical_pdb_cross_reference("4TTB;PDB:4TTC"),
            ["4ttb", "4ttc"],
        )
        self.assertEqual(parse_historical_pdb_cross_reference("IDJZ"), [])

    def test_quality_and_temporal_filters(self) -> None:
        requested = ["xok", "xbad", "emok", "nmr1", "late", "none"]
        summary = {
            pdb_id: [{"release_date": "20180101"}] for pdb_id in requested[:-1]
        }
        summary["late"] = [{"release_date": "20190101"}]
        experiment = {
            "xok": [
                {
                    "experimental_method_class": "x-ray",
                    "experimental_method": "X-ray diffraction",
                    "resolution": 2.5,
                }
            ],
            "xbad": [
                {
                    "experimental_method_class": "x-ray",
                    "experimental_method": "X-ray diffraction",
                    "resolution": 3.1,
                }
            ],
            "emok": [
                {
                    "experimental_method_class": "em",
                    "experimental_method": "Electron Microscopy",
                    "resolution": 4.0,
                }
            ],
            "nmr1": [
                {
                    "experimental_method_class": "nmr",
                    "experimental_method": "Solution NMR",
                    "resolution": None,
                }
            ],
            "late": [
                {
                    "experimental_method_class": "nmr",
                    "experimental_method": "Solution NMR",
                }
            ],
        }
        qualified, counts = qualified_pdb_entries(
            {
                "requested_pdb_ids": requested,
                "summary": summary,
                "experiment": experiment,
            },
            cutoff_date="20181130",
            xray_max_resolution=3.0,
            em_max_resolution=4.0,
        )
        self.assertEqual(set(qualified), {"xok", "emok", "nmr1"})
        self.assertEqual(counts["quality_or_method_excluded"], 1)
        self.assertEqual(counts["released_after_cutoff"], 1)
        self.assertEqual(counts["missing_summary"], 1)

    def test_sequence_verification_requires_exact_accession_and_sequence(self) -> None:
        records = [
            {
                "protein_id": "P1",
                "sequence": "ACD",
                "uniprot_accessions": ["P00001", "P00002-2"],
            }
        ]
        mapping, summary = exact_sequence_accession_map(
            records,
            [
                {"acc": "P00001", "sequence": "ACD"},
                {"acc": "P00002", "sequence": "ACD"},
                {"acc": "P00002-2", "sequence": "AAA"},
            ],
        )
        self.assertEqual(mapping, {"P00001": 0})
        self.assertEqual(summary["verified_exact_accessions"], 1)

    def test_disorder_like_states_and_margin_are_protected(self) -> None:
        records = [
            {
                "protein_id": "P1",
                "sequence": "ACDEFG",
                "disprot_ids": ["DP00001"],
            }
        ]
        historical = [
            {
                "disprot_id": "DP00001",
                "regions": [
                    {
                        "term_namespace": "Structural state",
                        "term_name": "Disorder",
                        "start": 2,
                        "end": 3,
                    },
                    {
                        "term_namespace": "Structural state",
                        "term_name": "Order",
                        "start": 5,
                        "end": 6,
                    },
                ],
            }
        ]
        masks, summary = protected_masks(
            records,
            historical,
            protected_states={"Disorder", "Molten globule"},
            boundary_margin=1,
        )
        self.assertEqual(masks[0], [True, True, True, True, False, False])
        self.assertEqual(summary["protected_residues"], 4)

    def test_merge_never_overwrites_positive_or_protected_positions(self) -> None:
        records = [
            {
                "protein_id": "P1",
                "sequence": "ACDEF",
                "labels": [1, -1, -1, -1, -1],
                "label_counts": {"positive": 1, "negative": 0, "unknown": 4},
            }
        ]
        segment = ObservedSegment(0, "1abc", "A", "P00001", 1, 5, 1, 5, "1", "5")
        quality = {
            "1abc": {
                "release_date": "20180101",
                "accepted_experiments": [
                    {
                        "experimental_method_class": "x-ray",
                        "experimental_method": "X-ray diffraction",
                        "resolution": 2.0,
                    }
                ],
            }
        }
        merged, summary = merge_observed_negatives(
            records,
            [segment],
            [[False, False, True, False, False]],
            quality,
        )
        self.assertEqual(merged[0]["labels"], [1, 0, -1, 0, 0])
        self.assertEqual(summary["negative_residues"], 3)
        self.assertEqual(summary["unknown_residues"], 1)

    def test_sifts_range_is_checked_against_verified_sequence(self) -> None:
        header = "PDB,CHAIN,SP_PRIMARY,RES_BEG,RES_END,PDB_BEG,PDB_END,SP_BEG,SP_END\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observed.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                handle.write(header)
                handle.write("1abc,A,P00001,1,3,1,3,1,3\n")
            segments, summary = read_observed_segments(
                path,
                qualified_pdb_ids={"1abc"},
                accession_to_record={"P00001": 0},
                records=[{"protein_id": "P1", "sequence": "ACD"}],
            )
        self.assertEqual(len(segments), 1)
        self.assertEqual(summary["sifts_segments_used"], 1)


if __name__ == "__main__":
    unittest.main()
