from __future__ import annotations

import unittest

from muse_idr.data.disprot import build_historical_disprot_records


def region(start: int, end: int, evidence: str) -> dict[str, object]:
    return {
        "term_namespace": "Structural state",
        "term_name": "Disorder",
        "start": start,
        "end": end,
        "ec_name": evidence,
        "region_id": f"R{start}-{end}",
    }


def entry(
    identifier: str,
    sequence: str,
    regions: list[dict[str, object]],
    *,
    name: str = "protein",
    accession: str | None = None,
) -> dict[str, object]:
    return {
        "disprot_id": identifier,
        "acc": accession,
        "name": name,
        "sequence": sequence,
        "length": len(sequence),
        "regions": regions,
    }


class HistoricalDisprotTests(unittest.TestCase):
    def test_allowed_regions_are_positive_and_excluded_regions_remain_unknown(self) -> None:
        entries = [
            entry(
                "DP00001",
                "ACDE",
                [region(1, 2, "NMR"), region(3, 4, "Missing density")],
                accession="P00001",
            )
        ]
        records, summary = build_historical_disprot_records(
            entries,
            current_ids={"DP00001"},
            previous_ids=set(),
            allowed_positive_evidence={"NMR"},
            excluded_nonpositive_evidence={"Missing density"},
            name_exclusion_substrings={"polyprotein"},
        )
        self.assertEqual(records[0]["labels"], [1, 1, -1, -1])
        self.assertEqual(records[0]["label_counts"]["negative"], 0)
        self.assertEqual(summary["positive_residues_after_merge"], 2)

    def test_duplicate_sequences_merge_positive_labels_and_aliases(self) -> None:
        entries = [
            entry("DP00001", "ACDE", [region(1, 1, "NMR")], accession="P00001"),
            entry("DP00002", "ACDE", [region(4, 4, "NMR")], accession="P00002"),
        ]
        records, _ = build_historical_disprot_records(
            entries,
            current_ids={"DP00001", "DP00002"},
            previous_ids={"DP00001"},
            allowed_positive_evidence={"NMR"},
            excluded_nonpositive_evidence=set(),
            name_exclusion_substrings=set(),
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["labels"], [1, -1, -1, 1])
        self.assertEqual(records[0]["disprot_ids"], ["DP00001", "DP00002"])
        self.assertEqual(records[0]["cohorts"], ["caid1_history_delta", "pre_caid1_history"])

    def test_unknown_evidence_and_invalid_coordinates_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unclassified"):
            build_historical_disprot_records(
                [entry("DP00001", "AC", [region(1, 1, "Mystery assay")])],
                current_ids={"DP00001"},
                previous_ids=set(),
                allowed_positive_evidence={"NMR"},
                excluded_nonpositive_evidence=set(),
                name_exclusion_substrings=set(),
            )
        with self.assertRaisesRegex(ValueError, "Out-of-range"):
            build_historical_disprot_records(
                [entry("DP00001", "AC", [region(1, 3, "NMR")])],
                current_ids={"DP00001"},
                previous_ids=set(),
                allowed_positive_evidence={"NMR"},
                excluded_nonpositive_evidence=set(),
                name_exclusion_substrings=set(),
            )

    def test_polyprotein_name_is_excluded(self) -> None:
        records, summary = build_historical_disprot_records(
            [entry("DP00001", "AC", [region(1, 1, "NMR")], name="Genome polyprotein")],
            current_ids={"DP00001"},
            previous_ids=set(),
            allowed_positive_evidence={"NMR"},
            excluded_nonpositive_evidence=set(),
            name_exclusion_substrings={"polyprotein"},
        )
        self.assertEqual(records, [])
        self.assertEqual(summary["polyprotein_records_excluded"], 1)


if __name__ == "__main__":
    unittest.main()
