"""Validated removal of candidate proteins homologous to protected test targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .caid import SequenceRecord, sequence_sha256
from .leakage import HomologyHit, is_forbidden_homolog


@dataclass(frozen=True)
class HomologyFilterResult:
    kept: list[dict[str, object]]
    removed: list[dict[str, object]]
    forbidden_hits: list[HomologyHit]

    @property
    def removed_ids(self) -> set[str]:
        return {str(record["protein_id"]) for record in self.removed}


def filter_candidate_records(
    records: Iterable[dict[str, object]],
    fasta_records: Iterable[SequenceRecord],
    hits: Iterable[HomologyHit],
) -> HomologyFilterResult:
    """Cross-check JSONL/FASTA and remove every query with a forbidden hit."""

    rows = list(records)
    fasta = list(fasta_records)
    row_ids = [str(row.get("protein_id", "")) for row in rows]
    fasta_ids = [record.identifier for record in fasta]
    if not all(row_ids) or len(row_ids) != len(set(row_ids)):
        raise ValueError("Candidate JSONL protein_id values must be non-empty and unique")
    if row_ids != fasta_ids:
        raise ValueError("Candidate JSONL and FASTA identifiers/order do not match")

    for row, fasta_record in zip(rows, fasta, strict=True):
        sequence = str(row.get("sequence", ""))
        labels = row.get("labels")
        if not sequence or sequence != fasta_record.sequence:
            raise ValueError(f"JSONL/FASTA sequence mismatch for {fasta_record.identifier}")
        if row.get("sequence_sha256") != sequence_sha256(sequence):
            raise ValueError(f"Sequence SHA-256 mismatch for {fasta_record.identifier}")
        if not isinstance(labels, list) or len(labels) != len(sequence):
            raise ValueError(f"Sequence/label length mismatch for {fasta_record.identifier}")
        if not set(labels) <= {-1, 0, 1}:
            raise ValueError(f"Unexpected residue label for {fasta_record.identifier}")

    forbidden_hits = [hit for hit in hits if is_forbidden_homolog(hit)]
    forbidden_ids = {hit.query for hit in forbidden_hits}
    missing = sorted(forbidden_ids - set(row_ids))
    if missing:
        raise ValueError(f"Homology TSV contains unknown candidate queries: {missing}")

    kept = [row for row in rows if str(row["protein_id"]) not in forbidden_ids]
    removed = [row for row in rows if str(row["protein_id"]) in forbidden_ids]
    if {str(row["protein_id"]) for row in removed} != forbidden_ids:
        raise ValueError("Not every forbidden query was removed exactly once")
    return HomologyFilterResult(kept, removed, forbidden_hits)


def label_summary(records: Iterable[dict[str, object]]) -> dict[str, int]:
    """Summarize residue labels and per-sequence label availability."""

    rows = list(records)
    summary = {
        "sequences": len(rows),
        "sequences_with_positive_labels": 0,
        "sequences_with_negative_labels": 0,
        "sequences_with_both_labels": 0,
        "positive_residues": 0,
        "negative_residues": 0,
        "unknown_residues": 0,
    }
    for row in rows:
        labels = list(row["labels"])
        positive = labels.count(1)
        negative = labels.count(0)
        unknown = labels.count(-1)
        summary["positive_residues"] += positive
        summary["negative_residues"] += negative
        summary["unknown_residues"] += unknown
        summary["sequences_with_positive_labels"] += int(positive > 0)
        summary["sequences_with_negative_labels"] += int(negative > 0)
        summary["sequences_with_both_labels"] += int(positive > 0 and negative > 0)
    return summary
