"""PDB/SIFTS negative labels with temporal, quality and conflict safeguards."""

from __future__ import annotations

import csv
import gzip
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .caid import normalize_sequence

PDB_ID_RE = re.compile(r"(?<![A-Z0-9])[0-9][A-Z0-9]{3}(?![A-Z0-9])", re.IGNORECASE)


def parse_historical_pdb_cross_reference(raw_value: str) -> list[str]:
    """Normalize legacy PDB references, including optional chain suffixes."""

    return [match.lower() for match in PDB_ID_RE.findall(raw_value.strip())]


@dataclass(frozen=True)
class ObservedSegment:
    """One SIFTS segment containing only experimentally observed residues."""

    record_index: int
    pdb_id: str
    chain: str
    accession: str
    uniprot_begin: int
    uniprot_end: int
    residue_begin: int
    residue_end: int
    pdb_begin: str
    pdb_end: str


def qualified_pdb_entries(
    metadata: Mapping[str, object],
    *,
    cutoff_date: str,
    xray_max_resolution: float,
    em_max_resolution: float,
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    """Select historical PDB entries with predeclared experimental quality."""

    requested = metadata.get("requested_pdb_ids")
    summaries = metadata.get("summary")
    experiments = metadata.get("experiment")
    if not isinstance(requested, list) or not isinstance(summaries, dict) or not isinstance(
        experiments, dict
    ):
        raise ValueError("Malformed PDBe metadata snapshot")
    reasons: Counter[str] = Counter()
    qualified: dict[str, dict[str, object]] = {}
    for raw_pdb_id in requested:
        pdb_id = str(raw_pdb_id).lower()
        summary_rows = summaries.get(pdb_id)
        experiment_rows = experiments.get(pdb_id)
        if not isinstance(summary_rows, list) or not summary_rows:
            reasons["missing_summary"] += 1
            continue
        if len(summary_rows) != 1 or not isinstance(summary_rows[0], dict):
            raise ValueError(f"Unexpected summary row count for {pdb_id}")
        release_date = str(summary_rows[0].get("release_date", ""))
        if len(release_date) != 8 or not release_date.isdigit():
            raise ValueError(f"Invalid PDB release date for {pdb_id}: {release_date!r}")
        if release_date > cutoff_date:
            reasons["released_after_cutoff"] += 1
            continue
        if not isinstance(experiment_rows, list) or not experiment_rows:
            reasons["missing_experiment"] += 1
            continue
        accepted: list[dict[str, object]] = []
        for detail in experiment_rows:
            if not isinstance(detail, dict):
                raise ValueError(f"Malformed experiment detail for {pdb_id}")
            method_class = str(detail.get("experimental_method_class", "")).lower()
            resolution_value = detail.get("resolution")
            resolution = float(resolution_value) if resolution_value is not None else None
            allowed = method_class == "nmr"
            allowed = allowed or (
                method_class == "x-ray"
                and resolution is not None
                and resolution <= xray_max_resolution
            )
            allowed = allowed or (
                method_class == "em"
                and resolution is not None
                and resolution <= em_max_resolution
            )
            if allowed:
                accepted.append(
                    {
                        "experimental_method_class": method_class,
                        "experimental_method": str(detail.get("experimental_method", "")),
                        "resolution": resolution,
                    }
                )
        if not accepted:
            reasons["quality_or_method_excluded"] += 1
            continue
        qualified[pdb_id] = {
            "release_date": release_date,
            "accepted_experiments": accepted,
        }
        reasons["qualified"] += 1
    reasons["requested"] = len(requested)
    return qualified, dict(sorted(reasons.items()))


def exact_sequence_accession_map(
    records: Sequence[Mapping[str, object]],
    verification_entries: Iterable[Mapping[str, object]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Map only exact accession+sequence matches; never collapse isoforms."""

    sequences_by_accession: dict[str, set[str]] = defaultdict(set)
    for entry in verification_entries:
        accession = str(entry.get("acc", "")).upper()
        sequence_value = entry.get("sequence")
        if accession and sequence_value:
            sequences_by_accession[accession].add(normalize_sequence(str(sequence_value)))
    mapping: dict[str, int] = {}
    verified_records: set[int] = set()
    candidate_accessions = 0
    for record_index, record in enumerate(records):
        sequence = normalize_sequence(str(record["sequence"]))
        raw_accessions = record.get("uniprot_accessions", [])
        if not isinstance(raw_accessions, list):
            raise ValueError(f"Malformed accessions for {record.get('protein_id')}")
        for raw_accession in raw_accessions:
            accession = str(raw_accession).upper()
            candidate_accessions += 1
            if sequence not in sequences_by_accession.get(accession, set()):
                continue
            if accession in mapping and mapping[accession] != record_index:
                raise ValueError(f"Accession maps to multiple candidate sequences: {accession}")
            mapping[accession] = record_index
            verified_records.add(record_index)
    return mapping, {
        "candidate_accession_values": candidate_accessions,
        "verified_exact_accessions": len(mapping),
        "records_with_verified_accession": len(verified_records),
    }


def protected_masks(
    records: Sequence[Mapping[str, object]],
    historical_entries: Iterable[Mapping[str, object]],
    *,
    protected_states: set[str],
    boundary_margin: int,
) -> tuple[list[list[bool]], dict[str, object]]:
    """Protect all disorder-like annotations and their boundary margins."""

    if boundary_margin < 0:
        raise ValueError("boundary_margin must be non-negative")
    entry_by_id = {
        str(entry.get("disprot_id", "")).upper(): entry for entry in historical_entries
    }
    masks: list[list[bool]] = []
    state_regions: Counter[str] = Counter()
    for record in records:
        sequence_length = len(str(record["sequence"]))
        mask = [False] * sequence_length
        disprot_ids = record.get("disprot_ids", [])
        if not isinstance(disprot_ids, list):
            raise ValueError(f"Malformed DisProt aliases for {record.get('protein_id')}")
        for raw_identifier in disprot_ids:
            identifier = str(raw_identifier).upper()
            entry = entry_by_id.get(identifier)
            if entry is None:
                raise ValueError(f"Historical DisProt entry missing: {identifier}")
            regions = entry.get("regions", [])
            if not isinstance(regions, list):
                raise ValueError(f"Malformed regions for {identifier}")
            for region in regions:
                if not isinstance(region, dict):
                    continue
                state = str(region.get("term_name", ""))
                if (
                    region.get("term_namespace") != "Structural state"
                    or state not in protected_states
                ):
                    continue
                start = int(region["start"])
                end = int(region["end"])
                if not 1 <= start <= end <= sequence_length:
                    raise ValueError(f"Protected region outside sequence for {identifier}")
                state_regions[state] += 1
                protected_start = max(1, start - boundary_margin)
                protected_end = min(sequence_length, end + boundary_margin)
                for index in range(protected_start - 1, protected_end):
                    mask[index] = True
        masks.append(mask)
    return masks, {
        "protected_state_region_counts": dict(sorted(state_regions.items())),
        "protected_residues": sum(sum(mask) for mask in masks),
    }


def read_observed_segments(
    path: Path,
    *,
    qualified_pdb_ids: set[str],
    accession_to_record: Mapping[str, int],
    records: Sequence[Mapping[str, object]],
) -> tuple[list[ObservedSegment], dict[str, int]]:
    """Read only SIFTS observed segments relevant to verified candidates."""

    segments: list[ObservedSegment] = []
    matched_pdb_ids: set[str] = set()
    matched_accessions: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        expected_fields = {
            "PDB",
            "CHAIN",
            "SP_PRIMARY",
            "RES_BEG",
            "RES_END",
            "PDB_BEG",
            "PDB_END",
            "SP_BEG",
            "SP_END",
        }
        if set(reader.fieldnames or []) != expected_fields:
            raise ValueError(f"Unexpected SIFTS observed-segment columns: {reader.fieldnames}")
        for row in reader:
            pdb_id = row["PDB"].lower()
            accession = row["SP_PRIMARY"].upper()
            if pdb_id not in qualified_pdb_ids or accession not in accession_to_record:
                continue
            record_index = accession_to_record[accession]
            uniprot_begin = int(row["SP_BEG"])
            uniprot_end = int(row["SP_END"])
            residue_begin = int(row["RES_BEG"])
            residue_end = int(row["RES_END"])
            sequence_length = len(str(records[record_index]["sequence"]))
            if not 1 <= uniprot_begin <= uniprot_end <= sequence_length:
                raise ValueError(
                    f"SIFTS range outside exact verified sequence: {pdb_id}/{accession} "
                    f"{uniprot_begin}-{uniprot_end} of {sequence_length}"
                )
            segments.append(
                ObservedSegment(
                    record_index=record_index,
                    pdb_id=pdb_id,
                    chain=row["CHAIN"],
                    accession=accession,
                    uniprot_begin=uniprot_begin,
                    uniprot_end=uniprot_end,
                    residue_begin=residue_begin,
                    residue_end=residue_end,
                    pdb_begin=row["PDB_BEG"],
                    pdb_end=row["PDB_END"],
                )
            )
            matched_pdb_ids.add(pdb_id)
            matched_accessions.add(accession)
    return segments, {
        "sifts_segments_used": len(segments),
        "sifts_pdb_entries_used": len(matched_pdb_ids),
        "sifts_accessions_used": len(matched_accessions),
    }


def merge_observed_negatives(
    records: Sequence[Mapping[str, object]],
    segments: Sequence[ObservedSegment],
    masks: Sequence[Sequence[bool]],
    qualified_pdbs: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Set PDB-observed, non-conflicting positions to zero."""

    if len(records) != len(masks):
        raise ValueError("records and protected masks must have equal length")
    merged = [deepcopy(dict(record)) for record in records]
    support_counts = [[0] * len(str(record["sequence"])) for record in records]
    segment_rows: list[list[dict[str, object]]] = [[] for _ in records]
    for segment in segments:
        record = merged[segment.record_index]
        labels = record.get("labels")
        if not isinstance(labels, list):
            raise ValueError(f"Malformed labels for {record.get('protein_id')}")
        eligible = 0
        for index in range(segment.uniprot_begin - 1, segment.uniprot_end):
            if masks[segment.record_index][index] or labels[index] == 1:
                continue
            labels[index] = 0
            support_counts[segment.record_index][index] += 1
            eligible += 1
        if eligible:
            segment_rows[segment.record_index].append(
                {
                    "pdb_id": segment.pdb_id,
                    "chain": segment.chain,
                    "uniprot_accession": segment.accession,
                    "uniprot_begin": segment.uniprot_begin,
                    "uniprot_end": segment.uniprot_end,
                    "residue_begin": segment.residue_begin,
                    "residue_end": segment.residue_end,
                    "pdb_begin": segment.pdb_begin,
                    "pdb_end": segment.pdb_end,
                    "eligible_negative_residues": eligible,
                    "quality": qualified_pdbs[segment.pdb_id],
                }
            )

    positive_sequences: set[int] = set()
    negative_sequences: set[int] = set()
    support_distribution: Counter[str] = Counter()
    for index, record in enumerate(merged):
        labels = record["labels"]
        positives = labels.count(1)
        negatives = labels.count(0)
        unknown = labels.count(-1)
        if positives:
            positive_sequences.add(index)
        if negatives:
            negative_sequences.add(index)
        record["pdb_observed_negative_segments"] = segment_rows[index]
        record["label_counts"] = {
            "positive": positives,
            "negative": negatives,
            "unknown": unknown,
        }
        for support in support_counts[index]:
            if support == 1:
                support_distribution["one_structure"] += 1
            elif 2 <= support <= 5:
                support_distribution["two_to_five_structures"] += 1
            elif support > 5:
                support_distribution["more_than_five_structures"] += 1

    labeled_sequences = positive_sequences | negative_sequences
    summary = {
        "sequences_with_positive_labels": len(positive_sequences),
        "sequences_with_negative_labels": len(negative_sequences),
        "sequences_with_both_labels": len(positive_sequences & negative_sequences),
        "trainable_sequences": len(labeled_sequences),
        "all_unknown_sequences": len(merged) - len(labeled_sequences),
        "positive_residues": sum(record["label_counts"]["positive"] for record in merged),
        "negative_residues": sum(record["label_counts"]["negative"] for record in merged),
        "unknown_residues": sum(record["label_counts"]["unknown"] for record in merged),
        **dict(sorted(support_distribution.items())),
    }
    return merged, summary
