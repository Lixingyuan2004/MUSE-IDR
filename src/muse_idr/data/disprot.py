"""Build strict historical DisProt labels without inventing negative residues."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from .caid import normalize_sequence, sequence_sha256


def load_disprot_entries(path: Path) -> list[dict[str, object]]:
    """Load a DisProt release JSON and validate its record envelope."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"Unexpected DisProt JSON structure: {path}")
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_entry in payload["data"]:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Malformed DisProt entry: {path}")
        identifier = str(raw_entry.get("disprot_id", "")).upper()
        if not identifier:
            raise ValueError(f"DisProt entry without disprot_id: {path}")
        if identifier in seen:
            raise ValueError(f"Duplicate DisProt ID {identifier}: {path}")
        seen.add(identifier)
        entries.append(raw_entry)
    declared_size = payload.get("size")
    if declared_size is not None and int(declared_size) != len(entries):
        raise ValueError(f"DisProt declared size does not match data length: {path}")
    return entries


def _normalized_region(
    entry: Mapping[str, object],
    region: Mapping[str, object],
    sequence_length: int,
) -> dict[str, object]:
    identifier = str(entry["disprot_id"]).upper()
    try:
        start = int(region["start"])
        end = int(region["end"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid coordinates in {identifier}: {region!r}") from error
    if not 1 <= start <= end <= sequence_length:
        raise ValueError(
            f"Out-of-range region in {identifier}: {start}-{end} for length {sequence_length}"
        )
    evidence = region.get("ec_name")
    if not isinstance(evidence, str) or not evidence:
        raise ValueError(f"Missing evidence name in {identifier}: {region!r}")
    return {
        "source_disprot_id": identifier,
        "region_id": str(region.get("region_id", "")),
        "start": start,
        "end": end,
        "evidence": evidence,
        "reference_source": str(region.get("reference_source", "")),
        "reference_id": str(region.get("reference_id", "")),
    }


def build_historical_disprot_records(
    entries: Iterable[Mapping[str, object]],
    *,
    current_ids: set[str],
    previous_ids: set[str],
    allowed_positive_evidence: set[str],
    excluded_nonpositive_evidence: set[str],
    name_exclusion_substrings: Iterable[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Create sequence-deduplicated `1/-1` records from a historical release."""

    overlap = allowed_positive_evidence & excluded_nonpositive_evidence
    if overlap:
        raise ValueError(
            f"Evidence appears in both allowed and excluded policies: {sorted(overlap)}"
        )
    excluded_name_terms = tuple(term.lower() for term in name_exclusion_substrings)
    entry_by_id: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        identifier = str(entry.get("disprot_id", "")).upper()
        if identifier in entry_by_id:
            raise ValueError(f"Duplicate DisProt ID: {identifier}")
        entry_by_id[identifier] = entry
    missing_entries = sorted(current_ids - entry_by_id.keys())
    if missing_entries:
        raise ValueError(f"Structural-state IDs missing from JSON: {missing_entries[:10]}")

    excluded_names: list[str] = []
    allowed_evidence_counts: Counter[str] = Counter()
    allowed_evidence_spans: Counter[str] = Counter()
    excluded_evidence_counts: Counter[str] = Counter()
    excluded_evidence_spans: Counter[str] = Counter()
    aliases: list[dict[str, object]] = []

    for identifier in sorted(current_ids):
        entry = entry_by_id[identifier]
        name = str(entry.get("name", ""))
        if any(term in name.lower() for term in excluded_name_terms):
            excluded_names.append(identifier)
            continue
        sequence = normalize_sequence(str(entry.get("sequence", "")))
        declared_length = int(entry.get("length", len(sequence)))
        if declared_length != len(sequence):
            raise ValueError(
                f"Declared length mismatch for {identifier}: {declared_length} != {len(sequence)}"
            )
        raw_regions = entry.get("regions", [])
        if not isinstance(raw_regions, list):
            raise ValueError(f"regions must be a list for {identifier}")
        labels = [-1] * len(sequence)
        positive_regions: list[dict[str, object]] = []
        excluded_regions: list[dict[str, object]] = []
        for raw_region in raw_regions:
            if not isinstance(raw_region, dict):
                raise ValueError(f"Malformed region for {identifier}")
            if (
                raw_region.get("term_namespace") != "Structural state"
                or raw_region.get("term_name") != "Disorder"
            ):
                continue
            region = _normalized_region(entry, raw_region, len(sequence))
            evidence = str(region["evidence"])
            span = int(region["end"]) - int(region["start"]) + 1
            if evidence in allowed_positive_evidence:
                positive_regions.append(region)
                allowed_evidence_counts[evidence] += 1
                allowed_evidence_spans[evidence] += span
                for index in range(int(region["start"]) - 1, int(region["end"])):
                    labels[index] = 1
            elif evidence in excluded_nonpositive_evidence:
                excluded_regions.append(region)
                excluded_evidence_counts[evidence] += 1
                excluded_evidence_spans[evidence] += span
            else:
                raise ValueError(
                    f"Unclassified structural-disorder evidence {evidence!r} in {identifier}"
                )
        accession = entry.get("acc")
        aliases.append(
            {
                "disprot_id": identifier,
                "accession": str(accession).upper() if accession else None,
                "name": name,
                "cohort": (
                    "pre_caid1_history" if identifier in previous_ids else "caid1_history_delta"
                ),
                "sequence": sequence,
                "labels": labels,
                "positive_regions": positive_regions,
                "excluded_regions": excluded_regions,
            }
        )

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for alias in aliases:
        groups[sequence_sha256(str(alias["sequence"]))].append(alias)

    records: list[dict[str, object]] = []
    for digest, group in sorted(groups.items()):
        sequence = str(group[0]["sequence"])
        merged_labels = [-1] * len(sequence)
        for alias in group:
            if alias["sequence"] != sequence:
                raise AssertionError("SHA-256 collision with different sequences")
            for index, label in enumerate(alias["labels"]):
                if label == 1:
                    merged_labels[index] = 1
        disprot_ids = sorted(str(alias["disprot_id"]) for alias in group)
        accessions = sorted(
            {str(alias["accession"]) for alias in group if alias["accession"] is not None}
        )
        records.append(
            {
                "protein_id": disprot_ids[0],
                "disprot_ids": disprot_ids,
                "uniprot_accessions": accessions,
                "names": sorted({str(alias["name"]) for alias in group}),
                "cohorts": sorted({str(alias["cohort"]) for alias in group}),
                "sequence_sha256": digest,
                "sequence": sequence,
                "labels": merged_labels,
                "positive_regions": [
                    region for alias in group for region in alias["positive_regions"]
                ],
                "excluded_nonpositive_regions": [
                    region for alias in group for region in alias["excluded_regions"]
                ],
                "label_counts": {
                    "positive": merged_labels.count(1),
                    "negative": 0,
                    "unknown": merged_labels.count(-1),
                },
            }
        )

    positive_sequences = sum(record["label_counts"]["positive"] > 0 for record in records)
    summary: dict[str, object] = {
        "current_structural_state_ids": len(current_ids),
        "previous_structural_state_ids": len(previous_ids),
        "caid1_delta_before_name_filter": len(current_ids - previous_ids),
        "caid1_delta_after_name_filter": sum(
            alias["cohort"] == "caid1_history_delta" for alias in aliases
        ),
        "polyprotein_records_excluded": len(excluded_names),
        "polyprotein_disprot_ids": sorted(excluded_names),
        "records_after_filter": len(aliases),
        "unique_sequences_after_merge": len(records),
        "sequences_with_positive_labels": positive_sequences,
        "sequences_without_positive_labels": len(records) - positive_sequences,
        "positive_residues_after_merge": sum(
            int(record["label_counts"]["positive"]) for record in records
        ),
        "unknown_residues_after_merge": sum(
            int(record["label_counts"]["unknown"]) for record in records
        ),
        "negative_residues": 0,
        "allowed_evidence_region_counts": dict(sorted(allowed_evidence_counts.items())),
        "allowed_evidence_residue_spans_before_merge": dict(
            sorted(allowed_evidence_spans.items())
        ),
        "excluded_evidence_region_counts": dict(sorted(excluded_evidence_counts.items())),
        "excluded_evidence_residue_spans": dict(sorted(excluded_evidence_spans.items())),
    }
    return records, summary
