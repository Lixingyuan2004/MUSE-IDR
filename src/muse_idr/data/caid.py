"""Strict parsers and builders for CAID holdout sentinels.

CAID labels may be validated in memory, but sentinel outputs deliberately omit
them so training-data code can reject targets without gaining access to answers.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")
CAID_LABELS = frozenset("01-")
DISPROT_ID_RE = re.compile(r"DP\d{5}", re.IGNORECASE)


@dataclass(frozen=True)
class SequenceRecord:
    """A normalized protein sequence and its source header."""

    identifier: str
    sequence: str
    header: str


@dataclass(frozen=True)
class CaidReferenceRecord(SequenceRecord):
    """A CAID reference record; labels must never enter sentinel output."""

    labels: str


def normalize_sequence(sequence: str) -> str:
    """Uppercase a protein sequence, remove whitespace and reject bad symbols."""

    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("Protein sequence is empty")
    invalid = sorted(set(normalized) - AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Invalid amino-acid symbols: {''.join(invalid)}")
    return normalized


def sequence_sha256(sequence: str) -> str:
    """Return SHA-256 of the normalized uppercase sequence."""

    return hashlib.sha256(normalize_sequence(sequence).encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    """Stream a file into SHA-256 without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_disprot_id(header: str) -> str:
    """Extract and normalize a DisProt identifier from a FASTA header."""

    match = DISPROT_ID_RE.search(header)
    if match is None:
        raise ValueError(f"No DisProt identifier in header: {header!r}")
    return match.group(0).upper()


def parse_fasta(path: Path) -> Iterator[SequenceRecord]:
    """Parse ordinary FASTA while enforcing unique identifiers and sequences."""

    header: str | None = None
    sequence_lines: list[str] = []
    seen: dict[str, str] = {}

    def emit() -> SequenceRecord | None:
        if header is None:
            return None
        sequence = normalize_sequence("".join(sequence_lines))
        identifier = header.split()[0]
        if identifier in seen:
            if seen[identifier] != sequence:
                raise ValueError(f"Duplicate FASTA ID with different sequence: {identifier}")
            raise ValueError(f"Duplicate FASTA ID: {identifier}")
        seen[identifier] = sequence
        return SequenceRecord(identifier=identifier, sequence=sequence, header=header)

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                record = emit()
                if record is not None:
                    yield record
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header at {path}:{line_number}")
                sequence_lines = []
            else:
                if header is None:
                    raise ValueError(
                        f"Expected FASTA header at {path}:{line_number}; "
                        "the download may be HTML"
                    )
                sequence_lines.append(line)

    record = emit()
    if record is not None:
        yield record
    elif header is None:
        raise ValueError(f"No FASTA records found in {path}")


def parse_caid_reference(path: Path) -> Iterator[CaidReferenceRecord]:
    """Parse the official three-line CAID reference format strictly."""

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"CAID reference does not start with '>' (possibly HTML): {path}")
    if len(lines) % 3:
        raise ValueError(f"CAID reference must contain header/sequence/labels triplets: {path}")

    seen: dict[str, str] = {}
    for index in range(0, len(lines), 3):
        header_line, sequence_line, labels = lines[index : index + 3]
        if not header_line.startswith(">"):
            raise ValueError(f"Expected CAID header at logical line {index + 1}: {path}")
        header = header_line[1:].strip()
        identifier = header.split()[0]
        sequence = normalize_sequence(sequence_line)
        if set(labels) - CAID_LABELS:
            invalid = "".join(sorted(set(labels) - CAID_LABELS))
            raise ValueError(f"Invalid CAID label symbols {invalid!r} for {identifier}")
        if len(sequence) != len(labels):
            raise ValueError(
                f"Sequence/label length mismatch for {identifier}: "
                f"{len(sequence)} != {len(labels)}"
            )
        if identifier in seen:
            if seen[identifier] != sequence:
                raise ValueError(f"Duplicate CAID ID with different sequence: {identifier}")
            raise ValueError(f"Duplicate CAID ID: {identifier}")
        seen[identifier] = sequence
        yield CaidReferenceRecord(
            identifier=identifier,
            sequence=sequence,
            header=header,
            labels=labels,
        )


def fasta_disprot_ids(path: Path) -> set[str]:
    """Return DisProt IDs from a release FASTA without parsing its label strings."""

    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.startswith(">"):
                identifiers.add(extract_disprot_id(raw_line[1:].strip()))
    if not identifiers:
        raise ValueError(f"No DisProt FASTA headers found in {path}")
    return identifiers


def reconstruct_caid1_targets(
    current_release_fasta: Path,
    exclude_release_fasta: Path,
    release_json: Path,
) -> list[tuple[SequenceRecord, str | None]]:
    """Reconstruct the conservative CAID1 target pool from official snapshots.

    Returns `(record, uniprot_accession)` pairs. Structural-state strings in the
    release FASTA files are used only to obtain identifiers; amino-acid sequences
    come from the matching DisProt JSON snapshot.
    """

    current_ids = fasta_disprot_ids(current_release_fasta)
    excluded_ids = fasta_disprot_ids(exclude_release_fasta)
    payload = json.loads(release_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"Unexpected DisProt JSON structure: {release_json}")

    entries: dict[str, dict[str, object]] = {}
    for entry in payload["data"]:
        if not isinstance(entry, dict) or "disprot_id" not in entry:
            raise ValueError(f"Malformed DisProt entry in {release_json}")
        entries[str(entry["disprot_id"]).upper()] = entry

    targets: list[tuple[SequenceRecord, str | None]] = []
    for identifier in sorted(current_ids - excluded_ids):
        entry = entries.get(identifier)
        if entry is None:
            raise ValueError(f"Missing JSON entry for {identifier}")
        name = str(entry.get("name", ""))
        if "polyprotein" in name.lower():
            continue
        sequence = normalize_sequence(str(entry.get("sequence", "")))
        accession_value = entry.get("acc")
        accession = str(accession_value).upper() if accession_value else None
        record = SequenceRecord(identifier=identifier, sequence=sequence, header=identifier)
        targets.append((record, accession))
    return targets


def write_fasta(records: Iterable[SequenceRecord], output: Path, width: int = 80) -> None:
    """Write normalized standard FASTA; intended for ignored holdout artifacts."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            header = record.header.strip()
            if not header:
                raise ValueError(f"Empty FASTA header for {record.identifier}")
            handle.write(f">{header}\n")
            for start in range(0, len(record.sequence), width):
                handle.write(record.sequence[start : start + width] + "\n")


def subset_sentinel(
    manifest: Mapping[str, object],
    protected_editions: Iterable[str],
) -> dict[str, object]:
    """Derive a label-free sentinel containing only selected CAID editions."""

    if manifest.get("labels_included") is not False:
        raise ValueError("Refusing to subset a manifest that may contain labels")
    editions = {edition.lower() for edition in protected_editions}
    if not editions:
        raise ValueError("At least one protected edition is required")

    raw_targets = manifest.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError("Sentinel targets must be a list")

    targets: list[dict[str, object]] = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict) or not isinstance(raw_target.get("sources"), list):
            raise ValueError("Malformed sentinel target")
        kept_sources = [
            str(source)
            for source in raw_target["sources"]
            if str(source).split(":", 1)[0].lower() in editions
        ]
        if kept_sources:
            targets.append({**raw_target, "sources": sorted(kept_sources)})

    all_ids = {
        str(identifier)
        for target in targets
        for identifier in target.get("reference_ids", [])
    }
    original_unmapped = {
        str(identifier) for identifier in manifest.get("unmapped_reference_id_values", [])
    }
    unmapped_ids = sorted(all_ids & original_unmapped)

    raw_sources = manifest.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("Sentinel sources must be a list")
    sources = [
        source
        for source in raw_sources
        if isinstance(source, dict)
        and (
            str(source.get("edition", "")).lower() in editions
            or source.get("role") == "disprot_to_uniprot_identifier_mapping"
        )
    ]

    return {
        "schema_version": manifest.get("schema_version", 1),
        "generated_utc": manifest.get("generated_utc"),
        "labels_included": False,
        "purpose": "external_test_training_exclusion",
        "protected_editions": sorted(editions),
        "normalization": manifest.get("normalization"),
        "hash_algorithm": manifest.get("hash_algorithm"),
        "policy": manifest.get("policy"),
        "summary": {
            "unique_reference_ids": len(all_ids),
            "unique_sequences": len(targets),
            "mapped_reference_ids": len(all_ids) - len(unmapped_ids),
            "unmapped_reference_ids": len(unmapped_ids),
        },
        "unmapped_reference_id_values": unmapped_ids,
        "sources": sources,
        "targets": sorted(targets, key=lambda row: str(row["sequence_sha256"])),
    }
