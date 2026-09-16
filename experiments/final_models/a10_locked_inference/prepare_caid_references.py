"""Validate CAID reference files and export sequence-only inference inputs.

The source references remain in a quarantine directory. This program validates
their structure but writes no labels to the inference directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ALLOWED_LABELS = frozenset("01-")
ALLOWED_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


@dataclass(frozen=True)
class ReferenceRecord:
    protein_id: str
    sequence: str
    labels: str

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_reference(path: Path) -> list[ReferenceRecord]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError(f"reference is not three non-empty lines per target: {path}")
    records: list[ReferenceRecord] = []
    seen: set[str] = set()
    for offset in range(0, len(lines), 3):
        header, sequence, labels = lines[offset : offset + 3]
        if not header.startswith(">") or not header[1:].strip():
            raise ValueError(f"invalid reference header at non-empty line {offset + 1}")
        protein_id = header[1:].split(maxsplit=1)[0]
        sequence = sequence.upper()
        if protein_id in seen:
            raise ValueError(f"duplicate reference protein ID: {protein_id}")
        invalid_residues = sorted(set(sequence) - ALLOWED_RESIDUES)
        if invalid_residues:
            raise ValueError(f"unsupported residues in {protein_id}: {invalid_residues}")
        invalid_labels = sorted(set(labels) - ALLOWED_LABELS)
        if invalid_labels:
            raise ValueError(f"invalid reference labels in {protein_id}: {invalid_labels}")
        if len(sequence) != len(labels):
            raise ValueError(
                f"sequence/label length mismatch for {protein_id}: {len(sequence)} != {len(labels)}"
            )
        seen.add(protein_id)
        records.append(ReferenceRecord(protein_id, sequence, labels))
    if not records:
        raise ValueError(f"empty CAID reference: {path}")
    return records


def read_training_sequences(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if {"labels", "label", "scores"} & set(row):
                raise ValueError(f"training audit manifest contains labels at line {line_number}")
            protein_id = str(row["id"])
            sequence = str(row["sequence"]).upper().replace(" ", "")
            if protein_id in sequences:
                raise ValueError(f"duplicate training protein ID: {protein_id}")
            sequences[protein_id] = sequence
    return sequences


def write_fasta(path: Path, records: Sequence[ReferenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f">{record.protein_id}\n{record.sequence}\n")
    temporary.replace(path)


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def dataset_summary(path: Path, records: Sequence[ReferenceRecord]) -> dict[str, object]:
    labels = "".join(record.labels for record in records)
    return {
        "source": str(path),
        "source_sha256": file_sha256(path),
        "proteins": len(records),
        "residues": sum(len(record.sequence) for record in records),
        "positive_labels": labels.count("1"),
        "negative_labels": labels.count("0"),
        "excluded_labels": labels.count("-"),
        "all_sequence_label_lengths_match": True,
        "label_alphabet": ["-", "0", "1"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nox-reference", type=Path, required=True)
    parser.add_argument("--pdb-reference", type=Path, required=True)
    parser.add_argument("--training-sequences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nox = parse_reference(args.nox_reference)
    pdb = parse_reference(args.pdb_reference)
    training = read_training_sequences(args.training_sequences)

    union_by_id: dict[str, ReferenceRecord] = {}
    for record in [*nox, *pdb]:
        previous = union_by_id.get(record.protein_id)
        if previous is not None and previous.sequence != record.sequence:
            raise ValueError(f"NOX/PDB sequence disagreement: {record.protein_id}")
        if previous is None:
            union_by_id[record.protein_id] = record
    union = list(union_by_id.values())

    training_hash_to_ids: dict[str, list[str]] = {}
    for protein_id, sequence in training.items():
        digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        training_hash_to_ids.setdefault(digest, []).append(protein_id)
    exact_sequence_overlaps = [
        {
            "caid_id": record.protein_id,
            "training_ids": training_hash_to_ids[record.sequence_sha256],
        }
        for record in union
        if record.sequence_sha256 in training_hash_to_ids
    ]
    exact_id_overlaps = [
        protein_id for protein_id in union_by_id if protein_id in training
    ]

    sequence_dir = args.output_dir / "sequences_only"
    nox_output = sequence_dir / "caid3_disorder_nox_sequences.fasta"
    pdb_output = sequence_dir / "caid3_disorder_pdb_sequences.fasta"
    union_output = sequence_dir / "caid3_disorder_union_sequences.fasta"
    write_fasta(nox_output, nox)
    write_fasta(pdb_output, pdb)
    write_fasta(union_output, union)

    report = {
        "schema_version": 1,
        "status": "pass",
        "purpose": "locked_external_test_input_preparation",
        "nox": dataset_summary(args.nox_reference, nox),
        "pdb": dataset_summary(args.pdb_reference, pdb),
        "nox_pdb_shared_ids": len(set(r.protein_id for r in nox) & set(r.protein_id for r in pdb)),
        "union_proteins": len(union),
        "union_residues": sum(len(record.sequence) for record in union),
        "sequence_outputs": {
            "nox": str(nox_output),
            "nox_sha256": file_sha256(nox_output),
            "pdb": str(pdb_output),
            "pdb_sha256": file_sha256(pdb_output),
            "union": str(union_output),
            "union_sha256": file_sha256(union_output),
        },
        "training_sequence_manifest": str(args.training_sequences),
        "training_sequence_manifest_sha256": file_sha256(args.training_sequences),
        "training_proteins": len(training),
        "exact_id_overlaps": exact_id_overlaps,
        "exact_sequence_overlaps": exact_sequence_overlaps,
        "exact_sequence_overlap_count": len(exact_sequence_overlaps),
        "homology_audit_pending": True,
        "caid_used_for_training_or_tuning": False,
        "inference_inputs_contain_labels": False,
    }
    report_path = args.output_dir / "caid3_preparation_report.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
