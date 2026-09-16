"""Build a conservative, explicitly non-formal S2 residue reference audit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.supplementary.s2_independent_holdout.audit_evidence import (  # noqa: E402
    read_sifts,
)
from muse_idr.data.pdb_negatives import (  # noqa: E402
    exact_sequence_accession_map,
    merge_observed_negatives,
    protected_masks,
    qualified_pdb_entries,
    read_observed_segments,
)


CUTOFF_DATE = "20260630"
XRAY_MAX_RESOLUTION = 3.0
EM_MAX_RESOLUTION = 4.0
PINS = {
    "data/raw/disprot/2026_06/disprot.json":
        "afed1664507df330c22ad5ddd59796af49851e1f9df3f88a7ec0fb87f37d0d9b",
    "data/raw/sifts/2026-08-21/uniprot_segments_observed.csv.gz":
        "4f08d0fb3b793f8aecef3ca1add79bf1f926fef415473afaa7f74ee2b463ea21",
    "data/raw/pdbe/2026-08-21/historical_disprot_pdb_metadata.json":
        "d78734812ee6f01a5dd1fbeedee0e1d9d92e3a2d353cffaa0ebfe4a290721e10",
    "data/raw/pdbe/2026-09-06/s2_candidate_pdb_metadata.json":
        "88e90ea382d1d1ee15ddfa1b59545ab6c824c64b3a548c3295dadbf9ebdf2d72",
    "outputs/supplementary/s2_independent_holdout/evidence_audit_20260906/"
    "candidate_evidence_inventory.jsonl":
        "5f33c1c02908e5dae46fdafde8206558d1a153cadf6a5685b630c540e91a2b5a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def subset_metadata(metadata: dict[str, Any], wanted: set[str]) -> dict[str, Any]:
    requested = [
        str(value).lower() for value in metadata["requested_pdb_ids"]
        if str(value).lower() in wanted
    ]
    return {
        "requested_pdb_ids": requested,
        "summary": {
            key: value for key, value in metadata["summary"].items()
            if str(key).lower() in wanted
        },
        "experiment": {
            key: value for key, value in metadata["experiment"].items()
            if str(key).lower() in wanted
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/supplementary/s2_independent_holdout/"
            "qualification_audit_20260906"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_dir}")

    verified: dict[str, str] = {}
    for relative, expected in PINS.items():
        path = ROOT / relative
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
        verified[relative] = observed

    evidence_rows = read_jsonl(
        ROOT / "outputs/supplementary/s2_independent_holdout/evidence_audit_20260906/"
        "candidate_evidence_inventory.jsonl"
    )
    current = read_json(ROOT / "data/raw/disprot/2026_06/disprot.json")["data"]
    current_by_id = {str(entry["disprot_id"]): entry for entry in current}
    records: list[dict[str, Any]] = []
    for row in evidence_rows:
        protein_id = str(row["protein_id"])
        entry = current_by_id[protein_id]
        sequence = str(entry["sequence"]).upper()
        if len(sequence) != int(row["length"]):
            raise ValueError(f"sequence length mismatch: {protein_id}")
        labels = [-1] * len(sequence)
        for raw_start, raw_end in row["provisional_positive_intervals"]:
            start, end = int(raw_start), int(raw_end)
            if not 1 <= start <= end <= len(sequence):
                raise ValueError(f"positive interval outside sequence: {protein_id}")
            for index in range(start - 1, end):
                labels[index] = 1
        records.append(
            {
                "protein_id": protein_id,
                "sequence": sequence,
                "labels": labels,
                "uniprot_accessions": [str(entry["acc"]).upper()],
                "disprot_ids": [protein_id],
                "source_entry_release": entry.get("released"),
                "reference_status": "provisional_not_for_model_selection_or_formal_reporting",
            }
        )

    accession_to_record, exact_summary = exact_sequence_accession_map(records, current)
    if exact_summary["records_with_verified_accession"] != len(records):
        raise ValueError("not every candidate has an exact accession+sequence verification")

    sifts_path = ROOT / "data/raw/sifts/2026-08-21/uniprot_segments_observed.csv.gz"
    raw_sifts, raw_unequal = read_sifts(sifts_path, set(accession_to_record))
    candidate_pdb_ids = {
        segment["pdb_id"] for values in raw_sifts.values() for segment in values
    }
    metadata_paths = [
        ROOT / "data/raw/pdbe/2026-08-21/historical_disprot_pdb_metadata.json",
        ROOT / "data/raw/pdbe/2026-09-06/s2_candidate_pdb_metadata.json",
    ]
    qualified: dict[str, dict[str, object]] = {}
    quality_reasons: Counter[str] = Counter()
    metadata_covered: set[str] = set()
    for path in metadata_paths:
        subset = subset_metadata(read_json(path), candidate_pdb_ids)
        metadata_covered.update(str(value).lower() for value in subset["requested_pdb_ids"])
        selected, reasons = qualified_pdb_entries(
            subset,
            cutoff_date=CUTOFF_DATE,
            xray_max_resolution=XRAY_MAX_RESOLUTION,
            em_max_resolution=EM_MAX_RESOLUTION,
        )
        overlap = set(qualified) & set(selected)
        if overlap:
            raise ValueError(f"PDB IDs duplicated across pinned snapshots: {sorted(overlap)[:5]}")
        qualified.update(selected)
        quality_reasons.update(reasons)
    if metadata_covered != candidate_pdb_ids:
        raise ValueError(
            f"candidate PDB metadata incomplete: {len(candidate_pdb_ids - metadata_covered)} missing"
        )

    candidate_entries = [current_by_id[record["protein_id"]] for record in records]
    masks, protected_summary = protected_masks(
        records,
        candidate_entries,
        protected_states={"disorder", "molten globule", "pre-molten globule"},
        boundary_margin=5,
    )
    segments, segment_summary_raw = read_observed_segments(
        sifts_path,
        qualified_pdb_ids=set(qualified),
        accession_to_record=accession_to_record,
        records=records,
    )
    unequal_segments = [
        segment for segment in segments
        if segment.residue_end - segment.residue_begin
        != segment.uniprot_end - segment.uniprot_begin
    ]
    segments = [segment for segment in segments if segment not in unequal_segments]
    merged, label_summary = merge_observed_negatives(records, segments, masks, qualified)

    both_class = [
        row for row in merged
        if row["label_counts"]["positive"] and row["label_counts"]["negative"]
    ]
    labeled = [
        row for row in merged
        if row["label_counts"]["positive"] or row["label_counts"]["negative"]
    ]
    report = {
        "schema_version": 1,
        "experiment": "s2_provisional_reference_qualification_audit",
        "status": "provisional_reference_built_formal_test_not_ready",
        "verified_source_hashes": verified,
        "policy": {
            "pdb_release_cutoff": CUTOFF_DATE,
            "xray_max_resolution_angstrom": XRAY_MAX_RESOLUTION,
            "em_max_resolution_angstrom": EM_MAX_RESOLUTION,
            "nmr_resolution_required": False,
            "require_exact_accession_and_sequence": True,
            "require_equal_pdb_and_uniprot_segment_span": True,
            "protected_state_boundary_margin_residues": 5,
        },
        "candidates": len(records),
        "candidate_pdb_ids": len(candidate_pdb_ids),
        "metadata_covered_pdb_ids": len(metadata_covered),
        "quality_filter_counts": dict(sorted(quality_reasons.items())),
        "qualified_pdb_ids": len(qualified),
        "exact_mapping": exact_summary,
        "protected_regions": protected_summary,
        "sifts_before_equal_span_filter": segment_summary_raw,
        "unequal_sifts_segments_excluded": len(unequal_segments),
        "raw_candidate_unequal_sifts_rows": raw_unequal,
        "sifts_segments_after_equal_span_filter": len(segments),
        "labels": label_summary,
        "proteins_with_both_positive_and_negative_labels": len(both_class),
        "provisional_labeled_proteins": len(labeled),
        "formal_benchmark": {
            "ready": False,
            "reason": (
                "DisProt unpublished-field semantics and region-level manual review "
                "remain unresolved; local-homology sensitivity analysis is pending"
            ),
            "caid_labels_accessed": False,
            "model_predictions_accessed": False,
            "locked_a10_modified": False,
        },
    }

    output_dir.mkdir(parents=True)
    with (output_dir / "provisional_reference.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "qualified_pdb_ids.txt").write_text(
        "\n".join(sorted(qualified)) + "\n", encoding="ascii", newline="\n"
    )
    (output_dir / "provisional_labeled_ids.txt").write_text(
        "\n".join(str(row["protein_id"]) for row in labeled) + "\n",
        encoding="ascii",
        newline="\n",
    )
    (output_dir / "s2_qualification_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = f"""# S2 provisional reference qualification audit

This output is deliberately **not** a frozen external benchmark.

## Quality-qualified structure evidence

- Candidate proteins: {len(records)}.
- Candidate PDB IDs with complete pinned metadata: {len(metadata_covered):,} / {len(candidate_pdb_ids):,}.
- Qualified PDB IDs after date, method, and resolution filters: {len(qualified):,}.
- SIFTS observed segments after equal-span filtering: {len(segments):,}.
- Exact accession and sequence mappings: {exact_summary['verified_exact_accessions']}.

## Provisional residue labels

- Positive residues: {label_summary['positive_residues']:,}.
- Qualified structure-derived negative residues: {label_summary['negative_residues']:,}.
- Unknown residues: {label_summary['unknown_residues']:,}.
- Proteins with positive labels: {label_summary['sequences_with_positive_labels']}.
- Proteins with negative labels: {label_summary['sequences_with_negative_labels']}.
- Proteins with both classes: {len(both_class)}.
- Proteins with at least one provisional label: {len(labeled)}.

## Guardrail

The numerical reference remains provisional because the `unpublished` field in
the pinned DisProt export is unresolved and region-level manual review has not
been completed. These labels must not be used for tuning, model selection, or
formal performance claims. No CAID labels or model predictions were accessed,
and locked A10 was not modified.
"""
    (output_dir / "s2_qualification_audit.md").write_text(
        markdown, encoding="utf-8", newline="\n"
    )
    lock_targets = [
        Path(__file__),
        *metadata_paths,
        *sorted(path for path in output_dir.iterdir() if "lock" not in path.name),
    ]
    (output_dir / "s2_qualification_audit_lock.sha256").write_text(
        "\n".join(
            f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in lock_targets
        ) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
