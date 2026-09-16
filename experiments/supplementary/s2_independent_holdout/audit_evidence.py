"""Audit residue-label evidence for the sequence-screened S2 candidates.

This script deliberately does not create a formal benchmark.  It separates
direct experimental disorder evidence from annotations requiring review and
inventories structure mappings whose PDB quality metadata is still missing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_RELEASE = "2026_06"
PINS = {
    "data/raw/disprot/2026_06/disprot.json":
        "afed1664507df330c22ad5ddd59796af49851e1f9df3f88a7ec0fb87f37d0d9b",
    "data/raw/sifts/2026-08-21/uniprot_segments_observed.csv.gz":
        "4f08d0fb3b793f8aecef3ca1add79bf1f926fef415473afaa7f74ee2b463ea21",
    "data/raw/pdbe/2026-08-21/historical_disprot_pdb_metadata.json":
        "d78734812ee6f01a5dd1fbeedee0e1d9d92e3a2d353cffaa0ebfe4a290721e10",
}

# Conservative, exact ECO allowlist corresponding to direct biophysical
# techniques accepted by the historical training policy.  Evidence based on
# predictions, missing coordinates, generic inference, or underspecified
# combinatorial evidence is never promoted automatically.
DIRECT_DISORDER_ECO = {
    "ECO:0001183",  # FRET
    "ECO:0001184",  # gel filtration
    "ECO:0001238",  # NMR
    "ECO:0005642",  # HSQC
    "ECO:0006165",  # NMR spectroscopy
    "ECO:0006198",  # proton NMR
    "ECO:0006200",  # circular dichroism
    "ECO:0006204",  # far-UV circular dichroism
    "ECO:0006208",  # cryogenic electron microscopy
    "ECO:0006210",  # SAXS
    "ECO:0006236",  # HDX-MS
    "ECO:0006275",  # analytical ultracentrifugation
    "ECO:0006285",  # magnetic resonance
    "ECO:0006289",  # spin-label EPR
    "ECO:0006334",  # differential scanning fluorimetry
    "ECO:0007064",  # dynamic light scattering
    "ECO:0007689",  # SDS-PAGE
}
EXCLUDED_ECO = {
    "ECO:0006220": "missing_xray_coordinates_not_direct_positive",
    "ECO:0006224": "missing_cryoem_coordinates_not_direct_positive",
    "ECO:0008031": "prediction_based_author_inference",
    "ECO:0008033": "intrinsic_disorder_prediction",
    "ECO:0008035": "prediction_based_author_inference",
    "ECO:0008037": "prediction_based_curator_inference",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    identifier: str | None = None
    parts: list[str] = []
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records[identifier] = "".join(parts)
            identifier = line[1:].split(maxsplit=1)[0]
            if not identifier or identifier in records:
                raise ValueError(f"invalid or duplicate FASTA identifier: {identifier}")
            parts = []
        elif identifier is None:
            raise ValueError("sequence before first FASTA header")
        else:
            parts.append("".join(line.split()).upper())
    if identifier is not None:
        records[identifier] = "".join(parts)
    if not records:
        raise ValueError(f"no FASTA records: {path}")
    return records


def interval(region: dict[str, Any], length: int) -> tuple[int, int] | None:
    try:
        start, end = int(region["start"]), int(region["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if start < 1 or end < start or end > length:
        return None
    return start, end


def classify_disorder_region(region: dict[str, Any], length: int) -> dict[str, Any]:
    reasons: list[str] = []
    coordinates = interval(region, length)
    eco = str(region.get("ec_id") or "")
    if coordinates is None:
        reasons.append("invalid_coordinates")
    if region.get("obsolete"):
        reasons.append("obsolete_annotation")
    release = str(region.get("released") or "")
    if not re.fullmatch(r"\d{4}_\d{2}", release) or release > SNAPSHOT_RELEASE:
        reasons.append("annotation_not_in_pinned_release")
    if eco in EXCLUDED_ECO:
        reasons.append(EXCLUDED_ECO[eco])
    elif eco not in DIRECT_DISORDER_ECO:
        reasons.append("evidence_requires_manual_review")
    if not region.get("validated"):
        reasons.append("curator_validation_not_recorded")
    if region.get("construct_alterations"):
        reasons.append("experimental_construct_differs_from_reference")

    provisional = not reasons
    # The snapshot sets `unpublished: true` for every candidate structural-state
    # annotation.  Its semantics could not be verified while the official API
    # was unavailable, so even otherwise strong evidence is not a formal label.
    formal = provisional and region.get("unpublished") is not True
    if provisional and not formal:
        reasons.append("unpublished_field_semantics_unresolved")
    return {
        "coordinates": coordinates,
        "provisional_direct_evidence": provisional,
        "formal_eligible": formal,
        "reasons": reasons,
    }


def union_positions(intervals: Iterable[tuple[int, int]], length: int) -> set[int]:
    positions: set[int] = set()
    for start, end in intervals:
        if start < 1 or end < start or end > length:
            raise ValueError("invalid interval")
        positions.update(range(start, end + 1))
    return positions


def read_sifts(path: Path, accessions: set[str]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    reader = csv.DictReader(io.StringIO("".join(lines)))
    expected = {"PDB", "CHAIN", "SP_PRIMARY", "RES_BEG", "RES_END", "SP_BEG", "SP_END"}
    if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
        raise ValueError(f"unexpected SIFTS columns: {reader.fieldnames}")
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unequal_spans = 0
    for row in reader:
        accession = str(row["SP_PRIMARY"]).strip().upper()
        if accession not in accessions:
            continue
        try:
            res_start, res_end = int(row["RES_BEG"]), int(row["RES_END"])
            sp_start, sp_end = int(row["SP_BEG"]), int(row["SP_END"])
        except ValueError:
            continue
        equal_span = res_end - res_start == sp_end - sp_start
        unequal_spans += int(not equal_span)
        result[accession].append(
            {
                "pdb_id": str(row["PDB"]).strip().lower(),
                "chain": str(row["CHAIN"]).strip(),
                "res_start": res_start,
                "res_end": res_end,
                "sp_start": sp_start,
                "sp_end": sp_end,
                "equal_span": equal_span,
            }
        )
    return result, unequal_spans


def pdbe_inventory(path: Path) -> tuple[set[str], set[str]]:
    payload = read_json(path)
    summary = {str(key).lower() for key in payload.get("summary", {})}
    experiment = {str(key).lower() for key in payload.get("experiment", {})}
    return summary, experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-fasta",
        type=Path,
        default=Path(
            "outputs/supplementary/s2_independent_holdout/"
            "source_audit_20260906/after_2023_06_sequence_candidates.fasta"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/supplementary/s2_independent_holdout/"
            "evidence_audit_20260906"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_dir}")

    verified_hashes: dict[str, str] = {}
    for relative, expected in PINS.items():
        path = ROOT / relative
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
        verified_hashes[relative] = observed

    fasta_path = args.candidate_fasta if args.candidate_fasta.is_absolute() else ROOT / args.candidate_fasta
    sequences = read_fasta(fasta_path)
    snapshot = read_json(ROOT / "data/raw/disprot/2026_06/disprot.json")
    entries = {str(row["disprot_id"]): row for row in snapshot["data"]}
    missing = sorted(set(sequences) - set(entries))
    if missing:
        raise ValueError(f"candidate identifiers absent from snapshot: {missing[:5]}")

    review_rows: list[dict[str, Any]] = []
    protein_rows: list[dict[str, Any]] = []
    all_states = Counter()
    evidence = Counter()
    provisional_positive_residues = 0
    proteins_with_provisional_positive = 0
    all_unpublished = True

    accession_to_id: dict[str, str] = {}
    for protein_id, sequence in sequences.items():
        entry = entries[protein_id]
        accession = str(entry.get("acc") or "").strip().upper()
        if not accession or accession in accession_to_id:
            raise ValueError(f"missing or duplicate candidate accession: {accession}")
        accession_to_id[accession] = protein_id
        intervals: list[tuple[int, int]] = []
        state_regions = [
            region for region in entry.get("regions", [])
            if str(region.get("term_namespace") or "").casefold() == "structural state"
        ]
        for region in state_regions:
            term = str(region.get("term_name") or "").casefold()
            all_states[term] += 1
            all_unpublished = all_unpublished and region.get("unpublished") is True
            if term != "disorder":
                continue
            decision = classify_disorder_region(region, len(sequence))
            evidence[(str(region.get("ec_id") or ""), str(region.get("ec_name") or ""))] += 1
            if decision["provisional_direct_evidence"]:
                assert decision["coordinates"] is not None
                intervals.append(decision["coordinates"])
            review_rows.append(
                {
                    "protein_id": protein_id,
                    "accession": accession,
                    "region_id": region.get("region_id"),
                    "start": region.get("start"),
                    "end": region.get("end"),
                    "ec_id": region.get("ec_id"),
                    "ec_name": region.get("ec_name"),
                    "reference_source": region.get("reference_source"),
                    "reference_id": region.get("reference_id"),
                    "released": region.get("released"),
                    "validated": bool(region.get("validated")),
                    "unpublished": region.get("unpublished"),
                    "obsolete": bool(region.get("obsolete")),
                    "has_construct_alterations": bool(region.get("construct_alterations")),
                    "provisional_direct_evidence": decision["provisional_direct_evidence"],
                    "formal_eligible": decision["formal_eligible"],
                    "reasons": ";".join(decision["reasons"]),
                }
            )
        positives = union_positions(intervals, len(sequence))
        provisional_positive_residues += len(positives)
        proteins_with_provisional_positive += int(bool(positives))
        protein_rows.append(
            {
                "protein_id": protein_id,
                "accession": accession,
                "length": len(sequence),
                "structural_state_regions": len(state_regions),
                "provisional_positive_residues": len(positives),
                "provisional_positive_intervals": [list(value) for value in sorted(set(intervals))],
                "formal_positive_residues": 0,
            }
        )

    sifts, unequal_spans = read_sifts(
        ROOT / "data/raw/sifts/2026-08-21/uniprot_segments_observed.csv.gz",
        set(accession_to_id),
    )
    candidate_pdb_ids = {
        segment["pdb_id"] for rows in sifts.values() for segment in rows
    }
    summary_ids, experiment_ids = pdbe_inventory(
        ROOT / "data/raw/pdbe/2026-08-21/historical_disprot_pdb_metadata.json"
    )
    locally_complete_pdbe = candidate_pdb_ids & summary_ids & experiment_ids
    missing_pdbe = sorted(candidate_pdb_ids - locally_complete_pdbe)

    # Delay all writes until the pinned inputs and parsers have succeeded. This
    # avoids leaving a misleading partial audit directory after a preflight error.
    output_dir.mkdir(parents=True)

    proteins_with_sifts = 0
    proteins_with_positive_and_sifts = 0
    raw_mapped_positions = 0
    for row in protein_rows:
        segments = sifts.get(row["accession"], [])
        positions = union_positions(
            (
                (segment["sp_start"], segment["sp_end"])
                for segment in segments
                if segment["equal_span"]
                and 1 <= segment["sp_start"] <= segment["sp_end"] <= row["length"]
            ),
            row["length"],
        )
        row["sifts_segments"] = len(segments)
        row["sifts_pdb_entries"] = len({segment["pdb_id"] for segment in segments})
        row["raw_sifts_mapped_residues"] = len(positions)
        row["qualified_negative_residues"] = 0
        proteins_with_sifts += int(bool(positions))
        proteins_with_positive_and_sifts += int(
            bool(positions) and row["provisional_positive_residues"] > 0
        )
        raw_mapped_positions += len(positions)

    review_path = output_dir / "annotation_review_queue.tsv"
    fieldnames = list(review_rows[0]) if review_rows else []
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_rows)
    with (output_dir / "candidate_evidence_inventory.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in protein_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "pdb_metadata_request_ids.txt").write_text(
        "\n".join(missing_pdbe) + ("\n" if missing_pdbe else ""),
        encoding="ascii",
        newline="\n",
    )

    reason_counts = Counter(
        reason for row in review_rows for reason in str(row["reasons"]).split(";") if reason
    )
    report = {
        "schema_version": 1,
        "experiment": "s2_independent_holdout_evidence_availability_audit",
        "status": "audit_complete_formal_test_not_ready",
        "candidate_source": str(fasta_path.relative_to(ROOT)).replace("\\", "/"),
        "candidate_source_sha256": sha256(fasta_path),
        "verified_source_hashes": verified_hashes,
        "candidates": len(sequences),
        "candidate_residues": sum(map(len, sequences.values())),
        "structural_state_regions": sum(all_states.values()),
        "structural_state_terms": dict(sorted(all_states.items())),
        "disorder_evidence_regions": len(review_rows),
        "evidence_counts": [
            {"ec_id": key[0], "ec_name": key[1], "regions": value}
            for key, value in sorted(evidence.items())
        ],
        "review_reason_counts": dict(sorted(reason_counts.items())),
        "provisional_direct_evidence": {
            "proteins": proteins_with_provisional_positive,
            "positive_residues": provisional_positive_residues,
            "definition": (
                "direct ECO allowlist, valid coordinates, non-obsolete, released by "
                "2026_06, curator validation recorded, no construct alteration"
            ),
            "formal_labels": False,
        },
        "annotation_release_guard": {
            "all_candidate_structural_state_regions_unpublished_true": all_unpublished,
            "official_api_status_during_audit": "HTTP 503",
            "consequence": "unpublished field semantics require resolution before labels are frozen",
        },
        "structure_negative_availability": {
            "proteins_with_raw_sifts_mapping": proteins_with_sifts,
            "proteins_with_provisional_positive_and_raw_sifts_mapping": proteins_with_positive_and_sifts,
            "candidate_pdb_ids": len(candidate_pdb_ids),
            "raw_sifts_mapped_residue_sum_by_protein": raw_mapped_positions,
            "unequal_coordinate_span_rows_excluded_from_residue_count": unequal_spans,
            "pdb_ids_with_local_summary_and_experiment_metadata": len(locally_complete_pdbe),
            "pdb_ids_requiring_metadata_fetch": len(missing_pdbe),
            "qualified_negative_residues": 0,
            "formal_labels": False,
        },
        "formal_benchmark": {
            "ready": False,
            "blocking_items": [
                "resolve DisProt unpublished-field semantics and review candidate annotations",
                "fetch and pin PDBe summary/experiment metadata for candidate PDB entries",
                "apply PDB release-date and method-specific resolution thresholds",
                "verify exact accession/sequence/coordinate mapping and resolve positive-negative conflicts",
                "perform local-homology sensitivity analysis and freeze the final cohort before prediction",
            ],
            "caid_labels_accessed": False,
            "model_predictions_accessed": False,
            "locked_a10_modified": False,
        },
    }
    write_json(output_dir / "s2_evidence_audit.json", report)

    markdown = f"""# S2 annotation and structure-evidence availability audit

This is a pre-prediction audit. It does not define a formal test set and does not
read model predictions or CAID labels.

## Result

- Sequence-screened candidates: {len(sequences)} proteins / {sum(map(len, sequences.values())):,} residues.
- Structural-state annotations: {sum(all_states.values())}; disorder evidence rows: {len(review_rows)}.
- Provisional direct-evidence positives: {proteins_with_provisional_positive} proteins / {provisional_positive_residues:,} residues.
- Raw SIFTS mappings: {proteins_with_sifts} proteins and {len(candidate_pdb_ids):,} distinct PDB entries.
- PDB entries with both summary and experiment metadata already available locally: {len(locally_complete_pdbe):,}.
- PDB entries still requiring pinned quality metadata: {len(missing_pdbe):,}.
- Formal positive labels: 0; formal qualified negative labels: 0.

## Why the benchmark is not ready

The direct-evidence count is provisional. Every candidate structural-state
annotation in the pinned JSON carries `unpublished: true`, while the official
DisProt API returned HTTP 503 during this audit. This field is therefore not
silently ignored: its semantics and the affected region histories must be
resolved before freezing labels.

SIFTS indicates where structures map to candidate sequences, but mapping alone
does not prove a reliable ordered negative. The missing PDBe metadata must be
fetched and pinned; release date, experimental method, resolution, exact
sequence mapping, unequal coordinate spans, protected disorder boundaries, and
positive/negative conflicts must then be checked.

Prediction-based inference and missing-coordinate annotations are excluded from
automatic positive labels. Underspecified or combinatorial evidence remains in
`annotation_review_queue.tsv` for manual review. MIADE construct alterations also
require review because the experimental construct can differ from the reference
sequence.

## Files

- `annotation_review_queue.tsv`: one row per disorder annotation and its disposition.
- `candidate_evidence_inventory.jsonl`: per-protein positive and SIFTS availability counts.
- `pdb_metadata_request_ids.txt`: exact candidate PDB IDs lacking complete local metadata.
- `s2_evidence_audit.json`: machine-readable audit and blockers.

No CAID labels or A10 predictions were accessed, and locked A10 was not modified.
"""
    (output_dir / "s2_evidence_audit.md").write_text(
        markdown, encoding="utf-8", newline="\n"
    )

    lock_targets = [
        Path(__file__),
        ROOT / "experiments/supplementary/s2_independent_holdout/test_audit_evidence.py",
        *sorted(path for path in output_dir.iterdir() if path.name != "s2_evidence_audit_lock.sha256"),
    ]
    lock_lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in lock_targets
    ]
    (output_dir / "s2_evidence_audit_lock.sha256").write_text(
        "\n".join(lock_lines) + "\n", encoding="ascii", newline="\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
