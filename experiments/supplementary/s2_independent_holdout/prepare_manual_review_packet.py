"""Prepare a prediction-blind manual review packet for S2 positive regions.

The script only joins already pinned DisProt evidence with the pre-freeze cohort.
It does not accept annotations automatically, read model predictions, or create a
formal benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
PINS = {
    "data/raw/disprot/2026_06/disprot.json": (
        "afed1664507df330c22ad5ddd59796af49851e1f9df3f88a7ec0fb87f37d0d9b"
    ),
    (
        "outputs/supplementary/s2_independent_holdout/evidence_audit_20260906/"
        "annotation_review_queue.tsv"
    ): "1d3f526c20fe6de79588d8736bae1341196c2e0ebd6661dad4bce373937f4a7f",
    (
        "outputs/supplementary/s2_independent_holdout/prefreeze_20260906/"
        "candidate_decisions.tsv"
    ): "f94ff8dab35d29075e80ef7bf0c348c56a062eb28f28ebfb14124d2664def768",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def compact_statements(statements: Iterable[dict[str, Any]]) -> str:
    values = [
        {
            "type": compact_text(row.get("type")),
            "text": compact_text(row.get("text")),
        }
        for row in statements
    ]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def evidence_fingerprint(row: dict[str, Any]) -> str:
    payload = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/supplementary/s2_independent_holdout/"
            "manual_review_packet_20260906"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite review packet: {output_dir}")

    for relative, expected in PINS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"input SHA256 mismatch: {relative}: {observed}")

    decisions = read_tsv(ROOT / next(key for key in PINS if key.endswith("candidate_decisions.tsv")))
    retained_ids = {
        row["protein_id"] for row in decisions if parse_bool(row["prefreeze_retained"])
    }
    retained_positive_ids = {
        row["protein_id"]
        for row in decisions
        if parse_bool(row["prefreeze_retained"]) and int(row["positive_residues"]) > 0
    }

    queue = read_tsv(ROOT / next(key for key in PINS if key.endswith("annotation_review_queue.tsv")))
    selected_queue = [
        row
        for row in queue
        if row["protein_id"] in retained_positive_ids
        and parse_bool(row["provisional_direct_evidence"])
    ]

    snapshot = json.loads(
        (ROOT / "data/raw/disprot/2026_06/disprot.json").read_text(
            encoding="utf-8-sig"
        )
    )
    entries = {str(row["disprot_id"]): row for row in snapshot["data"]}
    if len(entries) != len(snapshot["data"]):
        raise ValueError("duplicate DisProt identifiers in snapshot")

    region_index: dict[tuple[str, str], dict[str, Any]] = {}
    for protein_id in retained_positive_ids:
        entry = entries[protein_id]
        for region in entry.get("regions", []):
            key = (protein_id, str(region.get("region_id") or ""))
            if key in region_index:
                raise ValueError(f"duplicate region identifier: {key}")
            region_index[key] = region

    evidence_rows: list[dict[str, Any]] = []
    for queued in selected_queue:
        protein_id = queued["protein_id"]
        entry = entries[protein_id]
        region_id = queued["region_id"]
        region = region_index[(protein_id, region_id)]
        validated = region.get("validated") if isinstance(region.get("validated"), dict) else {}
        reference_id = compact_text(region.get("reference_id"))
        source = compact_text(region.get("reference_source")).casefold()
        objective = {
            "protein_id": protein_id,
            "accession": compact_text(entry.get("acc")),
            "protein_name": compact_text(entry.get("name")),
            "organism": compact_text(entry.get("organism")),
            "sequence_length": int(entry["length"]),
            "region_id": region_id,
            "start": int(region["start"]),
            "end": int(region["end"]),
            "region_length": int(region["end"]) - int(region["start"]) + 1,
            "term_id": compact_text(region.get("term_id")),
            "term_name": compact_text(region.get("term_name")),
            "eco_id": compact_text(region.get("ec_id")),
            "eco_name": compact_text(region.get("ec_name")),
            "reference_source": source,
            "reference_id": reference_id,
            "reference_citation": compact_text(region.get("reference_html")),
            "reference_url": (
                f"https://pubmed.ncbi.nlm.nih.gov/{reference_id}/"
                if source == "pmid" and reference_id
                else ""
            ),
            "disprot_url": f"https://disprot.org/{protein_id}",
            "annotation_released": compact_text(region.get("released")),
            "annotation_version": region.get("version", ""),
            "curator_name": compact_text(region.get("curator_name")),
            "curator_orcid": compact_text(region.get("curator_orcid")),
            "validator_name": compact_text(validated.get("curator_name")),
            "validator_timestamp": compact_text(validated.get("timestamp")),
            "snapshot_unpublished": region.get("unpublished"),
            "snapshot_obsolete": bool(region.get("obsolete")),
            "construct_alterations": compact_text(region.get("construct_alterations")),
            "statements_json": compact_statements(region.get("statement", [])),
        }
        objective["evidence_sha256"] = evidence_fingerprint(objective)
        evidence_rows.append(objective)

    evidence_rows.sort(
        key=lambda row: (row["protein_id"], row["start"], row["end"], row["region_id"])
    )
    observed_positive_ids = {row["protein_id"] for row in evidence_rows}
    if observed_positive_ids != retained_positive_ids:
        raise ValueError(
            "positive review coverage mismatch: "
            f"expected {sorted(retained_positive_ids)}, got {sorted(observed_positive_ids)}"
        )
    if not evidence_rows:
        raise RuntimeError("manual review packet would be empty")

    output_dir.mkdir(parents=True)
    evidence_fields = list(evidence_rows[0])
    evidence_path = output_dir / "positive_region_evidence.tsv"
    write_tsv(evidence_path, evidence_rows, evidence_fields)

    review_fields = evidence_fields + [
        "review_decision",
        "coordinate_scope_confirmed",
        "direct_disorder_support_confirmed",
        "reference_sequence_construct_confirmed",
        "public_annotation_status_confirmed",
        "reviewer",
        "review_date",
        "review_notes",
    ]
    review_rows = [
        {
            **row,
            "review_decision": "",
            "coordinate_scope_confirmed": "",
            "direct_disorder_support_confirmed": "",
            "reference_sequence_construct_confirmed": "",
            "public_annotation_status_confirmed": "",
            "reviewer": "",
            "review_date": "",
            "review_notes": "",
        }
        for row in evidence_rows
    ]
    form_path = output_dir / "manual_review_form_TEMPLATE.tsv"
    write_tsv(form_path, review_rows, review_fields)

    evidence_counts = Counter((row["eco_id"], row["eco_name"]) for row in evidence_rows)
    report = {
        "schema_version": 1,
        "experiment": "s2_prediction_blind_manual_positive_review_packet",
        "status": "manual_review_packet_ready_decisions_pending",
        "input_pins": PINS,
        "prefreeze_retained_proteins": len(retained_ids),
        "positive_review": {
            "proteins": len(retained_positive_ids),
            "regions": len(evidence_rows),
            "distinct_references": len({row["reference_id"] for row in evidence_rows}),
            "evidence_counts": [
                {"eco_id": key[0], "eco_name": key[1], "regions": count}
                for key, count in sorted(evidence_counts.items())
            ],
        },
        "permitted_review_decisions": ["accept", "reject", "uncertain"],
        "ready_for_prediction": False,
        "manual_decisions_completed": False,
        "caid_labels_accessed": False,
        "model_predictions_accessed": False,
        "locked_a10_modified": False,
    }
    report_path = output_dir / "manual_review_packet_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    instructions = f"""# S2 positive-region manual review packet

This packet was prepared before reading any MUSE-IDR or baseline prediction.
It contains {len(evidence_rows)} provisional positive regions from
{len(retained_positive_ids)} retained proteins and {report['positive_review']['distinct_references']}
distinct references. It is not a final reference set.

## Required review for every row

1. Open the cited article and the public DisProt entry/history.
2. Confirm that the reported experiment directly supports intrinsic disorder,
   rather than only flexibility, solvent exposure, prediction, aggregation, or
   an indirect functional inference.
3. Confirm that the evidence supports the complete annotated coordinate range.
   If only a smaller range is supported, mark the row `uncertain` and record the
   defensible coordinates in `review_notes`; do not silently edit this packet.
4. Confirm that the experimental construct and residue numbering map to the
   pinned full-length accession sequence. Construct alterations require an
   explicit mapping or rejection.
5. Resolve the snapshot `unpublished` flag against the public entry and version
   history. Public visibility alone does not validate the scientific evidence.
6. Enter exactly `accept`, `reject`, or `uncertain`, complete all four confirmation
   columns with `yes`/`no`, and record reviewer, ISO date, and notes.

An annotation is accepted only when all four confirmations are `yes`. Treat
missing evidence or unresolved ambiguity as `uncertain`, which will be excluded
from the formal positive reference. Decisions must not use model scores.

## Editing workflow

Do not edit `manual_review_form_TEMPLATE.tsv` in place because its initial hash is
locked. Copy it to a new file named `manual_review_form_COMPLETED.tsv`, complete
that copy, and run a separate validation/finalization step. The finalization step
must reject blank or inconsistent decisions and create a new immutable lock before
any prediction is run.
"""
    instructions_path = output_dir / "MANUAL_REVIEW_INSTRUCTIONS.md"
    instructions_path.write_text(instructions, encoding="utf-8", newline="\n")

    lock_targets = [
        Path(__file__),
        ROOT / "experiments/supplementary/s2_independent_holdout/test_manual_review_packet.py",
        evidence_path,
        form_path,
        report_path,
        instructions_path,
    ]
    lock_lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in lock_targets
    ]
    (output_dir / "manual_review_packet_lock.sha256").write_text(
        "\n".join(lock_lines) + "\n", encoding="ascii", newline="\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
