"""Record prediction-blind S2 human decisions without changing the locked template."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
REVIEW_FIELDS = (
    "review_decision",
    "coordinate_scope_confirmed",
    "direct_disorder_support_confirmed",
    "reference_sequence_construct_confirmed",
    "public_annotation_status_confirmed",
    "reviewer",
    "review_date",
    "review_notes",
)
CONFIRMATION_FIELDS = REVIEW_FIELDS[1:5]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"missing TSV header: {path}")
        return list(reader.fieldnames), list(reader)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_decision(decision: dict[str, Any]) -> None:
    outcome = str(decision.get("review_decision", ""))
    if outcome not in {"accept", "reject", "uncertain"}:
        raise ValueError(f"invalid review decision: {outcome!r}")
    confirmations = [str(decision.get(field, "")) for field in CONFIRMATION_FIELDS]
    if any(value not in {"yes", "no"} for value in confirmations):
        raise ValueError("all confirmation fields must be yes or no")
    if outcome == "accept" and any(value != "yes" for value in confirmations):
        raise ValueError("accepted regions require four yes confirmations")
    if outcome == "reject" and all(value == "yes" for value in confirmations):
        raise ValueError("rejected regions require at least one no confirmation")
    if not str(decision.get("review_notes", "")).strip():
        raise ValueError("review notes are required")
    if not str(decision.get("source_evidence_sha256", "")).strip():
        raise ValueError("source evidence hash is required")


def apply_decisions(
    rows: list[dict[str, str]],
    decisions: list[dict[str, Any]],
    reviewer: str,
    review_date: str,
) -> tuple[list[dict[str, str]], int]:
    if not reviewer.strip():
        raise ValueError("reviewer is required")
    date.fromisoformat(review_date)
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for decision in decisions:
        validate_decision(decision)
        key = (str(decision["protein_id"]), str(decision["region_id"]))
        if key in indexed:
            raise ValueError(f"duplicate decision: {key}")
        indexed[key] = decision

    seen: set[tuple[str, str]] = set()
    output: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        key = (row["protein_id"], row["region_id"])
        decision = indexed.get(key)
        if decision is not None:
            if row["evidence_sha256"] != decision["source_evidence_sha256"]:
                raise ValueError(f"source evidence hash mismatch: {key}")
            for field in REVIEW_FIELDS[:-3]:
                row[field] = str(decision[field])
            row["reviewer"] = reviewer
            row["review_date"] = review_date
            row["review_notes"] = str(decision["review_notes"])
            seen.add(key)
        output.append(row)
    missing = set(indexed) - seen
    if missing:
        raise ValueError(f"decisions not found in template: {sorted(missing)}")
    return output, len(seen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    template = resolve(args.template)
    decisions_path = resolve(args.decisions)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite review progress: {output_dir}")
    if not template.is_file() or not decisions_path.is_file():
        raise FileNotFoundError("template or decisions file is missing")

    payload = json.loads(decisions_path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported decision schema")
    if payload.get("prediction_blind") is not True:
        raise ValueError("manual decisions must be prediction blind")
    if payload.get("locked_a10_modified") is not False:
        raise ValueError("decision payload must affirm that locked A10 was unchanged")

    fieldnames, rows = read_tsv(template)
    if any(field not in fieldnames for field in REVIEW_FIELDS):
        raise ValueError("template is missing review fields")
    if any(any(row[field] for field in REVIEW_FIELDS) for row in rows):
        raise ValueError("locked template is not blank")

    merged, added = apply_decisions(
        rows,
        list(payload.get("decisions", [])),
        str(payload.get("reviewer", "")),
        str(payload.get("review_date", "")),
    )
    reviewed = [row for row in merged if row["review_decision"]]
    remaining = len(merged) - len(reviewed)
    counts = Counter(row["review_decision"] for row in reviewed)

    output_dir.mkdir(parents=True)
    form_path = output_dir / "manual_review_form_IN_PROGRESS.tsv"
    write_tsv(form_path, fieldnames, merged)
    report_path = output_dir / "manual_review_progress_report.json"
    report = {
        "schema_version": 1,
        "experiment": "s2_prediction_blind_manual_review_progress",
        "status": "manual_review_in_progress" if remaining else "manual_review_complete_pending_finalization",
        "template": template.relative_to(ROOT).as_posix(),
        "template_sha256": sha256(template),
        "decisions": decisions_path.relative_to(ROOT).as_posix(),
        "decisions_sha256": sha256(decisions_path),
        "reviewer": payload["reviewer"],
        "review_date": payload["review_date"],
        "decision_rows_added": added,
        "total_regions": len(merged),
        "reviewed_regions": len(reviewed),
        "remaining_regions": remaining,
        "decision_counts": dict(sorted(counts.items())),
        "ready_for_finalization": remaining == 0,
        "ready_for_prediction": False,
        "model_predictions_accessed": False,
        "locked_a10_modified": False,
        "output": form_path.relative_to(ROOT).as_posix(),
        "output_sha256": sha256(form_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lock_path = output_dir / "manual_review_progress_lock.sha256"
    locked = [template, decisions_path, form_path, report_path]
    lock_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in locked),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
