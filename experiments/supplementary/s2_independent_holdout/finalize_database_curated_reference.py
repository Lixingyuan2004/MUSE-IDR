"""Freeze the prediction-blind, database-curated S2 reference and sequence set."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
PINS = {
    "outputs/supplementary/s2_independent_holdout/prefreeze_20260906/prefreeze_reference.jsonl": "db155d088547397337f3bbe7a065ff36016e4a50a9aa5483cd6c756fe02ffca8",
    "outputs/supplementary/s2_independent_holdout/prefreeze_20260906/prefreeze_sequences.fasta": "d777bfeb9cd8cf82af003c10bcef8b5b3e09dbee276e7e16164f408e336d67df",
    "outputs/supplementary/s2_independent_holdout/prefreeze_20260906/candidate_decisions.tsv": "f94ff8dab35d29075e80ef7bf0c348c56a062eb28f28ebfb14124d2664def768",
    "outputs/supplementary/s2_independent_holdout/prefreeze_20260906/prefreeze_report.json": "2eb8ee11cb69e72d92483206b08357e3e968894522f5413dd178fbfde8dd1046",
    "outputs/supplementary/s2_independent_holdout/prefreeze_20260906/prefreeze_lock.sha256": "7d519bb46dbbacf2b442a963d8f855eeeb4e79a0614837b5ec667d17aad00073",
    "experiments/supplementary/s2_independent_holdout/database_curated_policy_20260906.json": None,
}
EXPECTED = {
    "proteins": 87,
    "residues": 56394,
    "positive_residues": 2999,
    "negative_residues": 27729,
    "unknown_residues": 25666,
    "proteins_with_positive": 33,
    "proteins_with_negative": 76,
    "proteins_with_both": 22,
}
ALLOWED_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def label_counts(labels: Iterable[int]) -> dict[str, int]:
    counts = Counter(labels)
    return {
        "positive": counts[1],
        "negative": counts[0],
        "unknown": counts[-1],
    }


def finalize_record(row: dict[str, Any], policy_id: str) -> dict[str, Any]:
    protein_id = str(row.get("protein_id", ""))
    sequence = str(row.get("sequence", ""))
    labels = row.get("labels")
    if not protein_id or not sequence or not isinstance(labels, list):
        raise ValueError("record is missing protein_id, sequence, or labels")
    if sequence != sequence.upper() or set(sequence) - ALLOWED_RESIDUES:
        raise ValueError(f"invalid sequence: {protein_id}")
    if len(sequence) != len(labels):
        raise ValueError(f"sequence/label length mismatch: {protein_id}")
    if set(labels) - {-1, 0, 1}:
        raise ValueError(f"invalid labels: {protein_id}")
    observed = label_counts(labels)
    if observed != row.get("label_counts"):
        raise ValueError(f"stored label counts mismatch: {protein_id}")
    if observed["positive"] + observed["negative"] == 0:
        raise ValueError(f"record has no evaluable labels: {protein_id}")
    if row.get("reference_status") != "provisional_not_for_model_selection_or_formal_reporting":
        raise ValueError(f"unexpected source reference status: {protein_id}")
    result = dict(row)
    result["reference_status"] = "final_database_curated_s2_reference"
    result["label_policy_id"] = policy_id
    result["manual_article_review_applied"] = False
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "proteins": len(rows),
        "residues": sum(len(row["sequence"]) for row in rows),
        "positive_residues": sum(row["label_counts"]["positive"] for row in rows),
        "negative_residues": sum(row["label_counts"]["negative"] for row in rows),
        "unknown_residues": sum(row["label_counts"]["unknown"] for row in rows),
        "proteins_with_positive": sum(row["label_counts"]["positive"] > 0 for row in rows),
        "proteins_with_negative": sum(row["label_counts"]["negative"] > 0 for row in rows),
        "proteins_with_both": sum(
            row["label_counts"]["positive"] > 0 and row["label_counts"]["negative"] > 0
            for row in rows
        ),
    }
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def write_fasta(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(f">{row['protein_id']}\n{row['sequence']}\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def write_labels(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["protein_id", "position", "residue", "label"])
                for row in rows:
                    for position, (residue, label) in enumerate(
                        zip(row["sequence"], row["labels"]), start=1
                    ):
                        writer.writerow([row["protein_id"], position, residue, label])


def write_cohort(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "protein_id", "length", "source_entry_release", "positive_residues",
        "negative_residues", "unknown_residues", "has_both_classes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            counts = row["label_counts"]
            writer.writerow(
                {
                    "protein_id": row["protein_id"],
                    "length": len(row["sequence"]),
                    "source_entry_release": row["source_entry_release"],
                    "positive_residues": counts["positive"],
                    "negative_residues": counts["negative"],
                    "unknown_residues": counts["unknown"],
                    "has_both_classes": counts["positive"] > 0 and counts["negative"] > 0,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/supplementary/s2_independent_holdout/final_database_curated_20260906"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite final reference: {output_dir}")

    policy_path = ROOT / "experiments/supplementary/s2_independent_holdout/database_curated_policy_20260906.json"
    PINS["experiments/supplementary/s2_independent_holdout/database_curated_policy_20260906.json"] = sha256(policy_path)
    for relative, expected in PINS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"input SHA256 mismatch: {relative}: {observed}")

    policy = json.loads(policy_path.read_text(encoding="utf-8-sig"))
    if policy.get("model_predictions_accessed_before_freeze") is not False:
        raise ValueError("policy does not preserve prediction blinding")
    if policy.get("locked_a10_modified") is not False:
        raise ValueError("policy indicates a locked A10 modification")
    policy_id = str(policy["policy_id"])

    source_path = ROOT / next(key for key in PINS if key.endswith("prefreeze_reference.jsonl"))
    source_rows = read_jsonl(source_path)
    if len({row["protein_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("duplicate protein identifiers")
    if [row["protein_id"] for row in source_rows] != sorted(row["protein_id"] for row in source_rows):
        raise ValueError("prefreeze records are not deterministically sorted")
    rows = [finalize_record(row, policy_id) for row in source_rows]
    statistics = summarize(rows)
    if statistics != EXPECTED:
        raise ValueError(f"final statistics mismatch: {statistics}")

    output_dir.mkdir(parents=True)
    reference_path = output_dir / "s2_reference.jsonl"
    fasta_path = output_dir / "s2_sequences.fasta"
    labels_path = output_dir / "s2_labels.tsv.gz"
    cohort_path = output_dir / "s2_cohort.tsv"
    write_jsonl(reference_path, rows)
    write_fasta(fasta_path, rows)
    write_labels(labels_path, rows)
    write_cohort(cohort_path, rows)

    release_counts = Counter(row["source_entry_release"] for row in rows)
    report = {
        "schema_version": 1,
        "experiment": "s2_final_database_curated_reference",
        "status": "pass",
        "policy_id": policy_id,
        "policy_sha256": sha256(policy_path),
        "input_pins": PINS,
        "statistics": statistics,
        "source_entry_release_counts": dict(sorted(release_counts.items())),
        "manual_article_review_required": False,
        "manual_review_pilot_applied_to_labels": False,
        "database_annotation_limit_recorded": True,
        "sequence_fasta_contains_labels": False,
        "unknown_label": -1,
        "model_scores_used_for_selection": False,
        "caid_labels_accessed": False,
        "model_predictions_accessed_before_freeze": False,
        "locked_a10_modified": False,
        "ready_for_prediction": True,
        "outputs": {
            path.name: sha256(path)
            for path in (reference_path, fasta_path, labels_path, cohort_path)
        },
    }
    report_path = output_dir / "s2_finalization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary_path = output_dir / "s2_finalization_report.md"
    summary_path.write_text(
        "\n".join(
            [
                "# S2 database-curated reference freeze",
                "",
                "S2 is a supplementary post-lock temporal/database-version validation set.",
                "Positive labels use strictly filtered DisProt experimental annotations;",
                "negative labels use quality-filtered PDBe/SIFTS observed residues.",
                "The set was frozen before any MUSE-IDR or baseline prediction.",
                "",
                f"- Proteins: {statistics['proteins']}",
                f"- Residues: {statistics['residues']}",
                f"- Positive labels: {statistics['positive_residues']}",
                f"- Negative labels: {statistics['negative_residues']}",
                f"- Unknown labels (-1): {statistics['unknown_residues']}",
                f"- Proteins containing both classes: {statistics['proteins_with_both']}",
                "- Exhaustive article-by-article re-curation: not performed",
                "- Manual-review pilot decisions applied to labels: no",
                "- Ready for prediction: yes",
                "",
                "CAID2 and CAID3 remain the primary external benchmarks. S2 must not be",
                "presented as a manually re-curated gold-standard benchmark.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    script_path = Path(__file__).resolve()
    test_path = script_path.with_name("test_finalize_database_curated_reference.py")
    locked_paths = [
        *(ROOT / relative for relative in PINS),
        script_path,
        test_path,
        reference_path,
        fasta_path,
        labels_path,
        cohort_path,
        report_path,
        summary_path,
    ]
    lock_path = output_dir / "s2_final_lock.sha256"
    lock_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in locked_paths),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
