"""Run local-homology sensitivity checks for S2 sequence candidates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MMSEQS_VERSION = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
PINS = {
    "outputs/supplementary/s2_independent_holdout/source_audit_20260906/"
    "after_2023_06_sequence_candidates.fasta":
        "5d1a1a3d06d2ccbcf0121d70249462050dfa3a41fb66dcd85df359cb0ed1dba8",
    "outputs/supplementary/s2_independent_holdout/source_audit_20260906/"
    "training_and_caid23_sequences.fasta":
        "05a3b1efa6ed013505fb1b0943d4e21be3b2f5a1423f697aa647c90cde6fc8ef",
}
FORMAT = "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta_ids(path: Path) -> list[str]:
    return [line[1:].split(maxsplit=1)[0] for line in path.read_text(encoding="ascii").splitlines() if line.startswith(">")]


def run_search(
    mmseqs: Path,
    query: Path,
    target: Path,
    output: Path,
    temporary: Path,
    max_sequences: int,
    threads: int,
) -> list[str]:
    command = [
        str(mmseqs), "easy-search", str(query), str(target), str(output), str(temporary),
        "--prefilter-mode", "2",
        "--max-seqs", str(max_sequences),
        "--alignment-mode", "3",
        "--min-seq-id", "0.20",
        "-c", "0.10",
        "--cov-mode", "0",
        "-e", "0.001",
        "-s", "7.5",
        "--mask", "1",
        "--comp-bias-corr", "1",
        "--threads", str(threads),
        "--format-output", FORMAT,
    ]
    subprocess.run(command, check=True)
    return command


def read_hits(path: Path, *, remove_self: bool) -> list[dict[str, Any]]:
    fields = FORMAT.split(",")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for values in csv.reader(handle, delimiter="\t"):
            if len(values) != len(fields):
                raise ValueError(f"unexpected MMseqs row: {values}")
            row = dict(zip(fields, values))
            if remove_self and row["query"] == row["target"]:
                continue
            for key in ("fident", "qcov", "tcov", "evalue", "bits"):
                row[key] = float(row[key])
            for key in ("alnlen", "qlen", "tlen"):
                row[key] = int(row[key])
            rows.append(row)
    return rows


def criteria(row: dict[str, Any]) -> list[str]:
    matched: list[str] = []
    if row["fident"] > 0.30 and row["qcov"] >= 0.80 and row["tcov"] >= 0.80:
        matched.append("global_30id_both80")
    if row["fident"] >= 0.30 and row["qcov"] >= 0.50 and row["alnlen"] >= 50:
        matched.append("candidate_half_30id")
    if row["fident"] >= 0.30 and row["qcov"] >= 0.50 and row["tcov"] >= 0.50 and row["alnlen"] >= 50:
        matched.append("reciprocal_half_30id")
    if row["fident"] >= 0.40 and row["alnlen"] >= 50:
        matched.append("local_40id_50aa")
    if row["fident"] >= 0.50 and row["alnlen"] >= 30:
        matched.append("local_50id_30aa")
    return matched


def summarize(rows: list[dict[str, Any]], candidate_ids: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    query_sets: dict[str, set[str]] = defaultdict(set)
    flags: dict[str, dict[str, Any]] = {
        identifier: {"protein_id": identifier} for identifier in candidate_ids
    }
    for row in rows:
        for rule in criteria(row):
            counts[rule] += 1
            query_sets[rule].add(str(row["query"]))
            current = flags[str(row["query"])].setdefault(rule, [])
            current.append(
                {
                    "target": row["target"],
                    "fident": row["fident"],
                    "alnlen": row["alnlen"],
                    "qcov": row["qcov"],
                    "tcov": row["tcov"],
                    "evalue": row["evalue"],
                }
            )
    summary = {
        "reported_alignments": len(rows),
        "criterion_alignment_counts": dict(sorted(counts.items())),
        "criterion_query_counts": {
            key: len(value) for key, value in sorted(query_sets.items())
        },
        "queries_flagged_by_any_sensitivity_rule": len(
            {query for values in query_sets.values() for query in values}
        ),
    }
    return summary, list(flags.values())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmseqs",
        type=Path,
        default=Path("tools/mmseqs2-windows-18-8cc5c/mmseqs/bin/mmseqs.exe"),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/supplementary/s2_independent_holdout/"
            "local_homology_audit_20260906"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    mmseqs = args.mmseqs if args.mmseqs.is_absolute() else ROOT / args.mmseqs
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_dir}")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    query = ROOT / next(key for key in PINS if "after_2023" in key)
    forbidden = ROOT / next(key for key in PINS if "training_and_caid23" in key)
    for relative, expected in PINS.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
    version = subprocess.run([str(mmseqs), "version"], check=True, capture_output=True, text=True).stdout.strip()
    if version != MMSEQS_VERSION:
        raise ValueError(f"MMseqs version mismatch: {version}")

    candidate_ids = read_fasta_ids(query)
    target_ids = read_fasta_ids(forbidden)
    output_dir.mkdir(parents=True)
    forbidden_tsv = output_dir / "candidate_vs_training_and_caid23_local.tsv"
    internal_tsv = output_dir / "candidate_internal_local.tsv"
    commands = [
        run_search(mmseqs, query, forbidden, forbidden_tsv, output_dir / "tmp_forbidden", len(target_ids), args.threads),
        run_search(mmseqs, query, query, internal_tsv, output_dir / "tmp_internal", len(candidate_ids), args.threads),
    ]
    forbidden_rows = read_hits(forbidden_tsv, remove_self=False)
    internal_rows = read_hits(internal_tsv, remove_self=True)
    forbidden_summary, forbidden_flags = summarize(forbidden_rows, candidate_ids)
    internal_summary, internal_flags = summarize(internal_rows, candidate_ids)
    report = {
        "schema_version": 1,
        "experiment": "s2_local_homology_sensitivity_audit",
        "status": "pass_descriptive_not_automatic_exclusion",
        "inputs": {relative: expected for relative, expected in PINS.items()},
        "mmseqs_version": version,
        "mmseqs_executable_sha256": sha256(mmseqs),
        "search_prefilter": {
            "minimum_identity": 0.20,
            "minimum_both_sequence_coverage": 0.10,
            "maximum_evalue": 0.001,
            "low_complexity_masking": True,
            "composition_bias_correction": True,
        },
        "criteria_are_sensitivity_flags_not_automatic_exclusions": True,
        "forbidden_reference": forbidden_summary,
        "candidate_internal": internal_summary,
        "caid_labels_accessed": False,
        "model_predictions_accessed": False,
        "locked_a10_modified": False,
    }
    write_json(output_dir / "homology_sensitivity_report.json", report)
    write_json(output_dir / "commands.json", commands)
    write_json(output_dir / "forbidden_query_flags.json", forbidden_flags)
    write_json(output_dir / "candidate_internal_flags.json", internal_flags)
    markdown = f"""# S2 local-homology sensitivity audit

The original global exclusion required identity >0.30 and at least 0.80
coverage on both sequences. This audit uses a permissive search prefilter and
reports several post-hoc local-similarity flags. A flag is not an automatic
exclusion because low-complexity IDR composition and short alignments require
biological review.

## Against training and CAID2/3 sequences

- Raw alignments: {forbidden_summary['reported_alignments']:,}.
- Candidates flagged by at least one sensitivity rule: {forbidden_summary['queries_flagged_by_any_sensitivity_rule']} / {len(candidate_ids)}.
- Original global-rule hits among the already screened candidates: {forbidden_summary['criterion_query_counts'].get('global_30id_both80', 0)}.

## Within the S2 candidate cohort

- Non-self raw alignments: {internal_summary['reported_alignments']:,}.
- Candidates flagged by at least one sensitivity rule: {internal_summary['queries_flagged_by_any_sensitivity_rule']} / {len(candidate_ids)}.
- Global-rule candidate redundancy: {internal_summary['criterion_query_counts'].get('global_30id_both80', 0)} candidates.

No model scores or CAID labels were read. Decisions based on these flags must be
made before S2 predictions are generated.
"""
    (output_dir / "homology_sensitivity_report.md").write_text(markdown, encoding="utf-8", newline="\n")
    lock_targets = [
        Path(__file__), forbidden_tsv, internal_tsv,
        output_dir / "homology_sensitivity_report.json",
        output_dir / "homology_sensitivity_report.md",
        output_dir / "commands.json",
        output_dir / "forbidden_query_flags.json",
        output_dir / "candidate_internal_flags.json",
    ]
    (output_dir / "homology_sensitivity_lock.sha256").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in lock_targets) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
