"""Prepare an S2 pre-freeze cohort without reading model predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
PINS = {
    "outputs/supplementary/s2_independent_holdout/qualification_audit_20260906/"
    "provisional_reference.jsonl":
        "7a9e8640c2ad0358e219c1c98ee9e86fc453ced7f9ecacc7cce50667e380ab6b",
    "outputs/supplementary/s2_independent_holdout/local_homology_audit_20260906/"
    "forbidden_query_flags.json":
        "080d32bf014be7c5c40adfd45b271263cbc56ac57f7884ef8931b5c2ec303a8e",
    "outputs/supplementary/s2_independent_holdout/local_homology_audit_20260906/"
    "candidate_internal_local.tsv":
        "6c7dfcd74593b93639eb8279fca107fd9003af40c8b3f6b49b70fa31852be31b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def local_similarity(row: dict[str, Any]) -> bool:
    identity = float(row["fident"])
    length = int(row["alnlen"])
    query_coverage = float(row["qcov"])
    target_coverage = float(row["tcov"])
    return any(
        (
            identity > 0.30 and query_coverage >= 0.80 and target_coverage >= 0.80,
            identity >= 0.30 and query_coverage >= 0.50 and length >= 50,
            identity >= 0.30 and query_coverage >= 0.50
            and target_coverage >= 0.50 and length >= 50,
            identity >= 0.40 and length >= 50,
            identity >= 0.50 and length >= 30,
        )
    )


def connected_components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = {node: set() for node in nodes}
    for first, second in edges:
        if first == second or first not in graph or second not in graph:
            continue
        graph[first].add(second)
        graph[second].add(first)
    components: list[list[str]] = []
    seen: set[str] = set()
    for start in sorted(graph):
        if start in seen:
            continue
        stack = [start]
        component: list[str] = []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(graph[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def known_count(record: dict[str, Any]) -> int:
    return sum(label in (0, 1) for label in record["labels"])


def choose_representative(component: list[str], by_id: dict[str, dict[str, Any]]) -> str:
    # Selection uses annotation completeness only, never model performance.
    return sorted(component, key=lambda identifier: (-known_count(by_id[identifier]), identifier))[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/supplementary/s2_independent_holdout/prefreeze_20260906"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing pre-freeze output: {output_dir}")
    for relative, expected in PINS.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")

    reference_path = ROOT / next(key for key in PINS if "provisional_reference" in key)
    forbidden_path = ROOT / next(key for key in PINS if "forbidden_query_flags" in key)
    internal_path = ROOT / next(key for key in PINS if "candidate_internal_local" in key)
    records = read_jsonl(reference_path)
    by_id = {str(record["protein_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate provisional protein identifier")

    forbidden_rows = json.loads(forbidden_path.read_text(encoding="utf-8"))
    external_flags = {
        str(row["protein_id"]) for row in forbidden_rows if len(row) > 1
    }
    labeled = {identifier for identifier, record in by_id.items() if known_count(record)}
    eligible_before_internal = labeled - external_flags

    fields = "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits".split(",")
    internal_edges: list[tuple[str, str]] = []
    with internal_path.open("r", encoding="utf-8", newline="") as handle:
        for values in csv.reader(handle, delimiter="\t"):
            if len(values) != len(fields):
                raise ValueError("malformed internal MMseqs row")
            row = dict(zip(fields, values))
            if local_similarity(row):
                internal_edges.append((str(row["query"]), str(row["target"])))
    components = connected_components(eligible_before_internal, internal_edges)
    representatives: dict[str, str] = {}
    component_ids: dict[str, str] = {}
    for index, component in enumerate(components, start=1):
        representative = choose_representative(component, by_id)
        component_id = f"S2C{index:04d}"
        for identifier in component:
            representatives[identifier] = representative
            component_ids[identifier] = component_id
    retained = {
        identifier for identifier in eligible_before_internal
        if representatives[identifier] == identifier
    }

    decisions: list[dict[str, Any]] = []
    for identifier in sorted(by_id):
        record = by_id[identifier]
        positives = record["label_counts"]["positive"]
        negatives = record["label_counts"]["negative"]
        reasons: list[str] = []
        if not known_count(record):
            reasons.append("no_provisional_labels")
        if identifier in external_flags:
            reasons.append("local_similarity_to_training_or_caid23")
        if identifier in eligible_before_internal and identifier not in retained:
            reasons.append(
                f"internal_similarity_cluster_represented_by:{representatives[identifier]}"
            )
        decisions.append(
            {
                "protein_id": identifier,
                "length": len(record["sequence"]),
                "positive_residues": positives,
                "negative_residues": negatives,
                "known_residues": positives + negatives,
                "external_similarity_flag": identifier in external_flags,
                "internal_cluster": component_ids.get(identifier, ""),
                "internal_representative": representatives.get(identifier, ""),
                "prefreeze_retained": identifier in retained,
                "exclusion_reasons": ";".join(reasons),
                "manual_annotation_review_required": positives > 0,
            }
        )

    retained_records = [by_id[identifier] for identifier in sorted(retained)]
    totals = {
        "proteins": len(retained_records),
        "residues": sum(len(record["sequence"]) for record in retained_records),
        "positive_residues": sum(record["label_counts"]["positive"] for record in retained_records),
        "negative_residues": sum(record["label_counts"]["negative"] for record in retained_records),
        "unknown_residues": sum(record["label_counts"]["unknown"] for record in retained_records),
        "proteins_with_positive": sum(record["label_counts"]["positive"] > 0 for record in retained_records),
        "proteins_with_negative": sum(record["label_counts"]["negative"] > 0 for record in retained_records),
        "proteins_with_both": sum(
            record["label_counts"]["positive"] > 0 and record["label_counts"]["negative"] > 0
            for record in retained_records
        ),
    }
    report = {
        "schema_version": 1,
        "experiment": "s2_prediction_blind_prefreeze",
        "status": "prefreeze_complete_manual_annotation_review_pending",
        "input_pins": PINS,
        "rules": {
            "exclude_no_provisional_labels": True,
            "exclude_any_predeclared_external_local_similarity_flag": True,
            "internal_similarity_graph_uses_any_predeclared_sensitivity_rule": True,
            "internal_representative": "largest known-residue count, then lowest DisProt ID",
            "model_score_used_for_selection": False,
        },
        "input_candidates": len(records),
        "input_provisionally_labeled": len(labeled),
        "external_similarity_exclusions": len(labeled & external_flags),
        "eligible_before_internal_deduplication": len(eligible_before_internal),
        "internal_components": len(components),
        "multi_member_internal_components": sum(len(component) > 1 for component in components),
        "internal_redundancy_exclusions": len(eligible_before_internal) - len(retained),
        "prefreeze": totals,
        "ready_for_prediction": False,
        "remaining_blockers": [
            "manual review of every retained positive-evidence region",
            "resolve and document DisProt unpublished-field semantics",
            "sign off the final protein list and regenerate an immutable final lock",
        ],
        "caid_labels_accessed": False,
        "model_predictions_accessed": False,
        "locked_a10_modified": False,
    }

    output_dir.mkdir(parents=True)
    with (output_dir / "prefreeze_reference.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for record in retained_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "prefreeze_sequences.fasta").write_text(
        "".join(
            f'>{record["protein_id"]}\n{record["sequence"]}\n'
            for record in retained_records
        ),
        encoding="ascii",
        newline="\n",
    )
    decision_path = output_dir / "candidate_decisions.tsv"
    with decision_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(decisions[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(decisions)
    (output_dir / "prefreeze_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = f"""# S2 prediction-blind pre-freeze cohort

All selection rules were applied before reading any model prediction.

- Input candidates: {len(records)}.
- Provisionally labeled candidates: {len(labeled)}.
- Labeled candidates excluded for local similarity to training/CAID2/3: {len(labeled & external_flags)}.
- Internal similarity representatives removed: {len(eligible_before_internal) - len(retained)}.
- Pre-freeze cohort: {totals['proteins']} proteins / {totals['residues']:,} residues.
- Positive / negative / unknown residues: {totals['positive_residues']:,} / {totals['negative_residues']:,} / {totals['unknown_residues']:,}.
- Proteins with both classes: {totals['proteins_with_both']}.

This cohort is not ready for prediction. Every retained positive annotation must
be manually reviewed and the DisProt `unpublished` field discrepancy must be
documented before the final reference and sequence-only FASTA are locked.
"""
    (output_dir / "prefreeze_report.md").write_text(markdown, encoding="utf-8", newline="\n")
    lock_targets = [
        Path(__file__),
        ROOT / "experiments/supplementary/s2_independent_holdout/test_prepare_prefreeze.py",
        *(output_dir / name for name in (
            "prefreeze_reference.jsonl", "prefreeze_sequences.fasta",
            "candidate_decisions.tsv", "prefreeze_report.json", "prefreeze_report.md",
        )),
    ]
    (output_dir / "prefreeze_lock.sha256").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in lock_targets) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
