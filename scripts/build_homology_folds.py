"""Build connected homology clusters and a deterministic balanced five-fold split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from muse_idr.data.caid import file_sha256, parse_fasta
from muse_idr.data.folds import assign_balanced_folds, build_homology_clusters, summarize_folds
from muse_idr.data.homology_filter import filter_candidate_records
from muse_idr.data.leakage import is_forbidden_homolog, parse_homology_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/data/homology_cluster_5fold.json")
    )
    return parser.parse_args()


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _validate_source(path: Path, source: dict[str, object]) -> None:
    if path.stat().st_size != int(source["bytes"]):
        raise ValueError(f"Source byte count changed: {path}")
    if file_sha256(path) != source["sha256"]:
        raise ValueError(f"Source SHA-256 changed: {path}")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config_path = _resolve(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = config["sources"]
    paths = {name: _resolve(root, source["path"]) for name, source in sources.items()}
    for name, path in paths.items():
        _validate_source(path, sources[name])

    records = _load_jsonl(paths["training_jsonl"])
    fasta = parse_fasta(paths["training_fasta"])
    filter_candidate_records(records, fasta, [])
    hits = parse_homology_tsv(paths["all_vs_all_tsv"])
    clusters, edges = build_homology_clusters(records, hits)
    split = config["split"]
    fold_count = int(split["folds"])
    assignments = assign_balanced_folds(
        clusters, folds=fold_count, seed=int(split["assignment_seed"])
    )
    summaries = summarize_folds(clusters, assignments, fold_count)
    cluster_by_member = {
        member: cluster for cluster in clusters for member in cluster.members
    }
    fold_by_member = {
        member: assignments[cluster.cluster_id]
        for member, cluster in cluster_by_member.items()
    }
    cross_fold_edges = [edge for edge in edges if fold_by_member[edge[0]] != fold_by_member[edge[1]]]
    if cross_fold_edges:
        raise ValueError(f"Homology edges cross folds: {cross_fold_edges[:5]}")
    if any(summary["positive_residues"] == 0 or summary["negative_residues"] == 0 for summary in summaries):
        raise ValueError("A validation fold lacks a residue class required for ROC-AUC")

    strict_hits = [hit for hit in hits if is_forbidden_homolog(hit)]
    observed = {
        "input_sequences": len(records),
        "mmseqs_rows": len(hits),
        "strict_rule_rows": len(strict_hits),
        "self_rows": sum(hit.query == hit.target for hit in strict_hits),
        "unique_undirected_nonself_edges": len(edges),
        "clusters": len(clusters),
        "non_singleton_clusters": sum(cluster.sequences > 1 for cluster in clusters),
        "largest_cluster_sequences": max(cluster.sequences for cluster in clusters),
        "cross_fold_homology_edges": len(cross_fold_edges),
    }
    if observed != config["expected"]:
        raise ValueError(f"Expected {config['expected']}, observed {observed}")

    outputs = config["outputs"]
    clusters_path = _resolve(root, outputs["clusters_jsonl"])
    assignments_path = _resolve(root, outputs["assignments_jsonl"])
    manifest_path = _resolve(root, outputs["manifest"])
    clusters_path.parent.mkdir(parents=True, exist_ok=True)
    with clusters_path.open("w", encoding="utf-8", newline="\n") as handle:
        for cluster in clusters:
            row = {
                "cluster_id": cluster.cluster_id,
                "fold": assignments[cluster.cluster_id],
                "members": list(cluster.members),
                "sequences": cluster.sequences,
                "positive_residues": cluster.positive_residues,
                "negative_residues": cluster.negative_residues,
                "unknown_residues": cluster.unknown_residues,
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    with assignments_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            identifier = str(record["protein_id"])
            cluster = cluster_by_member[identifier]
            labels = list(record["labels"])
            row = {
                "protein_id": identifier,
                "cluster_id": cluster.cluster_id,
                "fold": fold_by_member[identifier],
                "sequence_length": len(str(record["sequence"])),
                "positive_residues": labels.count(1),
                "negative_residues": labels.count(0),
                "unknown_residues": labels.count(-1),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task": "single_output_classic_idr",
        "training_readiness": "ready_for_baseline_training",
        "sources": sources,
        "homology_rule": config["homology_rule"],
        "split": split,
        "summary": observed,
        "fold_summaries": summaries,
        "audit": {
            "all_training_proteins_assigned_once": len(fold_by_member) == len(records),
            "homology_clusters_split_across_folds": 0,
            "folds_with_both_residue_classes": fold_count,
            "external_test_audit": paths["external_test_audit"].relative_to(root).as_posix(),
            "external_test_audit_status": "pass",
        },
        "outputs": {
            "clusters_jsonl": {
                "path": clusters_path.relative_to(root).as_posix(),
                "bytes": clusters_path.stat().st_size,
                "sha256": file_sha256(clusters_path),
            },
            "assignments_jsonl": {
                "path": assignments_path.relative_to(root).as_posix(),
                "bytes": assignments_path.stat().st_size,
                "sha256": file_sha256(assignments_path),
            },
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"summary": observed, "folds": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
