"""Remove CAID2/3 homologs from the historical candidate training set."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from muse_idr.data.caid import SequenceRecord, file_sha256, parse_fasta, write_fasta
from muse_idr.data.homology_filter import filter_candidate_records, label_summary
from muse_idr.data.leakage import audit_records, is_forbidden_homolog, parse_homology_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/caid23_homology_filter.json"),
    )
    return parser.parse_args()


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _validate_file(path: Path, source: dict[str, object]) -> None:
    if path.stat().st_size != int(source["bytes"]):
        raise ValueError(f"Source byte count changed: {path}")
    if file_sha256(path) != source["sha256"]:
        raise ValueError(f"Source SHA-256 changed: {path}")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(row)
    return records


def _fasta_record(record: dict[str, object]) -> SequenceRecord:
    aliases = [*record["disprot_ids"], *record["uniprot_accessions"]]
    return SequenceRecord(
        identifier=str(record["protein_id"]),
        sequence=str(record["sequence"]),
        header=" ".join(str(alias) for alias in aliases),
    )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config_path = _resolve(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = config["sources"]
    paths = {name: _resolve(root, source["path"]) for name, source in sources.items()}
    for name, path in paths.items():
        _validate_file(path, sources[name])

    records = _load_jsonl(paths["candidate_jsonl"])
    fasta_records = parse_fasta(paths["candidate_fasta"])
    hits = parse_homology_tsv(paths["homology_tsv"])
    result = filter_candidate_records(records, fasta_records, hits)
    post_filter_hits = [
        hit
        for hit in parse_homology_tsv(paths["post_filter_homology_tsv"])
        if is_forbidden_homolog(hit)
    ]
    formal_audit = json.loads(paths["post_filter_formal_audit"].read_text(encoding="utf-8"))
    if formal_audit != {
        "status": "pass",
        "homology_audit": "complete",
        "exact_or_identifier_findings": [],
        "forbidden_homology_hits": [],
    }:
        raise ValueError("Post-filter formal audit is not a clean pass")
    observed = {
        "input_sequences": len(records),
        "forbidden_hit_rows": len(result.forbidden_hits),
        "removed_sequences": len(result.removed),
        "kept_sequences": len(result.kept),
        "post_filter_forbidden_hit_rows": len(post_filter_hits),
    }
    if observed != config["expected"]:
        raise ValueError(f"Expected {config['expected']}, observed {observed}")

    outputs = config["outputs"]
    jsonl_path = _resolve(root, outputs["jsonl"])
    fasta_path = _resolve(root, outputs["fasta"])
    manifest_path = _resolve(root, outputs["manifest"])
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in result.kept:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_fasta((_fasta_record(record) for record in result.kept), fasta_path)

    sentinel_path = paths["holdout_sentinel"]
    direct_findings = audit_records(parse_fasta(fasta_path), sentinel_path)
    if direct_findings:
        raise ValueError("Direct CAID2/3 leakage remained after homology filtering")

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task": "single_output_classic_idr",
        "training_readiness": "blocked_pending_homology_cluster_split",
        "filter_policy": config["filter_policy"],
        "sources": sources,
        "summary": {
            **observed,
            "kept_labels": label_summary(result.kept),
            "removed_labels": label_summary(result.removed),
            "post_filter_direct_findings": len(direct_findings),
        },
        "removed_protein_ids": sorted(result.removed_ids),
        "post_filter_external_test_audit": {
            "status": formal_audit["status"],
            "homology_tsv": paths["post_filter_homology_tsv"].relative_to(root).as_posix(),
            "homology_tsv_sha256": file_sha256(paths["post_filter_homology_tsv"]),
            "formal_report": paths["post_filter_formal_audit"].relative_to(root).as_posix(),
            "findings": 0,
        },
        "outputs": {
            "jsonl": {
                "path": jsonl_path.relative_to(root).as_posix(),
                "bytes": jsonl_path.stat().st_size,
                "sha256": file_sha256(jsonl_path),
            },
            "fasta": {
                "path": fasta_path.relative_to(root).as_posix(),
                "bytes": fasta_path.stat().st_size,
                "sha256": file_sha256(fasta_path),
            },
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
