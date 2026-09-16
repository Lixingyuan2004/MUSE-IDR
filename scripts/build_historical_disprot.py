"""Build the 2018 historical DisProt/CAID1 positive-label candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from muse_idr.data.caid import SequenceRecord, fasta_disprot_ids, file_sha256, write_fasta
from muse_idr.data.disprot import build_historical_disprot_records, load_disprot_entries
from muse_idr.data.leakage import audit_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/historical_disprot_2018.json"),
    )
    return parser.parse_args()


def _resolve(root: Path, path_value: object) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else root / path


def _validate_source(path: Path, source: dict[str, object]) -> None:
    if path.stat().st_size != int(source["bytes"]):
        raise ValueError(f"Source byte count changed: {path}")
    actual_hash = file_sha256(path)
    if actual_hash != source["sha256"]:
        raise ValueError(f"Source SHA-256 changed: {path}")


def _validate_expected(actual: dict[str, object], expected: dict[str, object]) -> None:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            raise ValueError(f"Expected {key}={expected_value}, observed {actual_value}")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config_path = _resolve(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = config["sources"]
    source_paths = {name: _resolve(root, item["local_path"]) for name, item in sources.items()}
    for name, path in source_paths.items():
        _validate_source(path, sources[name])

    entries = load_disprot_entries(source_paths["release_json"])
    current_ids = fasta_disprot_ids(source_paths["current_structural_state_fasta"])
    previous_ids = fasta_disprot_ids(source_paths["previous_structural_state_fasta"])
    policy = config["evidence_policy"]
    records, summary = build_historical_disprot_records(
        entries,
        current_ids=current_ids,
        previous_ids=previous_ids,
        allowed_positive_evidence=set(policy["allowed_positive_evidence"]),
        excluded_nonpositive_evidence=set(policy["excluded_nonpositive_evidence"]),
        name_exclusion_substrings=config["name_exclusion_substrings"],
    )
    observed = {"release_json_records": len(entries), **summary}
    _validate_expected(observed, config["expected"])

    sentinel_path = _resolve(root, config["external_test_sentinel"])
    all_fasta_records = []
    for record in records:
        aliases = [*record["disprot_ids"], *record["uniprot_accessions"]]
        all_fasta_records.append(
            SequenceRecord(
                identifier=str(record["protein_id"]),
                sequence=str(record["sequence"]),
                header=" ".join(str(alias) for alias in aliases),
            )
        )
    direct_findings = audit_records(all_fasta_records, sentinel_path)
    forbidden_digests = {str(finding["sequence_sha256"]) for finding in direct_findings}
    forbidden_ids = {str(finding["candidate_id"]) for finding in direct_findings}
    filtered_records = [
        record
        for record in records
        if record["sequence_sha256"] not in forbidden_digests
        and record["protein_id"] not in forbidden_ids
    ]
    digest_by_id = {
        str(record["protein_id"]): str(record["sequence_sha256"]) for record in records
    }
    filtered_fasta_records = [
        record
        for record in all_fasta_records
        if record.identifier not in forbidden_ids
        and digest_by_id[record.identifier] not in forbidden_digests
    ]
    post_filter_findings = audit_records(filtered_fasta_records, sentinel_path)
    if post_filter_findings:
        raise AssertionError("CAID2/3 direct leakage remained after filtering")
    filtered_summary = {
        "direct_leakage_findings": len(direct_findings),
        "sequences_after_direct_filter": len(filtered_records),
        "sequences_with_positive_labels_after_direct_filter": sum(
            record["label_counts"]["positive"] > 0 for record in filtered_records
        ),
        "positive_residues_after_direct_filter": sum(
            int(record["label_counts"]["positive"]) for record in filtered_records
        ),
        "unknown_residues_after_direct_filter": sum(
            int(record["label_counts"]["unknown"]) for record in filtered_records
        ),
        "negative_residues_after_direct_filter": 0,
    }
    _validate_expected(filtered_summary, config["expected_after_direct_filter"])

    outputs = config["outputs"]
    records_path = _resolve(root, outputs["records_jsonl"])
    fasta_path = _resolve(root, outputs["candidate_fasta"])
    manifest_path = _resolve(root, outputs["manifest"])
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in filtered_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_fasta(filtered_fasta_records, fasta_path)

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": config["dataset_name"],
        "release": config["release"],
        "task": "single_output_classic_idr",
        "label_semantics": {"positive": 1, "negative": 0, "unknown": -1},
        "training_readiness": "blocked_pending_pdb_negatives_and_caid23_homology_audit",
        "evidence_policy": policy,
        "name_exclusion_substrings": config["name_exclusion_substrings"],
        "sources": {
            name: {
                **item,
                "local_path": Path(str(item["local_path"])).as_posix(),
            }
            for name, item in sources.items()
        },
        "summary": {**observed, **filtered_summary},
        "external_test_direct_filter": {
            "sentinel_path": sentinel_path.relative_to(root).as_posix(),
            "sentinel_sha256": file_sha256(sentinel_path),
            "findings": direct_findings,
            "post_filter_findings": post_filter_findings,
        },
        "outputs": {
            "records_jsonl": {
                "path": records_path.relative_to(root).as_posix(),
                "bytes": records_path.stat().st_size,
                "sha256": file_sha256(records_path),
            },
            "candidate_fasta": {
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
    print(json.dumps({**observed, **filtered_summary}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
