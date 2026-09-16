"""Build the label-free all-round provenance and CAID2/3 test sentinels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from muse_idr.data.caid import (
    SequenceRecord,
    fasta_disprot_ids,
    file_sha256,
    parse_caid_reference,
    reconstruct_caid1_targets,
    sequence_sha256,
    subset_sentinel,
    write_fasta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/holdout_sources.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/holdouts/caid_all_rounds_sentinel.json"),
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("data/manifests/holdouts/caid23_test_sentinel.json"),
        help="Label-free sentinel used by training-data audits.",
    )
    return parser.parse_args()


def source_metadata(root: Path, item: dict[str, object], records: int) -> dict[str, object]:
    path = root / str(item["local_path"])
    stat = path.stat()
    return {
        "local_path": path.relative_to(root).as_posix(),
        "url": str(item["url"]),
        "bytes": stat.st_size,
        "sha256": file_sha256(path),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_path = args.output if args.output.is_absolute() else root / args.output
    test_output_path = (
        args.test_output if args.test_output.is_absolute() else root / args.test_output
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    targets: dict[str, dict[str, object]] = {}
    target_sources: dict[str, set[str]] = defaultdict(set)
    target_ids: dict[str, set[str]] = defaultdict(set)
    target_accessions: dict[str, set[str]] = defaultdict(set)
    accession_by_reference_id: dict[str, set[str]] = defaultdict(set)
    external_test_records: dict[str, SequenceRecord] = {}
    sources: list[dict[str, object]] = []

    def add_target(
        record: SequenceRecord,
        source_name: str,
        accession: str | None = None,
    ) -> None:
        digest = sequence_sha256(record.sequence)
        existing = targets.setdefault(
            digest,
            {"sequence_sha256": digest, "length": len(record.sequence)},
        )
        if existing["length"] != len(record.sequence):
            raise AssertionError("SHA-256 collision with different sequence lengths")
        target_sources[digest].add(source_name)
        target_ids[digest].add(record.identifier.upper())
        if accession:
            target_accessions[digest].add(accession.upper())
            accession_by_reference_id[record.identifier.upper()].add(accession.upper())

    caid1 = config["caid1"]
    caid1_records = reconstruct_caid1_targets(
        root / caid1["current_release_fasta"]["local_path"],
        root / caid1["exclude_release_fasta"]["local_path"],
        root / caid1["release_json"]["local_path"],
    )
    if len(caid1_records) != int(caid1["expected_targets"]):
        raise ValueError(
            f"CAID1 reconstructed target count changed: {len(caid1_records)} "
            f"!= {caid1['expected_targets']}"
        )
    write_fasta(
        (record for record, _ in caid1_records),
        root / "data/caid_holdout/derived/caid1/targets.fasta",
    )
    for record, accession in caid1_records:
        add_target(record, "caid1:reconstructed", accession)
    for key in ("current_release_fasta", "exclude_release_fasta", "release_json"):
        item = caid1[key]
        item_path = root / item["local_path"]
        if key.endswith("fasta"):
            source_record_count = len(fasta_disprot_ids(item_path))
        else:
            source_record_count = len(json.loads(item_path.read_text(encoding="utf-8"))["data"])
        metadata = source_metadata(root, item, source_record_count)
        metadata.update({"edition": "caid1", "role": key})
        sources.append(metadata)

    for item in config["reference_files"]:
        path = root / item["local_path"]
        records = list(parse_caid_reference(path))
        if len(records) != int(item["expected_records"]):
            raise ValueError(
                f"Record count changed for {path}: {len(records)} != {item['expected_records']}"
            )
        source_name = f"{item['edition']}:{item['track']}"
        for record in records:
            add_target(record, source_name)
            if item["edition"] in {"caid2", "caid3"}:
                external_test_records.setdefault(sequence_sha256(record.sequence), record)
        metadata = source_metadata(root, item, len(records))
        metadata.update({"edition": item["edition"], "track": item["track"]})
        sources.append(metadata)

    write_fasta(
        sorted(external_test_records.values(), key=lambda record: record.identifier),
        root / "data/caid_holdout/derived/caid23/targets.fasta",
    )

    accessions_by_disprot: dict[str, set[str]] = defaultdict(set)
    for mapping_item in config["identifier_mappings"]:
        mapping_path = root / mapping_item["local_path"]
        mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping_entries = mapping_payload.get("data")
        if not isinstance(mapping_entries, list):
            raise ValueError(f"Unexpected identifier-mapping JSON structure: {mapping_path}")
        for entry in mapping_entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Malformed identifier-mapping entry: {mapping_path}")
            disprot_id = str(entry.get("disprot_id", "")).upper()
            accession = str(entry.get("acc", "")).upper()
            if disprot_id and accession:
                accessions_by_disprot[disprot_id].add(accession)
        mapping_metadata = source_metadata(root, mapping_item, len(mapping_entries))
        mapping_metadata.update(
            {
                "role": "disprot_to_uniprot_identifier_mapping",
                "release": mapping_item["release"],
            }
        )
        sources.append(mapping_metadata)
    for digest, identifiers in target_ids.items():
        for identifier in identifiers:
            for accession in accessions_by_disprot.get(identifier, set()):
                target_accessions[digest].add(accession)
                accession_by_reference_id[identifier].add(accession)

    target_rows: list[dict[str, object]] = []
    for digest, base in targets.items():
        target_rows.append(
            {
                **base,
                "reference_ids": sorted(target_ids[digest]),
                "uniprot_accessions": sorted(target_accessions[digest]),
                "sources": sorted(target_sources[digest]),
            }
        )

    all_ids = {identifier for values in target_ids.values() for identifier in values}
    mapped_ids = all_ids & accession_by_reference_id.keys()
    unmapped_ids = sorted(all_ids - mapped_ids)
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "labels_included": False,
        "purpose": "training_exclusion_only",
        "normalization": "uppercase_remove_whitespace_no_residue_substitution",
        "hash_algorithm": "sha256",
        "policy": {
            "remove_exact_sequence": True,
            "remove_identifier_and_isoform": True,
            "homology_identity_strictly_greater_than": 0.30,
            "homology_query_coverage_at_least": 0.80,
            "homology_target_coverage_at_least": 0.80,
        },
        "summary": {
            "source_records": sum(int(source["records"]) for source in sources),
            "unique_reference_ids": len(all_ids),
            "unique_sequences": len(target_rows),
            "mapped_reference_ids": len(mapped_ids),
            "unmapped_reference_ids": len(unmapped_ids),
        },
        "unmapped_reference_id_values": unmapped_ids,
        "sources": sources,
        "targets": sorted(target_rows, key=lambda row: str(row["sequence_sha256"])),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    test_manifest = subset_sentinel(manifest, {"caid2", "caid3"})
    test_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_path.write_text(
        json.dumps(test_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {"all_rounds": manifest["summary"], "external_test": test_manifest["summary"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
