"""Merge historical, quality-filtered PDB/SIFTS negatives into IDR labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from muse_idr.data.caid import SequenceRecord, file_sha256, write_fasta
from muse_idr.data.disprot import load_disprot_entries
from muse_idr.data.leakage import audit_records
from muse_idr.data.pdb_negatives import (
    exact_sequence_accession_map,
    merge_observed_negatives,
    protected_masks,
    qualified_pdb_entries,
    read_observed_segments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/pdb_negative_2018.json"),
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


def _validate_expected(summary: dict[str, object], expected: dict[str, object]) -> None:
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise ValueError(f"Expected {key}={expected_value}, observed {summary.get(key)}")


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
    paths = {name: _resolve(root, source["local_path"]) for name, source in sources.items()}
    for name, path in paths.items():
        _validate_file(path, sources[name])

    records = _load_jsonl(paths["positive_candidates"])
    historical_entries = load_disprot_entries(paths["historical_disprot"])
    verification_entries = load_disprot_entries(paths["sequence_verification_only"])
    metadata = json.loads(paths["pdbe_metadata"].read_text(encoding="utf-8"))
    quality = config["quality_policy"]
    qualified, quality_summary = qualified_pdb_entries(
        metadata,
        cutoff_date=config["historical_cutoff_date"],
        xray_max_resolution=float(quality["x-ray"]["maximum_resolution_angstrom"]),
        em_max_resolution=float(quality["em"]["maximum_resolution_angstrom"]),
    )
    accession_map, verification_summary = exact_sequence_accession_map(
        records, verification_entries
    )
    conflict = config["conflict_policy"]
    masks, protected_summary = protected_masks(
        records,
        historical_entries,
        protected_states=set(conflict["protected_structural_states"]),
        boundary_margin=int(conflict["protected_boundary_margin_residues"]),
    )
    segments, sifts_summary = read_observed_segments(
        paths["sifts_observed_segments"],
        qualified_pdb_ids=set(qualified),
        accession_to_record=accession_map,
        records=records,
    )
    merged, label_summary = merge_observed_negatives(records, segments, masks, qualified)
    trainable = [
        record
        for record in merged
        if record["label_counts"]["positive"] + record["label_counts"]["negative"] > 0
    ]

    sentinel_path = _resolve(root, config["external_test_sentinel"])
    post_merge_findings = audit_records(
        (_fasta_record(record) for record in trainable), sentinel_path
    )
    combined_summary: dict[str, object] = {
        "input_sequences": len(records),
        "qualified_pdb_entries": len(qualified),
        **quality_summary,
        **verification_summary,
        **protected_summary,
        **sifts_summary,
        **label_summary,
        "post_merge_direct_leakage_findings": len(post_merge_findings),
    }
    _validate_expected(combined_summary, config["expected"])
    if post_merge_findings:
        raise ValueError("Direct CAID2/3 leakage remained after PDB merge")

    outputs = config["outputs"]
    all_path = _resolve(root, outputs["all_merged_jsonl"])
    trainable_path = _resolve(root, outputs["trainable_jsonl"])
    fasta_path = _resolve(root, outputs["trainable_fasta"])
    manifest_path = _resolve(root, outputs["manifest"])
    all_path.parent.mkdir(parents=True, exist_ok=True)
    with all_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    with trainable_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in trainable:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_fasta((_fasta_record(record) for record in trainable), fasta_path)

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": config["dataset_name"],
        "task": "single_output_classic_idr",
        "training_readiness": "blocked_pending_caid23_homology_audit_and_cluster_split",
        "historical_cutoff_date": config["historical_cutoff_date"],
        "quality_policy": config["quality_policy"],
        "mapping_policy": config["mapping_policy"],
        "conflict_policy": config["conflict_policy"],
        "sequence_verification_only_prohibition": sources["sequence_verification_only"][
            "prohibition"
        ],
        "sources": sources,
        "summary": combined_summary,
        "external_test_direct_audit": {
            "sentinel_path": sentinel_path.relative_to(root).as_posix(),
            "sentinel_sha256": file_sha256(sentinel_path),
            "findings": post_merge_findings,
        },
        "outputs": {
            "all_merged_jsonl": {
                "path": all_path.relative_to(root).as_posix(),
                "bytes": all_path.stat().st_size,
                "sha256": file_sha256(all_path),
            },
            "trainable_jsonl": {
                "path": trainable_path.relative_to(root).as_posix(),
                "bytes": trainable_path.stat().st_size,
                "sha256": file_sha256(trainable_path),
            },
            "trainable_fasta": {
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
    print(json.dumps(combined_summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
