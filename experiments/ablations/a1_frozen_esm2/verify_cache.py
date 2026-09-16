"""Independently verify an A1 ESM representation cache against its sequence manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cache_embeddings import artifact_stem, atomic_write_json, file_sha256, load_sequence_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_sequence_manifest(args.manifest)
    index_path = args.cache_dir / "cache_index.jsonl"
    index_rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_id = {str(row["protein_id"]): row for row in index_rows}
    if len(by_id) != len(index_rows):
        raise ValueError("cache index contains duplicate protein IDs")
    expected_ids = {record.protein_id for record in records}
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        extra = sorted(set(by_id) - expected_ids)
        raise ValueError(f"cache/index mismatch: missing={missing[:5]}, extra={extra[:5]}")

    verified_residues = 0
    verified_bytes = 0
    hidden_sizes: set[int] = set()
    revisions: set[str] = set()
    for record in records:
        row = by_id[record.protein_id]
        stem = artifact_stem(record.protein_id)
        expected_relative = f"embeddings/{stem}.npy"
        if row["embedding_file"] != expected_relative:
            raise ValueError(f"unexpected cache path for {record.protein_id}")
        embedding_path = args.cache_dir / expected_relative
        embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        hidden_size = int(row["hidden_size"])
        if embedding.shape != (len(record.sequence), hidden_size):
            raise ValueError(f"shape mismatch for {record.protein_id}: {embedding.shape}")
        if embedding.dtype != np.float16:
            raise ValueError(f"dtype mismatch for {record.protein_id}: {embedding.dtype}")
        if not np.all(np.isfinite(embedding)):
            raise ValueError(f"non-finite cached value for {record.protein_id}")
        observed_hash = file_sha256(embedding_path)
        if observed_hash != row["embedding_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {record.protein_id}")
        if record.sequence_sha256 != row["sequence_sha256"]:
            raise ValueError(f"sequence hash mismatch for {record.protein_id}")
        if int(embedding.nbytes) != int(row["embedding_nbytes"]):
            raise ValueError(f"byte count mismatch for {record.protein_id}")
        verified_residues += len(record.sequence)
        verified_bytes += int(embedding.nbytes)
        hidden_sizes.add(hidden_size)
        revisions.add(str(row["model_revision"]))

    if len(hidden_sizes) != 1 or len(revisions) != 1:
        raise ValueError("cache mixes hidden sizes or model revisions")
    report = {
        "schema_version": 1,
        "status": "pass",
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": file_sha256(args.manifest),
        "cache_index_sha256": file_sha256(index_path),
        "proteins": len(records),
        "residues": verified_residues,
        "hidden_size": next(iter(hidden_sizes)),
        "model_revision": next(iter(revisions)),
        "verified_embedding_bytes": verified_bytes,
        "dtype": "float16",
        "all_finite": True,
        "all_file_hashes_match": True,
        "input_contains_residue_labels": False,
        "caid2_caid3_labels_accessed": False,
    }
    atomic_write_json(args.cache_dir / "cache_verification_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
