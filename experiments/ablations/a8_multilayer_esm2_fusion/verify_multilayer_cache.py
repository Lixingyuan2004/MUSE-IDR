"""Independently verify an A8 multi-layer cache against a label-free manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a8_multilayer_esm2_fusion.cache_multilayer import (  # noqa: E402
    artifact_stem,
    atomic_write_json,
    file_sha256,
    load_sequence_manifest,
)


def verify_cache(manifest: Path, cache_dir: Path) -> dict[str, object]:
    records = load_sequence_manifest(manifest)
    index_path = cache_dir / "cache_index.jsonl"
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
    selected_layers: set[tuple[int, ...]] = set()
    for record in records:
        row = by_id[record.protein_id]
        stem = artifact_stem(record.protein_id)
        expected_relative = f"embeddings/{stem}.npy"
        if row["embedding_file"] != expected_relative:
            raise ValueError(f"unexpected cache path for {record.protein_id}")
        if str(row.get("fold")) != record.fold:
            raise ValueError(f"fold mismatch for {record.protein_id}")
        embedding_path = cache_dir / expected_relative
        embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        hidden_size = int(row["hidden_size"])
        layers = tuple(int(value) for value in row["layer_indices"])
        if int(row["num_layers"]) != len(layers):
            raise ValueError(f"layer count mismatch for {record.protein_id}")
        expected_shape = (len(record.sequence), len(layers), hidden_size)
        if embedding.shape != expected_shape:
            raise ValueError(
                f"shape mismatch for {record.protein_id}: "
                f"{embedding.shape} != {expected_shape}"
            )
        if embedding.dtype != np.float16:
            raise ValueError(f"dtype mismatch for {record.protein_id}: {embedding.dtype}")
        if not np.all(np.isfinite(embedding)):
            raise ValueError(f"non-finite cached value for {record.protein_id}")
        if file_sha256(embedding_path) != row["embedding_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {record.protein_id}")
        if record.sequence_sha256 != row["sequence_sha256"]:
            raise ValueError(f"sequence hash mismatch for {record.protein_id}")
        if int(row["sequence_length"]) != len(record.sequence):
            raise ValueError(f"sequence length mismatch for {record.protein_id}")
        if int(embedding.nbytes) != int(row["embedding_nbytes"]):
            raise ValueError(f"byte count mismatch for {record.protein_id}")
        if row.get("caid2_caid3_labels_accessed") is not False:
            raise ValueError(f"CAID audit flag failed for {record.protein_id}")
        verified_residues += len(record.sequence)
        verified_bytes += int(embedding.nbytes)
        hidden_sizes.add(hidden_size)
        revisions.add(str(row["model_revision"]))
        selected_layers.add(layers)

    if len(hidden_sizes) != 1 or len(revisions) != 1 or len(selected_layers) != 1:
        raise ValueError("cache mixes hidden sizes, model revisions, or selected layers")
    layers = next(iter(selected_layers))
    report = {
        "schema_version": 1,
        "status": "pass",
        "manifest": manifest.as_posix(),
        "manifest_sha256": file_sha256(manifest),
        "cache_index_sha256": file_sha256(index_path),
        "proteins": len(records),
        "residues": verified_residues,
        "hidden_size": next(iter(hidden_sizes)),
        "layer_indices": list(layers),
        "num_layers": len(layers),
        "model_revision": next(iter(revisions)),
        "verified_embedding_bytes": verified_bytes,
        "dtype": "float16",
        "all_finite": True,
        "all_file_hashes_match": True,
        "input_contains_residue_labels": False,
        "caid2_caid3_labels_accessed": False,
    }
    atomic_write_json(cache_dir / "cache_verification_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_cache(args.manifest, args.cache_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
