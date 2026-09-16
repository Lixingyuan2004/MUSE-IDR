"""Cache one frozen ESM2-650M representation vector per protein residue."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from data import center_merge_weights, window_starts


@dataclass(frozen=True)
class SequenceRecord:
    protein_id: str
    sequence: str
    fold: str

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sequence_manifest(path: Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            forbidden = {"labels", "label", "scores"} & set(row)
            if forbidden:
                raise ValueError(
                    f"cache input must be label-free; found {sorted(forbidden)} at line {line_number}"
                )
            protein_id = str(row["id"])
            sequence = str(row["sequence"]).upper().replace(" ", "")
            fold = str(row.get("fold", ""))
            if not protein_id or protein_id in seen:
                raise ValueError(f"missing or duplicate protein id at line {line_number}")
            if not sequence:
                raise ValueError(f"empty sequence for {protein_id}")
            try:
                sequence.encode("ascii")
            except UnicodeEncodeError as error:
                raise ValueError(f"non-ASCII sequence for {protein_id}") from error
            seen.add(protein_id)
            records.append(SequenceRecord(protein_id, sequence, fold))
    if not records:
        raise ValueError(f"sequence manifest is empty: {path}")
    return records


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def atomic_save_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def artifact_stem(protein_id: str) -> str:
    return hashlib.sha256(protein_id.encode("utf-8")).hexdigest()[:24]


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def cached_entry_if_valid(
    record: SequenceRecord,
    embedding_path: Path,
    metadata_path: Path,
    expected: dict[str, object],
) -> dict[str, object] | None:
    if not embedding_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                return None
        embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (len(record.sequence), int(metadata["hidden_size"]))
        if embedding.shape != expected_shape or str(embedding.dtype) != str(metadata["dtype"]):
            return None
        if embedding.nbytes != int(metadata["embedding_nbytes"]):
            return None
        return metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


@torch.inference_mode()
def encode_record(
    record: SequenceRecord,
    tokenizer,
    backbone,
    device: torch.device,
    precision: str,
    max_residues: int,
    stride: int,
    window_batch_size: int,
) -> np.ndarray:
    starts = window_starts(len(record.sequence), max_residues, stride)
    hidden_size = int(backbone.config.hidden_size)
    weighted_hidden = np.zeros((len(record.sequence), hidden_size), dtype=np.float32)
    weight_sum = np.zeros(len(record.sequence), dtype=np.float32)

    spans = [(start, min(start + max_residues, len(record.sequence))) for start in starts]
    for batch_start in range(0, len(spans), window_batch_size):
        batch_spans = spans[batch_start : batch_start + window_batch_size]
        slices = [record.sequence[start:end] for start, end in batch_spans]
        encoded = tokenizer(
            slices,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        residue_mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()

        with autocast_context(device, precision):
            hidden = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden.float().cpu()

        for index, (start, end) in enumerate(batch_spans):
            residue_hidden = hidden[index][residue_mask[index]]
            expected_length = end - start
            if residue_hidden.shape != (expected_length, hidden_size):
                raise RuntimeError(
                    f"{record.protein_id}[{start}:{end}] token alignment failure: "
                    f"{tuple(residue_hidden.shape)} != {(expected_length, hidden_size)}"
                )
            weights = center_merge_weights(expected_length)
            weighted_hidden[start:end] += residue_hidden.numpy() * weights[:, None]
            weight_sum[start:end] += weights

    if np.any(weight_sum <= 0):
        raise RuntimeError(f"uncovered residues while caching {record.protein_id}")
    merged = weighted_hidden / weight_sum[:, None]
    if not np.all(np.isfinite(merged)):
        raise RuntimeError(f"non-finite representation for {record.protein_id}")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("cache_config.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    records = load_sequence_manifest(args.manifest)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]

    output_dir = args.output_dir
    embedding_dir = output_dir / "embeddings"
    metadata_dir = output_dir / "metadata"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    model_id = str(config["model_id"])
    requested_revision = str(config["revision"])
    precision = str(config["precision"]).lower()
    storage_dtype = np.dtype(str(config["storage_dtype"]))
    max_residues = int(config["max_residues"])
    stride = int(config["window_stride"])
    window_batch_size = int(config["window_batch_size"])
    resume = bool(config["resume"])
    if storage_dtype != np.float16:
        raise ValueError("A1 cache currently requires float16 storage")
    if window_batch_size < 1:
        raise ValueError("window_batch_size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and precision == "bf16":
        model_dtype = torch.bfloat16
    elif device.type == "cuda" and precision == "fp16":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=requested_revision)
    backbone = AutoModel.from_pretrained(
        model_id,
        revision=requested_revision,
        dtype=model_dtype,
        add_pooling_layer=False,
    ).to(device)
    backbone.requires_grad_(False)
    backbone.eval()
    resolved_revision = getattr(backbone.config, "_commit_hash", None)
    if resolved_revision != requested_revision:
        raise RuntimeError(
            f"resolved model revision {resolved_revision!r} != requested {requested_revision!r}"
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    started = perf_counter()
    entries: list[dict[str, object]] = []
    cached = 0
    generated = 0
    for index, record in enumerate(records, start=1):
        stem = artifact_stem(record.protein_id)
        embedding_path = embedding_dir / f"{stem}.npy"
        metadata_path = metadata_dir / f"{stem}.json"
        expected = {
            "protein_id": record.protein_id,
            "sequence_sha256": record.sequence_sha256,
            "sequence_length": len(record.sequence),
            "model_id": model_id,
            "model_revision": resolved_revision,
            "max_residues": max_residues,
            "window_stride": stride,
            "dtype": str(storage_dtype),
        }
        metadata = (
            cached_entry_if_valid(record, embedding_path, metadata_path, expected)
            if resume
            else None
        )
        if metadata is not None:
            cached += 1
        else:
            representation = encode_record(
                record,
                tokenizer,
                backbone,
                device,
                precision,
                max_residues,
                stride,
                window_batch_size,
            ).astype(storage_dtype, copy=False)
            atomic_save_numpy(embedding_path, representation)
            metadata = {
                **expected,
                "fold": record.fold,
                "hidden_size": int(representation.shape[1]),
                "embedding_file": embedding_path.relative_to(output_dir).as_posix(),
                "embedding_nbytes": int(representation.nbytes),
                "embedding_sha256": file_sha256(embedding_path),
                "finite": bool(np.all(np.isfinite(representation))),
                "caid2_caid3_labels_accessed": False,
            }
            atomic_write_json(metadata_path, metadata)
            generated += 1
        entries.append(metadata)
        print(
            json.dumps(
                {
                    "protein": index,
                    "total": len(records),
                    "protein_id": record.protein_id,
                    "length": len(record.sequence),
                    "source": "cache" if metadata is not None and index <= cached else "generated",
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    index_path = output_dir / "cache_index.jsonl"
    with index_path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, separators=(",", ":")) + "\n")

    report = {
        "schema_version": 1,
        "status": "pass",
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": file_sha256(args.manifest),
        "model_id": model_id,
        "model_revision": resolved_revision,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
        "precision": precision,
        "storage_dtype": str(storage_dtype),
        "max_residues": max_residues,
        "window_stride": stride,
        "proteins": len(entries),
        "generated": generated,
        "reused": cached,
        "total_residues": sum(int(entry["sequence_length"]) for entry in entries),
        "cache_bytes": sum(int(entry["embedding_nbytes"]) for entry in entries),
        "index_sha256": file_sha256(index_path),
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "input_contains_residue_labels": False,
        "caid2_caid3_labels_accessed": False,
    }
    atomic_write_json(output_dir / "cache_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
