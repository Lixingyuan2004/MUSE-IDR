from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from punch2_light_common import (
    DEFAULT_PROTT5_MODEL_ID,
    author_onehot,
    author_sequence_mapping,
    read_sequence_only_fasta,
    sha256,
)


def atomic_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if device.type != "cuda":
        raise ValueError(f"{name} feature extraction requires CUDA")
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def load_prott5(model_id: str, revision: str | None, device: torch.device, dtype: torch.dtype):
    from huggingface_hub import HfApi
    from transformers import T5EncoderModel, T5Tokenizer

    resolved_revision = revision or HfApi().model_info(model_id).sha
    if not resolved_revision or len(resolved_revision) != 40:
        raise RuntimeError("failed to resolve a full ProtT5 revision")
    tokenizer = T5Tokenizer.from_pretrained(
        model_id,
        revision=resolved_revision,
        do_lower_case=False,
    )
    model = T5EncoderModel.from_pretrained(
        model_id,
        revision=resolved_revision,
        dtype=dtype,
    ).to(device)
    model.eval()
    return tokenizer, model, resolved_revision


def extract_prott5(
    sequence: str,
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    mapped = author_sequence_mapping(sequence)
    spaced = " ".join(mapped)
    encoded = tokenizer(
        spaced,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    token_count = int(encoded["attention_mask"].sum().item())
    if token_count != len(sequence) + 1:
        raise ValueError(
            f"ProtT5 token/residue mismatch: {token_count} != {len(sequence)} + 1"
        )
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state[0, : len(sequence)]
    if hidden.shape != (len(sequence), 1024):
        raise ValueError(f"unexpected ProtT5 feature shape: {tuple(hidden.shape)}")
    result = hidden.detach().to(torch.float32).cpu().numpy()[None, :, :]
    if not np.all(np.isfinite(result)):
        raise ValueError("ProtT5 produced non-finite features")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_PROTT5_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    args = parser.parse_args()

    records = read_sequence_only_fasta(args.fasta)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.precision, device)
    tokenizer, model, revision = load_prott5(
        args.model_id,
        args.revision,
        device,
        dtype,
    )
    started = time.perf_counter()
    entries: list[dict[str, object]] = []
    total_residues = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for index, record in enumerate(records, start=1):
        onehot_path = args.output_dir / "onehot" / f"{record.protein_id}.npy"
        prottrans_path = args.output_dir / "protTrans" / f"{record.protein_id}.npy"
        onehot = author_onehot(record.sequence)
        prottrans = extract_prott5(record.sequence, tokenizer, model, device)
        atomic_numpy(onehot_path, onehot)
        atomic_numpy(prottrans_path, prottrans)
        entry = {
            "protein_id": record.protein_id,
            "length": len(record.sequence),
            "onehot": str(onehot_path.relative_to(args.output_dir)).replace("\\", "/"),
            "onehot_sha256": sha256(onehot_path),
            "prottrans": str(prottrans_path.relative_to(args.output_dir)).replace("\\", "/"),
            "prottrans_sha256": sha256(prottrans_path),
        }
        entries.append(entry)
        total_residues += len(record.sequence)
        print(
            json.dumps(
                {
                    "protein": index,
                    "total": len(records),
                    "protein_id": record.protein_id,
                    "length": len(record.sequence),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "experiment": "b4_punch2_light_official_feature_cache",
        "status": "pass",
        "input_fasta": str(args.fasta),
        "input_fasta_sha256": sha256(args.fasta),
        "input_contains_labels": False,
        "proteins": len(records),
        "residues": total_residues,
        "onehot_definition": "deemeng/embedding commit 28fe5c3; UZOB-to-X; 21 channels",
        "prott5_model_id": args.model_id,
        "prott5_revision": revision,
        "prott5_hidden_size": 1024,
        "prott5_truncation": False,
        "inference_precision": str(dtype).removeprefix("torch."),
        "storage_dtype": "float32",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "entries": entries,
        "caid_labels_accessed": False,
        "caid_labels_used_for_training_or_tuning": False,
    }
    if not math.isfinite(elapsed):
        raise RuntimeError("non-finite elapsed time")
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
