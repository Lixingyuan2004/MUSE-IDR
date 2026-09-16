"""Run an official LoRA-DR-Suite DisProt7 checkpoint on label-free FASTA.

Long proteins are split into overlapping windows.  Per-window two-class logits
are converted to a disorder-vs-order logit margin, averaged at overlapping
residues, and transformed to one classic-IDR probability per residue.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b4_baseline_reproduction.run_lora_official_smoke import (  # noqa: E402
    ALLOWED_MODEL_IDS,
    DEFAULT_MODEL_ID,
    resolve_device,
    resolve_dtype,
)


ALLOWED_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


@dataclass(frozen=True)
class FastaRecord:
    protein_id: str
    sequence: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def read_fasta(path: str | Path) -> list[FastaRecord]:
    """Read sequence-only FASTA and reject labels or aligned sequences."""

    path = Path(path)
    records: list[FastaRecord] = []
    seen: set[str] = set()
    header: str | None = None
    lines: list[str] = []

    def finish() -> None:
        nonlocal header, lines
        if header is None:
            return
        protein_id = header.split(maxsplit=1)[0]
        sequence = "".join(lines).replace(" ", "").upper()
        if not protein_id or protein_id in seen:
            raise ValueError(f"empty or duplicate FASTA identifier: {protein_id!r}")
        if not sequence:
            raise ValueError(f"empty FASTA sequence: {protein_id}")
        invalid = sorted(set(sequence) - ALLOWED_RESIDUES)
        if invalid:
            raise ValueError(
                f"unsupported residues in {protein_id}: {invalid}; "
                "CAID label/reference files are not valid inference inputs"
            )
        seen.add(protein_id)
        records.append(FastaRecord(protein_id, sequence))
        header = None
        lines = []

    with _open_text(path, "r") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"empty FASTA header at line {line_number}")
            else:
                if header is None:
                    raise ValueError(f"sequence before FASTA header at line {line_number}")
                lines.append("".join(line.split()))
    finish()
    if not records:
        raise ValueError(f"FASTA contains no records: {path}")
    return records


def window_starts(length: int, max_residues: int, stride: int) -> list[int]:
    if length < 1 or max_residues < 1 or stride < 1:
        raise ValueError("length and window sizes must be positive")
    if stride > max_residues:
        raise ValueError("window stride cannot exceed max residues")
    if length <= max_residues:
        return [0]
    starts = list(range(0, length - max_residues + 1, stride))
    final_start = length - max_residues
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def extract_window_margins(
    logits: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    expected_lengths: Sequence[int],
) -> list[torch.Tensor]:
    """Remove special/padding tokens and return class-1 minus class-0 logits."""

    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError(f"expected logits [batch, tokens, 2], got {tuple(logits.shape)}")
    if special_tokens_mask.shape != logits.shape[:2]:
        raise ValueError("special-token mask is not aligned with logits")
    if attention_mask.shape != logits.shape[:2]:
        raise ValueError("attention mask is not aligned with logits")
    if len(expected_lengths) != logits.shape[0]:
        raise ValueError("window length list is not aligned with logits")

    margins = logits.float()[..., 1] - logits.float()[..., 0]
    results: list[torch.Tensor] = []
    for row, expected in enumerate(expected_lengths):
        residue_mask = attention_mask[row].bool() & ~special_tokens_mask[row].bool()
        values = margins[row, residue_mask]
        if values.numel() != expected:
            raise ValueError(
                f"expected {expected} residues in window {row}, got {values.numel()}"
            )
        if not torch.isfinite(values).all():
            raise ValueError("non-finite window logit detected")
        results.append(values)
    return results


def merge_window_margins(
    sequence_length: int,
    starts: Sequence[int],
    margins: Sequence[torch.Tensor | np.ndarray],
) -> np.ndarray:
    """Uniformly average overlapping logit margins and return probabilities."""

    if len(starts) != len(margins) or not starts:
        raise ValueError("window starts and margins must be non-empty and aligned")
    sums = np.zeros(sequence_length, dtype=np.float64)
    counts = np.zeros(sequence_length, dtype=np.int32)
    for start, values in zip(starts, margins):
        array = np.asarray(values, dtype=np.float64)
        end = start + len(array)
        if start < 0 or end > sequence_length or not np.all(np.isfinite(array)):
            raise ValueError("invalid window placement or values")
        sums[start:end] += array
        counts[start:end] += 1
    if np.any(counts == 0):
        raise ValueError("sliding windows do not cover every residue")
    mean_margin = sums / counts
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(mean_margin, -80.0, 80.0)))
    result = probabilities.astype(np.float32)
    if result.shape != (sequence_length,) or not np.all(np.isfinite(result)):
        raise RuntimeError("invalid merged residue probabilities")
    return result


def verify_loaded_adapter(model, peft_config) -> dict[str, Any]:
    names = [name for name, _ in model.named_parameters()]
    lora_names = [name for name in names if "lora_A" in name or "lora_B" in name]
    active_head_names = [
        name
        for name in names
        if "modules_to_save.default" in name
        and ("classifier" in name or ".score" in name)
    ]
    target_modules = set(peft_config.target_modules or [])
    saved_modules = set(peft_config.modules_to_save or [])
    if not {"query", "value"}.issubset(target_modules):
        raise RuntimeError("official adapter does not contain query/value LoRA targets")
    if not ({"classifier", "score"} & saved_modules):
        raise RuntimeError("official adapter does not declare a saved classification head")
    if not lora_names or not active_head_names:
        raise RuntimeError("loaded model does not expose active LoRA and saved-head parameters")
    return {
        "target_modules": sorted(target_modules),
        "modules_to_save": sorted(saved_modules),
        "loaded_lora_parameter_tensors": len(lora_names),
        "loaded_saved_head_parameter_tensors": len(active_head_names),
        "lora_and_saved_head_loaded": True,
    }


def load_model(model_id: str, revision: str | None, device: torch.device, dtype: torch.dtype):
    from huggingface_hub import HfApi, hf_hub_download
    from peft import PeftConfig, PeftModel
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    api = HfApi()
    adapter_revision = revision or api.model_info(model_id).sha
    if not adapter_revision or len(adapter_revision) != 40:
        raise RuntimeError("failed to resolve full adapter revision")
    peft_config = PeftConfig.from_pretrained(model_id, revision=adapter_revision)
    base_model_id = peft_config.base_model_name_or_path
    if not base_model_id:
        raise RuntimeError("LoRA adapter does not declare a base model")
    base_revision = api.model_info(base_model_id).sha
    if not base_revision or len(base_revision) != 40:
        raise RuntimeError("failed to resolve full base-model revision")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, revision=base_revision)
    base_config = AutoConfig.from_pretrained(base_model_id, revision=base_revision)
    if base_config.model_type != "esm":
        raise RuntimeError(f"expected ESM base model, got {base_config.model_type!r}")
    base_config.num_labels = 2
    base_model = AutoModelForTokenClassification.from_pretrained(
        base_model_id,
        revision=base_revision,
        config=base_config,
        dtype=dtype,
    )
    model = PeftModel.from_pretrained(
        base_model,
        model_id,
        revision=adapter_revision,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    adapter_verification = verify_loaded_adapter(model, peft_config)

    adapter_config_path = hf_hub_download(
        model_id, "adapter_config.json", revision=adapter_revision
    )
    adapter_weights_path = hf_hub_download(
        model_id, "adapter_model.safetensors", revision=adapter_revision
    )
    provenance = {
        "model_id": model_id,
        "model_revision": adapter_revision,
        "base_model_id": base_model_id,
        "base_model_revision": base_revision,
        "adapter_config_sha256": file_sha256(adapter_config_path),
        "adapter_weights_sha256": file_sha256(adapter_weights_path),
        **adapter_verification,
    }
    return tokenizer, model, provenance


@torch.inference_mode()
def predict_sequence(
    sequence: str,
    tokenizer,
    model,
    device: torch.device,
    max_residues: int,
    stride: int,
    window_batch_size: int,
) -> tuple[np.ndarray, int]:
    starts = window_starts(len(sequence), max_residues, stride)
    windows = [sequence[start : start + max_residues] for start in starts]
    all_margins: list[torch.Tensor] = []
    for offset in range(0, len(windows), window_batch_size):
        batch = windows[offset : offset + window_batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special_tokens_mask = encoded.pop("special_tokens_mask")
        attention_mask = encoded["attention_mask"].clone()
        model_inputs = {name: value.to(device) for name, value in encoded.items()}
        logits = model(**model_inputs).logits.detach().cpu()
        all_margins.extend(
            extract_window_margins(
                logits,
                special_tokens_mask,
                attention_mask,
                [len(window) for window in batch],
            )
        )
    return merge_window_margins(len(sequence), starts, all_margins), len(windows)


def _atomic_text_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix.lower() == ".gz":
        handle = gzip.open(temporary, "wt", encoding="utf-8", newline="")
    else:
        handle = temporary.open("w", encoding="utf-8", newline="")
    return temporary, handle


def write_outputs(
    tsv_path: Path,
    caid_path: Path | None,
    records: Sequence[FastaRecord],
    predictions: dict[str, np.ndarray],
) -> None:
    temporary, handle = _atomic_text_writer(tsv_path)
    try:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["protein_id", "position", "residue", "idr_probability"])
        for record in records:
            scores = predictions[record.protein_id]
            if scores.shape != (len(record.sequence),):
                raise ValueError(f"prediction length mismatch: {record.protein_id}")
            for position, (residue, score) in enumerate(
                zip(record.sequence, scores), start=1
            ):
                writer.writerow([record.protein_id, position, residue, f"{float(score):.9g}"])
    finally:
        handle.close()
    temporary.replace(tsv_path)

    if caid_path is not None:
        temporary, handle = _atomic_text_writer(caid_path)
        try:
            for record in records:
                handle.write(f">{record.protein_id}\n")
                for position, (residue, score) in enumerate(
                    zip(record.sequence, predictions[record.protein_id]), start=1
                ):
                    handle.write(f"{position}\t{residue}\t{float(score):.9g}\n")
        finally:
            handle.close()
        temporary.replace(caid_path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--model-id", choices=sorted(ALLOWED_MODEL_IDS), default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caid-output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-residues", type=int, default=1022)
    parser.add_argument("--window-stride", type=int, default=511)
    parser.add_argument("--window-batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_batch_size < 1:
        raise ValueError("window batch size must be positive")
    records = read_fasta(args.fasta)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.precision, device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer, model, provenance = load_model(
        args.model_id, args.revision, device, dtype
    )

    predictions: dict[str, np.ndarray] = {}
    total_windows = 0
    for index, record in enumerate(records, start=1):
        scores, windows = predict_sequence(
            record.sequence,
            tokenizer,
            model,
            device,
            args.max_residues,
            args.window_stride,
            args.window_batch_size,
        )
        predictions[record.protein_id] = scores
        total_windows += windows
        print(
            json.dumps(
                {
                    "protein": index,
                    "total": len(records),
                    "protein_id": record.protein_id,
                    "length": len(record.sequence),
                    "windows": windows,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    write_outputs(args.output, args.caid_output, records, predictions)
    total_residues = sum(len(record.sequence) for record in records)
    score_values = np.concatenate(list(predictions.values()))
    report = {
        "schema_version": 1,
        "experiment": "b4_lora_dr_suite_official_external_inference",
        "status": "pass",
        **provenance,
        "training_regime": "DisProt_7_only",
        "input_fasta": str(args.fasta),
        "input_fasta_sha256": file_sha256(args.fasta),
        "input_contains_labels": False,
        "proteins": len(records),
        "residues": total_residues,
        "windows": total_windows,
        "max_residues": args.max_residues,
        "window_stride": args.window_stride,
        "window_batch_size": args.window_batch_size,
        "window_fusion": "uniform_mean_of_disorder_vs_order_logit_margin",
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "caid_output": str(args.caid_output) if args.caid_output else None,
        "caid_output_sha256": file_sha256(args.caid_output) if args.caid_output else None,
        "score_min": float(score_values.min()),
        "score_max": float(score_values.max()),
        "output_scores": int(score_values.size),
        "prediction_alignment": int(score_values.size) == total_residues,
        "output_heads": 1,
        "output_definition": "classic_idr_probability_per_residue",
        "soft_disorder_output": False,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "precision": str(dtype).removeprefix("torch."),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "caid_labels_accessed": False,
        "caid_labels_used_for_training_or_tuning": False,
        "scores_used_for_training_or_tuning": False,
    }
    if not report["prediction_alignment"]:
        raise RuntimeError("final residue prediction alignment failed")
    if not (math.isfinite(report["score_min"]) and math.isfinite(report["score_max"])):
        raise RuntimeError("non-finite output score summary")
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
