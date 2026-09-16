"""Load an official LoRA-DR-Suite checkpoint and run label-free smoke inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


DEFAULT_MODEL_ID = "CQSB/esm2_650M-LoRA-ID-DisProt7"
DEFAULT_SEQUENCE = "TAIWEQHTVTLHRAPGFGFGIAISGGRDNPHFQSGETSIVISDVLKG"
ALLOWED_MODEL_IDS = {
    "CQSB/esm2_35M-LoRA-ID-DisProt7",
    "CQSB/esm2_650M-LoRA-ID-DisProt7",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_residue_probabilities(
    logits: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    expected_residues: int,
) -> torch.Tensor:
    """Return positive-class probabilities after removing special tokens."""

    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 2:
        raise ValueError(f"expected logits [1, tokens, 2], got {tuple(logits.shape)}")
    if special_tokens_mask.shape != logits.shape[:2]:
        raise ValueError("special-token mask is not aligned with logits")
    probabilities = torch.softmax(logits.float(), dim=-1)[..., 1]
    residue_scores = probabilities[~special_tokens_mask.bool()]
    if residue_scores.numel() != expected_residues:
        raise ValueError(
            f"expected {expected_residues} residue scores, got {residue_scores.numel()}"
        )
    if not torch.isfinite(residue_scores).all():
        raise ValueError("non-finite residue probability detected")
    if not ((residue_scores >= 0) & (residue_scores <= 1)).all():
        raise ValueError("residue probabilities are outside [0, 1]")
    return residue_scores


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "fp32" or device.type == "cpu":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"unsupported precision: {precision}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, choices=sorted(ALLOWED_MODEL_IDS))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/b4_baseline_reproduction/lora_official_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = "".join(args.sequence.split()).upper()
    if not sequence or not sequence.isalpha():
        raise ValueError("sequence must contain amino-acid letters only")

    from huggingface_hub import HfApi
    from peft import PeftConfig, PeftModel
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    requested_revision = args.revision
    resolved_revision = requested_revision or HfApi().model_info(args.model_id).sha
    if not resolved_revision or len(resolved_revision) != 40:
        raise RuntimeError("failed to resolve a full Hugging Face model revision")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.precision, device)
    started = time.perf_counter()

    peft_config = PeftConfig.from_pretrained(
        args.model_id,
        revision=resolved_revision,
    )
    base_model_id = peft_config.base_model_name_or_path
    if not base_model_id:
        raise RuntimeError("LoRA adapter does not declare a base model")

    base_revision = HfApi().model_info(base_model_id).sha
    if not base_revision or len(base_revision) != 40:
        raise RuntimeError("failed to resolve a full base-model revision")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        revision=base_revision,
    )
    base_config = AutoConfig.from_pretrained(
        base_model_id,
        revision=base_revision,
    )
    if base_config.model_type != "esm":
        raise RuntimeError(
            f"expected ESM base configuration, got {base_config.model_type!r}"
        )
    base_config.num_labels = 2

    base_model = AutoModelForTokenClassification.from_pretrained(
        base_model_id,
        revision=base_revision,
        config=base_config,
        dtype=dtype,
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.model_id,
        revision=resolved_revision,
    ).to(device)
    model.eval()

    encoded = tokenizer(
        [sequence],
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    special_tokens_mask = encoded.pop("special_tokens_mask")
    model_inputs = {name: value.to(device) for name, value in encoded.items()}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        logits = model(**model_inputs).logits.detach().cpu()
    scores = extract_residue_probabilities(logits, special_tokens_mask, len(sequence))

    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    report = {
        "schema_version": 1,
        "experiment": "b4_lora_dr_suite_official_short_sequence_smoke",
        "status": "pass",
        "model_id": args.model_id,
        "model_revision": resolved_revision,
        "training_regime": "DisProt_7_only",
        "sequence_sha256": sha256_text(sequence),
        "residues": len(sequence),
        "output_scores": int(scores.numel()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "output_heads": 1,
        "output_definition": "classic_idr_probability_per_residue",
        "soft_disorder_output": False,
        "prediction_alignment": scores.numel() == len(sequence),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "precision": str(dtype).removeprefix("torch."),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
        "caid_input_used": False,
        "caid_labels_accessed": False,
        "caid_labels_used_for_training_or_tuning": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
