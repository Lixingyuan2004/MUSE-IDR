"""Load the real ESM2-650M checkpoint and verify residue/token alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from data import EsmWindowCollator
from model import FrozenEsmResidueClassifier


def example(protein_id: str, sequence: str) -> dict[str, object]:
    length = len(sequence)
    return {
        "record_index": 0,
        "protein_id": protein_id,
        "start": 0,
        "end": length,
        "sequence": sequence,
        "labels": np.zeros(length, dtype=np.int8),
        "loss_weight": np.ones(length, dtype=np.float32),
        "merge_weight": np.ones(length, dtype=np.float32),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_config = config["model"]
    model_id = str(model_config["pretrained_model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    collator = EsmWindowCollator(tokenizer)

    short_examples = [
        example("short_20", "ACDEFGHIKLMNPQRSTVWY"),
        example("short_31", "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIV"),
    ]
    batch = collator(short_examples)
    model = FrozenEsmResidueClassifier(
        pretrained_model=model_id,
        dropout=float(model_config["dropout"]),
        precision=str(model_config["precision"]),
        device=device,
    )
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        ).cpu()

    per_sequence = []
    for index, item in enumerate(short_examples):
        residue_logits = logits[index][batch["residue_mask"][index]]
        expected = len(str(item["sequence"]))
        if residue_logits.numel() != expected:
            raise RuntimeError(
                f"{item['protein_id']}: expected {expected} logits, got {residue_logits.numel()}"
            )
        probabilities = torch.sigmoid(residue_logits)
        if not torch.all(torch.isfinite(probabilities)):
            raise RuntimeError(f"non-finite probability for {item['protein_id']}")
        if not torch.all((probabilities >= 0) & (probabilities <= 1)):
            raise RuntimeError(f"probability outside [0,1] for {item['protein_id']}")
        per_sequence.append(
            {
                "protein_id": item["protein_id"],
                "residues": expected,
                "output_scores": int(residue_logits.numel()),
                "probability_min": float(probabilities.min()),
                "probability_max": float(probabilities.max()),
            }
        )

    # Tokenize the maximum supported residue window without running the 650M
    # forward pass on the local 6 GB GPU.
    boundary = example("boundary_1022", "A" * 1022)
    boundary_batch = collator([boundary])
    boundary_residues = int(boundary_batch["residue_mask"].sum())
    if boundary_residues != 1022:
        raise RuntimeError(f"1022-residue tokenizer boundary failed: {boundary_residues}")

    report = {
        "schema_version": 1,
        "experiment": "a1_frozen_esm2_smoke",
        "status": "pass",
        "model_id": model_id,
        "model_revision": getattr(model.backbone.config, "_commit_hash", None),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "sequences": per_sequence,
        "tokenizer_boundary_residues": boundary_residues,
        "output_heads": 1,
        "output_definition": "classic_idr_probability_per_residue",
        "caid2_caid3_labels_accessed": False,
    }
    output_path = Path(__file__).with_name("smoke_test_report.json")
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
