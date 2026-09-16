"""Smoke-test A2 shape, padding invariance, masking and gradients."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a2_multiscale_context.data import pad_cached_proteins
from experiments.ablations.a2_multiscale_context.model import MultiscaleContextIdrHead


def example(protein_id: str, length: int, hidden_size: int) -> dict[str, object]:
    generator = np.random.default_rng(length)
    labels = np.asarray(([0, 1, -1] * ((length + 2) // 3))[:length], dtype=np.float32)
    return {
        "protein_id": protein_id,
        "features": torch.from_numpy(
            generator.normal(size=(length, hidden_size)).astype(np.float16)
        ),
        "labels": torch.from_numpy(labels),
        "length": length,
    }


def main() -> None:
    torch.manual_seed(17)
    hidden_size = 16
    model = MultiscaleContextIdrHead(
        input_size=hidden_size,
        hidden_size=8,
        kernels=(3, 5),
        dropout=0.0,
    )
    short = example("short", 7, hidden_size)
    long = example("long", 13, hidden_size)
    short_batch = pad_cached_proteins([short])
    padded_batch = pad_cached_proteins([short, long])

    model.eval()
    with torch.inference_mode():
        short_logits = model(
            short_batch["features"].float(), short_batch["residue_mask"]
        )[0, :7]
        padded_logits = model(
            padded_batch["features"].float(), padded_batch["residue_mask"]
        )[0, :7]
    torch.testing.assert_close(short_logits, padded_logits, rtol=0, atol=1e-6)

    model.train()
    logits = model(padded_batch["features"].float(), padded_batch["residue_mask"])
    known = padded_batch["residue_mask"] & (padded_batch["labels"] != -1)
    loss = F.binary_cross_entropy_with_logits(logits[known], padded_batch["labels"][known])
    loss.backward()
    if model.classifier.weight.grad is None:
        raise RuntimeError("A2 classifier did not receive gradients")
    if model.output_heads != 1 or logits.shape != (2, 13):
        raise RuntimeError("A2 single-output shape check failed")
    if not torch.all(logits[~padded_batch["residue_mask"]] == 0):
        raise RuntimeError("A2 padded logits must be masked")

    print(
        json.dumps(
            {
                "status": "pass",
                "architecture": "gated_multiscale_depthwise_cnn",
                "kernels": list(model.kernels),
                "padding_invariant": True,
                "unknown_labels_excluded": True,
                "output_heads": model.output_heads,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "caid2_caid3_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
