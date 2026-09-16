"""Smoke-test A8 scalar mixing, padding invariance, masking, and gradients."""

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

from experiments.ablations.a8_multilayer_esm2_fusion.data import (
    pad_multilayer_cached_proteins,
)
from experiments.ablations.a8_multilayer_esm2_fusion.model import (
    MultilayerEsm2FusionIdrModel,
)


def example(
    protein_id: str, length: int, num_layers: int, hidden_size: int
) -> dict[str, object]:
    generator = np.random.default_rng(length)
    labels = np.asarray(([0, 1, -1] * ((length + 2) // 3))[:length], dtype=np.float32)
    return {
        "protein_id": protein_id,
        "features": torch.from_numpy(
            generator.normal(size=(length, num_layers, hidden_size)).astype(np.float16)
        ),
        "labels": torch.from_numpy(labels),
        "length": length,
    }


def main() -> None:
    torch.manual_seed(17)
    input_size = 16
    layers = (30, 31, 32, 33)
    model = MultilayerEsm2FusionIdrModel(
        input_size=input_size,
        hidden_size=8,
        layer_indices=layers,
        kernels=(3, 5),
        dropout=0.0,
        initial_last_layer_weight=0.7,
    )
    initial_weights = model.layer_weights.detach()
    torch.testing.assert_close(
        initial_weights,
        torch.tensor([0.1, 0.1, 0.1, 0.7]),
        rtol=1e-6,
        atol=1e-6,
    )
    short = example("short", 7, len(layers), input_size)
    long = example("long", 19, len(layers), input_size)
    short_batch = pad_multilayer_cached_proteins([short])
    padded_batch = pad_multilayer_cached_proteins([short, long])

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
    logits.retain_grad()
    known = padded_batch["residue_mask"] & (padded_batch["labels"] != -1)
    loss = F.binary_cross_entropy_with_logits(logits[known], padded_batch["labels"][known])
    loss.backward()
    if model.layer_logits.grad is None or not torch.all(
        torch.isfinite(model.layer_logits.grad)
    ):
        raise RuntimeError("A8 layer mixing logits did not receive finite gradients")
    if not torch.any(model.layer_logits.grad != 0):
        raise RuntimeError("A8 layer mixing logits received only zero gradients")
    if model.head.classifier.weight.grad is None:
        raise RuntimeError("A8 classifier did not receive gradients")
    if not torch.all(logits.grad[~known] == 0):
        raise RuntimeError("unknown or padding logits affected the supervised loss")
    if model.output_heads != 1 or logits.shape != (2, 19):
        raise RuntimeError("A8 single-output shape check failed")
    if not torch.all(logits[~padded_batch["residue_mask"]] == 0):
        raise RuntimeError("A8 padded logits must be masked")

    formal = MultilayerEsm2FusionIdrModel()
    print(
        json.dumps(
            {
                "status": "pass",
                "architecture": "learned_last4_esm2_scalar_mix_plus_a2",
                "layer_indices": list(formal.layer_indices),
                "initial_layer_weights": [
                    float(value) for value in formal.layer_weights.detach().tolist()
                ],
                "padding_invariant": True,
                "unknown_labels_excluded": True,
                "layer_mixture_receives_gradients": True,
                "output_heads": formal.output_heads,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in formal.parameters()
                ),
                "caid2_caid3_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
