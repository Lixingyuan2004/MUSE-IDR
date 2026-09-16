"""Smoke-test A7 padding invariance, masking and bidirectional gradients."""

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
from experiments.ablations.a7_bidirectional_sequence_context.model import (
    BidirectionalSequenceContextIdrHead,
)


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
    input_size = 16
    model = BidirectionalSequenceContextIdrHead(
        input_size=input_size,
        hidden_size=8,
        kernels=(3, 5),
        gru_hidden_size=4,
        gru_layers=1,
        dropout=0.0,
        sequence_dropout=0.0,
        sequence_gate_init=-2.0,
    )
    short = example("short", 7, input_size)
    long = example("long", 19, input_size)
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
    torch.testing.assert_close(short_logits, padded_logits, rtol=1e-6, atol=1e-6)

    model.train()
    logits = model(padded_batch["features"].float(), padded_batch["residue_mask"])
    logits.retain_grad()
    known = padded_batch["residue_mask"] & (padded_batch["labels"] != -1)
    loss = F.binary_cross_entropy_with_logits(logits[known], padded_batch["labels"][known])
    loss.backward()
    if model.classifier.weight.grad is None:
        raise RuntimeError("A7 classifier did not receive gradients")
    if model.bidirectional_gru.weight_ih_l0.grad is None:
        raise RuntimeError("A7 forward GRU did not receive gradients")
    if model.bidirectional_gru.weight_ih_l0_reverse.grad is None:
        raise RuntimeError("A7 reverse GRU did not receive gradients")
    if not torch.all(logits.grad[~known] == 0):
        raise RuntimeError("unknown or padding logits affected the supervised loss")
    if model.output_heads != 1 or logits.shape != (2, 19):
        raise RuntimeError("A7 single-output shape check failed")
    if not torch.all(logits[~padded_batch["residue_mask"]] == 0):
        raise RuntimeError("A7 padded logits must be masked")
    initial_gate = torch.sigmoid(model.sequence_gate.bias.detach())
    expected_gate = torch.full_like(initial_gate, torch.sigmoid(torch.tensor(-2.0)))
    torch.testing.assert_close(initial_gate, expected_gate)

    formal = BidirectionalSequenceContextIdrHead()
    print(
        json.dumps(
            {
                "status": "pass",
                "architecture": "a2_local_plus_gated_bidirectional_gru",
                "local_kernels": list(formal.kernels),
                "gru_hidden_size_per_direction": formal.gru_hidden_size,
                "gru_layers": formal.gru_layers,
                "bidirectional_sequence_context": formal.sequence_context_is_bidirectional,
                "padding_invariant": True,
                "unknown_labels_excluded": True,
                "forward_and_reverse_gru_receive_gradients": True,
                "initial_sequence_gate": float(initial_gate.mean()),
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
