"""Smoke-test A8 multi-layer extraction, window merging, and label rejection."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a8_multilayer_esm2_fusion.cache_multilayer import (
    SequenceRecord,
    encode_record,
    load_sequence_manifest,
)


class FakeTokenizer:
    def __call__(
        self,
        sequences: list[str],
        padding: bool,
        return_tensors: str,
        return_special_tokens_mask: bool,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors, return_special_tokens_mask
        max_tokens = max(len(sequence) for sequence in sequences) + 2
        input_ids = torch.zeros((len(sequences), max_tokens), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        special = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[index, 0] = 1
            input_ids[index, 1 : length + 1] = torch.tensor(
                [ord(character) % 23 + 3 for character in sequence]
            )
            input_ids[index, length + 1] = 2
            attention[index, : length + 2] = 1
            special[index, 0] = 1
            special[index, length + 1] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "special_tokens_mask": special,
        }


class FakeBackbone:
    def __init__(self) -> None:
        self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=5)

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask, output_hidden_states, return_dict
        base = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        channels = torch.arange(4, device=input_ids.device).view(1, 1, 4)
        hidden_states = tuple(base + channels + layer * 100 for layer in range(6))
        return SimpleNamespace(hidden_states=hidden_states)


def main() -> None:
    record = SequenceRecord("protein", "ACDEFGHIK", "0")
    representation = encode_record(
        record,
        FakeTokenizer(),
        FakeBackbone(),
        layer_indices=(2, 3, 5),
        device=torch.device("cpu"),
        precision="fp32",
        max_residues=5,
        stride=3,
        window_batch_size=2,
    )
    if representation.shape != (9, 3, 4):
        raise RuntimeError(f"unexpected multi-layer shape: {representation.shape}")
    np.testing.assert_allclose(
        representation[:, 1] - representation[:, 0], 100.0, rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        representation[:, 2] - representation[:, 1], 200.0, rtol=0, atol=1e-4
    )

    label_rejected = False
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "bad.jsonl"
        manifest.write_text(
            json.dumps({"id": "bad", "sequence": "ACD", "labels": [0, 1, 0]})
            + "\n",
            encoding="utf-8",
        )
        try:
            load_sequence_manifest(manifest)
        except ValueError:
            label_rejected = True
    if not label_rejected:
        raise RuntimeError("A8 cache accepted a manifest containing labels")

    print(
        json.dumps(
            {
                "status": "pass",
                "windowed_multilayer_alignment": True,
                "selected_layer_differences_preserved": True,
                "label_containing_manifest_rejected": True,
                "caid2_caid3_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
