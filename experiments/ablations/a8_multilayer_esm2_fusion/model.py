"""Learned scalar fusion of frozen ESM2 layers followed by the exact A2 head."""

from __future__ import annotations

import math

import torch
from torch import nn

from experiments.ablations.a2_multiscale_context.model import MultiscaleContextIdrHead


class MultilayerEsm2FusionIdrModel(nn.Module):
    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 256,
        layer_indices: tuple[int, ...] = (30, 31, 32, 33),
        kernels: tuple[int, ...] = (3, 7, 15, 31),
        dropout: float = 0.2,
        initial_last_layer_weight: float = 0.7,
    ) -> None:
        super().__init__()
        if len(layer_indices) < 2:
            raise ValueError("A8 requires at least two selected ESM2 layers")
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer indices must be unique")
        if not 0.0 < initial_last_layer_weight < 1.0:
            raise ValueError("initial_last_layer_weight must be between zero and one")
        self.input_size = int(input_size)
        self.layer_indices = tuple(int(value) for value in layer_indices)
        self.layer_logits = nn.Parameter(torch.zeros(len(self.layer_indices)))
        last_logit = math.log(
            (len(self.layer_indices) - 1)
            * initial_last_layer_weight
            / (1.0 - initial_last_layer_weight)
        )
        with torch.no_grad():
            self.layer_logits[-1] = last_logit
        self.head = MultiscaleContextIdrHead(
            input_size=input_size,
            hidden_size=hidden_size,
            kernels=kernels,
            dropout=dropout,
        )

    @property
    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(
        self, features: torch.Tensor, residue_mask: torch.Tensor
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "features must have shape [batch, residues, selected_layers, channels]"
            )
        if residue_mask.shape != features.shape[:2]:
            raise ValueError("residue_mask must match the first two feature dimensions")
        if features.shape[2] != len(self.layer_indices):
            raise ValueError(
                f"expected {len(self.layer_indices)} layers, got {features.shape[2]}"
            )
        if features.shape[3] != self.input_size:
            raise ValueError(
                f"expected input size {self.input_size}, got {features.shape[3]}"
            )
        weights = self.layer_weights.to(dtype=features.dtype)
        fused = torch.einsum("blnh,n->blh", features, weights)
        fused = fused * residue_mask.unsqueeze(-1).to(dtype=fused.dtype)
        return self.head(fused, residue_mask)

    @property
    def output_heads(self) -> int:
        return self.head.output_heads

    @property
    def kernels(self) -> tuple[int, ...]:
        return self.head.kernels
