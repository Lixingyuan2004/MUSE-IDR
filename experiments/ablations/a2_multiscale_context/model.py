"""Single-output multiscale context head for cached residue representations."""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseContextBranch(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("context kernels must be positive odd integers")
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,
            bias=False,
        )
        self.pointwise = nn.Conv1d(hidden_size, output_size, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.activation(self.depthwise(features)))


class MultiscaleContextIdrHead(nn.Module):
    """Predict one classic-IDR logit per residue using parallel local contexts."""

    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 256,
        kernels: tuple[int, ...] = (3, 7, 15, 31),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("at least one context kernel is required")
        if hidden_size % len(kernels) != 0:
            raise ValueError("hidden_size must be divisible by the number of kernels")

        branch_size = hidden_size // len(kernels)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.kernels = tuple(int(kernel) for kernel in kernels)
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_activation = nn.GELU()
        self.branches = nn.ModuleList(
            DepthwiseContextBranch(hidden_size, branch_size, kernel)
            for kernel in self.kernels
        )
        self.context_norm = nn.LayerNorm(hidden_size)
        self.context_projection = nn.Linear(hidden_size, hidden_size)
        self.context_activation = nn.GELU()
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self, features: torch.Tensor, residue_mask: torch.Tensor
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, residues, channels]")
        if residue_mask.shape != features.shape[:2]:
            raise ValueError("residue_mask must match the first two feature dimensions")
        if features.shape[-1] != self.input_size:
            raise ValueError(
                f"expected input size {self.input_size}, got {features.shape[-1]}"
            )

        mask = residue_mask.unsqueeze(-1).to(dtype=features.dtype)
        base = self.input_activation(self.input_projection(self.input_norm(features)))
        base = base * mask

        channel_first = base.transpose(1, 2)
        channel_mask = residue_mask.unsqueeze(1).to(dtype=features.dtype)
        context = torch.cat(
            [branch(channel_first) * channel_mask for branch in self.branches], dim=1
        ).transpose(1, 2)
        context = self.context_activation(
            self.context_projection(self.context_norm(context))
        )
        context = context * mask

        gate = torch.sigmoid(self.gate(torch.cat([base, context], dim=-1)))
        fused = base + self.dropout(gate * context)
        logits = self.classifier(self.output_norm(fused)).squeeze(-1)
        return logits.masked_fill(~residue_mask, 0.0)

    @property
    def output_heads(self) -> int:
        return int(self.classifier.out_features)
