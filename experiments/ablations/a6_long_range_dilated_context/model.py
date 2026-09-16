from __future__ import annotations

import torch
from torch import nn

from experiments.ablations.a2_multiscale_context.model import DepthwiseContextBranch


class DilatedResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dilation: int,
        dropout: float,
        residual_gate_init: float,
    ) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError("dilation must be positive")
        self.dilation = int(dilation)
        self.norm = nn.LayerNorm(hidden_size)
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            dilation=self.dilation,
            padding=self.dilation,
            groups=hidden_size,
            bias=False,
        )
        self.pointwise = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_gate_logit = nn.Parameter(
            torch.full((hidden_size,), float(residual_gate_init))
        )

    def forward(
        self, features: torch.Tensor, residue_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = residue_mask.unsqueeze(-1).to(dtype=features.dtype)
        normalized = self.norm(features) * mask
        update = self.depthwise(normalized.transpose(1, 2))
        update = self.pointwise(self.activation(update)).transpose(1, 2)
        update = update * mask
        gate = torch.sigmoid(self.residual_gate_logit).view(1, 1, -1)
        return (features + self.dropout(gate * update)) * mask


class LongRangeDilatedContextIdrHead(nn.Module):
    """A2 local context followed by a gated dilated residual stack."""

    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 256,
        kernels: tuple[int, ...] = (3, 7, 15, 31),
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.2,
        dilated_dropout: float = 0.1,
        residual_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("at least one local kernel is required")
        if not dilations:
            raise ValueError("at least one dilation is required")
        if hidden_size % len(kernels) != 0:
            raise ValueError("hidden_size must be divisible by the number of kernels")
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("local kernels must be positive odd integers")
        if any(dilation < 1 for dilation in dilations):
            raise ValueError("dilations must be positive integers")

        branch_size = hidden_size // len(kernels)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.kernels = tuple(int(kernel) for kernel in kernels)
        self.dilations = tuple(int(dilation) for dilation in dilations)

        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_activation = nn.GELU()
        self.local_branches = nn.ModuleList(
            DepthwiseContextBranch(hidden_size, branch_size, kernel)
            for kernel in self.kernels
        )
        self.local_context_norm = nn.LayerNorm(hidden_size)
        self.local_context_projection = nn.Linear(hidden_size, hidden_size)
        self.local_context_activation = nn.GELU()
        self.local_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.local_dropout = nn.Dropout(dropout)

        self.dilated_blocks = nn.ModuleList(
            DilatedResidualBlock(
                hidden_size,
                dilation,
                dilated_dropout,
                residual_gate_init,
            )
            for dilation in self.dilations
        )
        self.long_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.long_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    @property
    def dilated_stack_receptive_field(self) -> int:
        return 1 + 2 * sum(self.dilations)

    @property
    def maximum_input_receptive_field(self) -> int:
        return max(self.kernels) + 2 * sum(self.dilations)

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
        local_context = torch.cat(
            [branch(channel_first) * channel_mask for branch in self.local_branches],
            dim=1,
        ).transpose(1, 2)
        local_context = self.local_context_activation(
            self.local_context_projection(self.local_context_norm(local_context))
        )
        local_context = local_context * mask
        local_gate = torch.sigmoid(
            self.local_gate(torch.cat([base, local_context], dim=-1))
        )
        local_fused = base + self.local_dropout(local_gate * local_context)
        local_fused = local_fused * mask

        long_context = local_fused
        for block in self.dilated_blocks:
            long_context = block(long_context, residue_mask)
        long_gate = torch.sigmoid(
            self.long_gate(torch.cat([local_fused, long_context], dim=-1))
        )
        fused = local_fused + self.long_dropout(
            long_gate * (long_context - local_fused)
        )
        fused = fused * mask
        logits = self.classifier(self.output_norm(fused)).squeeze(-1)
        return logits.masked_fill(~residue_mask, 0.0)

    @property
    def output_heads(self) -> int:
        return int(self.classifier.out_features)
