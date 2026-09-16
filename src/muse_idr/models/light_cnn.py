"""A small full-length convolutional baseline with one residue output."""

from __future__ import annotations

import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(
        self, hidden_size: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and at least 3")
        self.norm = nn.LayerNorm(hidden_size)
        self.context = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size // 2),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.norm(inputs)
        hidden = self.context(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.projection(hidden)
        return residual + self.dropout(hidden)


class LightIDRCNN(nn.Module):
    """Map BxL residue tokens to exactly one BxL classic-IDR logit tensor."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int = 64,
        kernel_size: int = 9,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or hidden_size < 1 or not dilations:
            raise ValueError("Invalid model dimensions")
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.blocks = nn.ModuleList(
            ResidualConvBlock(hidden_size, kernel_size, dilation, dropout)
            for dilation in dilations
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.idr_head = nn.Linear(hidden_size, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, length]")
        hidden = self.embedding(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return self.idr_head(self.output_norm(hidden)).squeeze(-1)
