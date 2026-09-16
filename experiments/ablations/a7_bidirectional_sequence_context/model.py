from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from experiments.ablations.a2_multiscale_context.model import DepthwiseContextBranch


class BidirectionalSequenceContextIdrHead(nn.Module):
    """A2 local multiscale features plus gated padding-aware bidirectional GRU."""

    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 256,
        kernels: tuple[int, ...] = (3, 7, 15, 31),
        gru_hidden_size: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.2,
        sequence_dropout: float = 0.1,
        sequence_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("at least one local kernel is required")
        if hidden_size % len(kernels) != 0:
            raise ValueError("hidden_size must be divisible by the number of kernels")
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("local kernels must be positive odd integers")
        if gru_hidden_size < 1 or gru_layers < 1:
            raise ValueError("GRU hidden size and layer count must be positive")

        branch_size = hidden_size // len(kernels)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.kernels = tuple(int(kernel) for kernel in kernels)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_layers = int(gru_layers)

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

        self.bidirectional_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=sequence_dropout if gru_layers > 1 else 0.0,
        )
        sequence_size = gru_hidden_size * 2
        self.sequence_norm = nn.LayerNorm(sequence_size)
        self.sequence_projection = nn.Linear(sequence_size, hidden_size)
        self.sequence_activation = nn.GELU()
        self.sequence_gate = nn.Linear(hidden_size * 2, hidden_size)
        nn.init.zeros_(self.sequence_gate.weight)
        nn.init.constant_(self.sequence_gate.bias, float(sequence_gate_init))
        self.sequence_dropout = nn.Dropout(sequence_dropout)

        self.output_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def _local_features(
        self, features: torch.Tensor, residue_mask: torch.Tensor
    ) -> torch.Tensor:
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
        return (base + self.local_dropout(local_gate * local_context)) * mask

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
        if residue_mask.dtype != torch.bool:
            raise ValueError("residue_mask must be boolean")

        lengths = residue_mask.sum(dim=1, dtype=torch.long)
        if torch.any(lengths < 1):
            raise ValueError("every protein must contain at least one residue")
        positions = torch.arange(
            residue_mask.shape[1], device=residue_mask.device
        ).unsqueeze(0)
        expected_mask = positions < lengths.unsqueeze(1)
        if not torch.equal(expected_mask, residue_mask):
            raise ValueError("residue_mask must contain contiguous valid prefixes")

        mask = residue_mask.unsqueeze(-1).to(dtype=features.dtype)
        local_features = self._local_features(features, residue_mask)
        packed = pack_padded_sequence(
            local_features,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_sequence, _ = self.bidirectional_gru(packed)
        sequence_context, _ = pad_packed_sequence(
            packed_sequence,
            batch_first=True,
            total_length=features.shape[1],
        )
        sequence_context = self.sequence_activation(
            self.sequence_projection(self.sequence_norm(sequence_context))
        )
        sequence_context = sequence_context * mask
        sequence_gate = torch.sigmoid(
            self.sequence_gate(torch.cat([local_features, sequence_context], dim=-1))
        )
        fused = local_features + self.sequence_dropout(
            sequence_gate * sequence_context
        )
        fused = fused * mask
        logits = self.classifier(self.output_norm(fused)).squeeze(-1)
        return logits.masked_fill(~residue_mask, 0.0)

    @property
    def output_heads(self) -> int:
        return int(self.classifier.out_features)

    @property
    def sequence_context_is_bidirectional(self) -> bool:
        return bool(self.bidirectional_gru.bidirectional)
