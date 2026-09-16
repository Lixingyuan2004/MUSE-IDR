"""Controlled head variants for post-lock MUSE-IDR mechanism validation."""

from __future__ import annotations

import torch
from torch import nn

from experiments.ablations.a2_multiscale_context.model import (
    MultiscaleContextIdrHead,
)


class ContextFusionAblationIdrHead(MultiscaleContextIdrHead):
    """A2-compatible head with a predeclared alternative context fusion rule.

    ``ungated_add`` removes the learned gate and adds context directly. The
    ``additive_projection`` control retains the exact A2 gate-layer parameter
    count but replaces sigmoid multiplicative gating with a learned additive
    projection. This separates the effect of conditional gating from the
    effect of simply adding another trainable projection.
    """

    ALLOWED_FUSION_MODES = frozenset({"ungated_add", "additive_projection"})

    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 256,
        kernels: tuple[int, ...] = (3, 7, 15, 31),
        dropout: float = 0.2,
        fusion_mode: str = "ungated_add",
    ) -> None:
        if fusion_mode not in self.ALLOWED_FUSION_MODES:
            raise ValueError(
                f"unsupported fusion mode {fusion_mode!r}; "
                f"expected one of {sorted(self.ALLOWED_FUSION_MODES)}"
            )
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            kernels=kernels,
            dropout=dropout,
        )
        self.fusion_mode = fusion_mode
        if fusion_mode == "ungated_add":
            self.gate = None

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

        if self.fusion_mode == "ungated_add":
            update = context
        else:
            if self.gate is None:
                raise RuntimeError("additive projection is missing")
            update = self.gate(torch.cat([base, context], dim=-1))
            update = update * mask
        fused = base + self.dropout(update)
        logits = self.classifier(self.output_norm(fused)).squeeze(-1)
        return logits.masked_fill(~residue_mask, 0.0)


class UniformLayerEsm2FusionIdrModel(nn.Module):
    """Fixed uniform mean of selected ESM2 layers followed by the exact A2 head."""

    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 256,
        layer_indices: tuple[int, ...] = (30, 31, 32, 33),
        kernels: tuple[int, ...] = (3, 7, 15, 31),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(layer_indices) < 2:
            raise ValueError("uniform layer fusion requires at least two layers")
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer indices must be unique")
        self.input_size = int(input_size)
        self.layer_indices = tuple(int(value) for value in layer_indices)
        self.register_buffer(
            "_layer_weights",
            torch.full(
                (len(self.layer_indices),),
                1.0 / len(self.layer_indices),
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.head = MultiscaleContextIdrHead(
            input_size=input_size,
            hidden_size=hidden_size,
            kernels=kernels,
            dropout=dropout,
        )

    @property
    def layer_weights(self) -> torch.Tensor:
        return self._layer_weights

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
        weights = self.layer_weights.to(device=features.device, dtype=features.dtype)
        fused = torch.einsum("blnh,n->blh", features, weights)
        fused = fused * residue_mask.unsqueeze(-1).to(dtype=fused.dtype)
        return self.head(fused, residue_mask)

    @property
    def output_heads(self) -> int:
        return self.head.output_heads

    @property
    def kernels(self) -> tuple[int, ...]:
        return self.head.kernels
