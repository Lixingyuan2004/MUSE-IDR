from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from transformers import AutoModel


class FrozenEsmResidueClassifier(nn.Module):
    """Frozen ESM backbone and one trainable scalar logit per residue.

    The backbone is always kept in evaluation mode. Only ``dropout`` and
    ``classifier`` are trainable. The returned tensor still includes special
    token positions; the data collator supplies a mask that removes them.
    """

    def __init__(
        self,
        pretrained_model: str,
        dropout: float = 0.2,
        precision: str = "bf16",
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.precision = precision.lower()

        if self.device.type == "cuda" and self.precision == "bf16":
            backbone_dtype = torch.bfloat16
        elif self.device.type == "cuda" and self.precision == "fp16":
            backbone_dtype = torch.float16
        else:
            backbone_dtype = torch.float32

        self.backbone = AutoModel.from_pretrained(
            pretrained_model,
            torch_dtype=backbone_dtype,
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.backbone.to(self.device)

        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        self.classifier.to(self.device, dtype=torch.float32)

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen backbone must not introduce dropout noise during head
        # training. This also makes repeated inference deterministic.
        self.backbone.eval()
        return self

    def reset_head(self) -> None:
        self.classifier.reset_parameters()

    def trainable_parameters(self):
        return list(self.dropout.parameters()) + list(self.classifier.parameters())

    def _backbone_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.precision == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), self._backbone_context():
            hidden = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        # Keep the tiny trainable head in FP32 for stable optimization.
        hidden = hidden.float()
        return self.classifier(self.dropout(hidden)).squeeze(-1)
