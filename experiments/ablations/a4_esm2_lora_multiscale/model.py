from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch import nn
from transformers import AutoModel

from experiments.ablations.a2_multiscale_context.model import MultiscaleContextIdrHead


def _torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported precision: {name}")


def _cpu_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


class Esm2LoraMultiscaleIdrModel(nn.Module):
    """One-head, per-residue classic-IDR predictor.

    The ESM2 base weights always remain frozen.  With LoRA enabled, only the
    injected adapter matrices and the A2 context head are trainable.
    """

    output_heads = 1
    output_definition = "classic_idr_probability_per_residue"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        head_hidden_size: int,
        context_channels: int,
        context_kernels: list[int] | tuple[int, ...],
        context_dropout: float,
        lora_enabled: bool,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: list[str] | tuple[str, ...],
        gradient_checkpointing: bool,
        precision: str,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.device_ref = torch.device(device)
        self.precision_name = precision.lower()
        self.backbone_dtype = _torch_dtype(precision)
        if self.device_ref.type == "cpu" and self.backbone_dtype != torch.float32:
            self.backbone_dtype = torch.float32
        self.lora_enabled = bool(lora_enabled)
        self.gradient_checkpointing = bool(gradient_checkpointing and self.lora_enabled)

        hidden_size = int(getattr(backbone.config, "hidden_size"))
        if hidden_size != int(head_hidden_size):
            raise ValueError(
                f"Backbone hidden size {hidden_size} does not match head input "
                f"size {head_hidden_size}"
            )

        backbone.requires_grad_(False)
        backbone.to(device=self.device_ref, dtype=self.backbone_dtype)
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False

        if self.lora_enabled:
            if self.gradient_checkpointing:
                if hasattr(backbone, "gradient_checkpointing_enable"):
                    backbone.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                if hasattr(backbone, "enable_input_require_grads"):
                    backbone.enable_input_require_grads()
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=list(lora_target_modules),
                bias="none",
            )
            backbone = get_peft_model(backbone, peft_config)
            backbone.to(self.device_ref)

        self.backbone = backbone
        self.context_head = MultiscaleContextIdrHead(
            input_size=hidden_size,
            hidden_size=int(context_channels),
            kernels=tuple(int(k) for k in context_kernels),
            dropout=float(context_dropout),
        ).to(device=self.device_ref, dtype=torch.float32)
        self.assert_trainable_contract()

    @classmethod
    def from_pretrained_config(
        cls,
        config: Mapping[str, Any],
        device: torch.device | str,
    ) -> "Esm2LoraMultiscaleIdrModel":
        model_cfg = config["model"]
        revision = str(model_cfg["revision"])
        precision = str(config["training"].get("precision", "bf16"))
        dtype = _torch_dtype(precision)
        if torch.device(device).type == "cpu":
            dtype = torch.float32
        backbone = AutoModel.from_pretrained(
            str(model_cfg["name"]),
            revision=revision,
            torch_dtype=dtype,
        )
        lora_cfg = config["lora"]
        context_cfg = config["context"]
        return cls(
            backbone,
            head_hidden_size=int(model_cfg["hidden_size"]),
            context_channels=int(context_cfg["channels"]),
            context_kernels=context_cfg["kernels"],
            context_dropout=float(context_cfg["dropout"]),
            lora_enabled=bool(lora_cfg["enabled"]),
            lora_rank=int(lora_cfg["rank"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            lora_target_modules=lora_cfg["target_modules"],
            gradient_checkpointing=bool(config["training"].get("gradient_checkpointing", True)),
            precision=precision,
            device=device,
        )

    def _autocast_context(self):
        if self.device_ref.type != "cuda" or self.backbone_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.backbone_dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        with self._autocast_context():
            hidden = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        logits = self.context_head(hidden.float(), residue_mask.bool())
        return logits.masked_fill(~residue_mask.bool(), 0.0)

    def lora_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        if not self.lora_enabled:
            return []
        return [
            (name, parameter)
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]

    def head_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (f"context_head.{name}", parameter)
            for name, parameter in self.context_head.named_parameters()
            if parameter.requires_grad
        ]

    def assert_trainable_contract(self) -> None:
        unexpected = [
            name
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected trainable ESM2 base parameters: {unexpected[:8]}")
        if self.lora_enabled and not self.lora_named_parameters():
            raise RuntimeError("LoRA is enabled but no adapter parameter is trainable")
        if not self.lora_enabled and any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        ):
            raise RuntimeError("Frozen raw control has trainable backbone parameters")
        if not self.head_named_parameters():
            raise RuntimeError("The multiscale IDR head has no trainable parameters")

    def parameter_report(self) -> dict[str, int]:
        lora = sum(p.numel() for _, p in self.lora_named_parameters())
        head = sum(p.numel() for _, p in self.head_named_parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "total_parameters": int(total),
            "lora_trainable_parameters": int(lora),
            "head_trainable_parameters": int(head),
            "trainable_parameters": int(lora + head),
        }

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        if not self.lora_enabled:
            return {}
        return _cpu_state_dict(get_peft_model_state_dict(self.backbone))

    def load_adapter_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        if self.lora_enabled:
            set_peft_model_state_dict(self.backbone, dict(state))
        elif state:
            raise ValueError("Adapter state was supplied to a frozen-control model")

    def compact_state_dict(self) -> dict[str, Any]:
        return {
            "lora_enabled": self.lora_enabled,
            "adapter": self.adapter_state_dict(),
            "context_head": _cpu_state_dict(self.context_head.state_dict()),
            "output_heads": self.output_heads,
            "output_definition": self.output_definition,
        }

    def load_compact_state_dict(self, state: Mapping[str, Any]) -> None:
        if bool(state["lora_enabled"]) != self.lora_enabled:
            raise ValueError("Checkpoint LoRA mode does not match the model configuration")
        self.load_adapter_state_dict(state.get("adapter", {}))
        self.context_head.load_state_dict(state["context_head"], strict=True)
