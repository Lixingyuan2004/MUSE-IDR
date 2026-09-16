from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import EsmConfig, EsmModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a4_esm2_lora_multiscale.model import (
    Esm2LoraMultiscaleIdrModel,
)
from experiments.ablations.a4_esm2_lora_multiscale.train_a4 import weighted_bce_loss


def tiny_backbone() -> EsmModel:
    config = EsmConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=0,
        mask_token_id=3,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    return EsmModel(config, add_pooling_layer=False)


def make_model(lora_enabled: bool) -> Esm2LoraMultiscaleIdrModel:
    return Esm2LoraMultiscaleIdrModel(
        tiny_backbone(),
        head_hidden_size=32,
        context_channels=8,
        context_kernels=[3, 5],
        context_dropout=0.0,
        lora_enabled=lora_enabled,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=["query", "value"],
        gradient_checkpointing=False,
        precision="fp32",
        device="cpu",
    )


def main() -> None:
    torch.manual_seed(7)
    model = make_model(True)
    input_ids = torch.tensor([[1, 4, 5, 6, 7, 2, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.long)
    residue_mask = torch.tensor([[0, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    labels = torch.tensor([[-1, 1, 0, -1, 1, -1, -1]], dtype=torch.long)
    loss_weight = residue_mask.float()
    logits = model(input_ids, attention_mask, residue_mask)
    logits.retain_grad()
    loss, known = weighted_bce_loss(
        logits,
        labels,
        residue_mask,
        loss_weight,
        torch.tensor(1.0),
    )
    loss.backward()

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected = [
        name
        for name in trainable_names
        if "lora_" not in name and not name.startswith("context_head.")
    ]
    unknown_gradient = float(logits.grad[0, 3].abs().item())
    special_gradient = float((logits.grad[0, [0, 5, 6]]).abs().sum().item())
    adapter_state = model.adapter_state_dict()
    compact_state = model.compact_state_dict()

    # A real run reloads the same immutable pretrained backbone. Resetting the
    # tiny random backbone reproduces that contract without a network download.
    torch.manual_seed(7)
    clone = make_model(True)
    clone.load_compact_state_dict(compact_state)
    clone.eval()
    model.eval()
    with torch.no_grad():
        clone_logits = clone(input_ids, attention_mask, residue_mask)
        reference_logits = model(input_ids, attention_mask, residue_mask)
    checkpoint_round_trip = bool(torch.allclose(clone_logits, reference_logits, atol=1e-6))

    frozen_control = make_model(False)
    frozen_report = frozen_control.parameter_report()
    result = {
        "status": "pass",
        "architecture": "esm2_lora_plus_gated_multiscale_context",
        "output_heads": model.output_heads,
        "logit_shape": list(logits.shape),
        "known_labels_used": known,
        "unknown_label_output_gradient_zero": unknown_gradient == 0.0,
        "special_and_padding_output_gradient_zero": special_gradient == 0.0,
        "finite_gradients": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ),
        "only_lora_and_head_trainable": not unexpected,
        "adapter_tensors": len(adapter_state),
        "compact_checkpoint_excludes_full_backbone": not any(
            "encoder.layer" in key for key in compact_state.keys()
        ),
        "compact_checkpoint_round_trip": checkpoint_round_trip,
        "frozen_raw_control_lora_parameters": frozen_report["lora_trainable_parameters"],
        "frozen_raw_control_backbone_frozen": not any(
            parameter.requires_grad for parameter in frozen_control.backbone.parameters()
        ),
        "caid2_caid3_labels_accessed": False,
    }
    checks = [
        result["output_heads"] == 1,
        result["logit_shape"] == [1, 7],
        result["known_labels_used"] == 3,
        result["unknown_label_output_gradient_zero"],
        result["special_and_padding_output_gradient_zero"],
        result["finite_gradients"],
        result["only_lora_and_head_trainable"],
        result["adapter_tensors"] == 8,
        result["compact_checkpoint_excludes_full_backbone"],
        result["compact_checkpoint_round_trip"],
        result["frozen_raw_control_lora_parameters"] == 0,
        result["frozen_raw_control_backbone_frozen"],
    ]
    if not all(checks):
        result["status"] = "fail"
        print(json.dumps(result, indent=2))
        raise AssertionError("A4 unit smoke test failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
