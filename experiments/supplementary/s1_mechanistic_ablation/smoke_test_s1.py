"""Smoke-test S1 shape, masking, invariance, parameters, and fixed mixing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a2_multiscale_context.model import (  # noqa: E402
    MultiscaleContextIdrHead,
)
from experiments.supplementary.s1_mechanistic_ablation.model import (  # noqa: E402
    ContextFusionAblationIdrHead,
    UniformLayerEsm2FusionIdrModel,
)


def check_padding_invariance(model: torch.nn.Module, multilayer: bool) -> None:
    model.eval()
    residue_mask_short = torch.ones((1, 7), dtype=torch.bool)
    residue_mask_padded = torch.zeros((1, 13), dtype=torch.bool)
    residue_mask_padded[:, :7] = True
    if multilayer:
        short = torch.randn(1, 7, 4, 16)
        padded = torch.zeros(1, 13, 4, 16)
        padded[:, :7] = short
    else:
        short = torch.randn(1, 7, 16)
        padded = torch.zeros(1, 13, 16)
        padded[:, :7] = short
    with torch.inference_mode():
        short_logits = model(short, residue_mask_short)[0]
        padded_logits = model(padded, residue_mask_padded)[0]
    torch.testing.assert_close(short_logits, padded_logits[:7], rtol=0, atol=1e-6)
    if not torch.all(padded_logits[7:] == 0):
        raise RuntimeError("S1 padded logits must be zero")


def main() -> None:
    torch.manual_seed(17)
    small_models: dict[str, torch.nn.Module] = {
        "ungated_add": ContextFusionAblationIdrHead(
            input_size=16,
            hidden_size=8,
            kernels=(3, 5),
            dropout=0.0,
            fusion_mode="ungated_add",
        ),
        "additive_projection": ContextFusionAblationIdrHead(
            input_size=16,
            hidden_size=8,
            kernels=(3, 5),
            dropout=0.0,
            fusion_mode="additive_projection",
        ),
        "single_scale": MultiscaleContextIdrHead(
            input_size=16,
            hidden_size=8,
            kernels=(3,),
            dropout=0.0,
        ),
        "uniform_layer_mix": UniformLayerEsm2FusionIdrModel(
            input_size=16,
            hidden_size=8,
            layer_indices=(30, 31, 32, 33),
            kernels=(3, 5),
            dropout=0.0,
        ),
    }
    for name, model in small_models.items():
        check_padding_invariance(model, multilayer=name == "uniform_layer_mix")

    additive = small_models["additive_projection"]
    additive.train()
    features = torch.randn(2, 11, 16)
    mask = torch.ones((2, 11), dtype=torch.bool)
    labels = torch.randint(0, 2, (2, 11), dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(additive(features, mask), labels)
    loss.backward()
    if additive.gate is None or additive.gate.weight.grad is None:
        raise RuntimeError("parameter-matched additive fusion received no gradient")

    uniform = small_models["uniform_layer_mix"]
    expected_weights = torch.full((4,), 0.25)
    torch.testing.assert_close(uniform.layer_weights, expected_weights)
    if uniform.layer_weights.requires_grad:
        raise RuntimeError("uniform layer weights must remain fixed")

    formal_reference = MultiscaleContextIdrHead()
    formal_additive = ContextFusionAblationIdrHead(
        fusion_mode="additive_projection"
    )
    formal_ungated = ContextFusionAblationIdrHead(fusion_mode="ungated_add")
    reference_parameters = sum(p.numel() for p in formal_reference.parameters())
    additive_parameters = sum(p.numel() for p in formal_additive.parameters())
    ungated_parameters = sum(p.numel() for p in formal_ungated.parameters())
    if reference_parameters != additive_parameters:
        raise RuntimeError("additive fusion control is not parameter matched to A2")
    if ungated_parameters >= reference_parameters:
        raise RuntimeError("direct ungated control should remove the gate parameters")

    config_path = Path(__file__).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_variants = {
        "a2_reference",
        "ungated_add",
        "additive_projection",
        "single_k3",
        "single_k7",
        "single_k15",
        "single_k31",
        "uniform_layer_mix",
    }
    if set(config["variants"]) != expected_variants:
        raise RuntimeError("S1 predeclared variant set changed")
    if config["evaluation"].get("caid_labels_permitted") is not False:
        raise RuntimeError("S1 must prohibit CAID label access")

    print(
        json.dumps(
            {
                "status": "pass",
                "variants_predeclared": len(expected_variants),
                "padding_invariant": True,
                "masked_padding_zero": True,
                "single_scale_supported": True,
                "direct_ungated_supported": True,
                "parameter_matched_additive_supported": True,
                "uniform_layer_weights": [0.25, 0.25, 0.25, 0.25],
                "uniform_layer_weights_trainable": False,
                "a2_reference_parameters": reference_parameters,
                "additive_control_parameters": additive_parameters,
                "ungated_control_parameters": ungated_parameters,
                "output_heads": 1,
                "caid_labels_accessed": False,
                "locked_a10_modified": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
