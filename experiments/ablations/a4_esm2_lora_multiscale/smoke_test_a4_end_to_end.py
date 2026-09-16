from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import EsmConfig, EsmModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a4_esm2_lora_multiscale.data import ProteinRecord
from experiments.ablations.a4_esm2_lora_multiscale.model import (
    Esm2LoraMultiscaleIdrModel,
)
from experiments.ablations.a4_esm2_lora_multiscale.train_a4 import run_fold


class ToyEsmTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        alphabet = "ACDEFGHIKLMNPQRSTVWYXBZUO"
        self.ids = {amino_acid: index + 4 for index, amino_acid in enumerate(alphabet)}

    def __call__(self, sequences: list[str], **_: Any) -> dict[str, torch.Tensor]:
        rows = [
            [self.cls_token_id]
            + [self.ids.get(amino_acid, self.ids["X"]) for amino_acid in sequence]
            + [self.eos_token_id]
            for sequence in sequences
        ]
        width = max(map(len, rows))
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        special_tokens_mask = torch.ones((len(rows), width), dtype=torch.long)
        for row_index, row in enumerate(rows):
            input_ids[row_index, : len(row)] = torch.tensor(row)
            attention_mask[row_index, : len(row)] = 1
            if len(row) > 2:
                special_tokens_mask[row_index, 1 : len(row) - 1] = 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
        }


def tiny_model(config: dict[str, Any], device: torch.device) -> Esm2LoraMultiscaleIdrModel:
    esm_config = EsmConfig(
        vocab_size=32,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=64,
        pad_token_id=0,
        mask_token_id=3,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    backbone = EsmModel(esm_config, add_pooling_layer=False)
    return Esm2LoraMultiscaleIdrModel(
        backbone,
        head_hidden_size=24,
        context_channels=8,
        context_kernels=[3, 5],
        context_dropout=0.0,
        lora_enabled=bool(config["lora"]["enabled"]),
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules=["query", "value"],
        gradient_checkpointing=False,
        precision="fp32",
        device=device,
    )


def records() -> list[ProteinRecord]:
    return [
        ProteinRecord("train_a", "ACDEFGHIKL", np.array([0, 0, 0, 1, 1, 1, -1, 0, 1, 1]), "0"),
        ProteinRecord("train_b", "LMNPQRSTVWY", np.array([1, 1, 0, 0, 0, 1, 1, -1, 0, 1, 0]), "0"),
        ProteinRecord("train_c", "GGGGPPPPAA", np.array([0, 0, 1, 1, 1, 1, 0, 0, -1, 1]), "2"),
        ProteinRecord("train_d", "KKKKEEEEQQ", np.array([1, 1, 1, 0, 0, 0, 1, -1, 0, 1]), "2"),
        ProteinRecord("valid_a", "AAAAGGGGPP", np.array([0, 0, 0, 1, 1, 1, -1, 1, 0, 1]), "1"),
        ProteinRecord("valid_b", "VVVVSSSSQQQ", np.array([0, 0, 1, 1, 1, 0, 0, -1, 1, 1, 0]), "1"),
    ]


def main() -> None:
    config: dict[str, Any] = {
        "model": {"name": "offline-tiny-esm", "revision": "random-test", "hidden_size": 24},
        "data": {"max_residues": 8, "stride": 4},
        "context": {"channels": 8, "kernels": [3, 5], "dropout": 0.0},
        "lora": {
            "enabled": True,
            "rank": 2,
            "alpha": 4,
            "dropout": 0.0,
            "target_modules": ["query", "value"],
        },
        "training": {
            "precision": "fp32",
            "gradient_checkpointing": False,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "epochs": 2,
            "patience": 2,
            "lora_learning_rate": 0.001,
            "head_learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
        },
    }
    with tempfile.TemporaryDirectory(prefix="a4_smoke_") as temporary:
        output = Path(temporary)
        result = run_fold(
            records=records(),
            tokenizer=ToyEsmTokenizer(),
            config=config,
            fold="1",
            seed=17,
            output_dir=output,
            device=torch.device("cpu"),
            model_builder=tiny_model,
        )
        predictions = result["predictions"]
        report = result["report"]
        checkpoint_path = output / "fold_1" / "best_compact_checkpoint.pt"
        prediction_path = output / "fold_1" / "predictions.jsonl.gz"
        alignment = all(
            len(prediction["scores"]) == len(record.sequence)
            for prediction, record in zip(predictions, [r for r in records() if r.fold == "1"])
        )
        summary = {
            "status": "pass",
            "fold_training": True,
            "online_backbone_forward_passes_used": report["training_backbone_forward_passes"] > 0,
            "lora_parameters_trained": report["lora_trainable_parameters"] > 0,
            "checkpoint_written": checkpoint_path.is_file(),
            "prediction_file_written": prediction_path.is_file(),
            "prediction_alignment": alignment,
            "micro_roc_auc_defined": bool(np.isfinite(result["metrics"].micro_roc_auc)),
            "unknown_labels_excluded": result["metrics"].known_residues == 19,
            "output_heads": 1,
            "caid2_caid3_labels_accessed": False,
        }
        if not all(
            value
            for key, value in summary.items()
            if key not in {"status", "output_heads", "caid2_caid3_labels_accessed"}
        ) or summary["output_heads"] != 1:
            summary["status"] = "fail"
            print(json.dumps(summary, indent=2))
            raise AssertionError("A4 end-to-end smoke test failed")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
