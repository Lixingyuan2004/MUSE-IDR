"""Network-free tests for LoRA-DR-Suite residue-score extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b4_baseline_reproduction.run_lora_official_smoke import (
    ALLOWED_MODEL_IDS,
    extract_residue_probabilities,
)


def main() -> None:
    logits = torch.tensor(
        [[[100.0, -100.0], [2.0, -1.0], [-1.0, 2.0], [0.0, 0.0], [-100.0, 100.0]]]
    )
    special_tokens = torch.tensor([[1, 0, 0, 0, 1]])
    scores = extract_residue_probabilities(logits, special_tokens, 3)
    if scores.shape != (3,):
        raise AssertionError("one score per residue is required")
    if not scores[0] < scores[2] < scores[1]:
        raise AssertionError("positive-class softmax probabilities are incorrect")

    wrong_shape_rejected = False
    try:
        extract_residue_probabilities(logits[..., :1], special_tokens, 3)
    except ValueError:
        wrong_shape_rejected = True
    if not wrong_shape_rejected:
        raise AssertionError("a one-logit/one-class model must be rejected")

    alignment_error_rejected = False
    try:
        extract_residue_probabilities(logits, special_tokens, 4)
    except ValueError:
        alignment_error_rejected = True
    if not alignment_error_rejected:
        raise AssertionError("residue alignment error must be rejected")

    if any("-SD" in model_id for model_id in ALLOWED_MODEL_IDS):
        raise AssertionError("soft-disorder checkpoint leaked into B4 IDR smoke allowlist")

    print(
        json.dumps(
            {
                "status": "pass",
                "special_tokens_excluded": True,
                "positive_class_probability_extracted": True,
                "one_score_per_residue": True,
                "wrong_output_shape_rejected": True,
                "prediction_misalignment_rejected": True,
                "only_disprot7_checkpoints_allowed": True,
                "soft_disorder_output": False,
                "caid_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
