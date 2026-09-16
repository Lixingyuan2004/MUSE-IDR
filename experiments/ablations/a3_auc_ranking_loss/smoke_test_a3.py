"""Fast CPU checks for A3's sampled AUC ranking loss."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a3_auc_ranking_loss.loss import (  # noqa: E402
    combined_bce_auc_loss,
    sampled_pairwise_auc_loss,
)


def main() -> None:
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    good_logits = torch.tensor([2.0, 1.0, -1.0, -2.0], requires_grad=True)
    bad_logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    good_rank, good_pairs = sampled_pairwise_auc_loss(
        good_logits, labels, max_pairs=32
    )
    bad_rank, bad_pairs = sampled_pairwise_auc_loss(bad_logits, labels, max_pairs=32)
    if not good_rank < bad_rank or good_pairs != 4 or bad_pairs != 4:
        raise RuntimeError("pairwise AUC loss does not reward correct ordering")

    unknown_labels = torch.tensor([1.0, 0.0, -1.0, 1.0])
    known_mask = unknown_labels != -1
    logits_a = torch.tensor([1.0, -1.0, -100.0, 0.5], requires_grad=True)
    logits_b = torch.tensor([1.0, -1.0, 100.0, 0.5])
    positive_weight = torch.tensor(1.0)
    loss_a = combined_bce_auc_loss(
        logits_a,
        unknown_labels,
        known_mask,
        positive_weight,
        rank_weight=0.2,
        rank_margin=0.0,
        max_pairs=32,
    )
    loss_b = combined_bce_auc_loss(
        logits_b,
        unknown_labels,
        known_mask,
        positive_weight,
        rank_weight=0.2,
        rank_margin=0.0,
        max_pairs=32,
    )
    if not torch.allclose(loss_a.total, loss_b.total):
        raise RuntimeError("unknown label changed the combined loss")

    loss_a.total.backward()
    if logits_a.grad is None or not torch.all(torch.isfinite(logits_a.grad)):
        raise RuntimeError("combined A3 loss did not produce finite gradients")
    if float(logits_a.grad[2]) != 0.0:
        raise RuntimeError("unknown label received a gradient")

    torch.manual_seed(17)
    many_labels = torch.cat([torch.ones(100), torch.zeros(100)])
    many_logits = torch.linspace(-2.0, 2.0, steps=200)
    sampled_loss, sampled_pairs = sampled_pairwise_auc_loss(
        many_logits, many_labels, max_pairs=128
    )
    if sampled_pairs != 128 or not torch.isfinite(sampled_loss):
        raise RuntimeError("bounded pair sampling failed")

    print(
        json.dumps(
            {
                "status": "pass",
                "correct_ordering_has_lower_loss": True,
                "unknown_labels_excluded": True,
                "unknown_label_gradient_zero": True,
                "finite_gradients": True,
                "bounded_pair_sampling": True,
                "sampled_pairs": sampled_pairs,
                "caid2_caid3_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
