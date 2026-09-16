"""Loss functions for directly improving residue-level ROC-AUC ranking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossComponents:
    total: torch.Tensor
    bce: torch.Tensor
    ranking: torch.Tensor
    known_residues: int
    ranking_pairs: int


def sampled_pairwise_auc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_pairs: int,
    margin: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Return a logistic AUC surrogate over positive-negative logit pairs.

    All pairs are used when the Cartesian product is small. Otherwise, positive
    and negative residues are sampled independently with replacement. Sampling
    uses PyTorch's seeded device RNG, so formal runs remain reproducible.
    """

    if logits.ndim != 1 or labels.ndim != 1 or logits.shape != labels.shape:
        raise ValueError("logits and labels must be matching one-dimensional tensors")
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    if torch.any((labels != 0) & (labels != 1)):
        raise ValueError("pairwise AUC labels must be binary")

    positive_logits = logits[labels == 1]
    negative_logits = logits[labels == 0]
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        return logits.sum() * 0.0, 0

    available_pairs = positive_logits.numel() * negative_logits.numel()
    if available_pairs <= max_pairs:
        differences = positive_logits[:, None] - negative_logits[None, :]
        pair_count = int(available_pairs)
    else:
        pair_count = int(max_pairs)
        positive_indices = torch.randint(
            positive_logits.numel(), (pair_count,), device=logits.device
        )
        negative_indices = torch.randint(
            negative_logits.numel(), (pair_count,), device=logits.device
        )
        differences = (
            positive_logits[positive_indices] - negative_logits[negative_indices]
        )

    ranking_loss = F.softplus(float(margin) - differences).mean()
    return ranking_loss, pair_count


def combined_bce_auc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    known_mask: torch.Tensor,
    positive_weight: torch.Tensor,
    rank_weight: float,
    rank_margin: float,
    max_pairs: int,
) -> LossComponents:
    """Combine class-balanced BCE with the sampled AUC ranking surrogate."""

    if logits.shape != labels.shape or logits.shape != known_mask.shape:
        raise ValueError("logits, labels, and known_mask must have matching shapes")
    if known_mask.dtype != torch.bool:
        raise ValueError("known_mask must be boolean")
    if rank_weight < 0:
        raise ValueError("rank_weight must be non-negative")
    if not torch.any(known_mask):
        raise ValueError("combined loss received no known labels")

    known_logits = logits[known_mask]
    known_labels = labels[known_mask]
    if torch.any((known_labels != 0) & (known_labels != 1)):
        raise ValueError("known labels must be binary")

    bce_loss = F.binary_cross_entropy_with_logits(
        known_logits, known_labels, pos_weight=positive_weight
    )
    ranking_loss, pair_count = sampled_pairwise_auc_loss(
        known_logits,
        known_labels,
        max_pairs=max_pairs,
        margin=rank_margin,
    )
    total_loss = bce_loss + float(rank_weight) * ranking_loss
    return LossComponents(
        total=total_loss,
        bce=bce_loss,
        ranking=ranking_loss,
        known_residues=int(known_logits.numel()),
        ranking_pairs=pair_count,
    )
