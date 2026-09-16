"""Minimal, explicit training primitives for single-output residue models."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from muse_idr.evaluation import ResiduePrediction


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cosine_with_warmup_factor(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_factor: float = 0.0,
) -> float:
    """Return a deterministic linear-warmup then cosine-decay LR multiplier."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup_steps must be between zero and total_steps")
    if not 0.0 <= min_factor <= 1.0:
        raise ValueError("min_factor must be between zero and one")
    bounded_step = min(max(step, 0), total_steps)
    if warmup_steps > 0 and bounded_step < warmup_steps:
        return (bounded_step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps == 0:
        return 1.0
    progress = (bounded_step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def make_cosine_with_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_factor: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a batch-level scheduler whose entire trajectory is recorded by config."""

    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between zero and one")
    warmup_steps = int(round(total_steps * warmup_ratio))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup_factor(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_factor=min_factor,
        ),
    )


def masked_bce_with_logits(
    logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Exclude unknown and padded positions before computing weighted BCE."""

    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have identical shapes")
    known = labels != -1
    known_count = int(known.sum().item())
    if known_count == 0:
        raise ValueError("Batch contains no labeled residues")
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits[known], labels[known].to(dtype=logits.dtype), pos_weight=pos_weight
    )
    return loss, known_count


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    pos_weight: torch.Tensor,
    use_amp: bool,
    max_batches: int | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float | int]:
    model.train()
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_weighted_loss = 0.0
    total_known = 0
    processed_batches = 0
    for batch_index, batch in enumerate(batches):
        if max_batches is not None and batch_index >= max_batches:
            break
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(tokens)
            loss, known_count = masked_bce_with_logits(logits, labels, pos_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_succeeded = scaler.get_scale() >= scale_before_step
        if scheduler is not None and optimizer_step_succeeded:
            scheduler.step()
        total_weighted_loss += float(loss.detach().cpu()) * known_count
        total_known += known_count
        processed_batches += 1
    if total_known == 0:
        raise ValueError("No labeled residues were used for training")
    return {
        "loss": total_weighted_loss / total_known,
        "labeled_residues": total_known,
        "batches": processed_batches,
        "learning_rate_end": float(optimizer.param_groups[0]["lr"]),
    }


@torch.inference_mode()
def predict_residues(
    model: nn.Module,
    batches: Iterable[dict[str, object]],
    *,
    device: torch.device,
    use_amp: bool,
) -> list[ResiduePrediction]:
    model.eval()
    predictions: list[ResiduePrediction] = []
    for batch in batches:
        tokens = batch["tokens"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            probabilities = torch.sigmoid(model(tokens)).float().cpu()
        labels = batch["labels"]
        for index, (identifier, sequence, length) in enumerate(
            zip(batch["protein_ids"], batch["sequences"], batch["lengths"], strict=True)
        ):
            size = int(length)
            predictions.append(
                ResiduePrediction(
                    protein_id=identifier,
                    sequence=sequence,
                    labels=tuple(int(value) for value in labels[index, :size].tolist()),
                    scores=tuple(float(value) for value in probabilities[index, :size].tolist()),
                )
            )
    return predictions
