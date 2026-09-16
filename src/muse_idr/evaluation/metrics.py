"""Full-precision residue metrics independent of model implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .records import ResiduePrediction


def _validated_arrays(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("labels and scores must be one-dimensional with equal length")
    if len(y_true) == 0:
        raise ValueError("At least one labeled residue is required")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("Metric inputs may contain only binary labels 0/1")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("Metric scores must be finite")
    return y_true, y_score


def roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-correct ROC-AUC: P(score_pos > score_neg) + 0.5 * P(tie)."""

    y_true, y_score = _validated_arrays(labels, scores)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires at least one positive and one negative residue")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    group_positives = np.add.reduceat(sorted_labels, starts).astype(np.float64)
    group_sizes = np.diff(np.r_[starts, len(sorted_labels)]).astype(np.float64)
    group_negatives = group_sizes - group_positives
    negatives_below = np.cumsum(group_negatives) - group_negatives
    favorable_pairs = np.sum(group_positives * (negatives_below + 0.5 * group_negatives))
    return float(favorable_pairs / (positives * negatives))


def precision_recall_summary(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[float, float]:
    """Return trapezoidal PR-AUC and stepwise average precision."""

    y_true, y_score = _validated_arrays(labels, scores)
    positives = int(y_true.sum())
    if positives == 0:
        raise ValueError("Precision-recall metrics require at least one positive residue")

    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    group_positives = np.add.reduceat(sorted_labels, starts).astype(np.float64)
    group_sizes = np.diff(np.r_[starts, len(sorted_labels)]).astype(np.float64)
    true_positives = np.cumsum(group_positives)
    predicted_positives = np.cumsum(group_sizes)
    recall = np.r_[0.0, true_positives / positives]
    precision = np.r_[1.0, true_positives / predicted_positives]
    delta_recall = np.diff(recall)
    trapezoid_auc = np.sum(delta_recall * (precision[:-1] + precision[1:]) * 0.5)
    average_precision = np.sum(delta_recall * precision[1:])
    return float(trapezoid_auc), float(average_precision)


def threshold_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, object]:
    """Compute interpretable secondary metrics at a predeclared threshold."""

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    y_true, y_score = _validated_arrays(labels, scores)
    predicted = y_score >= threshold
    positive = y_true == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))

    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if recall is not None and precision + recall > 0.0
        else 0.0
    )
    specificity_denominator = tn + fp
    specificity = tn / specificity_denominator if specificity_denominator else None
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else None
    return {
        "threshold": float(threshold),
        "prediction_rule": "score_greater_than_or_equal_to_threshold",
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": balanced_accuracy,
    }


def _pool_known(
    records: Sequence[ResiduePrediction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    labels: list[int] = []
    scores: list[float] = []
    protein_indices: list[int] = []
    unknown = 0
    for protein_index, record in enumerate(records):
        for label, score in zip(record.labels, record.scores, strict=True):
            if label == -1:
                unknown += 1
                continue
            labels.append(label)
            scores.append(score)
            protein_indices.append(protein_index)
    if not labels:
        raise ValueError("No known 0/1 labels remain after masking unknown residues")
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    counts = {
        "proteins": len(records),
        "residues_total": sum(len(record.sequence) for record in records),
        "residues_labeled": len(labels),
        "residues_unknown_masked": unknown,
        "residues_positive": int(y_true.sum()),
        "residues_negative": int(len(y_true) - y_true.sum()),
    }
    return y_true, y_score, np.asarray(protein_indices, dtype=np.int64), counts


def _protein_macro_auc(records: Sequence[ResiduePrediction]) -> dict[str, object]:
    values: list[float] = []
    excluded_single_class = 0
    excluded_no_labels = 0
    for record in records:
        known = [index for index, label in enumerate(record.labels) if label != -1]
        if not known:
            excluded_no_labels += 1
            continue
        labels = [record.labels[index] for index in known]
        if len(set(labels)) < 2:
            excluded_single_class += 1
            continue
        scores = [record.scores[index] for index in known]
        values.append(roc_auc_score(labels, scores))
    return {
        "value": float(np.mean(values)) if values else None,
        "proteins_included": len(values),
        "proteins_excluded_single_class": excluded_single_class,
        "proteins_excluded_no_known_labels": excluded_no_labels,
    }


def _protein_bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    protein_indices: np.ndarray,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    _, evaluable_proteins = np.unique(protein_indices, return_inverse=True)
    protein_count = int(evaluable_proteins.max()) + 1
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_proteins = evaluable_proteins[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    group_index = np.cumsum(np.r_[0, np.diff(sorted_scores) != 0]).astype(np.int64)
    group_count = len(starts)
    rng = np.random.default_rng(seed)
    probabilities = np.full(protein_count, 1.0 / protein_count)
    values: list[float] = []

    for _ in range(replicates):
        protein_weights = rng.multinomial(protein_count, probabilities)
        residue_weights = protein_weights[sorted_proteins]
        positive_by_group = np.bincount(
            group_index,
            weights=residue_weights * (sorted_labels == 1),
            minlength=group_count,
        )
        negative_by_group = np.bincount(
            group_index,
            weights=residue_weights * (sorted_labels == 0),
            minlength=group_count,
        )
        positives = float(positive_by_group.sum())
        negatives = float(negative_by_group.sum())
        if positives == 0.0 or negatives == 0.0:
            continue
        negatives_below = np.cumsum(negative_by_group) - negative_by_group
        favorable = np.sum(positive_by_group * (negatives_below + 0.5 * negative_by_group))
        values.append(float(favorable / (positives * negatives)))

    if not values:
        raise ValueError("All protein bootstrap replicates contained only one class")
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(np.asarray(values), [alpha, 1.0 - alpha])
    return {
        "method": "protein_cluster_percentile_bootstrap",
        "confidence_level": confidence_level,
        "replicates_requested": replicates,
        "replicates_valid": len(values),
        "seed": seed,
        "lower": float(lower),
        "upper": float(upper),
    }


def evaluate_predictions(
    records: Sequence[ResiduePrediction],
    *,
    threshold: float = 0.5,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 1729,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Evaluate one universal IDR score using a frozen, model-agnostic protocol."""

    if not records:
        raise ValueError("At least one protein record is required")
    identifiers = [record.protein_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate protein_id values are not allowed")

    labels, scores, protein_indices, counts = _pool_known(records)
    micro_auc = roc_auc_score(labels, scores)
    pr_auc, average_precision = precision_recall_summary(labels, scores)
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "single_output_residue_classic_idr",
        "primary_metric": "residue_micro_roc_auc",
        "counts": counts,
        "metrics": {
            "residue_micro_roc_auc": micro_auc,
            "residue_pr_auc_trapezoid": pr_auc,
            "residue_average_precision": average_precision,
            "protein_macro_roc_auc": _protein_macro_auc(records),
            "threshold_metrics": threshold_metrics(labels, scores, threshold),
        },
    }
    if bootstrap_replicates:
        result["uncertainty"] = {
            "residue_micro_roc_auc": _protein_bootstrap_auc(
                labels,
                scores,
                protein_indices,
                bootstrap_replicates,
                bootstrap_seed,
                confidence_level,
            )
        }
    return result
