"""Strict, shared evaluation for residue-level IDR predictions."""

from .io import load_aligned_jsonl
from .metrics import evaluate_predictions, roc_auc_score
from .oof import load_fold_assignments, validate_and_order_oof_records
from .records import ResiduePrediction

__all__ = [
    "ResiduePrediction",
    "evaluate_predictions",
    "load_aligned_jsonl",
    "load_fold_assignments",
    "roc_auc_score",
    "validate_and_order_oof_records",
]
