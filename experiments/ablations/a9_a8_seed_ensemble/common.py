from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SEEDS = (17, 29, 43)
EPSILON = 1e-7


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fold_map(manifest: str | Path) -> dict[str, str]:
    fold_map: dict[str, str] = {}
    with Path(manifest).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            protein_id = str(
                row.get("protein_id", row.get("id", row.get("name", row.get("accession"))))
            )
            fold = row.get("fold", row.get("fold_id", row.get("split_fold")))
            if protein_id == "None" or fold is None:
                raise KeyError("Manifest needs a protein ID and fold field")
            if protein_id in fold_map:
                raise ValueError(f"Duplicate manifest protein ID: {protein_id}")
            fold_map[protein_id] = str(fold)
    if not fold_map:
        raise ValueError("Manifest is empty")
    return fold_map


def compute_metrics(
    protein_ids: Sequence[str], labels: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    if len(protein_ids) != labels.size or labels.shape != scores.shape:
        raise ValueError("Protein IDs, labels and scores must have matching lengths")
    known = np.isin(labels, [0, 1])
    y_true = labels[known]
    y_score = scores[known]
    if np.unique(y_true).size != 2:
        raise ValueError("Metrics require both known label classes")

    macro_values: list[float] = []
    start = 0
    while start < len(protein_ids):
        end = start + 1
        while end < len(protein_ids) and protein_ids[end] == protein_ids[start]:
            end += 1
        protein_known = np.isin(labels[start:end], [0, 1])
        protein_labels = labels[start:end][protein_known]
        if np.unique(protein_labels).size == 2:
            macro_values.append(
                float(
                    roc_auc_score(
                        protein_labels, scores[start:end][protein_known]
                    )
                )
            )
        start = end

    return {
        "micro_roc_auc": float(roc_auc_score(y_true, y_score)),
        "micro_pr_auc": float(average_precision_score(y_true, y_score)),
        "macro_roc_auc": float(np.mean(macro_values)),
        "known_residues": int(known.sum()),
        "positive_residues": int(np.sum(y_true == 1)),
        "negative_residues": int(np.sum(y_true == 0)),
        "proteins": len(set(protein_ids)),
        "macro_auc_proteins": len(macro_values),
    }


def logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def build_untuned_methods(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(SEEDS):
        raise ValueError("Expected [residues, 3 seeds] probabilities")
    return {
        "uniform_probability_mean": probabilities.mean(axis=1),
        "uniform_logit_mean": sigmoid(logits(probabilities).mean(axis=1)),
        "probability_median": np.median(probabilities, axis=1),
    }


def search_weights(
    labels: np.ndarray,
    score_matrix: np.ndarray,
    *,
    step: float,
    center: np.ndarray | None = None,
    radius: float | None = None,
) -> tuple[float, np.ndarray, int]:
    known = np.isin(labels, [0, 1])
    y_true = labels[known]
    matrix = score_matrix[known]
    divisions = int(round(1.0 / step))
    if not math.isclose(divisions * step, 1.0, abs_tol=1e-9):
        raise ValueError("Weight step must divide 1.0")
    best_auc = -float("inf")
    best_weights: np.ndarray | None = None
    candidates = 0
    for first in range(divisions + 1):
        for second in range(divisions - first + 1):
            weights = np.asarray(
                [first * step, second * step, 1.0 - (first + second) * step],
                dtype=np.float64,
            )
            if center is not None and radius is not None:
                if float(np.max(np.abs(weights - center))) > radius + 1e-12:
                    continue
            candidates += 1
            auc = float(roc_auc_score(y_true, matrix @ weights))
            if auc > best_auc:
                best_auc = auc
                best_weights = weights.copy()
    if best_weights is None:
        raise RuntimeError("Weight search evaluated no candidates")
    return best_auc, best_weights, candidates


def per_fold_metrics(
    protein_ids: Sequence[str],
    folds: Sequence[str],
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, dict[str, Any]]:
    fold_array = np.asarray(folds, dtype=object)
    protein_array = np.asarray(protein_ids, dtype=object)
    result: dict[str, dict[str, Any]] = {}
    for fold in sorted(set(folds)):
        mask = fold_array == fold
        result[fold] = compute_metrics(
            protein_array[mask].tolist(), labels[mask], scores[mask]
        )
    return result


def write_predictions(
    path: str | Path,
    keys: Sequence[tuple[str, int]],
    folds: Sequence[str],
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    methods: Mapping[str, np.ndarray],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    method_names = list(methods)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["protein_id", "position", "fold", "label"]
            + [f"seed_{seed}" for seed in SEEDS]
            + method_names
        )
        for index, ((protein_id, position), fold, label) in enumerate(
            zip(keys, folds, labels)
        ):
            writer.writerow(
                [protein_id, position, fold, int(label)]
                + [f"{value:.9g}" for value in probability_matrix[index]]
                + [f"{methods[name][index]:.9g}" for name in method_names]
            )
