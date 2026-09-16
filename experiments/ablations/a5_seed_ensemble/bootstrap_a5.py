from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a5_seed_ensemble.ensemble_a2 import logits, sigmoid


SEEDS = (17, 29, 43)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BootstrapInput:
    labels: np.ndarray
    seed_probabilities: np.ndarray
    protein_index: np.ndarray
    protein_ids: tuple[str, ...]
    protein_folds: tuple[str, ...]


class FixedScoreWeightedAuc:
    """Exact weighted ROC-AUC with one precomputed score ordering."""

    def __init__(self, labels: np.ndarray, scores: np.ndarray) -> None:
        if labels.shape != scores.shape or labels.ndim != 1:
            raise ValueError("labels and scores must be aligned one-dimensional arrays")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("FixedScoreWeightedAuc accepts known binary labels only")
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        self.order = order
        self.positive = (labels[order] == 1).astype(np.float64)
        self.negative = 1.0 - self.positive
        self.group_starts = np.flatnonzero(
            np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
        )

    def evaluate(self, sample_weight: np.ndarray) -> float:
        if sample_weight.ndim != 1 or sample_weight.size != self.order.size:
            raise ValueError("sample weights do not align with the AUC ordering")
        ordered_weight = sample_weight[self.order].astype(np.float64, copy=False)
        group_positive = np.add.reduceat(
            ordered_weight * self.positive, self.group_starts
        )
        group_negative = np.add.reduceat(
            ordered_weight * self.negative, self.group_starts
        )
        total_positive = float(group_positive.sum())
        total_negative = float(group_negative.sum())
        if total_positive <= 0.0 or total_negative <= 0.0:
            return float("nan")
        negative_below = np.cumsum(group_negative) - group_negative
        favorable_pairs = np.sum(
            group_positive * (negative_below + 0.5 * group_negative)
        )
        return float(favorable_pairs / (total_positive * total_negative))


def load_bootstrap_input(path: str | Path) -> BootstrapInput:
    required = {
        "protein_id",
        "fold",
        "label",
        "seed_17",
        "seed_29",
        "seed_43",
    }
    protein_lookup: dict[str, int] = {}
    protein_ids: list[str] = []
    protein_folds: list[str] = []
    labels: list[int] = []
    probabilities: list[tuple[float, float, float]] = []
    row_protein_index: list[int] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required prediction columns: {reader.fieldnames}")
        for row in reader:
            label = int(row["label"])
            if label not in (0, 1):
                continue
            protein_id = row["protein_id"]
            fold = row["fold"]
            if protein_id not in protein_lookup:
                protein_lookup[protein_id] = len(protein_ids)
                protein_ids.append(protein_id)
                protein_folds.append(fold)
            protein = protein_lookup[protein_id]
            if protein_folds[protein] != fold:
                raise ValueError(f"Protein occurs in multiple folds: {protein_id}")
            labels.append(label)
            probabilities.append(
                tuple(float(row[f"seed_{seed}"]) for seed in SEEDS)
            )
            row_protein_index.append(protein)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probability_array).all():
        raise ValueError("Non-finite probability in A5 prediction file")
    return BootstrapInput(
        labels=np.asarray(labels, dtype=np.int8),
        seed_probabilities=probability_array,
        protein_index=np.asarray(row_protein_index, dtype=np.int32),
        protein_ids=tuple(protein_ids),
        protein_folds=tuple(protein_folds),
    )


def paired_stratified_cluster_bootstrap(
    data: BootstrapInput,
    base_scores: np.ndarray,
    ensemble_scores: np.ndarray,
    *,
    replicates: int,
    random_seed: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    base_auc = FixedScoreWeightedAuc(data.labels, base_scores)
    ensemble_auc = FixedScoreWeightedAuc(data.labels, ensemble_scores)
    protein_folds = np.asarray(data.protein_folds, dtype=object)
    fold_groups = [
        np.flatnonzero(protein_folds == fold)
        for fold in sorted(set(data.protein_folds))
    ]
    number_of_proteins = len(data.protein_ids)
    root_rng = np.random.default_rng(random_seed)
    replicate_seeds = root_rng.integers(
        0, np.iinfo(np.uint64).max, size=replicates, dtype=np.uint64
    )

    def one(seed: np.uint64) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        protein_counts = np.zeros(number_of_proteins, dtype=np.int32)
        for group in fold_groups:
            sampled = rng.choice(group, size=group.size, replace=True)
            protein_counts += np.bincount(
                sampled, minlength=number_of_proteins
            ).astype(np.int32)
        row_weight = protein_counts[data.protein_index]
        return base_auc.evaluate(row_weight), ensemble_auc.evaluate(row_weight)

    base_samples = np.empty(replicates, dtype=np.float64)
    ensemble_samples = np.empty(replicates, dtype=np.float64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, (base, ensemble) in enumerate(pool.map(one, replicate_seeds)):
            base_samples[index] = base
            ensemble_samples[index] = ensemble
            if (index + 1) % 250 == 0 or index + 1 == replicates:
                print(
                    json.dumps(
                        {"bootstrap_completed": index + 1, "replicates": replicates}
                    ),
                    flush=True,
                )
    if not np.isfinite(base_samples).all() or not np.isfinite(ensemble_samples).all():
        raise RuntimeError("Bootstrap generated an undefined AUC replicate")
    return base_samples, ensemble_samples, ensemble_samples - base_samples


def interval(samples: np.ndarray, confidence: float = 0.95) -> list[float]:
    tail = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(samples, tail)),
        float(np.quantile(samples, 1.0 - tail)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ensemble-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260824)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_bootstrap_input(args.predictions)
    with Path(args.ensemble_report).open("r", encoding="utf-8") as handle:
        ensemble_report = json.load(handle)
    selected = ensemble_report["selected_for_future_external_test"]
    if selected["method"] != "uniform_logit_mean":
        raise ValueError("Bootstrap is registered for the formal equal-logit A5 model")

    base_scores = data.seed_probabilities[:, 0]
    ensemble_scores = sigmoid(logits(data.seed_probabilities).mean(axis=1))
    point_base = float(roc_auc_score(data.labels, base_scores))
    point_ensemble = float(roc_auc_score(data.labels, ensemble_scores))
    expected_base = ensemble_report["source_seed_metrics"]["17"]["micro_roc_auc"]
    expected_ensemble = ensemble_report["method_metrics"]["uniform_logit_mean"]["oof"][
        "micro_roc_auc"
    ]
    if not math.isclose(point_base, expected_base, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Seed 17 point AUC does not reproduce the A5 report")
    if not math.isclose(point_ensemble, expected_ensemble, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("A5 point AUC does not reproduce the A5 report")

    base_samples, ensemble_samples, delta_samples = paired_stratified_cluster_bootstrap(
        data,
        base_scores,
        ensemble_scores,
        replicates=args.replicates,
        random_seed=args.random_seed,
        workers=args.workers,
    )
    nonpositive = int(np.sum(delta_samples <= 0.0))
    report = {
        "schema_version": 1,
        "experiment": "a5_seed_ensemble_protein_cluster_bootstrap",
        "comparison": "uniform_logit_mean_vs_best_individual_seed_17",
        "sampling_unit": "protein",
        "paired": True,
        "stratified_by_fold": True,
        "replicates": args.replicates,
        "random_seed": args.random_seed,
        "proteins": len(data.protein_ids),
        "known_residues": int(data.labels.size),
        "fold_proteins": {
            fold: int(sum(value == fold for value in data.protein_folds))
            for fold in sorted(set(data.protein_folds))
        },
        "point_estimate": {
            "seed_17_micro_roc_auc": point_base,
            "a5_micro_roc_auc": point_ensemble,
            "paired_delta": point_ensemble - point_base,
        },
        "bootstrap": {
            "seed_17_auc_mean": float(base_samples.mean()),
            "seed_17_auc_95_percentile_ci": interval(base_samples),
            "a5_auc_mean": float(ensemble_samples.mean()),
            "a5_auc_95_percentile_ci": interval(ensemble_samples),
            "paired_delta_mean": float(delta_samples.mean()),
            "paired_delta_standard_deviation": float(delta_samples.std(ddof=1)),
            "paired_delta_95_percentile_ci": interval(delta_samples),
            "probability_delta_positive": float(np.mean(delta_samples > 0.0)),
            "one_sided_p_delta_le_zero_plus_one_correction": float(
                (nonpositive + 1) / (args.replicates + 1)
            ),
            "nonpositive_delta_replicates": nonpositive,
        },
        "predictions": str(args.predictions),
        "predictions_sha256": file_sha256(args.predictions),
        "ensemble_report": str(args.ensemble_report),
        "ensemble_report_sha256": file_sha256(args.ensemble_report),
        "caid1_caid2_caid3_used": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "a5_bootstrap_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    np.savez_compressed(
        output_dir / "a5_bootstrap_samples.npz",
        seed_17_auc=base_samples,
        a5_auc=ensemble_samples,
        paired_delta=delta_samples,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
