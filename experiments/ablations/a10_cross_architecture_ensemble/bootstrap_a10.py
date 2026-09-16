from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a9_a8_seed_ensemble.common import file_sha256


class FixedScoreWeightedAuc:
    def __init__(self, labels: np.ndarray, scores: np.ndarray) -> None:
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        self.order = order
        self.positive = (labels[order] == 1).astype(np.float64)
        self.negative = 1.0 - self.positive
        self.group_starts = np.flatnonzero(
            np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
        )

    def evaluate(self, sample_weight: np.ndarray) -> float:
        weights = sample_weight[self.order].astype(np.float64, copy=False)
        positives = np.add.reduceat(weights * self.positive, self.group_starts)
        negatives = np.add.reduceat(weights * self.negative, self.group_starts)
        total_positive = float(positives.sum())
        total_negative = float(negatives.sum())
        negative_below = np.cumsum(negatives) - negatives
        favorable = np.sum(positives * (negative_below + 0.5 * negatives))
        return float(favorable / (total_positive * total_negative))


def load_predictions(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    proteins: list[str] = []
    folds: list[str] = []
    lookup: dict[str, int] = {}
    row_protein: list[int] = []
    labels: list[int] = []
    baseline: list[float] = []
    candidate: list[float] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "protein_id", "fold", "label", "a2_uniform_logit_mean",
            "six_model_uniform_logit_mean",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing A10 prediction columns: {reader.fieldnames}")
        for row in reader:
            label = int(row["label"])
            if label not in (0, 1):
                continue
            protein = row["protein_id"]
            fold = row["fold"]
            if protein not in lookup:
                lookup[protein] = len(proteins)
                proteins.append(protein)
                folds.append(fold)
            index = lookup[protein]
            if folds[index] != fold:
                raise ValueError(f"Protein occurs in multiple folds: {protein}")
            row_protein.append(index)
            labels.append(label)
            baseline.append(float(row["a2_uniform_logit_mean"]))
            candidate.append(float(row["six_model_uniform_logit_mean"]))
    return (
        np.asarray(labels, dtype=np.int8),
        np.asarray(baseline, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
        tuple(proteins),
        tuple(folds),
        np.asarray(row_protein, dtype=np.int32),
    )


def interval(samples: np.ndarray) -> list[float]:
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


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
    labels, base, candidate, protein_ids, protein_folds, row_protein = load_predictions(
        args.predictions
    )
    with Path(args.ensemble_report).open("r", encoding="utf-8") as handle:
        ensemble_report = json.load(handle)
    point_base = float(roc_auc_score(labels, base))
    point_candidate = float(roc_auc_score(labels, candidate))
    expected_base = ensemble_report["method_metrics"]["a2_uniform_logit_mean"]["oof"]["micro_roc_auc"]
    expected_candidate = ensemble_report["method_metrics"]["six_model_uniform_logit_mean"]["oof"]["micro_roc_auc"]
    if not math.isclose(point_base, expected_base, rel_tol=0.0, abs_tol=2e-9):
        raise ValueError("A5 point AUC does not reproduce report")
    if not math.isclose(point_candidate, expected_candidate, rel_tol=0.0, abs_tol=2e-9):
        raise ValueError("A10 point AUC does not reproduce report")

    base_auc = FixedScoreWeightedAuc(labels, base)
    candidate_auc = FixedScoreWeightedAuc(labels, candidate)
    fold_array = np.asarray(protein_folds, dtype=object)
    fold_groups = [
        np.flatnonzero(fold_array == fold) for fold in sorted(set(protein_folds))
    ]
    root_rng = np.random.default_rng(args.random_seed)
    seeds = root_rng.integers(
        0, np.iinfo(np.uint64).max, size=args.replicates, dtype=np.uint64
    )

    def one(seed: np.uint64) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        counts = np.zeros(len(protein_ids), dtype=np.int32)
        for group in fold_groups:
            sampled = rng.choice(group, size=group.size, replace=True)
            counts += np.bincount(sampled, minlength=len(protein_ids)).astype(np.int32)
        weights = counts[row_protein]
        return base_auc.evaluate(weights), candidate_auc.evaluate(weights)

    base_samples = np.empty(args.replicates, dtype=np.float64)
    candidate_samples = np.empty(args.replicates, dtype=np.float64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, (base_value, candidate_value) in enumerate(pool.map(one, seeds)):
            base_samples[index] = base_value
            candidate_samples[index] = candidate_value
            if (index + 1) % 250 == 0 or index + 1 == args.replicates:
                print(json.dumps({"bootstrap_completed": index + 1}), flush=True)
    delta = candidate_samples - base_samples
    nonpositive = int(np.sum(delta <= 0.0))
    report = {
        "schema_version": 1,
        "experiment": "a10_cross_architecture_protein_cluster_bootstrap",
        "comparison": "six_model_uniform_logit_mean_vs_a5_a2_uniform_logit_mean",
        "sampling_unit": "protein",
        "paired": True,
        "stratified_by_fold": True,
        "replicates": args.replicates,
        "random_seed": args.random_seed,
        "proteins": len(protein_ids),
        "known_residues": int(labels.size),
        "point_estimate": {
            "a5_micro_roc_auc": point_base,
            "a10_micro_roc_auc": point_candidate,
            "paired_delta": point_candidate - point_base,
        },
        "bootstrap": {
            "paired_delta_mean": float(delta.mean()),
            "paired_delta_standard_deviation": float(delta.std(ddof=1)),
            "paired_delta_95_percentile_ci": interval(delta),
            "probability_delta_positive": float(np.mean(delta > 0.0)),
            "one_sided_p_delta_le_zero_plus_one_correction": float(
                (nonpositive + 1) / (args.replicates + 1)
            ),
            "nonpositive_delta_replicates": nonpositive,
        },
        "predictions_sha256": file_sha256(args.predictions),
        "ensemble_report_sha256": file_sha256(args.ensemble_report),
        "caid1_caid2_caid3_used": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "a10_bootstrap_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    np.savez_compressed(
        output_dir / "a10_bootstrap_samples.npz",
        a5_auc=base_samples,
        a10_auc=candidate_samples,
        paired_delta=delta,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
