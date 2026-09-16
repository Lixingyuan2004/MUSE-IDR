from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a9_a8_seed_ensemble.common import (
    SEEDS,
    compute_metrics,
    file_sha256,
    load_fold_map,
    logits,
    per_fold_metrics,
    sigmoid,
)
from experiments.ablations.a9_a8_seed_ensemble.ensemble_a8 import read_oof_file


FAMILIES = ("a2", "a8")


def load_family(
    root: str | Path,
    family: str,
) -> tuple[
    list[tuple[str, int]],
    np.ndarray,
    np.ndarray,
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    root = Path(root)
    keys: list[tuple[str, int]] | None = None
    labels: np.ndarray | None = None
    columns: list[np.ndarray] = []
    metrics: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for seed in SEEDS:
        seed_dir = root / f"seed_{seed}"
        prediction_path = seed_dir / "oof_predictions.csv"
        metrics_path = seed_dir / "oof_metrics.json"
        current_keys, current_labels, scores, digest = read_oof_file(prediction_path)
        if keys is None:
            keys = current_keys
            labels = current_labels
        elif current_keys != keys or not np.array_equal(current_labels, labels):
            raise ValueError(f"{family} seed {seed} OOF rows are not aligned")
        columns.append(scores)
        hashes[f"seed_{seed}_predictions"] = digest
        hashes[f"seed_{seed}_metrics"] = file_sha256(metrics_path)
        with metrics_path.open("r", encoding="utf-8") as handle:
            recorded = json.load(handle)["oof_metrics"]
        reproduced = compute_metrics(
            [key[0] for key in current_keys], current_labels, scores
        )
        for metric in ("micro_roc_auc", "micro_pr_auc", "macro_roc_auc"):
            if not math.isclose(
                reproduced[metric], recorded[metric], rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{family} seed {seed} {metric} does not reproduce")
        metrics[str(seed)] = reproduced
    assert keys is not None and labels is not None
    return keys, labels, np.stack(columns, axis=1), metrics, hashes


def search_family_logit_weight(
    labels: np.ndarray,
    a2_logit: np.ndarray,
    a8_logit: np.ndarray,
    step: float = 0.01,
) -> tuple[float, float, int]:
    divisions = int(round(1.0 / step))
    if not math.isclose(divisions * step, 1.0, abs_tol=1e-9):
        raise ValueError("Weight step must divide 1.0")
    known = np.isin(labels, [0, 1])
    y_true = labels[known]
    best_auc = -float("inf")
    best_a8_weight = 0.0
    for index in range(divisions + 1):
        a8_weight = index * step
        scores = (1.0 - a8_weight) * a2_logit[known] + a8_weight * a8_logit[known]
        auc = float(roc_auc_score(y_true, scores))
        if auc > best_auc:
            best_auc = auc
            best_a8_weight = a8_weight
    return best_auc, best_a8_weight, divisions + 1


def write_predictions(
    path: str | Path,
    keys: Sequence[tuple[str, int]],
    folds: Sequence[str],
    labels: np.ndarray,
    a2: np.ndarray,
    a8: np.ndarray,
    methods: Mapping[str, np.ndarray],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    method_names = list(methods)
    source_names = [f"a2_seed_{seed}" for seed in SEEDS] + [
        f"a8_seed_{seed}" for seed in SEEDS
    ]
    sources = np.concatenate([a2, a8], axis=1)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["protein_id", "position", "fold", "label"]
            + source_names
            + method_names
        )
        for index, ((protein_id, position), fold, label) in enumerate(
            zip(keys, folds, labels)
        ):
            writer.writerow(
                [protein_id, position, fold, int(label)]
                + [f"{value:.9g}" for value in sources[index]]
                + [f"{methods[name][index]:.9g}" for name in method_names]
            )


def build_ensemble(
    *,
    a2_root: str | Path,
    a8_root: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    a2_keys, a2_labels, a2, a2_metrics, a2_hashes = load_family(a2_root, "a2")
    a8_keys, a8_labels, a8, a8_metrics, a8_hashes = load_family(a8_root, "a8")
    if a2_keys != a8_keys or not np.array_equal(a2_labels, a8_labels):
        raise ValueError("A2 and A8 OOF rows/labels are not exactly aligned")
    keys = a2_keys
    labels = a2_labels
    protein_ids = [key[0] for key in keys]
    fold_map = load_fold_map(manifest)
    missing = sorted(set(protein_ids) - set(fold_map))
    if missing:
        raise KeyError(f"Predicted proteins missing from manifest: {missing[:8]}")
    folds = [fold_map[protein_id] for protein_id in protein_ids]

    a2_logit = logits(a2).mean(axis=1)
    a8_logit = logits(a8).mean(axis=1)
    all_probabilities = np.concatenate([a2, a8], axis=1)
    methods = {
        "a2_uniform_logit_mean": sigmoid(a2_logit),
        "a8_uniform_logit_mean": sigmoid(a8_logit),
        "six_model_uniform_logit_mean": sigmoid(logits(all_probabilities).mean(axis=1)),
        "six_model_uniform_probability_mean": all_probabilities.mean(axis=1),
    }
    tuned_auc, tuned_a8_weight, candidates = search_family_logit_weight(
        labels, a2_logit, a8_logit
    )
    methods["oof_tuned_family_logit_weighted"] = sigmoid(
        (1.0 - tuned_a8_weight) * a2_logit + tuned_a8_weight * a8_logit
    )
    method_metrics: dict[str, Any] = {}
    for name, scores in methods.items():
        method_metrics[name] = {
            "oof": compute_metrics(protein_ids, labels, scores),
            "by_fold": per_fold_metrics(protein_ids, folds, labels, scores),
            "uses_oof_labels_to_choose_weights": name.startswith("oof_tuned_"),
        }

    formal_name = "six_model_uniform_logit_mean"
    baseline_name = "a2_uniform_logit_mean"
    formal_auc = method_metrics[formal_name]["oof"]["micro_roc_auc"]
    baseline_auc = method_metrics[baseline_name]["oof"]["micro_roc_auc"]
    recommended = formal_name if formal_auc > baseline_auc else baseline_name
    known = np.isin(labels, [0, 1])
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "a10_cross_architecture_ensemble",
        "families": list(FAMILIES),
        "seeds_per_family": list(SEEDS),
        "a2_root": str(a2_root),
        "a8_root": str(a8_root),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "alignment_verified_across_all_six_models": True,
        "rows": int(labels.size),
        "known_residues": int(known.sum()),
        "source_prediction_and_metric_sha256": {"a2": a2_hashes, "a8": a8_hashes},
        "source_seed_metrics": {"a2": a2_metrics, "a8": a8_metrics},
        "known_residue_six_model_prediction_correlation": np.corrcoef(
            all_probabilities[known].T
        ).tolist(),
        "known_residue_family_ensemble_correlation": float(
            np.corrcoef(methods[baseline_name][known], methods["a8_uniform_logit_mean"][known])[0, 1]
        ),
        "method_metrics": method_metrics,
        "formal_candidate": {
            "method": formal_name,
            "model_weights": [1.0 / 6.0] * 6,
            "weights_fitted_to_labels": False,
            "micro_roc_auc": formal_auc,
            "delta_vs_a5_a2_ensemble": formal_auc - baseline_auc,
        },
        "exploratory_family_weight_search": {
            "warning": "Uses classic-dataset OOF labels and is not the formal model.",
            "a2_weight": 1.0 - tuned_a8_weight,
            "a8_weight": tuned_a8_weight,
            "micro_roc_auc": tuned_auc,
            "candidates": candidates,
            "caid_used": False,
        },
        "recommended_for_future_external_test": {
            "method": recommended,
            "reason": "Higher classic-dataset OOF micro ROC-AUC among fixed label-free candidates.",
            "caid_used_for_selection": False,
        },
        "single_output_score": True,
        "soft_disorder_output": False,
        "caid1_caid2_caid3_used": False,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "a10_ensemble_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_predictions(
        output_dir / "a10_oof_predictions.csv.gz",
        keys,
        folds,
        labels,
        a2,
        a8,
        methods,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2-root", required=True)
    parser.add_argument("--a8-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_ensemble(
        a2_root=args.a2_root,
        a8_root=args.a8_root,
        manifest=args.manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
