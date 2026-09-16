from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a9_a8_seed_ensemble.common import (
    SEEDS,
    build_untuned_methods,
    compute_metrics,
    file_sha256,
    load_fold_map,
    logits,
    per_fold_metrics,
    search_weights,
    sigmoid,
    write_predictions,
)


def read_oof_file(
    path: str | Path,
) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray, str]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["protein_id", "position", "label", "probability"]
        if reader.fieldnames != required:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        keys: list[tuple[str, int]] = []
        labels: list[int] = []
        probabilities: list[float] = []
        for row in reader:
            keys.append((row["protein_id"], int(row["position"])))
            labels.append(int(row["label"]))
            probabilities.append(float(row["probability"]))
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate protein-position keys in {path}")
    label_array = np.asarray(labels, dtype=np.int8)
    if not np.isin(label_array, [-1, 0, 1]).all():
        raise ValueError(f"Invalid residue label in {path}")
    scores = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"Invalid probability in {path}")
    return keys, label_array, scores, file_sha256(path)


def build_ensemble(
    *,
    input_root: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    coarse_step: float = 0.05,
    fine_step: float = 0.01,
    fine_radius: float = 0.11,
) -> dict[str, Any]:
    input_root = Path(input_root)
    manifest = Path(manifest)
    output_dir = Path(output_dir)
    fold_map = load_fold_map(manifest)
    source_metrics: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    source_metric_hashes: dict[str, str] = {}
    probability_columns: list[np.ndarray] = []
    keys: list[tuple[str, int]] | None = None
    labels: np.ndarray | None = None

    for seed in SEEDS:
        seed_dir = input_root / f"seed_{seed}"
        prediction_path = seed_dir / "oof_predictions.csv"
        metrics_path = seed_dir / "oof_metrics.json"
        current_keys, current_labels, scores, digest = read_oof_file(prediction_path)
        if keys is None:
            keys = current_keys
            labels = current_labels
        elif current_keys != keys or not np.array_equal(current_labels, labels):
            raise ValueError(f"Seed {seed} OOF rows are not exactly aligned")
        probability_columns.append(scores)
        source_hashes[str(seed)] = digest
        source_metric_hashes[str(seed)] = file_sha256(metrics_path)

        with metrics_path.open("r", encoding="utf-8") as handle:
            archived = json.load(handle)["oof_metrics"]
        reproduced = compute_metrics(
            [key[0] for key in current_keys], current_labels, scores
        )
        for metric in ("micro_roc_auc", "micro_pr_auc", "macro_roc_auc"):
            if not math.isclose(
                reproduced[metric], archived[metric], rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Seed {seed} {metric} mismatch: "
                    f"{reproduced[metric]} != {archived[metric]}"
                )
        source_metrics[str(seed)] = reproduced

    assert keys is not None and labels is not None
    protein_ids = [key[0] for key in keys]
    missing_folds = sorted(set(protein_ids) - set(fold_map))
    if missing_folds:
        raise KeyError(f"Predicted proteins missing from manifest: {missing_folds[:8]}")
    folds = [fold_map[protein_id] for protein_id in protein_ids]
    probabilities = np.stack(probability_columns, axis=1)
    methods = build_untuned_methods(probabilities)

    # These label-tuned variants are diagnostic only. They are never selected as the
    # formal model and are explicitly marked as development estimates in the report.
    coarse_probability = search_weights(labels, probabilities, step=coarse_step)
    fine_probability = search_weights(
        labels,
        probabilities,
        step=fine_step,
        center=coarse_probability[1],
        radius=fine_radius,
    )
    logit_matrix = logits(probabilities)
    coarse_logit = search_weights(labels, logit_matrix, step=coarse_step)
    fine_logit = search_weights(
        labels,
        logit_matrix,
        step=fine_step,
        center=coarse_logit[1],
        radius=fine_radius,
    )
    methods["oof_tuned_probability_weighted"] = probabilities @ fine_probability[1]
    methods["oof_tuned_logit_weighted"] = sigmoid(logit_matrix @ fine_logit[1])

    method_metrics: dict[str, Any] = {}
    for name, scores in methods.items():
        method_metrics[name] = {
            "oof": compute_metrics(protein_ids, labels, scores),
            "by_fold": per_fold_metrics(protein_ids, folds, labels, scores),
            "uses_oof_labels_to_choose_weights": name.startswith("oof_tuned_"),
        }

    known = np.isin(labels, [0, 1])
    best_seed = max(
        SEEDS, key=lambda seed: source_metrics[str(seed)]["micro_roc_auc"]
    )
    best_seed_auc = source_metrics[str(best_seed)]["micro_roc_auc"]
    selected_name = "uniform_logit_mean"
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "a9_a8_seed_ensemble",
        "base_model": "a8_multilayer_esm2_fusion",
        "seeds": list(SEEDS),
        "input_root": str(input_root),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "alignment_verified": True,
        "rows": int(labels.size),
        "known_residues": int(known.sum()),
        "source_prediction_sha256": source_hashes,
        "source_metric_sha256": source_metric_hashes,
        "source_seed_metrics": source_metrics,
        "known_residue_prediction_correlation": np.corrcoef(
            probabilities[known].T
        ).tolist(),
        "untuned_methods_do_not_access_labels": [
            "uniform_probability_mean",
            "uniform_logit_mean",
            "probability_median",
        ],
        "weight_search": {
            "warning": (
                "Weights use the same classic-dataset OOF labels reported below. "
                "Their metric is exploratory development performance only; the "
                "formal model remains the label-free equal-logit ensemble."
            ),
            "seed_order": list(SEEDS),
            "probability": {
                "coarse_auc": coarse_probability[0],
                "coarse_weights": coarse_probability[1].tolist(),
                "coarse_candidates": coarse_probability[2],
                "fine_auc": fine_probability[0],
                "fine_weights": fine_probability[1].tolist(),
                "fine_candidates": fine_probability[2],
            },
            "logit": {
                "coarse_auc": coarse_logit[0],
                "coarse_weights": coarse_logit[1].tolist(),
                "coarse_candidates": coarse_logit[2],
                "fine_auc": fine_logit[0],
                "fine_weights": fine_logit[1].tolist(),
                "fine_candidates": fine_logit[2],
            },
        },
        "method_metrics": method_metrics,
        "best_individual_seed": int(best_seed),
        "best_individual_micro_roc_auc": best_seed_auc,
        "fair_uniform_logit_delta_vs_best_seed": (
            method_metrics[selected_name]["oof"]["micro_roc_auc"] - best_seed_auc
        ),
        "selected_for_future_external_test": {
            "method": selected_name,
            "seed_order": list(SEEDS),
            "weights": [1.0 / len(SEEDS)] * len(SEEDS),
            "selection_dataset": "none_equal_weights_are_label_free",
            "selection_rationale": (
                "Equal-logit averaging is fixed without fitting ensemble weights "
                "to OOF or CAID labels and matches the formal A5 protocol."
            ),
            "caid_used_for_selection": False,
        },
        "single_output_score": True,
        "soft_disorder_output": False,
        "caid1_caid2_caid3_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "a9_ensemble_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_predictions(
        output_dir / "a9_oof_predictions.csv.gz",
        keys,
        folds,
        labels,
        probabilities,
        methods,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coarse-step", type=float, default=0.05)
    parser.add_argument("--fine-step", type=float, default=0.01)
    parser.add_argument("--fine-radius", type=float, default=0.11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_ensemble(
        input_root=args.input_root,
        manifest=args.manifest,
        output_dir=args.output_dir,
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
        fine_radius=args.fine_radius,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
