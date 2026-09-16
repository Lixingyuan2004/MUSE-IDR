from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SEEDS = (17, 29, 43)
EPSILON = 1e-7


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload(tar: tarfile.TarFile, member: str) -> tuple[bytes, str]:
    extracted = tar.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"Archive member is not a regular file: {member}")
    payload = extracted.read()
    return payload, hashlib.sha256(payload).hexdigest()


def read_oof_member(
    tar: tarfile.TarFile, member: str
) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray, str]:
    payload, digest = _payload(tar, member)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    required = ["protein_id", "position", "label", "probability"]
    if reader.fieldnames != required:
        raise ValueError(f"Unexpected columns in {member}: {reader.fieldnames}")
    keys: list[tuple[str, int]] = []
    labels: list[int] = []
    probabilities: list[float] = []
    for row in reader:
        keys.append((row["protein_id"], int(row["position"])))
        labels.append(int(row["label"]))
        probabilities.append(float(row["probability"]))
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate protein-position keys in {member}")
    scores = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"Invalid probability in {member}")
    return keys, np.asarray(labels, dtype=np.int8), scores, digest


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coarse-step", type=float, default=0.05)
    parser.add_argument("--fine-step", type=float, default=0.01)
    parser.add_argument("--fine-radius", type=float, default=0.11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive)
    fold_map = load_fold_map(args.manifest)
    source_metrics: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    probability_columns: list[np.ndarray] = []
    keys: list[tuple[str, int]] | None = None
    labels: np.ndarray | None = None

    with tarfile.open(archive_path, "r:gz") as tar:
        for seed in SEEDS:
            member = (
                f"outputs/ablations/a2_multiscale_context/seed_{seed}/"
                "oof_predictions.csv"
            )
            current_keys, current_labels, scores, digest = read_oof_member(tar, member)
            if keys is None:
                keys = current_keys
                labels = current_labels
            elif current_keys != keys or not np.array_equal(current_labels, labels):
                raise ValueError(f"Seed {seed} OOF rows are not exactly aligned")
            probability_columns.append(scores)
            source_hashes[str(seed)] = digest

            metrics_member = (
                f"outputs/ablations/a2_multiscale_context/seed_{seed}/oof_metrics.json"
            )
            metrics_payload, _ = _payload(tar, metrics_member)
            archived = json.loads(metrics_payload)["oof_metrics"]
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

    coarse_probability = search_weights(
        labels, probabilities, step=args.coarse_step
    )
    fine_probability = search_weights(
        labels,
        probabilities,
        step=args.fine_step,
        center=coarse_probability[1],
        radius=args.fine_radius,
    )
    logit_matrix = logits(probabilities)
    coarse_logit = search_weights(labels, logit_matrix, step=args.coarse_step)
    fine_logit = search_weights(
        labels,
        logit_matrix,
        step=args.fine_step,
        center=coarse_logit[1],
        radius=args.fine_radius,
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
    uniform_beats_tuned_folds = sum(
        method_metrics["uniform_logit_mean"]["by_fold"][fold]["micro_roc_auc"]
        > method_metrics["oof_tuned_logit_weighted"]["by_fold"][fold]["micro_roc_auc"]
        for fold in method_metrics["uniform_logit_mean"]["by_fold"]
    )
    report = {
        "schema_version": 1,
        "experiment": "a5_seed_ensemble",
        "base_model": "a2_multiscale_context",
        "seeds": list(SEEDS),
        "archive": str(archive_path),
        "archive_sha256": file_sha256(archive_path),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "alignment_verified": True,
        "rows": int(labels.size),
        "known_residues": int(known.sum()),
        "source_prediction_sha256": source_hashes,
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
                "Their OOF metric is development/tuning performance, not an "
                "unbiased external estimate. CAID remains untouched."
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
            method_metrics["uniform_logit_mean"]["oof"]["micro_roc_auc"]
            - best_seed_auc
        ),
        "uniform_logit_beats_oof_tuned_logit_folds": uniform_beats_tuned_folds,
        "selected_for_future_external_test": {
            "method": selected_name,
            "seed_order": list(SEEDS),
            "weights": [1.0 / len(SEEDS)] * len(SEEDS),
            "selection_dataset": "none_equal_weights_are_label_free",
            "selection_rationale": (
                "Best label-free ensemble on pooled OOF and more robust than "
                "OOF-tuned logit weights in four of five folds."
            ),
            "caid_used_for_selection": False,
        },
        "single_output_score": True,
        "caid1_caid2_caid3_used": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "a5_ensemble_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_predictions(
        output_dir / "a5_oof_predictions.csv.gz",
        keys,
        folds,
        labels,
        probabilities,
        methods,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
