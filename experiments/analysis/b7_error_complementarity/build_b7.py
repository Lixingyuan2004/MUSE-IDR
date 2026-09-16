"""Build post-lock error, robustness, and complementarity analyses.

B7 is descriptive.  It verifies every locked input before reading CAID labels,
uses only predeclared strata and the fixed 0.5 threshold, and never changes a
model, ensemble weight, threshold, loss, or hyperparameter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b1_caid3_fair_comparison.build_b1 import (  # noqa: E402
    ALLOWED_RESIDUE_NORMALIZATIONS,
    Prediction,
    Reference,
    parse_reference,
)
from experiments.analysis.b6_paired_model_statistics.build_b6 import (  # noqa: E402
    MODEL_ORDER,
    full_precision_metric_vector,
    parse_prediction_tsv,
    sha256,
    verify_locked_file,
)


TRACK_ORDER = ("disorder_nox", "disorder_pdb")
METRIC_NAMES = (
    "roc_auc_full_precision",
    "aucpr_trapezoid_full_precision",
    "aps_full_precision",
    "f1_at_0_5_full_precision",
    "mcc_at_0_5_full_precision",
    "fmax_full_precision",
)


@dataclass(frozen=True)
class AlignedProtein:
    protein_id: str
    sequence_length: int
    known_positions: np.ndarray
    labels: np.ndarray
    scores: tuple[np.ndarray, ...]
    segment_lengths: np.ndarray
    transition_distances: np.ndarray

    @property
    def disorder_fraction(self) -> float:
        return float(np.mean(self.labels))


def length_bin(length: int) -> str:
    if length <= 200:
        return "1-200"
    if length <= 500:
        return "201-500"
    if length <= 1000:
        return "501-1000"
    return "1001+"


def disorder_fraction_bin(value: float) -> str:
    if value <= 0.10:
        return "0-0.10"
    if value <= 0.30:
        return "(0.10,0.30]"
    if value <= 0.60:
        return "(0.30,0.60]"
    return "(0.60,1.00]"


def segment_length_bin(length: int) -> str:
    if length <= 15:
        return "1-15"
    if length <= 30:
        return "16-30"
    if length <= 100:
        return "31-100"
    return "101+"


def transition_distance_bin(distance: float) -> str:
    if not math.isfinite(distance):
        return "no-transition"
    if distance <= 2:
        return "0-2"
    if distance <= 5:
        return "3-5"
    if distance <= 15:
        return "6-15"
    return "16+"


def annotate_reference_labels(labels: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return known positions, same-state run lengths, and transition distances.

    Masked residues break a contiguous region.  For a 0/1 transition between
    adjacent residues, both residues next to the boundary have distance zero.
    A contiguous known-label region with no transition receives infinity.
    """
    known_positions: list[int] = []
    segment_lengths: list[int] = []
    transition_distances: list[float] = []
    index = 0
    while index < len(labels):
        if labels[index] == "-":
            index += 1
            continue
        region_start = index
        while index < len(labels) and labels[index] != "-":
            index += 1
        region_end = index
        region = labels[region_start:region_end]
        region_segment_lengths = np.empty(len(region), dtype=np.int32)
        run_start = 0
        while run_start < len(region):
            run_end = run_start + 1
            while run_end < len(region) and region[run_end] == region[run_start]:
                run_end += 1
            region_segment_lengths[run_start:run_end] = run_end - run_start
            run_start = run_end

        distances = np.full(len(region), np.inf, dtype=np.float64)
        transitions = [
            boundary
            for boundary in range(len(region) - 1)
            if region[boundary] != region[boundary + 1]
        ]
        positions = np.arange(len(region), dtype=np.int32)
        for boundary in transitions:
            distances = np.minimum(
                distances,
                np.minimum(np.abs(positions - boundary), np.abs(positions - boundary - 1)),
            )
        known_positions.extend(range(region_start, region_end))
        segment_lengths.extend(region_segment_lengths.tolist())
        transition_distances.extend(distances.tolist())

    return (
        np.asarray(known_positions, dtype=np.int32),
        np.asarray(segment_lengths, dtype=np.int32),
        np.asarray(transition_distances, dtype=np.float64),
    )


def align_track(
    reference: dict[str, Reference],
    predictions: dict[str, dict[str, Prediction]],
) -> tuple[list[AlignedProtein], dict[str, object]]:
    if tuple(predictions) != MODEL_ORDER:
        raise ValueError("prediction model order does not match the lock")
    proteins: list[AlignedProtein] = []
    normalized_mismatches = {model: 0 for model in MODEL_ORDER}
    for protein_id, record in reference.items():
        known_positions, segment_lengths, transition_distances = (
            annotate_reference_labels(record.labels)
        )
        if known_positions.size == 0:
            continue
        labels = np.fromiter(
            (int(record.labels[int(position)]) for position in known_positions),
            dtype=np.int8,
        )
        model_scores: list[np.ndarray] = []
        for model in MODEL_ORDER:
            prediction = predictions[model].get(protein_id)
            if prediction is None:
                raise ValueError(f"missing prediction: {model} {protein_id}")
            if len(prediction.sequence) != len(record.sequence):
                raise ValueError(f"sequence length mismatch: {model} {protein_id}")
            for expected, observed in zip(record.sequence, prediction.sequence):
                if expected == observed:
                    continue
                if (expected, observed) not in ALLOWED_RESIDUE_NORMALIZATIONS:
                    raise ValueError(
                        f"substantive residue mismatch: {model} {protein_id} "
                        f"{expected!r}->{observed!r}"
                    )
                normalized_mismatches[model] += 1
            scores = np.asarray(prediction.scores[known_positions], dtype=np.float64)
            if scores.shape != labels.shape or not np.all(np.isfinite(scores)):
                raise ValueError(f"invalid aligned scores: {model} {protein_id}")
            if np.any((scores < 0.0) | (scores > 1.0)):
                raise ValueError(f"scores outside [0,1]: {model} {protein_id}")
            model_scores.append(scores)
        proteins.append(
            AlignedProtein(
                protein_id=protein_id,
                sequence_length=len(record.sequence),
                known_positions=known_positions,
                labels=labels,
                scores=tuple(model_scores),
                segment_lengths=segment_lengths,
                transition_distances=transition_distances,
            )
        )
    if not proteins:
        raise ValueError("track has no known labels")
    return proteins, {
        "reference_proteins": len(reference),
        "aligned_proteins": len(proteins),
        "known_residues": int(sum(protein.labels.size for protein in proteins)),
        "normalized_residue_mismatches": normalized_mismatches,
        "substantive_residue_mismatches": 0,
    }


def threshold_counts(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, int | float]:
    predicted = y_score >= 0.5
    positive = y_true == 1
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "f1": f1, "mcc": mcc}


def descriptive_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.size != y_score.size:
        raise ValueError("descriptive metrics require aligned vectors")
    if y_true.size == 0:
        raise ValueError("descriptive metrics require at least one residue")
    counts = threshold_counts(y_true, y_score)
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(y_score, epsilon, 1.0 - epsilon)
    result: dict[str, object] = {
        "residues": int(y_true.size),
        "positive_labels": int(np.sum(y_true == 1)),
        "negative_labels": int(np.sum(y_true == 0)),
        "prevalence": float(np.mean(y_true)),
        "mean_score": float(np.mean(y_score)),
        "calibration_bias": float(np.mean(y_score) - np.mean(y_true)),
        "brier_score": float(np.mean((y_score - y_true) ** 2)),
        "mean_absolute_error": float(np.mean(np.abs(y_score - y_true))),
        "log_loss": float(
            -np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1.0 - clipped))
        ),
        "error_rate_at_0_5": float(np.mean((y_score >= 0.5) != (y_true == 1))),
        "f1_at_0_5_full_precision": float(counts["f1"]),
        "mcc_at_0_5_full_precision": float(counts["mcc"]),
        "tp": counts["tp"],
        "tn": counts["tn"],
        "fp": counts["fp"],
        "fn": counts["fn"],
    }
    if np.unique(y_true).size == 2:
        vector = full_precision_metric_vector(y_true, y_score)
        for key, value in zip(METRIC_NAMES, vector):
            result[key] = float(value)
    else:
        for key in METRIC_NAMES:
            if key not in result:
                result[key] = None
    return result


def concatenate(
    proteins: Iterable[AlignedProtein], model_index: int
) -> tuple[np.ndarray, np.ndarray]:
    selected = list(proteins)
    if not selected:
        raise ValueError("cannot concatenate an empty protein group")
    return (
        np.concatenate([protein.labels for protein in selected]),
        np.concatenate([protein.scores[model_index] for protein in selected]),
    )


def model_metric_row(
    dataset: str,
    track: str,
    model: str,
    proteins: list[AlignedProtein],
    **context: object,
) -> dict[str, object]:
    labels, scores = concatenate(proteins, MODEL_ORDER.index(model))
    return {
        "dataset": dataset,
        "track": track,
        "model": model,
        "proteins": len(proteins),
        **context,
        **descriptive_metrics(labels, scores),
    }


def subgroup_rows(
    dataset: str, track: str, proteins: list[AlignedProtein]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    definitions = {
        "protein_length": lambda protein: length_bin(protein.sequence_length),
        "known_label_disorder_fraction": lambda protein: disorder_fraction_bin(
            protein.disorder_fraction
        ),
    }
    for stratification, classifier in definitions.items():
        groups: dict[str, list[AlignedProtein]] = {}
        for protein in proteins:
            groups.setdefault(classifier(protein), []).append(protein)
        for stratum, group in groups.items():
            for model in MODEL_ORDER:
                rows.append(
                    model_metric_row(
                        dataset,
                        track,
                        model,
                        group,
                        stratification=stratification,
                        stratum=stratum,
                    )
                )
    return rows


def boundary_rows(
    dataset: str, track: str, proteins: list[AlignedProtein]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: dict[str, list[tuple[np.ndarray, tuple[np.ndarray, ...]]]] = {}
    for protein in proteins:
        labels = protein.labels
        for stratum in sorted({transition_distance_bin(x) for x in protein.transition_distances}):
            mask = np.fromiter(
                (transition_distance_bin(x) == stratum for x in protein.transition_distances),
                dtype=bool,
            )
            groups.setdefault(stratum, []).append(
                (labels[mask], tuple(scores[mask] for scores in protein.scores))
            )
    for stratum, blocks in groups.items():
        labels = np.concatenate([block[0] for block in blocks])
        for model_index, model in enumerate(MODEL_ORDER):
            scores = np.concatenate([block[1][model_index] for block in blocks])
            rows.append(
                {
                    "dataset": dataset,
                    "track": track,
                    "stratification": "distance_to_nearest_observed_transition",
                    "stratum": stratum,
                    "model": model,
                    "proteins": len(blocks),
                    **descriptive_metrics(labels, scores),
                }
            )
    return rows


def segment_rows(
    dataset: str, track: str, proteins: list[AlignedProtein]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[tuple[np.ndarray, tuple[np.ndarray, ...]]]] = {}
    for protein in proteins:
        keys = [
            ("disordered" if label else "ordered", segment_length_bin(int(length)))
            for label, length in zip(protein.labels, protein.segment_lengths)
        ]
        for key in sorted(set(keys)):
            mask = np.fromiter((candidate == key for candidate in keys), dtype=bool)
            groups.setdefault(key, []).append(
                (
                    protein.labels[mask],
                    tuple(scores[mask] for scores in protein.scores),
                )
            )
    for (true_state, stratum), blocks in groups.items():
        labels = np.concatenate([block[0] for block in blocks])
        for model_index, model in enumerate(MODEL_ORDER):
            scores = np.concatenate([block[1][model_index] for block in blocks])
            rows.append(
                {
                    "dataset": dataset,
                    "track": track,
                    "true_state": true_state,
                    "segment_length_bin": stratum,
                    "model": model,
                    "proteins": len(blocks),
                    **descriptive_metrics(labels, scores),
                }
            )
    return rows


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = 0.5 * (start + end - 1)
        start = end
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def pairwise_complementarity(
    labels: np.ndarray, a10_scores: np.ndarray, comparator_scores: np.ndarray
) -> dict[str, object]:
    a10_predicted = a10_scores >= 0.5
    comparator_predicted = comparator_scores >= 0.5
    truth = labels == 1
    a10_correct = a10_predicted == truth
    comparator_correct = comparator_predicted == truth
    both_correct = int(np.sum(a10_correct & comparator_correct))
    a10_only = int(np.sum(a10_correct & ~comparator_correct))
    comparator_only = int(np.sum(~a10_correct & comparator_correct))
    both_wrong = int(np.sum(~a10_correct & ~comparator_correct))
    total = int(labels.size)
    union_errors = a10_only + comparator_only + both_wrong
    return {
        "residues": total,
        "score_pearson": correlation(a10_scores, comparator_scores),
        "score_spearman": correlation(rankdata(a10_scores), rankdata(comparator_scores)),
        "mean_absolute_score_difference": float(
            np.mean(np.abs(a10_scores - comparator_scores))
        ),
        "threshold_disagreement_residues": int(
            np.sum(a10_predicted != comparator_predicted)
        ),
        "threshold_disagreement_rate": float(
            np.mean(a10_predicted != comparator_predicted)
        ),
        "both_correct": both_correct,
        "a10_only_correct": a10_only,
        "comparator_only_correct": comparator_only,
        "both_wrong": both_wrong,
        "both_correct_rate": both_correct / total,
        "a10_only_correct_rate": a10_only / total,
        "comparator_only_correct_rate": comparator_only / total,
        "both_wrong_rate": both_wrong / total,
        "either_model_correct_rate": (both_correct + a10_only + comparator_only) / total,
        "a10_net_correct_rate_advantage": (a10_only - comparator_only) / total,
        "error_overlap_jaccard": both_wrong / union_errors if union_errors else None,
    }


def complementarity_rows(
    dataset: str, track: str, proteins: list[AlignedProtein]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str, list[AlignedProtein]]] = [("overall", "all", proteins)]
    for stratification, classifier in (
        ("protein_length", lambda protein: length_bin(protein.sequence_length)),
        (
            "known_label_disorder_fraction",
            lambda protein: disorder_fraction_bin(protein.disorder_fraction),
        ),
    ):
        groups: dict[str, list[AlignedProtein]] = {}
        for protein in proteins:
            groups.setdefault(classifier(protein), []).append(protein)
        scopes.extend((stratification, key, value) for key, value in groups.items())

    a10_index = MODEL_ORDER.index("A10-locked")
    for scope, stratum, group in scopes:
        labels = np.concatenate([protein.labels for protein in group])
        a10_scores = np.concatenate([protein.scores[a10_index] for protein in group])
        for comparator in MODEL_ORDER[1:]:
            comparator_index = MODEL_ORDER.index(comparator)
            comparator_scores = np.concatenate(
                [protein.scores[comparator_index] for protein in group]
            )
            rows.append(
                {
                    "dataset": dataset,
                    "track": track,
                    "scope": scope,
                    "stratum": stratum,
                    "proteins": len(group),
                    "first_model": "A10-locked",
                    "second_model": comparator,
                    **pairwise_complementarity(labels, a10_scores, comparator_scores),
                }
            )
    return rows


def per_protein_rows(
    dataset: str, track: str, proteins: list[AlignedProtein]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for protein in proteins:
        for model_index, model in enumerate(MODEL_ORDER):
            rows.append(
                {
                    "dataset": dataset,
                    "track": track,
                    "protein_id": protein.protein_id,
                    "sequence_length": protein.sequence_length,
                    "known_label_disorder_fraction": protein.disorder_fraction,
                    "protein_length_bin": length_bin(protein.sequence_length),
                    "disorder_fraction_bin": disorder_fraction_bin(
                        protein.disorder_fraction
                    ),
                    "model": model,
                    **descriptive_metrics(protein.labels, protein.scores[model_index]),
                }
            )
    return rows


def robustness_rows(point_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_key = {
        (str(row["dataset"]), str(row["track"]), str(row["model"])): row
        for row in point_rows
    }
    for track in TRACK_ORDER:
        for model in MODEL_ORDER:
            caid2 = by_key[("CAID2", track, model)]
            caid3 = by_key[("CAID3", track, model)]
            for metric in (
                "roc_auc_full_precision",
                "aucpr_trapezoid_full_precision",
                "aps_full_precision",
                "f1_at_0_5_full_precision",
                "mcc_at_0_5_full_precision",
                "fmax_full_precision",
                "brier_score",
                "error_rate_at_0_5",
            ):
                first = float(caid2[metric])
                second = float(caid3[metric])
                rows.append(
                    {
                        "scope": "CAID3_minus_CAID2_same_track",
                        "track": track,
                        "model": model,
                        "metric": metric,
                        "caid2": first,
                        "caid3": second,
                        "delta": second - first,
                        "absolute_delta": abs(second - first),
                    }
                )
    for model in MODEL_ORDER:
        model_rows = [row for row in point_rows if row["model"] == model]
        for metric in (
            "roc_auc_full_precision",
            "aucpr_trapezoid_full_precision",
            "aps_full_precision",
            "f1_at_0_5_full_precision",
            "mcc_at_0_5_full_precision",
            "fmax_full_precision",
            "brier_score",
            "error_rate_at_0_5",
        ):
            values = [float(row[metric]) for row in model_rows]
            rows.append(
                {
                    "scope": "range_across_four_external_tracks",
                    "track": "all",
                    "model": model,
                    "metric": metric,
                    "minimum": min(values),
                    "maximum": max(values),
                    "range": max(values) - min(values),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(
    point_rows: list[dict[str, object]],
    complementarity: list[dict[str, object]],
) -> str:
    lines = [
        "# B7 locked-model error, robustness, and complementarity analysis",
        "",
        "B7 is descriptive and post-lock. B6 remains the confirmatory statistical analysis.",
        "No CAID labels were used to train, tune, select, or recalibrate A10.",
        "",
        "## Overall full-precision metrics",
        "",
        "| Dataset | Track | Model | ROC-AUC | AUPRC | APS | F1@0.5 | MCC@0.5 | Brier |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in point_rows:
        lines.append(
            "| {dataset} | {track} | {model} | {roc:.6f} | {auprc:.6f} | "
            "{aps:.6f} | {f1:.6f} | {mcc:.6f} | {brier:.6f} |".format(
                roc=row["roc_auc_full_precision"],
                auprc=row["aucpr_trapezoid_full_precision"],
                aps=row["aps_full_precision"],
                f1=row["f1_at_0_5_full_precision"],
                mcc=row["mcc_at_0_5_full_precision"],
                brier=row["brier_score"],
                **row,
            )
        )
    lines += [
        "",
        "## Overall threshold-error complementarity",
        "",
        "| Dataset | Track | Comparator | Disagreement | A10-only correct | Comparator-only correct | Either correct | Score Spearman |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in complementarity:
        if row["scope"] != "overall":
            continue
        spearman = row["score_spearman"]
        lines.append(
            "| {dataset} | {track} | {second_model} | {disagree:.4f} | "
            "{a10:.4f} | {other:.4f} | {either:.4f} | {spearman:.4f} |".format(
                disagree=row["threshold_disagreement_rate"],
                a10=row["a10_only_correct_rate"],
                other=row["comparator_only_correct_rate"],
                either=row["either_model_correct_rate"],
                spearman=float(spearman) if spearman is not None else float("nan"),
                **row,
            )
        )
    lines += [
        "",
        "## Interpretation constraints",
        "",
        "- Subgroup and complementarity results are descriptive, not new confirmatory tests.",
        "- The fixed 0.5 threshold is used only for error diagnostics.",
        "- Fmax is reported for characterization and is not used to modify the locked model.",
        "- Masked CAID labels are excluded; complete proteins remain the reporting unit where applicable.",
        "- Any future model change must be named and evaluated as a new model, never as A10-locked.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("b7_lock_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/analysis/b7_error_complementarity/formal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if manifest.get("status") != "locked":
        raise ValueError("B7 manifest is not locked")
    if manifest.get("model_selection_completed_before_caid_label_access") is not True:
        raise ValueError("model selection was not completed before label access")
    if manifest.get("caid_labels_used_for_training_or_tuning") is not False:
        raise ValueError("CAID labels were used for training or tuning")
    protocol = manifest.get("protocol", {})
    if protocol.get("analysis_type") != "descriptive_post_lock":
        raise ValueError("B7 must remain a descriptive post-lock analysis")
    if protocol.get("new_hypothesis_tests") is not False:
        raise ValueError("B7 cannot introduce new hypothesis tests")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_verification: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    protein_rows: list[dict[str, object]] = []
    subgroups: list[dict[str, object]] = []
    boundaries: list[dict[str, object]] = []
    segments: list[dict[str, object]] = []
    complementarity: list[dict[str, object]] = []
    alignment_reports: list[dict[str, object]] = []

    for dataset_name, dataset in manifest["datasets"].items():
        predictions: dict[str, dict[str, Prediction]] = {}
        for model in MODEL_ORDER:
            entry = dataset["predictions"][model]
            path = PROJECT_ROOT / entry["path"]
            lock_verification.append(
                {"dataset": dataset_name, "kind": "prediction", "model": model}
                | verify_locked_file(path, entry["sha256"])
            )
            predictions[model] = parse_prediction_tsv(path)

        for track in TRACK_ORDER:
            reference_entry = dataset["references"][track]
            reference_path = PROJECT_ROOT / reference_entry["path"]
            lock_verification.append(
                {"dataset": dataset_name, "kind": "reference", "track": track}
                | verify_locked_file(reference_path, reference_entry["sha256"])
            )
            reference = parse_reference(reference_path.read_text(encoding="utf-8-sig"))
            proteins, alignment = align_track(reference, predictions)
            alignment_reports.append(
                {"dataset": dataset_name, "track": track, **alignment}
            )
            for model in MODEL_ORDER:
                point_rows.append(model_metric_row(dataset_name, track, model, proteins))
            protein_rows.extend(per_protein_rows(dataset_name, track, proteins))
            subgroups.extend(subgroup_rows(dataset_name, track, proteins))
            boundaries.extend(boundary_rows(dataset_name, track, proteins))
            segments.extend(segment_rows(dataset_name, track, proteins))
            complementarity.extend(
                complementarity_rows(dataset_name, track, proteins)
            )
            print(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "track": track,
                        "proteins": len(proteins),
                        "known_residues": alignment["known_residues"],
                    }
                ),
                flush=True,
            )

    robustness = robustness_rows(point_rows)
    outputs = {
        "point_metrics": args.output_dir / "b7_point_metrics.csv",
        "per_protein_metrics": args.output_dir / "b7_per_protein_metrics.csv.gz",
        "subgroup_metrics": args.output_dir / "b7_subgroup_metrics.csv",
        "boundary_metrics": args.output_dir / "b7_boundary_metrics.csv",
        "segment_metrics": args.output_dir / "b7_segment_metrics.csv",
        "complementarity": args.output_dir / "b7_pairwise_complementarity.csv",
        "robustness": args.output_dir / "b7_cross_dataset_robustness.csv",
    }
    write_csv(outputs["point_metrics"], point_rows)
    # gzip is intentionally handled by csv through a temporary plain file-free path.
    import gzip

    columns: list[str] = []
    for row in protein_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with gzip.open(
        outputs["per_protein_metrics"], "wt", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(protein_rows)
    write_csv(outputs["subgroup_metrics"], subgroups)
    write_csv(outputs["boundary_metrics"], boundaries)
    write_csv(outputs["segment_metrics"], segments)
    write_csv(outputs["complementarity"], complementarity)
    write_csv(outputs["robustness"], robustness)

    summary_path = args.output_dir / "b7_summary.md"
    summary_path.write_text(
        markdown_summary(point_rows, complementarity), encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "experiment": "b7_error_robustness_and_complementarity",
        "status": "pass",
        "analysis_type": "descriptive_post_lock",
        "confirmatory_inference_source": "B6",
        "models": list(MODEL_ORDER),
        "datasets": list(manifest["datasets"]),
        "tracks": list(TRACK_ORDER),
        "protocol": protocol,
        "lock_manifest": str(args.manifest),
        "lock_manifest_sha256": sha256(args.manifest),
        "all_input_hashes_match": True,
        "lock_verification": lock_verification,
        "alignment": alignment_reports,
        "row_counts": {
            "point_metrics": len(point_rows),
            "per_protein_metrics": len(protein_rows),
            "subgroup_metrics": len(subgroups),
            "boundary_metrics": len(boundaries),
            "segment_metrics": len(segments),
            "pairwise_complementarity": len(complementarity),
            "cross_dataset_robustness": len(robustness),
        },
        "outputs": {key: str(path) for key, path in outputs.items()}
        | {"summary": str(summary_path)},
        "new_hypothesis_tests_performed": False,
        "model_selection_completed_before_caid_label_access": True,
        "caid_labels_used_for_training_or_tuning": False,
        "caid_labels_used_for_post_lock_descriptive_analysis": True,
    }
    report_path = args.output_dir / "b7_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                **report["row_counts"],
                "output_dir": str(args.output_dir),
                "new_hypothesis_tests_performed": False,
                "caid_labels_used_for_training_or_tuning": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
