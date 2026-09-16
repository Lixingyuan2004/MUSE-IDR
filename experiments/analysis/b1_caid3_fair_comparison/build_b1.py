"""Build the B1 CAID3 fair-comparison report from locked artifacts.

The script never trains or tunes a model. It reads the already locked A10
prediction, the CAID3 references, and the public CAID3 prediction archive. It
then recomputes all methods with one parser and one metric implementation.
CAID labels are used only for post-lock evaluation and statistical analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import auc, precision_recall_curve


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_A10_ARCHIVE = PROJECT_ROOT / "release/a10_caid3_final_results_20260825.tar.gz"
DEFAULT_PREDICTIONS_ZIP = (
    PROJECT_ROOT
    / "data/external/caid3_locked/official_predictions/caid3_predictions.zip"
)
DEFAULT_BASELINES = Path(__file__).with_name("published_baselines.json")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/analysis/b1_caid3_fair_comparison"

EXPECTED_A10_ARCHIVE_SHA256 = (
    "93b906a45339c918d00d4e670da4c5a792395fe4570b8b45c12b7e4da68f15c8"
)
EXPECTED_CAID3_PREDICTIONS_SHA256 = (
    "188868f53646d8395ef46d8d5e022c400e4234b21d38cfcf3ee91d719f19d84a"
)

TRACK_MEMBERS = {
    "disorder_nox": (
        "data/external/caid3_locked/reference_quarantine/"
        "caid3_disorder_nox_reference.fasta"
    ),
    "disorder_pdb": (
        "data/external/caid3_locked/reference_quarantine/"
        "caid3_disorder_pdb_reference.fasta"
    ),
}
A10_PREDICTION_MEMBER = (
    "outputs/final/a10/caid3_evaluation/official_format/a10_locked.caid"
)
A10_EVALUATION_MEMBER = (
    "outputs/final/a10/caid3_evaluation/a10_caid3_evaluation_report.json"
)
A10_INFERENCE_MEMBER = "outputs/final/a10/caid3_union_report.json"
A10_OFFICIAL_METRIC_MEMBERS = {
    "disorder_nox": {
        "default": (
            "outputs/final/a10/caid3_official/nox/"
            "caid3_disorder_nox_reference.analysis.all.dataset.default.metrics.csv"
        ),
        "f1s": (
            "outputs/final/a10/caid3_official/nox/"
            "caid3_disorder_nox_reference.analysis.all.dataset.f1s.metrics.csv"
        ),
    },
    "disorder_pdb": {
        "default": (
            "outputs/final/a10/caid3_official/pdb/"
            "caid3_disorder_pdb_reference.analysis.all.dataset.default.metrics.csv"
        ),
        "f1s": (
            "outputs/final/a10/caid3_official/pdb/"
            "caid3_disorder_pdb_reference.analysis.all.dataset.f1s.metrics.csv"
        ),
    },
}

ALLOWED_RESIDUE_NORMALIZATIONS = {
    ("U", "C"),
    ("U", "X"),
    ("O", "K"),
    ("O", "X"),
    ("B", "D"),
    ("B", "N"),
    ("B", "X"),
    ("Z", "E"),
    ("Z", "Q"),
    ("Z", "X"),
    ("J", "I"),
    ("J", "L"),
    ("J", "X"),
}

CORE_METHODS = {
    "disorder_nox": (
        "A10-locked",
        "ESMDisPred-2PDB",
        "ESMDisPred-1",
        "ESMDisPred-2",
        "DisoFLAG-IDR",
        "DisorderUnetLM",
        "flDPnn3a",
        "UdonPred-combined",
        "DisPredict3",
        "rawMSA",
        "rawMSA-disorder",
        "PUNCH2",
        "PUNCH2-Light",
    ),
    "disorder_pdb": (
        "A10-locked",
        "PUNCH2",
        "PUNCH2-Light",
        "AlphaFold-rsa",
        "SPOT-Disorder2",
        "AlphaFold3-rsa",
        "LMDisorder",
        "ESMDisPred-2PDB",
        "PredIDR2-Seq-Art",
        "PredIDR2-Prof-Art",
        "PredIDR2-Prof-Rnd",
    ),
}


@dataclass(frozen=True)
class Reference:
    protein_id: str
    sequence: str
    labels: str


@dataclass(frozen=True)
class Prediction:
    protein_id: str
    sequence: str
    scores: np.ndarray


@dataclass(frozen=True)
class ProteinPair:
    protein_id: str
    labels: np.ndarray
    first_scores: np.ndarray
    second_scores: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_text(archive: tarfile.TarFile, member: str) -> str:
    handle = archive.extractfile(member)
    if handle is None:
        raise FileNotFoundError(f"missing archive member: {member}")
    return handle.read().decode("utf-8-sig")


def parse_official_metric_row(text: str, model_name: str = "a10_locked") -> dict[str, float]:
    rows = list(csv.DictReader(io.StringIO(text)))
    matches = [row for row in rows if row.get("") == model_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one official metric row for {model_name}")
    return {
        key: float(value)
        for key, value in matches[0].items()
        if key and value not in {None, ""}
    }


def parse_reference(text: str) -> dict[str, Reference]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError("reference must use three non-empty lines per protein")
    records: dict[str, Reference] = {}
    for index in range(0, len(lines), 3):
        header, sequence, labels = lines[index : index + 3]
        if not header.startswith(">"):
            raise ValueError(f"invalid reference header: {header!r}")
        protein_id = header[1:].split()[0]
        sequence = sequence.upper()
        if not protein_id or protein_id in records:
            raise ValueError(f"invalid or duplicate reference id: {protein_id!r}")
        if len(sequence) != len(labels) or set(labels) - {"0", "1", "-"}:
            raise ValueError(f"invalid sequence/label alignment: {protein_id}")
        records[protein_id] = Reference(protein_id, sequence, labels)
    if not records:
        raise ValueError("reference is empty")
    return records


def parse_caid_predictions(text: str) -> dict[str, Prediction]:
    residues: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    position_origins: dict[str, int] = {}
    current: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0]
            if not current or current in residues:
                raise ValueError(f"invalid or duplicate prediction id at line {line_number}")
            residues[current] = []
            scores[current] = []
            continue
        if current is None:
            raise ValueError(f"prediction row before header at line {line_number}")
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"prediction row needs position/residue/score: line {line_number}")
        position = int(fields[0])
        residue = fields[1].upper()
        score = float(fields[2])
        if not residues[current]:
            if position not in {0, 1}:
                raise ValueError(
                    f"prediction positions must start at 0 or 1 for {current}: "
                    f"line {line_number}"
                )
            position_origins[current] = position
        expected_position = position_origins[current] + len(residues[current])
        if position != expected_position:
            raise ValueError(f"non-contiguous positions for {current} at line {line_number}")
        if len(residue) != 1 or not math.isfinite(score):
            raise ValueError(f"invalid prediction value at line {line_number}")
        residues[current].append(residue)
        scores[current].append(score)
    if not residues:
        raise ValueError("prediction file is empty")
    return {
        protein_id: Prediction(
            protein_id,
            "".join(residues[protein_id]),
            np.asarray(scores[protein_id], dtype=np.float64),
        )
        for protein_id in residues
    }


def binary_counts(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(y_score, kind="mergesort")[::-1]
    ordered_scores = y_score[order]
    ordered_true = y_true[order] == 1
    distinct = np.where(np.diff(ordered_scores))[0]
    threshold_indices = np.r_[distinct, ordered_true.size - 1]
    true_positives = np.cumsum(ordered_true, dtype=np.float64)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    thresholds = ordered_scores[threshold_indices]
    return false_positives, true_positives, thresholds


def binary_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    false_positives, true_positives, _ = binary_counts(y_true, y_score)
    tpr = np.r_[0.0, true_positives / true_positives[-1]]
    fpr = np.r_[0.0, false_positives / false_positives[-1]]
    return fpr, tpr


def roc_auc_full(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = binary_curve(y_true, y_score)
    return float(np.trapezoid(tpr, fpr))


def roc_auc_caid(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = binary_curve(y_true, np.round(y_score, 3))
    return float(np.round(np.trapezoid(np.round(tpr, 3), np.round(fpr, 3)), 3))


def caid_curve_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    """Reproduce BioComputingUP/CAID vectorized_metrics at revision 7bab7e8."""
    caid_scores = np.round(y_score, 3)
    false_positives, true_positives, thresholds = binary_counts(y_true, caid_scores)
    precision = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros_like(true_positives),
        where=(true_positives + false_positives) != 0,
    )
    recall = true_positives / true_positives[-1]
    caid_precision = np.round(np.r_[1.0, precision], 3)
    caid_recall = np.round(np.r_[0.0, recall], 3)
    caid_thresholds = np.round(np.r_[thresholds[0] + 1.0, thresholds], 3)
    aucpr = float(np.round(np.trapezoid(caid_precision, caid_recall), 3))
    aps = float(
        np.round(
            -np.sum(
                np.diff(caid_recall[::-1]) * caid_precision[::-1][:-1]
            ),
            3,
        )
    )
    f1_by_threshold = np.divide(
        2.0 * caid_precision[1:] * caid_recall[1:],
        caid_precision[1:] + caid_recall[1:],
        out=np.zeros_like(caid_precision[1:]),
        where=(caid_precision[1:] + caid_recall[1:]) != 0,
    )
    caid_f1_by_threshold = np.round(f1_by_threshold, 3)
    best_index = int(np.argmax(caid_f1_by_threshold))
    best_confusion = confusion_metrics(
        y_true, caid_scores, float(thresholds[best_index])
    )
    return {
        "aucpr_trapezoid": aucpr,
        "aps": aps,
        "fmax": float(caid_f1_by_threshold[best_index]),
        "mcc_at_fmax": float(np.round(best_confusion["mcc"], 3)),
        "fmax_threshold": float(caid_thresholds[best_index + 1]),
    }


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score, kind="mergesort")[::-1]
    ordered_scores = y_score[order]
    ordered_true = y_true[order] == 1
    distinct = np.where(np.diff(ordered_scores))[0]
    indices = np.r_[distinct, ordered_true.size - 1]
    tp = np.cumsum(ordered_true, dtype=np.float64)[indices]
    fp = 1 + indices - tp
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def confusion_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predicted = y_score >= threshold
    positive = y_true == 1
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "mcc": float(mcc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def fmax_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1))
    result = confusion_metrics(y_true, y_score, float(thresholds[best_index]))
    result["fmax"] = float(f1[best_index])
    return result


def align_method(
    reference: dict[str, Reference], predictions: dict[str, Prediction]
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int | float]]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    per_protein: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    predicted_residues = 0
    normalized_residue_mismatches = 0
    substantive_residue_mismatches = 0
    proteins_with_substantive_mismatches: set[str] = set()
    total_residues = sum(len(record.sequence) for record in reference.values())
    for protein_id, record in reference.items():
        prediction = predictions.get(protein_id)
        if prediction is None:
            continue
        if len(prediction.sequence) != len(record.sequence):
            raise ValueError(f"prediction/reference sequence length mismatch: {protein_id}")
        residue_differences = [
            (expected, observed)
            for expected, observed in zip(record.sequence, prediction.sequence)
            if expected != observed
        ]
        unsupported = [
            pair for pair in residue_differences
            if pair not in ALLOWED_RESIDUE_NORMALIZATIONS
        ]
        normalized_residue_mismatches += len(residue_differences) - len(unsupported)
        substantive_residue_mismatches += len(unsupported)
        if unsupported:
            proteins_with_substantive_mismatches.add(protein_id)
        known = np.fromiter((label != "-" for label in record.labels), dtype=bool)
        y_true = np.fromiter(
            (int(label) for label in record.labels if label != "-"), dtype=np.int8
        )
        y_score = prediction.scores[known]
        if y_true.size and np.unique(y_true).size:
            per_protein[protein_id] = (y_true, y_score)
            labels.append(y_true)
            scores.append(y_score)
        predicted_residues += len(record.sequence)
    if not labels:
        raise ValueError("method has no aligned known labels")
    y_true_all = np.concatenate(labels)
    y_score_all = np.concatenate(scores)
    if np.unique(y_true_all).tolist() != [0, 1]:
        raise ValueError("method evaluation requires both classes")
    summary: dict[str, int | float] = {
        "reference_proteins": len(reference),
        "predicted_proteins": len(per_protein),
        "protein_coverage": len(per_protein) / len(reference),
        "reference_residues": total_residues,
        "predicted_residues": predicted_residues,
        "residue_coverage": predicted_residues / total_residues,
        "known_residues": int(y_true_all.size),
        "positive_labels": int(y_true_all.sum()),
        "negative_labels": int(y_true_all.size - y_true_all.sum()),
        "scores_outside_unit_interval": int(
            np.sum((y_score_all < 0.0) | (y_score_all > 1.0))
        ),
        "normalized_residue_mismatches": normalized_residue_mismatches,
        "substantive_residue_mismatches": substantive_residue_mismatches,
        "proteins_with_substantive_mismatches": len(
            proteins_with_substantive_mismatches
        ),
    }
    return y_true_all, y_score_all, per_protein, summary


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    full_precision_default = confusion_metrics(y_true, y_score, 0.5)
    caid_default = confusion_metrics(y_true, np.round(y_score, 3), 0.5)
    full_precision_optimum = fmax_metrics(y_true, y_score)
    caid = caid_curve_metrics(y_true, y_score)
    return {
        "roc_auc_caid_compatible": roc_auc_caid(y_true, y_score),
        "roc_auc_full_precision": roc_auc_full(y_true, y_score),
        "aucpr_trapezoid": caid["aucpr_trapezoid"],
        "aps": caid["aps"],
        "f1_at_0_5": float(np.round(caid_default["f1"], 3)),
        "mcc_at_0_5": float(np.round(caid_default["mcc"], 3)),
        "fmax": caid["fmax"],
        "mcc_at_fmax": caid["mcc_at_fmax"],
        "fmax_threshold": caid["fmax_threshold"],
        "aucpr_trapezoid_full_precision": float(auc(recall, precision)),
        "aps_full_precision": average_precision(y_true, y_score),
        "f1_at_0_5_full_precision": full_precision_default["f1"],
        "mcc_at_0_5_full_precision": full_precision_default["mcc"],
        "fmax_full_precision": full_precision_optimum["fmax"],
        "mcc_at_fmax_full_precision": full_precision_optimum["mcc"],
        "fmax_threshold_full_precision": full_precision_optimum["threshold"],
    }


def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    result = np.empty(values.size, dtype=np.float64)
    result[order] = ranks
    return result


def fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    methods, total = predictions.shape
    negative_count = total - positive_count
    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    tx = np.empty((methods, positive_count), dtype=np.float64)
    ty = np.empty((methods, negative_count), dtype=np.float64)
    tz = np.empty((methods, total), dtype=np.float64)
    for method in range(methods):
        tx[method] = compute_midrank(positive[method])
        ty[method] = compute_midrank(negative[method])
        tz[method] = compute_midrank(predictions[method])
    aucs = tz[:, :positive_count].sum(axis=1) / positive_count / negative_count
    aucs -= (positive_count + 1.0) / (2.0 * negative_count)
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    covariance = np.cov(v01) / positive_count + np.cov(v10) / negative_count
    return aucs, np.atleast_2d(covariance)


def paired_delong(
    y_true: np.ndarray, first_scores: np.ndarray, second_scores: np.ndarray
) -> dict[str, float]:
    order = np.argsort(-y_true, kind="stable")
    predictions = np.vstack((first_scores[order], second_scores[order]))
    aucs, covariance = fast_delong(predictions, int(y_true.sum()))
    contrast = np.asarray([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    delta = float(aucs[0] - aucs[1])
    if variance <= 0.0:
        z_score = math.inf if delta else 0.0
        p_value = 0.0 if delta else 1.0
    else:
        z_score = abs(delta) / math.sqrt(variance)
        p_value = math.erfc(z_score / math.sqrt(2.0))
    return {
        "first_auc": float(aucs[0]),
        "second_auc": float(aucs[1]),
        "delta_first_minus_second": delta,
        "variance": variance,
        "z_score_absolute": float(z_score),
        "two_sided_p_value": float(p_value),
    }


def make_common_pairs(
    first: dict[str, tuple[np.ndarray, np.ndarray]],
    second: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[ProteinPair]:
    pairs: list[ProteinPair] = []
    for protein_id in sorted(set(first) & set(second)):
        first_labels, first_scores = first[protein_id]
        second_labels, second_scores = second[protein_id]
        if not np.array_equal(first_labels, second_labels):
            raise ValueError(f"paired labels do not align: {protein_id}")
        pairs.append(ProteinPair(protein_id, first_labels, first_scores, second_scores))
    if not pairs:
        raise ValueError("no common proteins for paired comparison")
    return pairs


def flatten_pairs(pairs: Iterable[ProteinPair]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = list(pairs)
    return (
        np.concatenate([pair.labels for pair in pairs]),
        np.concatenate([pair.first_scores for pair in pairs]),
        np.concatenate([pair.second_scores for pair in pairs]),
    )


def paired_protein_bootstrap(
    pairs: list[ProteinPair], replicates: int, random_seed: int
) -> dict[str, object]:
    labels, first_scores, second_scores = flatten_pairs(pairs)
    point_delta = roc_auc_full(labels, first_scores) - roc_auc_full(labels, second_scores)
    if replicates <= 0:
        return {
            "replicates": 0,
            "point_delta": point_delta,
            "samples": [],
        }
    rng = np.random.default_rng(random_seed)
    samples: list[float] = []
    size = len(pairs)
    for _ in range(replicates):
        selected = rng.integers(0, size, size=size)
        sampled = [pairs[int(index)] for index in selected]
        sampled_labels, sampled_first, sampled_second = flatten_pairs(sampled)
        if np.unique(sampled_labels).size < 2:
            continue
        samples.append(
            roc_auc_full(sampled_labels, sampled_first)
            - roc_auc_full(sampled_labels, sampled_second)
        )
    if len(samples) != replicates:
        raise RuntimeError("a bootstrap replicate lost one class")
    values = np.asarray(samples, dtype=np.float64)
    nonpositive = int(np.sum(values <= 0.0))
    nonnegative = int(np.sum(values >= 0.0))
    p_delta_le_zero = (nonpositive + 1) / (replicates + 1)
    p_delta_ge_zero = (nonnegative + 1) / (replicates + 1)
    return {
        "sampling_unit": "protein",
        "paired": True,
        "replicates": replicates,
        "random_seed": random_seed,
        "point_delta": float(point_delta),
        "mean_delta": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "percentile_95_ci": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "probability_delta_positive": float(np.mean(values > 0.0)),
        "one_sided_p_delta_le_zero_plus_one_correction": p_delta_le_zero,
        "one_sided_p_delta_ge_zero_plus_one_correction": p_delta_ge_zero,
        "two_sided_empirical_p_plus_one_correction": min(
            1.0, 2.0 * min(p_delta_le_zero, p_delta_ge_zero)
        ),
        "nonpositive_replicates": nonpositive,
        "nonnegative_replicates": nonnegative,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def fmt(value: object) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.3f}"


def build_markdown(
    core_rows: list[dict[str, object]],
    literature_rows: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    inference: dict[str, object],
    rank_context: dict[str, dict[str, int | str]],
) -> str:
    lines = [
        "# B1 CAID3 fair comparison",
        "",
        "A10 was locked before CAID3 labels were accessed. This document is post-lock evaluation only; it must not be used to retune A10.",
        "",
        "The main ROC-AUC/AUPRC/APS/F1/MCC/Fmax columns exactly reproduce the locked CAID metric implementation (revision 7bab7e8). Full-precision diagnostic values remain in b1_report.json.",
        "",
        "Rank context:",
        (
            f"- NOX: raw prediction archive recomputation rank "
            f"{rank_context['disorder_nox']['raw_prediction_archive_recomputation_rank']}; "
            f"published CAID3 table insertion rank "
            f"{rank_context['disorder_nox']['published_table_insertion_rank']}."
        ),
        (
            f"- PDB: raw prediction archive recomputation rank "
            f"{rank_context['disorder_pdb']['raw_prediction_archive_recomputation_rank']}; "
            f"published CAID3 table insertion rank "
            f"{rank_context['disorder_pdb']['published_table_insertion_rank']}."
        ),
        "",
    ]
    for track in ("disorder_nox", "disorder_pdb"):
        lines += [
            f"## {track}",
            "",
            "| Rank | Method | ROC-AUC | AUPRC | APS | F1@0.5 | MCC@0.5 | Fmax | Coverage |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in [item for item in core_rows if item["track"] == track]:
            lines.append(
                f"| {row['rank_by_roc_auc']} | {row['method']} | "
                f"{fmt(row['roc_auc_caid_compatible'])} | {fmt(row['aucpr_trapezoid'])} | "
                f"{fmt(row['aps'])} | {fmt(row['f1_at_0_5'])} | {fmt(row['mcc_at_0_5'])} | "
                f"{fmt(row['fmax'])} | {100.0 * float(row['protein_coverage']):.1f}% |"
            )
        lines.append("")
    lines += [
        "## Paired statistics",
        "",
        "| Track | First | Second | Common proteins | AUC delta | DeLong p (two-sided) | Protein bootstrap 95% CI |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in pairwise:
        bootstrap = row.get("protein_bootstrap")
        if isinstance(bootstrap, dict) and "percentile_95_ci" in bootstrap:
            ci = bootstrap["percentile_95_ci"]
            ci_text = f"[{float(ci[0]):+.4f}, {float(ci[1]):+.4f}]"
        else:
            ci_text = "not run"
        delong = row["delong"]
        lines.append(
            f"| {row['track']} | {row['first']} | {row['second']} | "
            f"{row['common_proteins']} | {float(delong['delta_first_minus_second']):+.4f} | "
            f"{float(delong['two_sided_p_value']):.4g} | {ci_text} |"
        )
    lines += [
        "",
        "## LoRA-DR-Suite literature reference (not the same reference set)",
        "",
        "The LoRA-DR-Suite paper reports a 148-sequence CAID3_NOX set, whereas the locked official reference used here contains 204 proteins. These rows are contextual only and are excluded from the formal ranking.",
        "",
        "| Method | ROC-AUC | PR-AUC | F1 | MCC | Fmax |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in literature_rows:
        lines.append(
            f"| {row['method']} | {fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} | "
            f"{fmt(row['f1'])} | {fmt(row['mcc'])} | {fmt(row['fmax'])} |"
        )
    lines += [
        "",
        "## Efficiency",
        "",
        f"- A10 inference: {float(inference['elapsed_seconds']):.3f} seconds for {inference['proteins']} proteins / {inference['residues']} residues on {inference['device_name']}.",
        f"- Peak CUDA memory: {int(inference['peak_cuda_memory_bytes']) / 1024 ** 3:.3f} GiB.",
        "- A10 output remains one classic-IDR probability per residue; no soft-disorder output.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a10-archive", type=Path, default=DEFAULT_A10_ARCHIVE)
    parser.add_argument("--official-predictions", type=Path, default=DEFAULT_PREDICTIONS_ZIP)
    parser.add_argument("--published-baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")
    archive_hash = sha256(args.a10_archive)
    predictions_hash = sha256(args.official_predictions)
    if archive_hash != EXPECTED_A10_ARCHIVE_SHA256:
        raise ValueError(f"unexpected A10 archive SHA256: {archive_hash}")
    if predictions_hash != EXPECTED_CAID3_PREDICTIONS_SHA256:
        raise ValueError(f"unexpected CAID3 predictions SHA256: {predictions_hash}")
    published = json.loads(args.published_baselines.read_text(encoding="utf-8"))

    with tarfile.open(args.a10_archive, "r:gz") as a10_archive:
        references = {
            track: parse_reference(archive_text(a10_archive, member))
            for track, member in TRACK_MEMBERS.items()
        }
        a10_predictions = parse_caid_predictions(
            archive_text(a10_archive, A10_PREDICTION_MEMBER)
        )
        locked_evaluation = json.loads(
            archive_text(a10_archive, A10_EVALUATION_MEMBER)
        )
        inference = json.loads(archive_text(a10_archive, A10_INFERENCE_MEMBER))
        official_a10_metrics = {
            track: {
                threshold_name: parse_official_metric_row(
                    archive_text(a10_archive, member)
                )
                for threshold_name, member in members.items()
            }
            for track, members in A10_OFFICIAL_METRIC_MEMBERS.items()
        }

    prediction_sets: dict[str, dict[str, Prediction]] = {"A10-locked": a10_predictions}
    required = sorted(set(CORE_METHODS["disorder_nox"]) | set(CORE_METHODS["disorder_pdb"]))
    with zipfile.ZipFile(args.official_predictions) as official_zip:
        names = set(official_zip.namelist())
        for method in required:
            if method == "A10-locked":
                continue
            member = f"predictions/merged/{method}.caid"
            if member not in names:
                raise FileNotFoundError(f"official method missing from prediction zip: {method}")
            prediction_sets[method] = parse_caid_predictions(
                official_zip.read(member).decode("utf-8-sig")
            )

    rows: list[dict[str, object]] = []
    per_protein: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
    for track, methods in CORE_METHODS.items():
        per_protein[track] = {}
        for method in methods:
            try:
                y_true, y_score, proteins, coverage = align_method(
                    references[track], prediction_sets[method]
                )
            except ValueError as error:
                raise ValueError(f"{track}/{method}: {error}") from error
            metrics = compute_metrics(y_true, y_score)
            per_protein[track][method] = proteins
            rows.append(
                {
                    "track": track,
                    "method": method,
                    "source": "locked_a10" if method == "A10-locked" else "official_caid3_predictions",
                    **coverage,
                    **metrics,
                }
            )

    for track in TRACK_MEMBERS:
        ranked = sorted(
            [row for row in rows if row["track"] == track],
            key=lambda row: (-float(row["roc_auc_caid_compatible"]), str(row["method"])),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank_by_roc_auc"] = rank

    a10_rank_context: dict[str, dict[str, int | str]] = {}
    for track in TRACK_MEMBERS:
        a10_row = next(
            row for row in rows if row["track"] == track and row["method"] == "A10-locked"
        )
        a10_auc = float(a10_row["roc_auc_caid_compatible"])
        published_auc_values = [
            float(item["roc_auc"])
            for item in published["official_caid3_tables"][track]
        ]
        a10_rank_context[track] = {
            "raw_prediction_archive_recomputation_rank": int(
                a10_row["rank_by_roc_auc"]
            ),
            "published_table_insertion_rank": 1
            + sum(value > a10_auc for value in published_auc_values),
            "published_rank_definition": (
                "1 + number of methods in the published CAID3 table with "
                "strictly higher ROC-AUC"
            ),
        }

    # B1 must exactly reproduce the locked official CAID outputs before comparing methods.
    for track in TRACK_MEMBERS:
        a10_row = next(
            row for row in rows if row["track"] == track and row["method"] == "A10-locked"
        )
        expected_default = official_a10_metrics[track]["default"]
        expected_f1s = official_a10_metrics[track]["f1s"]
        invariants = {
            "roc_auc_caid_compatible": expected_default["aucroc"],
            "aucpr_trapezoid": expected_default["aucpr"],
            "aps": expected_default["aps"],
            "f1_at_0_5": expected_default["f1s"],
            "mcc_at_0_5": expected_default["mcc"],
            "fmax": expected_f1s["f1s"],
            "mcc_at_fmax": expected_f1s["mcc"],
            "fmax_threshold": expected_f1s["thr"],
        }
        locked_auc = float(
            locked_evaluation["tracks"][track]["auc_roc_caid_compatible"]
        )
        if locked_auc != expected_default["aucroc"]:
            raise RuntimeError(f"locked/official A10 AUC disagreement for {track}")
        for metric_name, expected_value in invariants.items():
            computed_value = float(a10_row[metric_name])
            if abs(computed_value - float(expected_value)) > 1e-12:
                raise RuntimeError(
                    f"A10 official invariant failed for {track}/{metric_name}: "
                    f"computed={computed_value}, official={expected_value}"
                )

    core_rows = sorted(rows, key=lambda row: (str(row["track"]), int(row["rank_by_roc_auc"])))

    published_checks: list[dict[str, object]] = []
    for track, published_rows in published["official_caid3_tables"].items():
        computed_by_method = {
            str(row["method"]): row for row in rows if row["track"] == track
        }
        for item in published_rows:
            computed = computed_by_method[str(item["method"])]
            published_checks.append(
                {
                    "track": track,
                    "method": item["method"],
                    "published_roc_auc": item["roc_auc"],
                    "recomputed_roc_auc": computed["roc_auc_caid_compatible"],
                    "auc_delta_recomputed_minus_published": (
                        float(computed["roc_auc_caid_compatible"])
                        - float(item["roc_auc"])
                    ),
                    "published_aps": item["aps"],
                    "recomputed_aps": computed["aps"],
                    "published_coverage": item["coverage"],
                    "recomputed_protein_coverage": computed["protein_coverage"],
                }
            )

    comparisons = {
        "disorder_nox": ("ESMDisPred-2PDB", "PUNCH2", "PUNCH2-Light"),
        "disorder_pdb": ("PUNCH2", "PUNCH2-Light"),
    }
    bootstrap_targets = {
        (track, method)
        for track, methods in comparisons.items()
        for method in methods
    }
    pairwise: list[dict[str, object]] = []
    for comparison_index, (track, methods) in enumerate(comparisons.items()):
        for method_index, method in enumerate(methods):
            pairs = make_common_pairs(
                per_protein[track]["A10-locked"], per_protein[track][method]
            )
            labels, first_scores, second_scores = flatten_pairs(pairs)
            result: dict[str, object] = {
                "track": track,
                "first": "A10-locked",
                "second": method,
                "common_proteins": len(pairs),
                "common_known_residues": int(labels.size),
                "delong": paired_delong(labels, first_scores, second_scores),
                "protein_bootstrap": None,
            }
            if (track, method) in bootstrap_targets:
                seed = args.random_seed + 100 * comparison_index + method_index
                result["protein_bootstrap"] = paired_protein_bootstrap(
                    pairs, args.bootstrap_replicates, seed
                )
            pairwise.append(result)

    literature_rows = list(published["lora_dr_suite_reference_only"]["rows"])
    report = {
        "schema_version": 1,
        "experiment": "b1_caid3_fair_comparison",
        "status": "pass",
        "objective": "ROC-AUC primary; AUPRC/APS/F1/MCC secondary",
        "metric_protocol": {
            "official_compatible_columns": [
                "roc_auc_caid_compatible",
                "aucpr_trapezoid",
                "aps",
                "f1_at_0_5",
                "mcc_at_0_5",
                "fmax",
                "mcc_at_fmax",
                "fmax_threshold",
            ],
            "official_caid_revision": "7bab7e8880d7f949e480fdb377c5a5c8546ae336",
            "full_precision_values_are_diagnostic_only": True,
        },
        "a10_locked_before_caid_label_access": True,
        "caid_labels_used_for_training_or_tuning": False,
        "caid_labels_used_for_post_lock_evaluation": True,
        "a10_archive": str(args.a10_archive),
        "a10_archive_sha256": archive_hash,
        "official_predictions": str(args.official_predictions),
        "official_predictions_sha256": predictions_hash,
        "official_caid_revision": locked_evaluation["official_caid_revision"],
        "metrics": core_rows,
        "a10_rank_context": a10_rank_context,
        "published_recomputation_checks": published_checks,
        "pairwise_statistics": pairwise,
        "lora_dr_suite_reference_only": published["lora_dr_suite_reference_only"],
        "a10_efficiency": {
            key: inference[key]
            for key in (
                "proteins",
                "residues",
                "elapsed_seconds",
                "peak_cuda_memory_bytes",
                "device_name",
                "precision",
            )
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "b1_report.json", report)
    write_csv(args.output_dir / "b1_core_rankings.csv", core_rows)
    write_csv(args.output_dir / "b1_published_recomputation_checks.csv", published_checks)
    write_csv(
        args.output_dir / "b1_lora_dr_suite_reference_only.csv", literature_rows
    )
    (args.output_dir / "b1_summary.md").write_text(
        build_markdown(
            core_rows, literature_rows, pairwise, inference, a10_rank_context
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "output_dir": str(args.output_dir),
                "a10_ranks": {
                    track: next(
                        int(row["rank_by_roc_auc"])
                        for row in core_rows
                        if row["track"] == track and row["method"] == "A10-locked"
                    )
                    for track in TRACK_MEMBERS
                },
                "pairwise_comparisons": len(pairwise),
                "bootstrap_replicates_per_selected_comparison": args.bootstrap_replicates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
