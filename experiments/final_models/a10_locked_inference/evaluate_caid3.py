"""Freeze A10 predictions and evaluate CAID3 NOX/PDB references.

This evaluator reproduces the CAID dataset-level ROC-AUC convention while
also reporting an unrounded full-precision ROC-AUC. Reference labels are used
only after the A10 model and prediction file have been locked.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.final_models.a10_locked_inference.prepare_caid_references import (
    ReferenceRecord,
    parse_reference,
)


OFFICIAL_CAID_REPOSITORY = "https://github.com/BioComputingUP/CAID"
OFFICIAL_CAID_REVISION = "7bab7e8880d7f949e480fdb377c5a5c8546ae336"


@dataclass(frozen=True)
class PredictionRow:
    protein_id: str
    position: int
    residue: str
    score: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def read_predictions(path: Path) -> dict[str, list[PredictionRow]]:
    rows_by_protein: dict[str, list[PredictionRow]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    with open_text(path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"protein_id", "position", "residue", "idr_probability"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"prediction columns must include {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            protein_id = str(row["protein_id"])
            position = int(row["position"])
            residue = str(row["residue"]).upper()
            score = float(row["idr_probability"])
            key = (protein_id, position)
            if not protein_id or position <= 0 or len(residue) != 1:
                raise ValueError(f"invalid prediction row at line {line_number}")
            if key in seen:
                raise ValueError(f"duplicate prediction position: {key}")
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"invalid prediction score at line {line_number}: {score}")
            seen.add(key)
            rows_by_protein[protein_id].append(
                PredictionRow(protein_id, position, residue, score)
            )
    if not rows_by_protein:
        raise ValueError("prediction file is empty")
    for protein_id, rows in rows_by_protein.items():
        rows.sort(key=lambda item: item.position)
        expected = list(range(1, len(rows) + 1))
        observed = [row.position for row in rows]
        if observed != expected:
            raise ValueError(f"non-contiguous prediction positions: {protein_id}")
    return dict(rows_by_protein)


def align_reference(
    records: Iterable[ReferenceRecord],
    predictions: dict[str, list[PredictionRow]],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    labels: list[int] = []
    scores: list[float] = []
    proteins = 0
    residues = 0
    excluded = 0
    for record in records:
        rows = predictions.get(record.protein_id)
        if rows is None:
            raise ValueError(f"missing prediction target: {record.protein_id}")
        predicted_sequence = "".join(row.residue for row in rows)
        if predicted_sequence != record.sequence:
            raise ValueError(f"prediction/reference sequence mismatch: {record.protein_id}")
        proteins += 1
        residues += len(record.sequence)
        for label, row in zip(record.labels, rows):
            if label == "-":
                excluded += 1
                continue
            labels.append(int(label))
            scores.append(row.score)
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).tolist() != [0, 1]:
        raise ValueError("evaluation requires known positive and negative labels")
    summary = {
        "proteins": proteins,
        "residues": residues,
        "known_residues": int(y_true.size),
        "positive_labels": int(y_true.sum()),
        "negative_labels": int(y_true.size - y_true.sum()),
        "excluded_labels": excluded,
        "prediction_alignment": True,
    }
    return y_true, y_score, summary


def binary_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(y_score, kind="mergesort")[::-1]
    ordered_scores = y_score[order]
    ordered_true = y_true[order] == 1
    distinct = np.where(np.diff(ordered_scores))[0]
    threshold_indices = np.r_[distinct, ordered_true.size - 1]
    true_positives = np.cumsum(ordered_true, dtype=np.float64)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    tpr = np.r_[0.0, true_positives / true_positives[-1]]
    fpr = np.r_[0.0, false_positives / false_positives[-1]]
    return fpr, tpr


def auc_roc_full_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = binary_curve(y_true, y_score)
    return float(np.trapezoid(tpr, fpr))


def auc_roc_caid_compatible(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # CAID's parser rounds scores to three decimals. Its roc() then rounds the
    # FPR/TPR coordinates to three decimals before trapezoidal integration.
    fpr, tpr = binary_curve(y_true, np.round(y_score, 3))
    fpr = np.round(fpr, 3)
    tpr = np.round(tpr, 3)
    return float(np.round(np.trapezoid(tpr, fpr), 3))


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


def write_caid_predictions(
    path: Path,
    records: Iterable[ReferenceRecord],
    predictions: dict[str, list[PredictionRow]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f">{record.protein_id}\n")
            for row in predictions[record.protein_id]:
                handle.write(f"{row.position}\t{row.residue}\t{row.score:.9g}\n")
    temporary.replace(path)


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def evaluate_track(
    name: str,
    reference_path: Path,
    predictions: dict[str, list[PredictionRow]],
) -> tuple[list[ReferenceRecord], dict[str, object]]:
    records = parse_reference(reference_path)
    y_true, y_score, summary = align_reference(records, predictions)
    summary.update(
        {
            "track": name,
            "reference": str(reference_path),
            "reference_sha256": file_sha256(reference_path),
            "auc_roc_caid_compatible": auc_roc_caid_compatible(y_true, y_score),
            "auc_roc_full_precision": auc_roc_full_precision(y_true, y_score),
            "average_precision_full_precision": average_precision(y_true, y_score),
            "caid_masked_labels_excluded": True,
        }
    )
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--nox-reference", type=Path, required=True)
    parser.add_argument("--pdb-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_report = json.loads(args.prediction_report.read_text(encoding="utf-8-sig"))
    prediction_sha256 = file_sha256(args.predictions)
    if inference_report.get("status") != "pass":
        raise ValueError("locked inference report did not pass")
    if inference_report.get("output_sha256") != prediction_sha256:
        raise ValueError("prediction file hash does not match locked inference report")
    if inference_report.get("scores_used_for_training_or_tuning") is not False:
        raise ValueError("prediction report does not prove a locked external test")

    predictions = read_predictions(args.predictions)
    nox_records, nox_metrics = evaluate_track("disorder_nox", args.nox_reference, predictions)
    pdb_records, pdb_metrics = evaluate_track("disorder_pdb", args.pdb_reference, predictions)
    nox_ids = {record.protein_id for record in nox_records}
    pdb_ids = {record.protein_id for record in pdb_records}
    if not nox_ids.issubset(pdb_ids):
        raise ValueError("CAID3 NOX targets are not a subset of PDB targets")
    if set(predictions) != pdb_ids:
        raise ValueError("prediction target set does not equal the CAID3 union/PDB target set")

    caid_path = args.output_dir / "official_format" / "a10_locked.caid"
    write_caid_predictions(caid_path, pdb_records, predictions)
    report = {
        "schema_version": 1,
        "experiment": "a10_locked_caid3_evaluation",
        "status": "pass",
        "model_selection_completed_before_label_access": True,
        "prediction_file": str(args.predictions),
        "prediction_file_sha256": prediction_sha256,
        "prediction_report": str(args.prediction_report),
        "prediction_report_sha256": file_sha256(args.prediction_report),
        "locked_members": inference_report["locked_members"],
        "ensemble_method": inference_report["ensemble_method"],
        "tracks": {"disorder_nox": nox_metrics, "disorder_pdb": pdb_metrics},
        "official_caid_prediction": str(caid_path),
        "official_caid_prediction_sha256": file_sha256(caid_path),
        "official_compatibility_note": (
            "Scores are rounded to 3 decimals and ROC coordinates are rounded to 3 decimals "
            "before integration, matching BioComputingUP/CAID vectorized_metrics."
        ),
        "official_caid_repository": OFFICIAL_CAID_REPOSITORY,
        "official_caid_revision": OFFICIAL_CAID_REVISION,
        "single_output_score": True,
        "soft_disorder_output": False,
        "caid_labels_used_for_training_or_tuning": False,
    }
    report_path = args.output_dir / "a10_caid3_evaluation_report.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
