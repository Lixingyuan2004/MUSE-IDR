"""Evaluate three locked predictors on the frozen S2 holdout.

All prediction, protocol and reference hashes are verified before semantic
label access. The analysis follows the protocol frozen before either baseline
was run and uses paired whole-protein bootstrap samples.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b1_caid3_fair_comparison.build_b1 import (  # noqa: E402
    compute_metrics,
    confusion_metrics,
)
from experiments.analysis.b6_paired_model_statistics.build_b6 import (  # noqa: E402
    METRIC_ORDER,
    bootstrap_summary,
    full_precision_metric_vector,
    holm_adjust,
)
from experiments.supplementary.s2_independent_holdout.evaluate_s2_a10 import (  # noqa: E402
    EXPECTED_COUNTS,
    extended_metrics,
    read_cohort,
    read_labels,
    verify_lock_manifest,
)


PROTOCOL = PROJECT_ROOT / (
    "experiments/supplementary/s2_independent_holdout/"
    "s2_baseline_comparison_protocol_20260907.json"
)
PROTOCOL_SHA256 = "a231176522a0bd1bd5a0566c74d9006ab9ea6a65100d58b371b617d3533413e2"
PROTOCOL_ERRATUM = PROJECT_ROOT / (
    "experiments/supplementary/s2_independent_holdout/"
    "s2_protocol_erratum_20260907.json"
)
PROTOCOL_ERRATUM_SHA256 = (
    "57a53fae2f8fce476cb694a39919bad553b4202dfcf91b81a5650cb4c13cfecc"
)
MALFORMED_PROTOCOL_LORA_WEIGHTS_SHA256 = (
    "02f1e288f36f369786db68df5270affa693d2359a5badb0275ce81977448fe"
)
LORA_ADAPTER_WEIGHTS_SHA256 = (
    "02f1e288f36f369786db68df5270affa693d2359a5badb0275c28e81977448fe"
)
REFERENCE_DIR = PROJECT_ROOT / (
    "outputs/supplementary/s2_independent_holdout/final_database_curated_20260906"
)
A10_DIR = PROJECT_ROOT / (
    "outputs/supplementary/s2_independent_holdout/a10_prediction_20260907"
)
PUNCH_DIR = PROJECT_ROOT / (
    "outputs/supplementary/s2_independent_holdout/"
    "punch2_light_released13_prediction_20260907"
)
LORA_DIR = PROJECT_ROOT / (
    "outputs/supplementary/s2_independent_holdout/"
    "lora_dr_suite_650m_disprot7_offline_prediction_20260907"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / (
    "outputs/supplementary/s2_independent_holdout/"
    "three_model_evaluation_20260907"
)

MODEL_ORDER = (
    "MUSE-IDR A10",
    "PUNCH2-Light Released-13",
    "LoRA-DR-Suite ESM2-650M DisProt7",
)
PREDICTION_PATHS = {
    MODEL_ORDER[0]: A10_DIR / "muse_idr_a10_s2_predictions.tsv.gz",
    MODEL_ORDER[1]: PUNCH_DIR
    / "predictions/punch2_light_released13_predictions.tsv.gz",
    MODEL_ORDER[2]: LORA_DIR / "lora_dr_suite_650m_disprot7_predictions.tsv.gz",
}
PREDICTION_SHA256 = {
    MODEL_ORDER[0]: "70555772455afb707200036cf93266236013d18accdce1c0289a3a8236ab541b",
    MODEL_ORDER[1]: "c041c1d4d43ca9cf0dd5a0d0f67388f9cfcf7e0bdc59cea52fcf2495f955b6c3",
    MODEL_ORDER[2]: "77365cf86829299674c82494879e01736be50b4b14190710278a1e11639a334f",
}
BASELINE_LOCKS = {
    MODEL_ORDER[1]: (
        PUNCH_DIR / "s2_baseline_prediction_lock.sha256",
        "55285b50e421237f21766df6cebd63d62d6f86ebbbdd1718588007ff91f5e0db",
    ),
    MODEL_ORDER[2]: (
        LORA_DIR / "s2_baseline_prediction_lock.sha256",
        "cebca1195196f431635635b6d10b5d7a4ff9bea6eb8e9cd8b3970baff040675c",
    ),
}
POINT_METRICS = (*METRIC_ORDER, "brier_score", "macro_roc_auc")
PRIMARY_METRIC = "roc_auc_full_precision"
SECONDARY_METRIC = "aucpr_trapezoid_full_precision"


@dataclass(frozen=True)
class MultiProteinBlock:
    protein_id: str
    length: int
    labels: np.ndarray
    scores: tuple[np.ndarray, ...]
    source_entry_release: str


_WORKER_BLOCKS: list[MultiProteinBlock] | None = None
_WORKER_SEED = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_exact_file(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return {"path": str(path), "sha256": actual, "match": True}


def read_lock_entries(path: Path, expected_lock_sha256: str) -> dict[str, str]:
    verify_exact_file(path, expected_lock_sha256)
    entries: dict[str, str] = {}
    for number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            digest, relative_text = raw_line.split("  ", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"invalid lock line {number}: {path}") from error
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe lock member: {relative_text}")
        key = relative.as_posix()
        if key in entries:
            raise ValueError(f"duplicate lock member: {key}")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"invalid SHA256 at lock line {number}: {path}")
        entries[key] = digest
    if not entries:
        raise ValueError(f"empty result lock: {path}")
    return entries


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def verify_selected_lock_members(
    entries: dict[str, str], paths: Sequence[Path]
) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for path in paths:
        relative = project_relative(path)
        if relative not in entries:
            raise ValueError(f"required result is absent from lock: {relative}")
        result = verify_exact_file(path, entries[relative])
        verified.append({"relative_path": relative, **result})
    return verified


def read_prediction(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    residues: dict[str, list[str]] = {}
    values: dict[str, list[float]] = {}
    origins: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        common = {"protein_id", "residue", "idr_probability"}
        if not common.issubset(fields):
            raise ValueError(f"unexpected prediction header: {fields}")
        position_key = "position" if "position" in fields else "residue_index"
        if position_key not in fields:
            raise ValueError(f"prediction has no position column: {path}")
        if set(fields) != common | {position_key}:
            raise ValueError(f"unexpected extra prediction columns: {fields}")
        for line_number, row in enumerate(reader, start=2):
            protein_id = str(row["protein_id"])
            residue = str(row["residue"]).upper()
            position = int(row[position_key])
            probability = float(row["idr_probability"])
            if not protein_id or len(residue) != 1:
                raise ValueError(f"invalid prediction record at {path}:{line_number}")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"invalid probability at {path}:{line_number}")
            if protein_id not in residues:
                if position not in (0, 1):
                    raise ValueError(
                        f"positions must start at zero or one: {path}:{line_number}"
                    )
                residues[protein_id] = []
                values[protein_id] = []
                origins[protein_id] = position
            expected_position = origins[protein_id] + len(residues[protein_id])
            if position != expected_position:
                raise ValueError(
                    f"non-contiguous prediction for {protein_id}: "
                    f"{position} != {expected_position}"
                )
            residues[protein_id].append(residue)
            values[protein_id].append(probability)
    if not residues:
        raise ValueError(f"empty prediction file: {path}")
    return {
        protein_id: (
            "".join(residues[protein_id]),
            np.asarray(values[protein_id], dtype=np.float64),
        )
        for protein_id in residues
    }


def align_three_models(
    labels: dict[str, tuple[str, np.ndarray]],
    predictions: dict[str, dict[str, tuple[str, np.ndarray]]],
    cohort: dict[str, dict[str, str]],
) -> list[MultiProteinBlock]:
    expected_ids = set(labels)
    if set(cohort) != expected_ids:
        raise ValueError("S2 cohort and label protein sets differ")
    for model in MODEL_ORDER:
        if set(predictions[model]) != expected_ids:
            raise ValueError(f"S2 prediction protein set differs: {model}")
    blocks: list[MultiProteinBlock] = []
    for protein_id in labels:
        sequence, full_labels = labels[protein_id]
        if int(cohort[protein_id]["length"]) != len(sequence):
            raise ValueError(f"cohort length mismatch: {protein_id}")
        known = full_labels != -1
        model_scores: list[np.ndarray] = []
        for model in MODEL_ORDER:
            predicted_sequence, scores = predictions[model][protein_id]
            if predicted_sequence != sequence:
                raise ValueError(f"prediction sequence mismatch: {model} {protein_id}")
            if scores.shape != full_labels.shape:
                raise ValueError(f"prediction length mismatch: {model} {protein_id}")
            model_scores.append(scores[known])
        blocks.append(
            MultiProteinBlock(
                protein_id=protein_id,
                length=len(sequence),
                labels=full_labels[known],
                scores=tuple(model_scores),
                source_entry_release=str(cohort[protein_id]["source_entry_release"]),
            )
        )
    return blocks


def flatten_blocks(
    blocks: Sequence[MultiProteinBlock], model_index: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate([block.labels for block in blocks])
    scores = np.concatenate([block.scores[model_index] for block in blocks])
    return labels, scores


def protein_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size != 2:
        return float("nan")
    return float(compute_metrics(labels, scores)["roc_auc_full_precision"])


def point_metric_rows(blocks: Sequence[MultiProteinBlock]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODEL_ORDER):
        labels, scores = flatten_blocks(blocks, model_index)
        metrics = extended_metrics(labels, scores)
        macro_values = np.asarray(
            [protein_roc_auc(block.labels, block.scores[model_index]) for block in blocks]
        )
        row = {
            "model": model,
            "proteins": len(blocks),
            "residues": sum(block.length for block in blocks),
            "known_residues": int(labels.size),
            "positive_residues": int(np.sum(labels == 1)),
            "negative_residues": int(np.sum(labels == 0)),
            **metrics,
            "macro_roc_auc": float(np.nanmean(macro_values)),
            "macro_roc_auc_proteins": int(np.sum(np.isfinite(macro_values))),
        }
        vector = full_precision_metric_vector(labels, scores)
        for index, metric in enumerate(METRIC_ORDER):
            if not np.isclose(float(row[metric]), vector[index], rtol=0.0, atol=1e-12):
                raise RuntimeError(f"metric implementation mismatch: {model} {metric}")
        if model == MODEL_ORDER[1]:
            official = confusion_metrics(labels, scores, 0.35)
            row["official_threshold"] = 0.35
            row["f1_at_official_threshold"] = float(official["f1"])
            row["mcc_at_official_threshold"] = float(official["mcc"])
        else:
            row["official_threshold"] = None
            row["f1_at_official_threshold"] = None
            row["mcc_at_official_threshold"] = None
        rows.append(row)
    return rows


def per_protein_rows(blocks: Sequence[MultiProteinBlock]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        positives = int(np.sum(block.labels == 1))
        negatives = int(np.sum(block.labels == 0))
        for model_index, model in enumerate(MODEL_ORDER):
            scores = block.scores[model_index]
            default = confusion_metrics(block.labels, scores, 0.5)
            rows.append(
                {
                    "protein_id": block.protein_id,
                    "source_entry_release": block.source_entry_release,
                    "model": model,
                    "length": block.length,
                    "known_residues": int(block.labels.size),
                    "positive_residues": positives,
                    "negative_residues": negatives,
                    "roc_auc_full_precision": protein_roc_auc(block.labels, scores)
                    if positives and negatives
                    else None,
                    "f1_at_0_5_full_precision": float(default["f1"]),
                    "mcc_at_0_5_full_precision": float(default["mcc"]),
                    "brier_score": float(np.mean((scores - block.labels) ** 2)),
                }
            )
    return rows


def _init_worker(blocks: list[MultiProteinBlock], seed: int) -> None:
    global _WORKER_BLOCKS, _WORKER_SEED
    _WORKER_BLOCKS = blocks
    _WORKER_SEED = seed


def _bootstrap_chunk(start: int, count: int) -> tuple[int, np.ndarray]:
    if _WORKER_BLOCKS is None:
        raise RuntimeError("bootstrap worker not initialized")
    blocks = _WORKER_BLOCKS
    macro_values = np.asarray(
        [
            [protein_roc_auc(block.labels, scores) for scores in block.scores]
            for block in blocks
        ],
        dtype=np.float64,
    )
    output = np.empty((count, len(MODEL_ORDER), len(POINT_METRICS)), dtype=np.float64)
    for offset in range(count):
        replicate = start + offset
        rng = np.random.default_rng(np.random.SeedSequence([_WORKER_SEED, replicate]))
        selected = rng.integers(0, len(blocks), size=len(blocks))
        labels = np.concatenate([blocks[int(index)].labels for index in selected])
        if np.unique(labels).size != 2:
            raise RuntimeError(f"bootstrap replicate {replicate} lost one class")
        for model_index in range(len(MODEL_ORDER)):
            scores = np.concatenate(
                [blocks[int(index)].scores[model_index] for index in selected]
            )
            output[offset, model_index, : len(METRIC_ORDER)] = (
                full_precision_metric_vector(labels, scores)
            )
            output[offset, model_index, len(METRIC_ORDER)] = float(
                np.mean((scores - labels) ** 2)
            )
            output[offset, model_index, len(METRIC_ORDER) + 1] = float(
                np.nanmean(macro_values[selected, model_index])
            )
    return start, output


def bootstrap_samples(
    blocks: list[MultiProteinBlock],
    replicates: int,
    workers: int,
    seed: int,
    progress_every: int,
) -> np.ndarray:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    workers = max(1, min(workers, replicates))
    chunk_size = max(1, min(25, math.ceil(replicates / (workers * 4))))
    chunks = [
        (start, min(chunk_size, replicates - start))
        for start in range(0, replicates, chunk_size)
    ]
    output = np.empty(
        (replicates, len(MODEL_ORDER), len(POINT_METRICS)), dtype=np.float64
    )
    completed = 0
    next_progress = progress_every
    if workers == 1:
        _init_worker(blocks, seed)
        for start, count in chunks:
            _, values = _bootstrap_chunk(start, count)
            output[start : start + count] = values
            completed += count
            if progress_every and completed >= next_progress:
                print(json.dumps({"bootstrap_completed": completed}), flush=True)
                next_progress += progress_every
        return output
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(blocks, seed)
    ) as executor:
        futures = {
            executor.submit(_bootstrap_chunk, start, count): (start, count)
            for start, count in chunks
        }
        for future in as_completed(futures):
            start, count = futures[future]
            returned_start, values = future.result()
            if returned_start != start or values.shape != (
                count,
                len(MODEL_ORDER),
                len(POINT_METRICS),
            ):
                raise RuntimeError("bootstrap worker returned malformed output")
            output[start : start + count] = values
            completed += count
            if progress_every and completed >= next_progress:
                print(json.dumps({"bootstrap_completed": completed}), flush=True)
                next_progress += progress_every
    if not np.all(np.isfinite(output)):
        raise RuntimeError("bootstrap output contains non-finite values")
    return output


def metric_summary_rows(
    point_rows: Sequence[dict[str, Any]], samples: np.ndarray
) -> list[dict[str, Any]]:
    if samples.shape[1:] != (len(MODEL_ORDER), len(POINT_METRICS)):
        raise ValueError("unexpected bootstrap sample shape")
    point_by_model = {str(row["model"]): row for row in point_rows}
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODEL_ORDER):
        for metric_index, metric in enumerate(POINT_METRICS):
            values = samples[:, model_index, metric_index]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "point_estimate": float(point_by_model[model][metric]),
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_sample_sd": float(np.std(values, ddof=1)),
                    "ci_95_percentile_low": float(np.quantile(values, 0.025)),
                    "ci_95_percentile_high": float(np.quantile(values, 0.975)),
                    "bootstrap_unit": "whole_protein_cluster",
                    "bootstrap_replicates": int(samples.shape[0]),
                }
            )
    return rows


def pairwise_rows(
    point_rows: Sequence[dict[str, Any]], samples: np.ndarray
) -> list[dict[str, Any]]:
    point_by_model = {str(row["model"]): row for row in point_rows}
    rows: list[dict[str, Any]] = []
    for comparator_index in (1, 2):
        comparator = MODEL_ORDER[comparator_index]
        for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
            metric_index = POINT_METRICS.index(metric)
            point_delta = (
                float(point_by_model[MODEL_ORDER[0]][metric])
                - float(point_by_model[comparator][metric])
            )
            deltas = samples[:, 0, metric_index] - samples[:, comparator_index, metric_index]
            summary = bootstrap_summary(point_delta, deltas)
            is_primary = metric == PRIMARY_METRIC
            rows.append(
                {
                    "first_model": MODEL_ORDER[0],
                    "second_model": comparator,
                    "delta_definition": "first_minus_second",
                    "metric": metric,
                    "analysis_role": "primary" if is_primary else "key_secondary",
                    "point_delta": summary["point_delta"],
                    "bootstrap_mean_delta": summary["mean_delta"],
                    "bootstrap_sample_sd": summary["standard_deviation"],
                    "ci_95_percentile_low": summary["percentile_95_ci"][0],
                    "ci_95_percentile_high": summary["percentile_95_ci"][1],
                    "probability_delta_positive": summary["probability_delta_positive"],
                    "one_sided_p_delta_le_zero": summary["one_sided_p_delta_le_zero_plus_one_correction"] if is_primary else None,
                    "two_sided_empirical_p": summary["two_sided_empirical_p_plus_one_correction"] if is_primary else None,
                    "holm_family": "two_predeclared_primary_roc_auc_contrasts" if is_primary else None,
                    "holm_two_sided_p": None,
                    "bootstrap_unit": "whole_protein_cluster",
                    "bootstrap_replicates": int(samples.shape[0]),
                    "inferential_status": "predeclared_primary_contrast" if is_primary else "descriptive_key_secondary_no_extra_test_family",
                }
            )
    primary_rows = [row for row in rows if row["metric"] == PRIMARY_METRIC]
    adjusted = holm_adjust([float(row["two_sided_empirical_p"]) for row in primary_rows])
    for row, value in zip(primary_rows, adjusted, strict=True):
        row["holm_two_sided_p"] = float(value)
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label}: {observed!r} != {expected!r}")


def validate_prelabel_inputs() -> dict[str, Any]:
    protocol_file = verify_exact_file(PROTOCOL, PROTOCOL_SHA256)
    protocol = load_json(PROTOCOL)
    erratum_file = verify_exact_file(PROTOCOL_ERRATUM, PROTOCOL_ERRATUM_SHA256)
    erratum = load_json(PROTOCOL_ERRATUM)
    require_equal(protocol.get("schema_version"), 1, "protocol schema")
    require_equal(
        protocol.get("protocol_status"),
        "frozen_before_s2_baseline_prediction",
        "protocol status",
    )
    comparator_names = tuple(
        str(row.get("name")) for row in protocol.get("formal_comparators", [])
    )
    require_equal(comparator_names, MODEL_ORDER[1:], "formal comparator names")
    statistics = protocol.get("statistics", {})
    require_equal(
        statistics.get("bootstrap_replicates"), 2000, "protocol bootstrap replicates"
    )
    require_equal(statistics.get("bootstrap_seed"), 20260907, "protocol seed")
    require_equal(
        statistics.get("multiple_testing"),
        "Holm correction across the two primary ROC-AUC contrasts",
        "protocol multiplicity",
    )
    reference_lock = verify_lock_manifest(REFERENCE_DIR / "s2_final_lock.sha256")
    a10_lock = verify_lock_manifest(A10_DIR / "s2_a10_prediction_lock.sha256")
    prediction_files = {
        model: verify_exact_file(PREDICTION_PATHS[model], PREDICTION_SHA256[model])
        for model in MODEL_ORDER
    }

    punch_entries = read_lock_entries(*BASELINE_LOCKS[MODEL_ORDER[1]])
    punch_members = verify_selected_lock_members(
        punch_entries,
        [
            PREDICTION_PATHS[MODEL_ORDER[1]],
            PUNCH_DIR / "feature_report_fp32.json",
            PUNCH_DIR / "inference_report.json",
            PUNCH_DIR / "s2_baseline_prediction_report.json",
        ],
    )
    lora_entries = read_lock_entries(*BASELINE_LOCKS[MODEL_ORDER[2]])
    lora_members = verify_selected_lock_members(
        lora_entries,
        [
            PROJECT_ROOT
            / "experiments/supplementary/s2_independent_holdout/"
            "predict_lora_official_offline.py",
            PROJECT_ROOT
            / "experiments/supplementary/s2_independent_holdout/"
            "run_s2_baseline_predictions.py",
            PREDICTION_PATHS[MODEL_ORDER[2]],
            LORA_DIR / "inference_report.json",
            LORA_DIR / "s2_baseline_prediction_report.json",
        ],
    )

    punch_driver = load_json(PUNCH_DIR / "s2_baseline_prediction_report.json")
    punch_inference = load_json(PUNCH_DIR / "inference_report.json")
    lora_driver = load_json(LORA_DIR / "s2_baseline_prediction_report.json")
    lora_inference = load_json(LORA_DIR / "inference_report.json")
    for name, report, prediction_hash in (
        (MODEL_ORDER[1], punch_driver, PREDICTION_SHA256[MODEL_ORDER[1]]),
        (MODEL_ORDER[2], lora_driver, PREDICTION_SHA256[MODEL_ORDER[2]]),
    ):
        require_equal(report.get("status"), "pass", f"{name} driver status")
        require_equal(report.get("mode"), "prediction", f"{name} driver mode")
        require_equal(
            report.get("protocol_sha256"), PROTOCOL_SHA256, f"{name} protocol hash"
        )
        require_equal(
            report.get("prediction_sha256"), prediction_hash, f"{name} prediction hash"
        )
        require_equal(report.get("proteins"), 87, f"{name} proteins")
        require_equal(report.get("residues"), 56394, f"{name} residues")
        require_equal(
            report.get("prediction_validation", {}).get("complete_coverage"),
            True,
            f"{name} coverage",
        )
        require_equal(
            report.get("s2_reference_or_labels_accessed"), False, f"{name} label access"
        )
        require_equal(report.get("locked_a10_modified"), False, f"{name} A10 state")
        require_equal(
            report.get("scores_used_for_training_tuning_calibration_or_selection"),
            False,
            f"{name} score use",
        )
        require_equal(
            report.get("ready_for_label_side_evaluation"),
            True,
            f"{name} readiness",
        )

    punch_protocol = protocol["formal_comparators"][0]
    require_equal(
        punch_inference.get("source_revision"),
        punch_protocol["source_revision"],
        "PUNCH2-Light source revision",
    )
    require_equal(punch_inference.get("released_members_loaded"), 13, "PUNCH members")
    require_equal(
        punch_inference.get("variants", {}).get("released13", {}).get("output_sha256"),
        PREDICTION_SHA256[MODEL_ORDER[1]],
        "PUNCH output hash",
    )
    require_equal(punch_inference.get("caid_labels_accessed"), False, "PUNCH labels")

    lora_protocol = protocol["formal_comparators"][1]
    require_equal(
        lora_protocol.get("adapter_weights_sha256"),
        MALFORMED_PROTOCOL_LORA_WEIGHTS_SHA256,
        "documented malformed protocol adapter hash",
    )
    require_equal(len(lora_protocol["adapter_weights_sha256"]), 62, "malformed hash length")
    require_equal(
        erratum.get("protocol_sha256"), PROTOCOL_SHA256, "erratum protocol hash"
    )
    require_equal(
        erratum.get("recorded_value"),
        MALFORMED_PROTOCOL_LORA_WEIGHTS_SHA256,
        "erratum recorded value",
    )
    require_equal(
        erratum.get("corrected_value"),
        LORA_ADAPTER_WEIGHTS_SHA256,
        "erratum corrected value",
    )
    require_equal(erratum.get("analysis_impact"), "none", "erratum impact")
    require_equal(erratum.get("predictions_changed"), False, "erratum predictions")
    require_equal(
        erratum.get("statistical_protocol_changed"), False, "erratum statistics"
    )
    provenance_fields = (
        "model_id",
        "model_revision",
        "base_model_id",
        "base_model_revision",
        "adapter_config_sha256",
    )
    for field in provenance_fields:
        require_equal(
            lora_inference.get(field), lora_protocol[field], f"LoRA provenance {field}"
        )
    require_equal(
        lora_driver.get("model_provenance", {}).get("adapter_weights_sha256"),
        LORA_ADAPTER_WEIGHTS_SHA256,
        "LoRA driver adapter weights hash",
    )
    require_equal(
        lora_inference.get("adapter_weights_sha256"),
        LORA_ADAPTER_WEIGHTS_SHA256,
        "LoRA inference adapter weights hash",
    )
    require_equal(lora_inference.get("offline_cached_loading"), True, "LoRA offline")
    require_equal(
        lora_inference.get("network_requests_required"), False, "LoRA network"
    )
    require_equal(
        lora_inference.get("lora_and_saved_head_loaded"), True, "LoRA parameter load"
    )
    require_equal(
        lora_inference.get("output_sha256"),
        PREDICTION_SHA256[MODEL_ORDER[2]],
        "LoRA output hash",
    )
    require_equal(lora_inference.get("caid_labels_accessed"), False, "LoRA labels")
    return {
        "protocol": protocol_file,
        "protocol_erratum": erratum_file,
        "protocol_erratum_scope": "malformed_62_character_provenance_hash_only",
        "reference_lock": reference_lock,
        "a10_prediction_lock": a10_lock,
        "prediction_files": prediction_files,
        "baseline_result_locks": {
            MODEL_ORDER[1]: {
                "path": str(BASELINE_LOCKS[MODEL_ORDER[1]][0]),
                "sha256": BASELINE_LOCKS[MODEL_ORDER[1]][1],
                "verified_selected_members": punch_members,
            },
            MODEL_ORDER[2]: {
                "path": str(BASELINE_LOCKS[MODEL_ORDER[2]][0]),
                "sha256": BASELINE_LOCKS[MODEL_ORDER[2]][1],
                "verified_selected_members": lora_members,
            },
        },
        "hashes_verified_before_semantic_label_access": True,
    }


def build_summary(
    point_rows: Sequence[dict[str, Any]],
    interval_rows: Sequence[dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
) -> str:
    intervals = {
        (str(row["model"]), str(row["metric"])): row for row in interval_rows
    }
    labels = {
        "roc_auc_full_precision": "ROC-AUC",
        "aucpr_trapezoid_full_precision": "AUPRC",
        "aps_full_precision": "APS",
        "f1_at_0_5_full_precision": "F1@0.5",
        "mcc_at_0_5_full_precision": "MCC@0.5",
        "fmax_full_precision": "Fmax",
        "brier_score": "Brier",
        "macro_roc_auc": "Macro ROC-AUC",
    }
    display_metrics = tuple(labels)
    lines = [
        "# S2 three-model locked comparison",
        "",
        "All protocol, reference, prediction, and provenance hashes were verified "
        "before semantic label access. All three methods were evaluated on the same "
        "87 proteins and the same 30,728 known residues. Unknown labels (-1) were "
        "excluded before every metric.",
        "",
        "## Point estimates and paired protein-bootstrap intervals",
        "",
        "| Model | " + " | ".join(labels[metric] for metric in display_metrics) + " |",
        "| --- | " + " | ".join("---:" for _ in display_metrics) + " |",
    ]
    for point in point_rows:
        model = str(point["model"])
        cells = []
        for metric in display_metrics:
            interval = intervals[(model, metric)]
            cells.append(
                f"{float(point[metric]):.6f} "
                f"[{float(interval['ci_95_percentile_low']):.6f}, "
                f"{float(interval['ci_95_percentile_high']):.6f}]"
            )
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "The brackets are 95% percentile intervals from 2,000 paired whole-protein "
        "bootstrap replicates. Lower Brier score is better; higher values are better "
        "for the other metrics.",
        "",
        "## Predeclared primary ROC-AUC contrasts",
        "",
        "All deltas are MUSE-IDR A10 minus comparator. Holm correction covers exactly "
        "the two predeclared primary ROC-AUC contrasts.",
        "",
        "| Comparator | ROC-AUC delta | 95% CI | P(delta > 0) | Raw two-sided p | Holm p | Interpretation |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in comparisons:
        if row["metric"] != PRIMARY_METRIC:
            continue
        low = float(row["ci_95_percentile_low"])
        high = float(row["ci_95_percentile_high"])
        if low > 0.0:
            interpretation = "supports higher MUSE-IDR ROC-AUC"
        elif high < 0.0:
            interpretation = "supports lower MUSE-IDR ROC-AUC"
        else:
            interpretation = "95% CI includes zero"
        lines.append(
            "| {second_model} | {point_delta:+.6f} | [{low:+.6f}, {high:+.6f}] | "
            "{probability_delta_positive:.4f} | {two_sided_empirical_p:.6g} | "
            "{holm_two_sided_p:.6g} | {interpretation} |".format(
                low=low, high=high, interpretation=interpretation, **row
            )
        )

    lines += [
        "",
        "## Key secondary AUPRC contrasts",
        "",
        "AUPRC is the predeclared key secondary metric. Its paired deltas and intervals "
        "are reported descriptively; no additional post hoc hypothesis-test family was "
        "introduced.",
        "",
        "| Comparator | AUPRC delta | 95% CI | P(delta > 0) |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in comparisons:
        if row["metric"] != SECONDARY_METRIC:
            continue
        lines.append(
            "| {second_model} | {point_delta:+.6f} | "
            "[{ci_95_percentile_low:+.6f}, {ci_95_percentile_high:+.6f}] | "
            "{probability_delta_positive:.4f} |".format(**row)
        )

    punch = next(row for row in point_rows if row["model"] == MODEL_ORDER[1])
    lines += [
        "",
        "## Threshold and interpretation constraints",
        "",
        "- F1 and MCC in the main table use the same fixed threshold of 0.5 for all "
        "three models and are descriptive.",
        f"- At PUNCH2-Light's separately declared official threshold of 0.35, its "
        f"F1 is {float(punch['f1_at_official_threshold']):.6f} and MCC is "
        f"{float(punch['mcc_at_official_threshold']):.6f}; these values are not mixed "
        "with the common-threshold comparison.",
        "- Macro ROC-AUC averages only the 22 proteins containing both known classes.",
        "- No S2 labels or results were used for training, fine-tuning, calibration, "
        "threshold tuning, ensemble reweighting, or A10 reselection.",
        "- This independent temporal holdout supplements the locked CAID2/CAID3 "
        "results; it does not justify a universal state-of-the-art claim.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite S2 comparison: {output_dir}")
    if args.replicates != 2000 or args.seed != 20260907:
        raise ValueError("formal S2 evaluation requires 2000 replicates and seed 20260907")

    # The complete pre-label audit deliberately precedes read_labels().
    verification = validate_prelabel_inputs()
    cohort = read_cohort(REFERENCE_DIR / "s2_cohort.tsv")
    labels = read_labels(REFERENCE_DIR / "s2_labels.tsv.gz")
    predictions = {
        model: read_prediction(PREDICTION_PATHS[model]) for model in MODEL_ORDER
    }
    blocks = align_three_models(labels, predictions, cohort)
    reference_labels, _ = flatten_blocks(blocks, 0)
    observed_counts = {
        "proteins": len(blocks),
        "residues": sum(block.length for block in blocks),
        "known_residues": int(reference_labels.size),
        "positive_residues": int(np.sum(reference_labels == 1)),
        "negative_residues": int(np.sum(reference_labels == 0)),
        "unknown_residues": int(
            sum(block.length - block.labels.size for block in blocks)
        ),
    }
    if observed_counts != EXPECTED_COUNTS:
        raise ValueError(f"S2 frozen count mismatch: {observed_counts} != {EXPECTED_COUNTS}")
    two_class_proteins = sum(np.unique(block.labels).size == 2 for block in blocks)
    if two_class_proteins != 22:
        raise ValueError(f"S2 two-class protein mismatch: {two_class_proteins} != 22")

    points = point_metric_rows(blocks)
    proteins = per_protein_rows(blocks)
    samples = bootstrap_samples(
        blocks, args.replicates, args.workers, args.seed, args.progress_every
    )
    intervals = metric_summary_rows(points, samples)
    comparisons = pairwise_rows(points, samples)
    summary = build_summary(points, intervals, comparisons)

    output_dir.mkdir(parents=True)
    point_path = output_dir / "s2_three_model_point_metrics.csv"
    interval_path = output_dir / "s2_three_model_metric_intervals.csv"
    comparison_path = output_dir / "s2_three_model_pairwise_statistics.csv"
    protein_path = output_dir / "s2_three_model_per_protein_metrics.csv"
    sample_path = output_dir / "s2_three_model_bootstrap_samples.npz"
    report_path = output_dir / "s2_three_model_report.json"
    summary_path = output_dir / "s2_three_model_summary.md"
    write_csv(point_path, points)
    write_csv(interval_path, intervals)
    write_csv(comparison_path, comparisons)
    write_csv(protein_path, proteins)
    np.savez_compressed(
        sample_path,
        samples=samples,
        model_names=np.asarray(MODEL_ORDER),
        metric_names=np.asarray(POINT_METRICS),
        seed=np.asarray(args.seed),
        bootstrap_unit=np.asarray("whole_protein_cluster"),
    )
    summary_path.write_text(summary, encoding="utf-8", newline="\n")
    report = {
        "schema_version": 1,
        "experiment": "s2_locked_three_model_comparison",
        "status": "pass",
        "protocol_status": "frozen_before_s2_baseline_prediction",
        "verification": verification,
        "cohort": observed_counts,
        "two_class_proteins": two_class_proteins,
        "positive_prevalence_among_known": EXPECTED_COUNTS["positive_residues"]
        / EXPECTED_COUNTS["known_residues"],
        "model_order": list(MODEL_ORDER),
        "point_metrics": points,
        "bootstrap": {
            "unit": "whole_protein_cluster",
            "paired_across_all_three_models": True,
            "replicates": args.replicates,
            "seed": args.seed,
            "workers": args.workers,
            "intervals": intervals,
        },
        "comparisons": comparisons,
        "primary_metric": PRIMARY_METRIC,
        "primary_contrasts": 2,
        "primary_multiple_testing": "Holm across exactly two ROC-AUC contrasts",
        "key_secondary_metric": SECONDARY_METRIC,
        "unknown_labels_excluded": True,
        "common_threshold": 0.5,
        "punch_official_threshold_reported_separately": 0.35,
        "s2_labels_used_for_training_fine_tuning_or_calibration": False,
        "s2_used_for_threshold_tuning_or_ensemble_reweighting": False,
        "locked_a10_modified_or_reselected": False,
        "additional_baselines_run": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    test_path = Path(__file__).with_name("test_evaluate_s2_three_models.py").resolve()
    lock_members = [
        Path(__file__).resolve(),
        test_path,
        PROTOCOL,
        PROTOCOL_ERRATUM,
        REFERENCE_DIR / "s2_final_lock.sha256",
        A10_DIR / "s2_a10_prediction_lock.sha256",
        BASELINE_LOCKS[MODEL_ORDER[1]][0],
        BASELINE_LOCKS[MODEL_ORDER[2]][0],
        *(PREDICTION_PATHS[model] for model in MODEL_ORDER),
        point_path,
        interval_path,
        comparison_path,
        protein_path,
        sample_path,
        report_path,
        summary_path,
    ]
    for path in lock_members:
        if not path.is_file():
            raise FileNotFoundError(f"cannot lock missing file: {path}")
    lock_path = output_dir / "s2_three_model_result_lock.sha256"
    lock_path.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(PROJECT_ROOT).as_posix()}\n"
            for path in lock_members
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "models": len(MODEL_ORDER),
                "proteins": len(blocks),
                "known_residues": int(reference_labels.size),
                "bootstrap_replicates": args.replicates,
                "primary_contrasts": 2,
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "locked_a10_modified_or_reselected": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
