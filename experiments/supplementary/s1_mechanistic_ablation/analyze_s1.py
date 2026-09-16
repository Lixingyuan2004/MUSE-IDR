"""Analyze locked S1 OOF predictions without modifying or reselecting A10.

The primary analysis builds one equal-logit three-seed ensemble per S1
variant, then compares it with the matching locked A2 or A8 ensemble using a
paired, fold-stratified whole-protein bootstrap.  The seven predeclared
ROC-AUC comparisons share one Holm correction family.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a1_frozen_esm2.data import (  # noqa: E402
    ProteinRecord,
    load_manifest,
)
from experiments.ablations.a1_frozen_esm2.train_cached_a1 import (  # noqa: E402
    metrics_from_records,
)


SEEDS = (17, 29, 43)
S1_VARIANTS = (
    "ungated_add",
    "additive_projection",
    "single_k3",
    "single_k7",
    "single_k15",
    "single_k31",
    "uniform_layer_mix",
)
MODEL_ORDER = (
    "a2_reference",
    "a8_learned_mix",
    *S1_VARIANTS,
)
DISPLAY_NAMES = {
    "a2_reference": "A2 gated four-scale",
    "a8_learned_mix": "A8 learned layer mix",
    "ungated_add": "Direct ungated addition",
    "additive_projection": "Parameter-matched additive projection",
    "single_k3": "Single scale k=3",
    "single_k7": "Single scale k=7",
    "single_k15": "Single scale k=15",
    "single_k31": "Single scale k=31",
    "uniform_layer_mix": "Fixed uniform layer mix",
}
COMPARISONS = (
    (
        "gate_vs_direct_add",
        "a2_reference",
        "ungated_add",
        "sigmoid gate versus direct residual addition",
    ),
    (
        "gate_vs_parameter_matched_additive",
        "a2_reference",
        "additive_projection",
        "sigmoid gate versus a parameter-matched additive projection",
    ),
    (
        "multiscale_vs_k3",
        "a2_reference",
        "single_k3",
        "four-scale context versus kernel 3 only",
    ),
    (
        "multiscale_vs_k7",
        "a2_reference",
        "single_k7",
        "four-scale context versus kernel 7 only",
    ),
    (
        "multiscale_vs_k15",
        "a2_reference",
        "single_k15",
        "four-scale context versus kernel 15 only",
    ),
    (
        "multiscale_vs_k31",
        "a2_reference",
        "single_k31",
        "four-scale context versus kernel 31 only",
    ),
    (
        "learned_vs_uniform_layer_mix",
        "a8_learned_mix",
        "uniform_layer_mix",
        "learned last-four-layer mix versus a fixed uniform mean",
    ),
)

EXPECTED_HASHES = {
    "manifest": "df6a4deee4a65d009f4faf41c6b8930f9db8600fa28b6711da7ff48ad2c70bf4",
    "s1_config": "8e7c5b5a1641f53644f21f34e45572447652ed50d12359ceff16e201ba214671",
    "s1_model": "ad6b7dd2f92da7d224062e6da3d049179d6c2b319e9d3598a53a3a3c6de9879e",
    "s1_train": "94f87a6c7e5c5eb857ff2d667294fe39e339355b6301e189974938c683182e48",
    "a5_predictions": "220ce95fb398690df4348cfe7199fcbeb797862bf062441e48a66362ba1b1622",
    "a5_report": "acf246daa0add31ef1aec233e5ef56012d417fe041da7596d970435039f41825",
    "a9_predictions": "57432c063e47b7e66fc3cfcef000e248afbc7e166bb76eff0faed7b9b4b1be1c",
    "a9_report": "d7fbac75a3281d00ea14c5396bca54f3fbd168c381d28dddd06a2893d3552fee",
    "a1_a10_summary": "0a89a15fe14f6833095a04a97e5e209976a525a053db435fb08cc179b88383e8",
}


@dataclass(frozen=True)
class ProteinBlock:
    protein_id: str
    fold: str
    labels: np.ndarray
    scores: np.ndarray


_WORKER_BLOCKS: list[ProteinBlock] | None = None
_WORKER_FOLD_INDICES: tuple[np.ndarray, ...] | None = None
_WORKER_SEED = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {label}: {actual} != {expected}")
    return {"label": label, "path": str(path), "sha256": actual}


def verify_s1_lock(project_root: Path, s1_root: Path) -> dict[str, Any]:
    lock_path = s1_root / "s1_result_lock.sha256"
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    failures: list[str] = []
    verified = 0
    for line_number, line in enumerate(
        lock_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative_text = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"malformed S1 lock line {line_number}") from exc
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe S1 lock path: {relative_text}")
        path = project_root / relative
        if not path.is_file():
            failures.append(f"missing: {relative_text}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch: {relative_text}")
        verified += 1
    if failures:
        raise ValueError("S1 result-lock verification failed: " + "; ".join(failures))
    if verified != 358:
        raise ValueError(f"expected 358 locked S1 files, verified {verified}")
    return {
        "label": "s1_result_lock",
        "path": str(lock_path),
        "sha256": sha256(lock_path),
        "verified_files": verified,
    }


def open_csv(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _empty_score_map(records: Sequence[ProteinRecord]) -> dict[str, np.ndarray]:
    return {
        record.protein_id: np.full(len(record.sequence), np.nan, dtype=np.float64)
        for record in records
    }


def read_simple_predictions(
    path: Path, records: Sequence[ProteinRecord]
) -> dict[str, np.ndarray]:
    record_map = {record.protein_id: record for record in records}
    scores = _empty_score_map(records)
    rows = 0
    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        required = {"protein_id", "position", "label", "probability"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError(f"prediction schema mismatch: {path}")
        for row in reader:
            protein_id = row["protein_id"]
            if protein_id not in record_map:
                raise ValueError(f"unknown protein in {path}: {protein_id}")
            record = record_map[protein_id]
            index = int(row["position"]) - 1
            if index < 0 or index >= len(record.sequence):
                raise ValueError(f"invalid residue position in {path}: {protein_id}")
            if int(row["label"]) != int(record.labels[index]):
                raise ValueError(f"label mismatch in {path}: {protein_id}:{index + 1}")
            value = float(row["probability"])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid probability in {path}: {value}")
            if math.isfinite(float(scores[protein_id][index])):
                raise ValueError(f"duplicate prediction in {path}: {protein_id}:{index + 1}")
            scores[protein_id][index] = value
            rows += 1
    expected_rows = sum(len(record.sequence) for record in records)
    if rows != expected_rows:
        raise ValueError(f"row count mismatch in {path}: {rows} != {expected_rows}")
    if any(not np.all(np.isfinite(value)) for value in scores.values()):
        raise ValueError(f"incomplete predictions: {path}")
    return scores


def equal_logit_mean(seed_scores: Sequence[np.ndarray]) -> np.ndarray:
    if len(seed_scores) != len(SEEDS):
        raise ValueError("equal-logit mean requires exactly three seeds")
    shape = seed_scores[0].shape
    if any(value.shape != shape for value in seed_scores):
        raise ValueError("seed predictions are not aligned")
    epsilon = np.finfo(np.float64).eps
    stacked = np.stack(seed_scores, axis=0).astype(np.float64, copy=False)
    stacked = np.clip(stacked, epsilon, 1.0 - epsilon)
    mean_logit = np.mean(np.log(stacked / (1.0 - stacked)), axis=0)
    return 1.0 / (1.0 + np.exp(-mean_logit))


def build_s1_ensemble(
    variant: str,
    s1_root: Path,
    records: Sequence[ProteinRecord],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    seed_maps: list[dict[str, np.ndarray]] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_root = s1_root / variant / f"seed_{seed}"
        report_path = seed_root / "oof_metrics.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not all(
            (
                report.get("status") == "pass",
                report.get("variant") == variant,
                report.get("seed") == seed,
                report.get("complete_oof") is True,
                report.get("selected_folds") == ["0", "1", "2", "3", "4"],
                report.get(
                    "caid1_caid2_caid3_used_for_training_tuning_or_reselection"
                )
                is False,
                report.get("locked_a10_modified") is False,
            )
        ):
            raise ValueError(f"invalid completed S1 report: {report_path}")
        predictions = read_simple_predictions(
            seed_root / "oof_predictions.csv", records
        )
        recomputed = metrics_from_records(records, predictions)
        reported = report["oof_metrics"]
        for metric in ("micro_roc_auc", "micro_pr_auc", "macro_roc_auc"):
            if not math.isclose(
                float(recomputed[metric]),
                float(reported[metric]),
                rel_tol=0.0,
                abs_tol=5e-8,
            ):
                raise ValueError(f"recomputed {metric} mismatch: {variant} seed {seed}")
        seed_maps.append(predictions)
        seed_rows.append(
            {
                "model": variant,
                "display_name": DISPLAY_NAMES[variant],
                "seed": seed,
                **{key: reported[key] for key in (
                    "micro_roc_auc",
                    "micro_pr_auc",
                    "macro_roc_auc",
                )},
            }
        )
    ensemble = {
        record.protein_id: equal_logit_mean(
            [seed_map[record.protein_id] for seed_map in seed_maps]
        )
        for record in records
    }
    return ensemble, seed_rows


def read_reference_ensemble(
    path: Path,
    records: Sequence[ProteinRecord],
) -> tuple[dict[str, np.ndarray], float]:
    record_map = {record.protein_id: record for record in records}
    scores = _empty_score_map(records)
    rows = 0
    max_difference = 0.0
    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        required = {
            "protein_id",
            "position",
            "fold",
            "label",
            "seed_17",
            "seed_29",
            "seed_43",
            "uniform_logit_mean",
        }
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError(f"reference prediction schema mismatch: {path}")
        for row in reader:
            protein_id = row["protein_id"]
            if protein_id not in record_map:
                raise ValueError(f"unknown reference protein: {protein_id}")
            record = record_map[protein_id]
            index = int(row["position"]) - 1
            if index < 0 or index >= len(record.sequence):
                raise ValueError(f"invalid reference position: {protein_id}")
            if row["fold"] != record.fold:
                raise ValueError(f"fold mismatch: {protein_id}")
            if int(row["label"]) != int(record.labels[index]):
                raise ValueError(f"reference label mismatch: {protein_id}:{index + 1}")
            seeds = np.asarray(
                [float(row[f"seed_{seed}"]) for seed in SEEDS], dtype=np.float64
            )
            recomputed = float(equal_logit_mean([value[None] for value in seeds])[0])
            stored = float(row["uniform_logit_mean"])
            max_difference = max(max_difference, abs(recomputed - stored))
            if abs(recomputed - stored) > 1e-9:
                raise ValueError(f"reference logit-mean mismatch: {protein_id}:{index + 1}")
            if math.isfinite(float(scores[protein_id][index])):
                raise ValueError(f"duplicate reference prediction: {protein_id}:{index + 1}")
            scores[protein_id][index] = stored
            rows += 1
    expected_rows = sum(len(record.sequence) for record in records)
    if rows != expected_rows:
        raise ValueError(f"reference row count mismatch: {rows} != {expected_rows}")
    if any(not np.all(np.isfinite(value)) for value in scores.values()):
        raise ValueError(f"incomplete reference predictions: {path}")
    return scores, max_difference


def reference_seed_rows(report: dict[str, Any], model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        metrics = report["source_seed_metrics"][str(seed)]
        rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "seed": seed,
                **{key: metrics[key] for key in (
                    "micro_roc_auc",
                    "micro_pr_auc",
                    "macro_roc_auc",
                )},
            }
        )
    return rows


def roc_auc_full_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("both classes are required for ROC-AUC")
    order = np.argsort(y_score, kind="mergesort")[::-1]
    ordered_scores = y_score[order]
    ordered_positive = y_true[order] == 1
    distinct = np.where(np.diff(ordered_scores))[0]
    indices = np.r_[distinct, ordered_positive.size - 1]
    tp = np.cumsum(ordered_positive, dtype=np.float64)[indices]
    fp = 1 + indices - tp
    recall = tp / positives
    fpr = fp / negatives
    return float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, fpr]))


def build_blocks(
    records: Sequence[ProteinRecord],
    predictions: dict[str, dict[str, np.ndarray]],
) -> list[ProteinBlock]:
    blocks: list[ProteinBlock] = []
    for record in records:
        known = record.labels >= 0
        labels = record.labels[known].astype(np.int8, copy=False)
        score_matrix = np.column_stack(
            [predictions[model][record.protein_id][known] for model in MODEL_ORDER]
        )
        if not np.all(np.isfinite(score_matrix)):
            raise ValueError(f"non-finite ensemble scores: {record.protein_id}")
        blocks.append(ProteinBlock(record.protein_id, record.fold, labels, score_matrix))
    return blocks


def _init_worker(
    blocks: list[ProteinBlock], fold_indices: tuple[np.ndarray, ...], seed: int
) -> None:
    global _WORKER_BLOCKS, _WORKER_FOLD_INDICES, _WORKER_SEED
    _WORKER_BLOCKS = blocks
    _WORKER_FOLD_INDICES = fold_indices
    _WORKER_SEED = seed


def _bootstrap_chunk(start: int, count: int) -> tuple[int, np.ndarray]:
    if _WORKER_BLOCKS is None or _WORKER_FOLD_INDICES is None:
        raise RuntimeError("bootstrap worker is not initialized")
    blocks = _WORKER_BLOCKS
    output = np.empty((count, len(MODEL_ORDER)), dtype=np.float64)
    for offset in range(count):
        replicate = start + offset
        rng = np.random.default_rng(np.random.SeedSequence([_WORKER_SEED, replicate]))
        selected: list[int] = []
        for indices in _WORKER_FOLD_INDICES:
            draws = rng.integers(0, len(indices), size=len(indices))
            selected.extend(indices[draws].tolist())
        sampled = [blocks[index] for index in selected]
        labels = np.concatenate([block.labels for block in sampled])
        scores = np.concatenate([block.scores for block in sampled], axis=0)
        for model_index in range(len(MODEL_ORDER)):
            output[offset, model_index] = roc_auc_full_precision(
                labels, scores[:, model_index]
            )
    return start, output


def bootstrap_auc_samples(
    blocks: list[ProteinBlock],
    replicates: int,
    workers: int,
    random_seed: int,
    progress_every: int,
) -> np.ndarray:
    folds = sorted({block.fold for block in blocks})
    fold_indices = tuple(
        np.asarray(
            [index for index, block in enumerate(blocks) if block.fold == fold],
            dtype=np.int64,
        )
        for fold in folds
    )
    if len(fold_indices) != 5 or any(len(value) == 0 for value in fold_indices):
        raise ValueError("formal S1 bootstrap requires five non-empty folds")
    workers = max(1, min(int(workers), int(replicates)))
    chunk_size = max(1, min(10, math.ceil(replicates / (workers * 4))))
    chunks = [
        (start, min(chunk_size, replicates - start))
        for start in range(0, replicates, chunk_size)
    ]
    output = np.empty((replicates, len(MODEL_ORDER)), dtype=np.float64)
    completed = 0
    next_progress = progress_every
    if workers == 1:
        _init_worker(blocks, fold_indices, random_seed)
        for start, count in chunks:
            _, values = _bootstrap_chunk(start, count)
            output[start : start + count] = values
            completed += count
            if progress_every and completed >= next_progress:
                print(json.dumps({"bootstrap_completed": completed}), flush=True)
                next_progress += progress_every
        return output
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(blocks, fold_indices, random_seed),
    ) as executor:
        pending = {
            executor.submit(_bootstrap_chunk, start, count): (start, count)
            for start, count in chunks
        }
        for future in as_completed(pending):
            expected_start, expected_count = pending[future]
            returned_start, values = future.result()
            if returned_start != expected_start or values.shape != (
                expected_count,
                len(MODEL_ORDER),
            ):
                raise RuntimeError("bootstrap worker returned malformed output")
            output[returned_start : returned_start + expected_count] = values
            completed += expected_count
            if progress_every and completed >= next_progress:
                print(json.dumps({"bootstrap_completed": completed}), flush=True)
                next_progress += progress_every
    return output


def bootstrap_summary(point_delta: float, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    nonpositive = int(np.sum(values <= 0.0))
    nonnegative = int(np.sum(values >= 0.0))
    denominator = values.size + 1
    p_le_zero = (nonpositive + 1) / denominator
    p_ge_zero = (nonnegative + 1) / denominator
    return {
        "point_delta": float(point_delta),
        "mean_delta": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "percentile_95_ci": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "probability_delta_positive": float(np.mean(values > 0.0)),
        "one_sided_p_delta_le_zero_plus_one_correction": float(p_le_zero),
        "one_sided_p_delta_ge_zero_plus_one_correction": float(p_ge_zero),
        "two_sided_empirical_p_plus_one_correction": float(
            min(1.0, 2.0 * min(p_le_zero, p_ge_zero))
        ),
        "nonpositive_replicates": nonpositive,
        "nonnegative_replicates": nonnegative,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=np.float64), kind="mergesort")
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * float(p_values[int(index)])
        running = max(running, candidate)
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def seed_summary_rows(seed_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        selected = [row for row in seed_rows if row["model"] == model]
        if len(selected) != len(SEEDS):
            raise ValueError(f"expected three seed metrics for {model}")
        row: dict[str, Any] = {
            "model": model,
            "display_name": DISPLAY_NAMES[model],
            "seeds": "/".join(str(seed) for seed in SEEDS),
        }
        for metric in ("micro_roc_auc", "micro_pr_auc", "macro_roc_auc"):
            values = [float(value[metric]) for value in selected]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sample_sd"] = statistics.stdev(values)
        output.append(row)
    return output


def interpretation(row: dict[str, Any]) -> str:
    low, high = row["percentile_95_ci"]
    adjusted = float(row["holm_adjusted_p"])
    if low > 0.0 and adjusted < 0.05:
        return "supports_reference_module"
    if high < 0.0 and adjusted < 0.05:
        return "control_outperforms_reference"
    return "no_holm_significant_difference"


def markdown_summary(
    seed_summary: Sequence[dict[str, Any]],
    point_rows: Sequence[dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# S1 post-lock mechanistic ablation",
        "",
        "S1 is explanation-only. It does not modify, reweight, or reselect the locked A10 model. ",
        "All 358 S1 result files and all locked A2/A8 references were SHA256-verified before label access.",
        "",
        "## Three-seed robustness",
        "",
        "| Model | ROC-AUC mean +/- SD | PR-AUC mean +/- SD | Macro ROC-AUC mean +/- SD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in seed_summary:
        lines.append(
            "| {display_name} | {micro_roc_auc_mean:.6f} +/- {micro_roc_auc_sample_sd:.6f} | "
            "{micro_pr_auc_mean:.6f} +/- {micro_pr_auc_sample_sd:.6f} | "
            "{macro_roc_auc_mean:.6f} +/- {macro_roc_auc_sample_sd:.6f} |".format(
                **row
            )
        )
    lines += [
        "",
        "## Fixed equal-logit three-seed ensembles",
        "",
        "| Model | ROC-AUC | PR-AUC | Macro ROC-AUC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in point_rows:
        lines.append(
            "| {display_name} | {micro_roc_auc:.6f} | {micro_pr_auc:.6f} | "
            "{macro_roc_auc:.6f} |".format(**row)
        )
    lines += [
        "",
        "## Paired fold-stratified whole-protein bootstrap",
        "",
        "Positive deltas favor the reference module. The seven predeclared ROC-AUC comparisons share one Holm family.",
        "",
        "| Comparison | Reference - control ROC delta | 95% CI | Raw p | Holm p | Interpretation |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in comparisons:
        low, high = row["percentile_95_ci"]
        lines.append(
            "| {mechanism} | {point_delta:+.6f} | [{low:+.6f}, {high:+.6f}] | "
            "{raw:.6g} | {holm:.6g} | {interpretation} |".format(
                low=low,
                high=high,
                raw=row["two_sided_empirical_p_plus_one_correction"],
                holm=row["holm_adjusted_p"],
                **row,
            )
        )
    lines += [
        "",
        "## Guardrails",
        "",
        "- Bootstrap resampling uses whole proteins and preserves the five OOF fold sizes.",
        "- Equal-logit ensemble weights are fixed at 1/3 and are not fitted to labels.",
        "- CAID1/2/3 labels are not used in S1 training, tuning, weighting, or model selection.",
        "- S1 conclusions explain components; they do not change the locked A10 model or its external results.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--s1-root",
        type=Path,
        default=Path("outputs/supplementary/s1_mechanistic_ablation"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl"),
    )
    parser.add_argument(
        "--a5-predictions",
        type=Path,
        default=Path("outputs/ablations/a5_seed_ensemble/a5_oof_predictions.csv.gz"),
    )
    parser.add_argument(
        "--a5-report",
        type=Path,
        default=Path("outputs/ablations/a5_seed_ensemble/a5_ensemble_report.json"),
    )
    parser.add_argument(
        "--a9-predictions",
        type=Path,
        default=Path(
            "models/weights/a10_locked/a8_a9_a10/outputs/ablations/"
            "a9_a8_seed_ensemble/a9_oof_predictions.csv.gz"
        ),
    )
    parser.add_argument(
        "--a9-report",
        type=Path,
        default=Path(
            "models/weights/a10_locked/a8_a9_a10/outputs/ablations/"
            "a9_a8_seed_ensemble/a9_ensemble_report.json"
        ),
    )
    parser.add_argument(
        "--a1-a10-summary",
        type=Path,
        default=Path("outputs/analysis/a1_a10_20260824/a1_a10_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/supplementary/s1_mechanistic_analysis/formal"),
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> None:
    args = parse_args()
    if args.replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    started = perf_counter()
    project_root = args.project_root.resolve()
    s1_root = resolve(project_root, args.s1_root).resolve()
    manifest_path = resolve(project_root, args.manifest).resolve()
    a5_predictions_path = resolve(project_root, args.a5_predictions).resolve()
    a5_report_path = resolve(project_root, args.a5_report).resolve()
    a9_predictions_path = resolve(project_root, args.a9_predictions).resolve()
    a9_report_path = resolve(project_root, args.a9_report).resolve()
    summary_path = resolve(project_root, args.a1_a10_summary).resolve()
    output_dir = resolve(project_root, args.output_dir).resolve()

    # Every byte-level check happens before the label-bearing manifest is read.
    verification: list[dict[str, Any]] = [
        verify_s1_lock(project_root, s1_root),
        verify_file(manifest_path, EXPECTED_HASHES["manifest"], "manifest"),
        verify_file(
            project_root / "experiments/supplementary/s1_mechanistic_ablation/config.yaml",
            EXPECTED_HASHES["s1_config"],
            "s1_config",
        ),
        verify_file(
            project_root / "experiments/supplementary/s1_mechanistic_ablation/model.py",
            EXPECTED_HASHES["s1_model"],
            "s1_model",
        ),
        verify_file(
            project_root / "experiments/supplementary/s1_mechanistic_ablation/train_s1.py",
            EXPECTED_HASHES["s1_train"],
            "s1_train",
        ),
        verify_file(
            a5_predictions_path,
            EXPECTED_HASHES["a5_predictions"],
            "a5_predictions",
        ),
        verify_file(a5_report_path, EXPECTED_HASHES["a5_report"], "a5_report"),
        verify_file(
            a9_predictions_path,
            EXPECTED_HASHES["a9_predictions"],
            "a9_predictions",
        ),
        verify_file(a9_report_path, EXPECTED_HASHES["a9_report"], "a9_report"),
        verify_file(
            summary_path,
            EXPECTED_HASHES["a1_a10_summary"],
            "a1_a10_summary",
        ),
    ]

    records = load_manifest(manifest_path)
    if len(records) != 1133 or {record.fold for record in records} != set("01234"):
        raise ValueError("formal S1 manifest must contain 1,133 proteins in five folds")

    a5_report = json.loads(a5_report_path.read_text(encoding="utf-8"))
    a9_report = json.loads(a9_report_path.read_text(encoding="utf-8"))
    for name, report in (("A5", a5_report), ("A9", a9_report)):
        if report.get("manifest_sha256") != EXPECTED_HASHES["manifest"]:
            raise ValueError(f"{name} uses a different manifest")
        if report.get("seeds") != list(SEEDS):
            raise ValueError(f"{name} uses different seeds")
        if report.get("selected_for_future_external_test", {}).get("method") != "uniform_logit_mean":
            raise ValueError(f"{name} did not select fixed equal-logit averaging")
        if report["selected_for_future_external_test"].get("caid_used_for_selection") is not False:
            raise ValueError(f"{name} reports CAID use")

    a2_predictions, a5_max_difference = read_reference_ensemble(
        a5_predictions_path, records
    )
    a8_predictions, a9_max_difference = read_reference_ensemble(
        a9_predictions_path, records
    )
    predictions: dict[str, dict[str, np.ndarray]] = {
        "a2_reference": a2_predictions,
        "a8_learned_mix": a8_predictions,
    }
    seed_rows = reference_seed_rows(a5_report, "a2_reference")
    seed_rows.extend(reference_seed_rows(a9_report, "a8_learned_mix"))
    for variant in S1_VARIANTS:
        predictions[variant], rows = build_s1_ensemble(variant, s1_root, records)
        seed_rows.extend(rows)
        print(json.dumps({"ensemble_built": variant}), flush=True)

    point_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        metrics = metrics_from_records(records, predictions[model])
        point_rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "ensemble_method": "three_seed_uniform_logit_mean",
                **metrics,
            }
        )
    point_by_model = {row["model"]: row for row in point_rows}
    expected_reference = {
        "a2_reference": a5_report["method_metrics"]["uniform_logit_mean"]["oof"],
        "a8_learned_mix": a9_report["method_metrics"]["uniform_logit_mean"]["oof"],
    }
    for model, expected in expected_reference.items():
        for metric in ("micro_roc_auc", "micro_pr_auc", "macro_roc_auc"):
            if not math.isclose(
                float(point_by_model[model][metric]),
                float(expected[metric]),
                rel_tol=0.0,
                abs_tol=5e-8,
            ):
                raise ValueError(f"locked {model} metric mismatch: {metric}")

    blocks = build_blocks(records, predictions)
    samples = bootstrap_auc_samples(
        blocks,
        args.replicates,
        args.workers,
        args.seed,
        args.progress_every,
    )
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("bootstrap generated non-finite samples")

    model_index = {model: index for index, model in enumerate(MODEL_ORDER)}
    comparison_rows: list[dict[str, Any]] = []
    for comparison_id, reference, control, mechanism in COMPARISONS:
        point_delta = (
            float(point_by_model[reference]["micro_roc_auc"])
            - float(point_by_model[control]["micro_roc_auc"])
        )
        deltas = samples[:, model_index[reference]] - samples[:, model_index[control]]
        comparison_rows.append(
            {
                "comparison_id": comparison_id,
                "mechanism": mechanism,
                "reference_model": reference,
                "reference_display_name": DISPLAY_NAMES[reference],
                "control_model": control,
                "control_display_name": DISPLAY_NAMES[control],
                **bootstrap_summary(point_delta, deltas),
            }
        )
    adjusted = holm_adjust(
        [
            float(row["two_sided_empirical_p_plus_one_correction"])
            for row in comparison_rows
        ]
    )
    for row, value in zip(comparison_rows, adjusted):
        row["holm_adjusted_p"] = value
        row["interpretation"] = interpretation(row)

    seed_summary = seed_summary_rows(seed_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "s1_seed_metrics.csv", seed_rows)
    write_csv(output_dir / "s1_seed_summary.csv", seed_summary)
    write_csv(output_dir / "s1_ensemble_metrics.csv", point_rows)
    write_csv(output_dir / "s1_pairwise_bootstrap.csv", comparison_rows)
    np.savez_compressed(
        output_dir / "s1_bootstrap_auc_samples.npz",
        samples=samples,
        model_order=np.asarray(MODEL_ORDER),
    )
    (output_dir / "s1_summary.md").write_text(
        markdown_summary(seed_summary, point_rows, comparison_rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "status": "pass",
        "experiment": "s1_post_lock_mechanistic_ablation_analysis",
        "purpose": "explanation_only_no_a10_reselection",
        "input_verification": verification,
        "reference_logit_reconstruction_max_abs_difference": {
            "A5": a5_max_difference,
            "A9": a9_max_difference,
        },
        "proteins": len(records),
        "known_residues": int(sum(np.sum(record.labels >= 0) for record in records)),
        "seeds": list(SEEDS),
        "models": list(MODEL_ORDER),
        "comparisons": len(COMPARISONS),
        "primary_metric": "micro_roc_auc",
        "bootstrap": {
            "paired": True,
            "sampling_unit": "whole_protein",
            "stratified_by_oof_fold": True,
            "replicates": args.replicates,
            "random_seed": args.seed,
            "workers": args.workers,
        },
        "multiple_testing": {
            "method": "Holm",
            "family": "seven_predeclared_primary_roc_auc_comparisons",
            "hypotheses": len(COMPARISONS),
        },
        "seed_summary": seed_summary,
        "ensemble_metrics": point_rows,
        "pairwise_bootstrap": comparison_rows,
        "elapsed_seconds": perf_counter() - started,
        "caid1_caid2_caid3_used_for_training_tuning_weighting_or_reselection": False,
        "locked_a10_modified": False,
    }
    (output_dir / "s1_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "models": len(MODEL_ORDER),
                "comparisons": len(COMPARISONS),
                "bootstrap_replicates": args.replicates,
                "output_dir": str(output_dir),
                "caid_labels_used": False,
                "locked_a10_modified": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
