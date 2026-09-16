"""Evaluate locked A10 predictions on the frozen database-curated S2 holdout.

All immutable prediction and reference inputs are SHA256-verified before
semantic label access. This is post-lock evaluation only and must never be
used to train, tune, recalibrate, reweight, or reselect A10.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
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
    full_precision_metric_vector,
)


EXPECTED_ARCHIVE_SHA256 = (
    "10aa2b49d8aa7243aaef8d88ab1314f59415a9f398feda89315c951e57a4d7e6"
)
EXPECTED_PREDICTION_SHA256 = (
    "70555772455afb707200036cf93266236013d18accdce1c0289a3a8236ab541b"
)
EXPECTED_COUNTS = {
    "proteins": 87,
    "residues": 56_394,
    "known_residues": 30_728,
    "positive_residues": 2_999,
    "negative_residues": 27_729,
    "unknown_residues": 25_666,
}
DEFAULT_ARCHIVE = PROJECT_ROOT / "release/s2_a10_prediction_results_20260907.tar.gz"
DEFAULT_REFERENCE_DIR = (
    PROJECT_ROOT
    / "outputs/supplementary/s2_independent_holdout/final_database_curated_20260906"
)
DEFAULT_PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs/supplementary/s2_independent_holdout/a10_prediction_20260907"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/supplementary/s2_independent_holdout/evaluation_20260907"
)
BOOTSTRAP_METRICS = (*METRIC_ORDER, "brier_score", "macro_roc_auc")


@dataclass(frozen=True)
class ProteinBlock:
    protein_id: str
    sequence: str
    labels: np.ndarray
    scores: np.ndarray
    source_entry_release: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected.lower():
        raise ValueError(f"SHA256 mismatch for {path}: {observed} != {expected}")
    return {"path": str(path), "sha256": observed, "match": True}


def verify_lock_manifest(lock_path: Path, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    seen: set[str] = set()
    count = 0
    for line_number, raw_line in enumerate(
        lock_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            expected, relative_text = raw_line.split("  ", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"invalid lock line {line_number}: {lock_path}") from error
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe locked path: {relative_text}")
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"duplicate locked path: {normalized}")
        seen.add(normalized)
        target = root / relative
        if not target.is_file():
            raise FileNotFoundError(f"missing locked file: {target}")
        observed = sha256(target)
        if observed != expected.lower():
            raise ValueError(
                f"locked-file SHA256 mismatch for {normalized}: "
                f"{observed} != {expected.lower()}"
            )
        count += 1
    if not count:
        raise ValueError(f"empty SHA256 lock: {lock_path}")
    return {
        "lock_path": str(lock_path),
        "lock_sha256": sha256(lock_path),
        "verified_files": count,
        "all_match": True,
    }


def read_cohort(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"protein_id", "length", "source_entry_release"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"unexpected S2 cohort header: {reader.fieldnames}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            protein_id = str(row["protein_id"])
            if protein_id in rows:
                raise ValueError(f"duplicate cohort protein: {protein_id}")
            rows[protein_id] = dict(row)
    return rows


def read_labels(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    residues: dict[str, list[str]] = {}
    values: dict[str, list[int]] = {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["protein_id", "position", "residue", "label"]:
            raise ValueError(f"unexpected S2 label header: {reader.fieldnames}")
        for row in reader:
            protein_id = str(row["protein_id"])
            position = int(row["position"])
            label = int(row["label"])
            if label not in (-1, 0, 1):
                raise ValueError(f"unsupported label for {protein_id}: {label}")
            expected_position = len(residues.setdefault(protein_id, [])) + 1
            if position != expected_position:
                raise ValueError(
                    f"non-contiguous label position for {protein_id}: "
                    f"{position} != {expected_position}"
                )
            residues[protein_id].append(str(row["residue"]))
            values.setdefault(protein_id, []).append(label)
    return {
        protein_id: (
            "".join(residues[protein_id]),
            np.asarray(values[protein_id], dtype=np.int8),
        )
        for protein_id in residues
    }


def read_predictions(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    residues: dict[str, list[str]] = {}
    values: dict[str, list[float]] = {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_header = [
            "protein_id", "position", "residue", "idr_probability"
        ]
        if reader.fieldnames != expected_header:
            raise ValueError(f"unexpected S2 prediction header: {reader.fieldnames}")
        for row in reader:
            protein_id = str(row["protein_id"])
            position = int(row["position"])
            score = float(row["idr_probability"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"invalid probability for {protein_id}: {score}")
            expected_position = len(residues.setdefault(protein_id, [])) + 1
            if position != expected_position:
                raise ValueError(
                    f"non-contiguous prediction position for {protein_id}: "
                    f"{position} != {expected_position}"
                )
            residues[protein_id].append(str(row["residue"]))
            values.setdefault(protein_id, []).append(score)
    return {
        protein_id: (
            "".join(residues[protein_id]),
            np.asarray(values[protein_id], dtype=np.float64),
        )
        for protein_id in residues
    }


def align_blocks(
    labels: dict[str, tuple[str, np.ndarray]],
    predictions: dict[str, tuple[str, np.ndarray]],
    cohort: dict[str, dict[str, str]],
) -> list[ProteinBlock]:
    if set(labels) != set(predictions) or set(labels) != set(cohort):
        raise ValueError("S2 label, prediction, and cohort protein sets differ")
    blocks: list[ProteinBlock] = []
    for protein_id in labels:
        sequence, protein_labels = labels[protein_id]
        predicted_sequence, protein_scores = predictions[protein_id]
        if sequence != predicted_sequence:
            raise ValueError(f"S2 prediction sequence mismatch: {protein_id}")
        if protein_labels.shape != protein_scores.shape:
            raise ValueError(f"S2 prediction length mismatch: {protein_id}")
        row = cohort[protein_id]
        if int(row["length"]) != len(sequence):
            raise ValueError(f"S2 cohort length mismatch: {protein_id}")
        blocks.append(
            ProteinBlock(
                protein_id,
                sequence,
                protein_labels,
                protein_scores,
                str(row["source_entry_release"]),
            )
        )
    return blocks


def flatten_known(blocks: Sequence[ProteinBlock]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([block.labels[block.labels != -1] for block in blocks]),
        np.concatenate([block.scores[block.labels != -1] for block in blocks]),
    )


def extended_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    if np.unique(y_true).tolist() != [0, 1]:
        raise ValueError("metrics require both classes")
    result = dict(compute_metrics(y_true, y_score))
    default = confusion_metrics(y_true, y_score, 0.5)
    result.update(
        {
            "brier_score": float(np.mean((y_score - y_true) ** 2)),
            "log_loss": float(
                -np.mean(
                    y_true * np.log(np.clip(y_score, 1e-12, 1.0))
                    + (1 - y_true)
                    * np.log(np.clip(1.0 - y_score, 1e-12, 1.0))
                )
            ),
            "precision_at_0_5_full_precision": default["tp"]
            / (default["tp"] + default["fp"])
            if default["tp"] + default["fp"]
            else 0.0,
            "recall_at_0_5_full_precision": default["tp"]
            / (default["tp"] + default["fn"]),
            "specificity_at_0_5_full_precision": default["tn"]
            / (default["tn"] + default["fp"]),
            "accuracy_at_0_5_full_precision": (default["tp"] + default["tn"])
            / y_true.size,
            "confusion_at_0_5": {
                key: int(default[key]) for key in ("tp", "tn", "fp", "fn")
            },
        }
    )
    return result


def per_protein_rows(blocks: Sequence[ProteinBlock]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        known = block.labels != -1
        y_true, y_score = block.labels[known], block.scores[known]
        positives, negatives = int(y_true.sum()), int(y_true.size - y_true.sum())
        default = confusion_metrics(y_true, y_score, 0.5)
        metric = compute_metrics(y_true, y_score) if positives and negatives else None
        rows.append(
            {
                "protein_id": block.protein_id,
                "source_entry_release": block.source_entry_release,
                "length": len(block.sequence),
                "known_residues": int(y_true.size),
                "positive_residues": positives,
                "negative_residues": negatives,
                "roc_auc_full_precision": (
                    metric["roc_auc_full_precision"] if metric else None
                ),
                "aps_full_precision": metric["aps_full_precision"] if metric else None,
                "f1_at_0_5_full_precision": default["f1"],
                "mcc_at_0_5_full_precision": default["mcc"],
                "brier_score": float(np.mean((y_score - y_true) ** 2)),
            }
        )
    return rows


def macro_roc_auc(rows: Sequence[dict[str, Any]]) -> float:
    values = [
        float(row["roc_auc_full_precision"])
        for row in rows
        if row["roc_auc_full_precision"] is not None
    ]
    if not values:
        raise ValueError("no two-class proteins for macro ROC-AUC")
    return float(np.mean(values))


def bootstrap_samples(
    blocks: Sequence[ProteinBlock], replicates: int, seed: int, progress_every: int
) -> np.ndarray:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    protein_rows = per_protein_rows(blocks)
    output = np.empty((replicates, len(BOOTSTRAP_METRICS)), dtype=np.float64)
    for replicate in range(replicates):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        indices = rng.integers(0, len(blocks), size=len(blocks))
        sampled_blocks = [blocks[int(index)] for index in indices]
        y_true, y_score = flatten_known(sampled_blocks)
        if np.unique(y_true).tolist() != [0, 1]:
            raise RuntimeError(f"bootstrap replicate {replicate} lost one class")
        output[replicate] = np.r_[
            full_precision_metric_vector(y_true, y_score),
            np.mean((y_score - y_true) ** 2),
            macro_roc_auc([protein_rows[int(index)] for index in indices]),
        ]
        if progress_every and (replicate + 1) % progress_every == 0:
            print(json.dumps({"bootstrap_completed": replicate + 1}), flush=True)
    return output


def summarize_bootstrap(samples: np.ndarray) -> dict[str, dict[str, float]]:
    if samples.shape[1] != len(BOOTSTRAP_METRICS) or not np.all(np.isfinite(samples)):
        raise ValueError("invalid S2 bootstrap samples")
    return {
        metric: {
            "mean": float(np.mean(samples[:, index])),
            "sample_sd": float(np.std(samples[:, index], ddof=1)),
            "ci_95_percentile_low": float(np.quantile(samples[:, index], 0.025)),
            "ci_95_percentile_high": float(np.quantile(samples[:, index], 0.975)),
        }
        for index, metric in enumerate(BOOTSTRAP_METRICS)
    }


def release_strata_rows(blocks: Sequence[ProteinBlock]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for release in sorted({block.source_entry_release for block in blocks}):
        selected = [block for block in blocks if block.source_entry_release == release]
        y_true, y_score = flatten_known(selected)
        positives, negatives = int(y_true.sum()), int(y_true.size - y_true.sum())
        metric = extended_metrics(y_true, y_score) if positives and negatives else None
        rows.append(
            {
                "source_entry_release": release,
                "proteins": len(selected),
                "known_residues": int(y_true.size),
                "positive_residues": positives,
                "negative_residues": negatives,
                "roc_auc_full_precision": (
                    metric["roc_auc_full_precision"] if metric else None
                ),
                "aps_full_precision": metric["aps_full_precision"] if metric else None,
                "f1_at_0_5_full_precision": (
                    metric["f1_at_0_5_full_precision"] if metric else None
                ),
                "mcc_at_0_5_full_precision": (
                    metric["mcc_at_0_5_full_precision"] if metric else None
                ),
                "brier_score": metric["brier_score"] if metric else None,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    point: dict[str, Any], bootstrap: dict[str, dict[str, float]], macro_auc: float
) -> str:
    def row(label: str, metric: str, point_value: float) -> str:
        item = bootstrap[metric]
        return (
            f"| {label} | {point_value:.6f} | "
            f"[{item['ci_95_percentile_low']:.6f}, "
            f"{item['ci_95_percentile_high']:.6f}] |"
        )

    known = EXPECTED_COUNTS["known_residues"]
    lines = [
        "# S2 database-curated temporal holdout evaluation",
        "",
        "All prediction and reference hashes were verified before semantic label "
        "access. A10 remained locked; S2 was not used for training, tuning, "
        "recalibration, weighting, or model selection.",
        "",
        "## Cohort",
        "",
        f"- Proteins: {EXPECTED_COUNTS['proteins']}",
        f"- Residues: {EXPECTED_COUNTS['residues']}",
        f"- Known labels: {known} "
        f"({100.0 * known / EXPECTED_COUNTS['residues']:.2f}%)",
        f"- Positive / negative / unknown: {EXPECTED_COUNTS['positive_residues']} / "
        f"{EXPECTED_COUNTS['negative_residues']} / "
        f"{EXPECTED_COUNTS['unknown_residues']}",
        f"- Positive prevalence among known labels: "
        f"{100.0 * EXPECTED_COUNTS['positive_residues'] / known:.2f}%",
        "- Reference policy: database-curated DisProt annotations; no partial "
        "manual relabeling was applied.",
        "",
        "## Locked MUSE-IDR A10 results",
        "",
        "| Metric | Point estimate | Protein-bootstrap 95% CI |",
        "| --- | ---: | ---: |",
        row("ROC-AUC", "roc_auc_full_precision", point["roc_auc_full_precision"]),
        row(
            "AUPRC",
            "aucpr_trapezoid_full_precision",
            point["aucpr_trapezoid_full_precision"],
        ),
        row("APS", "aps_full_precision", point["aps_full_precision"]),
        row("F1 at 0.5", "f1_at_0_5_full_precision", point["f1_at_0_5_full_precision"]),
        row("MCC at 0.5", "mcc_at_0_5_full_precision", point["mcc_at_0_5_full_precision"]),
        row("Fmax (descriptive)", "fmax_full_precision", point["fmax_full_precision"]),
        row("Brier score", "brier_score", point["brier_score"]),
        row("Macro ROC-AUC (two-class proteins)", "macro_roc_auc", macro_auc),
        "",
        "## Interpretation constraints",
        "",
        "- Full-precision scores are primary; unknown labels (-1) are excluded.",
        "- The fixed 0.5 threshold is locked. Fmax is descriptive and is not used "
        "to retune A10.",
        "- Macro ROC-AUC includes only proteins containing both known classes.",
        "- S2 supplements, but does not replace, the CAID2/CAID3 benchmarks.",
        "- No S2 superiority claim is supported until comparator predictions are "
        "evaluated on this exact reference.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    reference_dir = args.reference_dir.resolve()
    prediction_dir = args.prediction_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite S2 evaluation: {output_dir}")

    # These checks deliberately precede read_labels().
    archive_verification = verify_file(archive, EXPECTED_ARCHIVE_SHA256)
    prediction_lock = verify_lock_manifest(
        prediction_dir / "s2_a10_prediction_lock.sha256"
    )
    reference_lock = verify_lock_manifest(reference_dir / "s2_final_lock.sha256")
    prediction_path = prediction_dir / "muse_idr_a10_s2_predictions.tsv.gz"
    verify_file(prediction_path, EXPECTED_PREDICTION_SHA256)

    cohort = read_cohort(reference_dir / "s2_cohort.tsv")
    labels = read_labels(reference_dir / "s2_labels.tsv.gz")
    predictions = read_predictions(prediction_path)
    blocks = align_blocks(labels, predictions, cohort)
    y_true, y_score = flatten_known(blocks)
    observed_counts = {
        "proteins": len(blocks),
        "residues": sum(len(block.sequence) for block in blocks),
        "known_residues": int(y_true.size),
        "positive_residues": int(np.sum(y_true == 1)),
        "negative_residues": int(np.sum(y_true == 0)),
        "unknown_residues": int(sum(np.sum(block.labels == -1) for block in blocks)),
    }
    if observed_counts != EXPECTED_COUNTS:
        raise ValueError(
            f"S2 frozen count mismatch: {observed_counts} != {EXPECTED_COUNTS}"
        )

    point = extended_metrics(y_true, y_score)
    protein_rows = per_protein_rows(blocks)
    macro_auc = macro_roc_auc(protein_rows)
    two_class_proteins = sum(
        row["roc_auc_full_precision"] is not None for row in protein_rows
    )
    samples = bootstrap_samples(
        blocks, args.replicates, args.seed, args.progress_every
    )
    bootstrap = summarize_bootstrap(samples)
    strata_rows = release_strata_rows(blocks)

    output_dir.mkdir(parents=True)
    point_path = output_dir / "s2_point_metrics.json"
    protein_path = output_dir / "s2_per_protein_metrics.csv"
    strata_path = output_dir / "s2_release_strata_metrics.csv"
    samples_path = output_dir / "s2_bootstrap_samples.npz"
    report_path = output_dir / "s2_evaluation_report.json"
    summary_path = output_dir / "s2_summary.md"
    point_path.write_text(
        json.dumps(point, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_csv(protein_path, protein_rows)
    write_csv(strata_path, strata_rows)
    np.savez_compressed(
        samples_path,
        samples=samples,
        metric_names=np.asarray(BOOTSTRAP_METRICS),
        seed=np.asarray(args.seed),
    )
    summary_path.write_text(
        build_summary(point, bootstrap, macro_auc), encoding="utf-8", newline="\n"
    )
    report = {
        "schema_version": 1,
        "experiment": "s2_database_curated_temporal_holdout_evaluation",
        "status": "pass",
        "prediction_archive_verification": archive_verification,
        "prediction_lock_verification": prediction_lock,
        "reference_lock_verification": reference_lock,
        "hashes_verified_before_semantic_label_access": True,
        "cohort": observed_counts,
        "positive_prevalence_among_known": EXPECTED_COUNTS["positive_residues"]
        / EXPECTED_COUNTS["known_residues"],
        "point_metrics": point,
        "macro_roc_auc_two_class_proteins": macro_auc,
        "two_class_proteins": two_class_proteins,
        "bootstrap": {
            "unit": "whole_protein_cluster",
            "replicates": args.replicates,
            "seed": args.seed,
            "summary": bootstrap,
        },
        "reference_policy": "database_curated_disprot_no_partial_manual_relabeling",
        "s2_role": "supplementary_temporal_database_curated_holdout",
        "unknown_labels_excluded": True,
        "fixed_threshold": 0.5,
        "fmax_used_for_tuning": False,
        "comparator_predictions_evaluated": False,
        "scores_used_for_training_tuning_recalibration_weighting_or_selection": False,
        "locked_a10_modified": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    lock_members = [
        Path(__file__).resolve(),
        Path(__file__).with_name("test_evaluate_s2_a10.py").resolve(),
        archive,
        reference_dir / "s2_final_lock.sha256",
        prediction_dir / "s2_a10_prediction_lock.sha256",
        point_path,
        protein_path,
        strata_path,
        samples_path,
        report_path,
        summary_path,
    ]
    lock_path = output_dir / "s2_evaluation_lock.sha256"
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
                "proteins": len(blocks),
                "known_residues": int(y_true.size),
                "two_class_proteins": two_class_proteins,
                "roc_auc": point["roc_auc_full_precision"],
                "auprc": point["aucpr_trapezoid_full_precision"],
                "aps": point["aps_full_precision"],
                "f1_at_0_5": point["f1_at_0_5_full_precision"],
                "mcc_at_0_5": point["mcc_at_0_5_full_precision"],
                "fmax": point["fmax_full_precision"],
                "brier_score": point["brier_score"],
                "macro_roc_auc": macro_auc,
                "bootstrap_replicates": args.replicates,
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "locked_a10_modified": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
