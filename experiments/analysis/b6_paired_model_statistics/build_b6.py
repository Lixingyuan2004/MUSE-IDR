"""Run post-lock paired protein-cluster statistics for A10 and baselines.

The script verifies every reference and prediction SHA256 before reading labels,
recomputes the locked point estimates, and then resamples whole proteins.  CAID
labels are evaluation-only and are never used to choose models, weights, losses,
thresholds, or hyperparameters.
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

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b1_caid3_fair_comparison.build_b1 import (  # noqa: E402
    Prediction,
    align_method,
    compute_metrics,
    flatten_pairs,
    make_common_pairs,
    paired_delong,
    parse_reference,
)


MODEL_ORDER = (
    "A10-locked",
    "LoRA-DR-Suite 650M",
    "PUNCH2-Light Paper-8",
    "PUNCH2-Light Released-13",
)
METRIC_ORDER = (
    "roc_auc_full_precision",
    "aucpr_trapezoid_full_precision",
    "aps_full_precision",
    "f1_at_0_5_full_precision",
    "mcc_at_0_5_full_precision",
    "fmax_full_precision",
)
TRACK_ORDER = ("disorder_nox", "disorder_pdb")


@dataclass(frozen=True)
class ProteinBlock:
    protein_id: str
    labels: np.ndarray
    scores: tuple[np.ndarray, ...]


_WORKER_BLOCKS: list[ProteinBlock] | None = None
_WORKER_SEED = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_locked_file(path: Path, expected_sha256: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected_sha256}")
    return {
        "path": str(path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "match": True,
    }


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def parse_prediction_tsv(path: Path) -> dict[str, Prediction]:
    residues: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    origins: dict[str, int] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        required = {"protein_id", "residue", "idr_probability"}
        if not required.issubset(fields):
            raise ValueError(f"prediction TSV has the wrong header: {path}")
        position_key = "position" if "position" in fields else "residue_index"
        if position_key not in fields:
            raise ValueError(f"prediction TSV has no position column: {path}")
        for line_number, row in enumerate(reader, start=2):
            protein_id = str(row["protein_id"])
            residue = str(row["residue"]).upper()
            position = int(row[position_key])
            score = float(row["idr_probability"])
            if not protein_id or len(residue) != 1 or not math.isfinite(score):
                raise ValueError(f"invalid prediction at {path}:{line_number}")
            if protein_id not in residues:
                if position not in {0, 1}:
                    raise ValueError(f"positions must start at 0 or 1: {path}:{line_number}")
                residues[protein_id] = []
                scores[protein_id] = []
                origins[protein_id] = position
            expected_position = origins[protein_id] + len(residues[protein_id])
            if position != expected_position:
                raise ValueError(
                    f"non-contiguous positions for {protein_id}: "
                    f"{position} != {expected_position}"
                )
            residues[protein_id].append(residue)
            scores[protein_id].append(score)
    if not residues:
        raise ValueError(f"empty prediction TSV: {path}")
    return {
        protein_id: Prediction(
            protein_id=protein_id,
            sequence="".join(residues[protein_id]),
            scores=np.asarray(scores[protein_id], dtype=np.float64),
        )
        for protein_id in residues
    }


def full_precision_metric_vector(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Compute six full-precision metrics with one stable score sort."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.size != y_score.size:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("both classes are required")

    order = np.argsort(y_score, kind="mergesort")[::-1]
    ordered_scores = y_score[order]
    ordered_positive = y_true[order] == 1
    distinct = np.where(np.diff(ordered_scores))[0]
    indices = np.r_[distinct, ordered_positive.size - 1]
    tp = np.cumsum(ordered_positive, dtype=np.float64)[indices]
    fp = 1 + indices - tp
    precision = tp / (tp + fp)
    recall = tp / positives
    fpr = fp / negatives

    roc_auc = float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, fpr]))
    aucpr = float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))
    aps = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    f1_curve = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    fmax = float(np.max(f1_curve))

    predicted = y_score >= 0.5
    positive = y_true == 1
    tp_default = int(np.sum(predicted & positive))
    tn_default = int(np.sum(~predicted & ~positive))
    fp_default = int(np.sum(predicted & ~positive))
    fn_default = int(np.sum(~predicted & positive))
    default_precision = (
        tp_default / (tp_default + fp_default) if tp_default + fp_default else 0.0
    )
    default_recall = tp_default / (tp_default + fn_default)
    f1_default = (
        2.0 * default_precision * default_recall / (default_precision + default_recall)
        if default_precision + default_recall
        else 0.0
    )
    denominator = math.sqrt(
        (tp_default + fp_default)
        * (tp_default + fn_default)
        * (tn_default + fp_default)
        * (tn_default + fn_default)
    )
    mcc_default = (
        (tp_default * tn_default - fp_default * fn_default) / denominator
        if denominator
        else 0.0
    )
    return np.asarray(
        [roc_auc, aucpr, aps, f1_default, mcc_default, fmax],
        dtype=np.float64,
    )


def build_protein_blocks(
    per_model: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
) -> list[ProteinBlock]:
    protein_sets = [set(per_model[model]) for model in MODEL_ORDER]
    if not all(protein_set == protein_sets[0] for protein_set in protein_sets[1:]):
        raise ValueError("models do not cover the same known-label proteins")
    blocks: list[ProteinBlock] = []
    for protein_id in sorted(protein_sets[0]):
        labels = per_model[MODEL_ORDER[0]][protein_id][0]
        model_scores: list[np.ndarray] = []
        for model in MODEL_ORDER:
            model_labels, scores = per_model[model][protein_id]
            if not np.array_equal(labels, model_labels):
                raise ValueError(f"labels do not align for {protein_id}: {model}")
            model_scores.append(scores)
        blocks.append(ProteinBlock(protein_id, labels, tuple(model_scores)))
    if not blocks:
        raise ValueError("no common protein blocks")
    return blocks


def _init_worker(blocks: list[ProteinBlock], random_seed: int) -> None:
    global _WORKER_BLOCKS, _WORKER_SEED
    _WORKER_BLOCKS = blocks
    _WORKER_SEED = random_seed


def _bootstrap_chunk(start: int, count: int) -> tuple[int, np.ndarray]:
    if _WORKER_BLOCKS is None:
        raise RuntimeError("bootstrap worker was not initialized")
    blocks = _WORKER_BLOCKS
    output = np.empty((count, len(MODEL_ORDER), len(METRIC_ORDER)), dtype=np.float64)
    for offset in range(count):
        replicate_index = start + offset
        rng = np.random.default_rng(np.random.SeedSequence([_WORKER_SEED, replicate_index]))
        selected = rng.integers(0, len(blocks), size=len(blocks))
        sampled = [blocks[int(index)] for index in selected]
        labels = np.concatenate([block.labels for block in sampled])
        if np.unique(labels).size != 2:
            raise RuntimeError(f"bootstrap replicate {replicate_index} lost one class")
        for model_index in range(len(MODEL_ORDER)):
            scores = np.concatenate([block.scores[model_index] for block in sampled])
            output[offset, model_index] = full_precision_metric_vector(labels, scores)
    return start, output


def bootstrap_metric_samples(
    blocks: list[ProteinBlock],
    replicates: int,
    workers: int,
    random_seed: int,
    progress_every: int,
) -> np.ndarray:
    if replicates <= 0:
        return np.empty((0, len(MODEL_ORDER), len(METRIC_ORDER)), dtype=np.float64)
    workers = max(1, min(workers, replicates))
    chunk_size = max(1, min(25, math.ceil(replicates / (workers * 4))))
    chunks = [
        (start, min(chunk_size, replicates - start))
        for start in range(0, replicates, chunk_size)
    ]
    output = np.empty((replicates, len(MODEL_ORDER), len(METRIC_ORDER)), dtype=np.float64)
    completed = 0
    next_progress = progress_every
    if workers == 1:
        _init_worker(blocks, random_seed)
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
        initargs=(blocks, random_seed),
    ) as executor:
        future_to_chunk = {
            executor.submit(_bootstrap_chunk, start, count): (start, count)
            for start, count in chunks
        }
        for future in as_completed(future_to_chunk):
            start, count = future_to_chunk[future]
            returned_start, values = future.result()
            if returned_start != start or values.shape[0] != count:
                raise RuntimeError("bootstrap worker returned a malformed chunk")
            output[start : start + count] = values
            completed += count
            if progress_every and completed >= next_progress:
                print(json.dumps({"bootstrap_completed": completed}), flush=True)
                next_progress += progress_every
    return output


def bootstrap_summary(point_delta: float, values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be a finite non-empty vector")
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


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=np.float64), kind="mergesort")
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[int(index)]
        running = max(running, candidate)
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


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


def markdown_summary(comparisons: list[dict[str, object]]) -> str:
    lines = [
        "# B6 paired protein-cluster statistics",
        "",
        "All predictions and references were SHA256-verified before label access. "
        "Bootstrap resampling uses whole proteins, and all deltas are A10 minus comparator.",
        "",
        "## ROC-AUC comparisons",
        "",
        "| Dataset | Track | Comparator | Full-precision ROC delta | 95% CI | Bootstrap p (Holm) | DeLong p (Holm) |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in comparisons:
        if row["metric"] != "roc_auc_full_precision":
            continue
        ci = row["percentile_95_ci"]
        lines.append(
            "| {dataset} | {track} | {second_model} | {point_delta:+.6f} | "
            "[{low:+.6f}, {high:+.6f}] | {boot:.6g} | {delong:.6g} |".format(
                low=ci[0],
                high=ci[1],
                boot=row["bootstrap_holm_p"],
                delong=row["delong_holm_p"],
                **row,
            )
        )
    lines += [
        "",
        "## Interpretation policy",
        "",
        "- A positive delta with a 95% CI entirely above zero supports A10 superiority.",
        "- A CI crossing zero is reported as comparable or as a higher point estimate, not as significant superiority.",
        "- Primary Holm correction covers A10 vs LoRA and A10 vs Released-13 across all four tracks and six metrics.",
        "- Paper-8 comparisons are supplementary and corrected in a separate family.",
        "- CAID2/CAID3 labels remain evaluation-only and must not be used to retune A10.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("b6_lock_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/analysis/b6_paired_model_statistics/formal",
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if manifest.get("status") != "locked":
        raise ValueError("B6 manifest is not locked")
    if manifest.get("model_selection_completed_before_caid_label_access") is not True:
        raise ValueError("model selection was not completed before CAID label access")
    if manifest.get("caid_labels_used_for_training_or_tuning") is not False:
        raise ValueError("CAID labels were used for training or tuning")
    if args.replicates <= 0:
        raise ValueError("formal B6 requires positive bootstrap replicates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_verification: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    sample_arrays: dict[str, np.ndarray] = {}

    for dataset_name, dataset in manifest["datasets"].items():
        predictions: dict[str, dict[str, Prediction]] = {}
        for model in MODEL_ORDER:
            entry = dataset["predictions"][model]
            prediction_path = PROJECT_ROOT / entry["path"]
            lock_verification.append(
                {"dataset": dataset_name, "kind": "prediction", "model": model}
                | verify_locked_file(prediction_path, entry["sha256"])
            )
            predictions[model] = parse_prediction_tsv(prediction_path)

        for track_index, track in enumerate(TRACK_ORDER):
            reference_entry = dataset["references"][track]
            reference_path = PROJECT_ROOT / reference_entry["path"]
            lock_verification.append(
                {"dataset": dataset_name, "kind": "reference", "track": track}
                | verify_locked_file(reference_path, reference_entry["sha256"])
            )
            reference = parse_reference(reference_path.read_text(encoding="utf-8-sig"))
            per_model: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
            full_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for model in MODEL_ORDER:
                labels, scores, per_protein, coverage = align_method(
                    reference, predictions[model]
                )
                if coverage["protein_coverage"] != 1.0 or coverage["residue_coverage"] != 1.0:
                    raise ValueError(f"incomplete coverage: {dataset_name} {track} {model}")
                if coverage["substantive_residue_mismatches"] != 0:
                    raise ValueError(f"residue mismatch: {dataset_name} {track} {model}")
                metrics = compute_metrics(labels, scores)
                expected_roc = dataset["expected_roc_auc_caid"][track][model]
                if metrics["roc_auc_caid_compatible"] != expected_roc:
                    raise ValueError(
                        f"locked ROC mismatch for {dataset_name} {track} {model}: "
                        f"{metrics['roc_auc_caid_compatible']} != {expected_roc}"
                    )
                point_rows.append(
                    {
                        "dataset": dataset_name,
                        "track": track,
                        "model": model,
                        **coverage,
                        **metrics,
                    }
                )
                per_model[model] = per_protein
                full_arrays[model] = (labels, scores)

            blocks = build_protein_blocks(per_model)
            track_seed = args.seed + 1000 * list(manifest["datasets"]).index(dataset_name) + track_index
            print(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "track": track,
                        "proteins": len(blocks),
                        "replicates": args.replicates,
                        "workers": args.workers,
                        "seed": track_seed,
                    }
                ),
                flush=True,
            )
            samples = bootstrap_metric_samples(
                blocks,
                args.replicates,
                args.workers,
                track_seed,
                args.progress_every,
            )
            sample_arrays[f"{dataset_name}__{track}"] = samples
            point_vectors = {
                model: full_precision_metric_vector(*full_arrays[model])
                for model in MODEL_ORDER
            }
            for comparator in manifest["primary_comparators"] + manifest["supplementary_comparators"]:
                comparator_index = MODEL_ORDER.index(comparator)
                pairs = make_common_pairs(per_model[MODEL_ORDER[0]], per_model[comparator])
                labels, first_scores, second_scores = flatten_pairs(pairs)
                delong = paired_delong(labels, first_scores, second_scores)
                for metric_index, metric in enumerate(METRIC_ORDER):
                    point_delta = (
                        point_vectors[MODEL_ORDER[0]][metric_index]
                        - point_vectors[comparator][metric_index]
                    )
                    summary = bootstrap_summary(
                        point_delta,
                        samples[:, 0, metric_index]
                        - samples[:, comparator_index, metric_index],
                    )
                    comparison_rows.append(
                        {
                            "dataset": dataset_name,
                            "track": track,
                            "first_model": MODEL_ORDER[0],
                            "second_model": comparator,
                            "comparison_family": (
                                "primary"
                                if comparator in manifest["primary_comparators"]
                                else "supplementary"
                            ),
                            "metric": metric,
                            "common_proteins": len(pairs),
                            "replicates": args.replicates,
                            "random_seed": track_seed,
                            **summary,
                            "delong_two_sided_p": (
                                delong["two_sided_p_value"]
                                if metric == "roc_auc_full_precision"
                                else None
                            ),
                        }
                    )

    for family in ("primary", "supplementary"):
        family_rows = [row for row in comparison_rows if row["comparison_family"] == family]
        adjusted = holm_adjust(
            [float(row["two_sided_empirical_p_plus_one_correction"]) for row in family_rows]
        )
        for row, value in zip(family_rows, adjusted):
            row["bootstrap_holm_p"] = value
        roc_rows = [row for row in family_rows if row["metric"] == "roc_auc_full_precision"]
        delong_adjusted = holm_adjust([float(row["delong_two_sided_p"]) for row in roc_rows])
        for row, value in zip(roc_rows, delong_adjusted):
            row["delong_holm_p"] = value
        for row in family_rows:
            if row["metric"] != "roc_auc_full_precision":
                row["delong_holm_p"] = None

    write_csv(args.output_dir / "b6_point_metrics.csv", point_rows)
    write_csv(args.output_dir / "b6_pairwise_statistics.csv", comparison_rows)
    np.savez_compressed(
        args.output_dir / "b6_bootstrap_samples.npz",
        **sample_arrays,
    )
    (args.output_dir / "b6_summary.md").write_text(
        markdown_summary(comparison_rows), encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "experiment": "b6_paired_model_statistics",
        "status": "pass",
        "sampling_unit": "protein",
        "paired": True,
        "bootstrap_replicates": args.replicates,
        "workers": args.workers,
        "random_seed": args.seed,
        "model_order": MODEL_ORDER,
        "metric_order": METRIC_ORDER,
        "primary_comparators": manifest["primary_comparators"],
        "supplementary_comparators": manifest["supplementary_comparators"],
        "multiple_testing": {
            "method": "Holm",
            "primary_family": "2 comparators x 4 tracks x 6 metrics = 48 hypotheses",
            "supplementary_family": "1 comparator x 4 tracks x 6 metrics = 24 hypotheses",
        },
        "lock_manifest": str(args.manifest),
        "lock_manifest_sha256": sha256(args.manifest),
        "all_input_hashes_match": True,
        "lock_verification": lock_verification,
        "point_metrics": point_rows,
        "pairwise_statistics": comparison_rows,
        "model_selection_completed_before_caid_label_access": True,
        "caid_labels_used_for_training_or_tuning": False,
        "caid_labels_used_for_post_lock_evaluation": True,
    }
    (args.output_dir / "b6_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "point_metric_rows": len(point_rows),
                "pairwise_metric_rows": len(comparison_rows),
                "bootstrap_replicates_per_track": args.replicates,
                "output_dir": str(args.output_dir),
                "caid_labels_used_for_training_or_tuning": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
