"""Evaluate the locked A10 model against official CAID2 predictions.

The script performs post-lock external evaluation only.  It reads the A10
predictions produced from the label-free CAID2 union FASTA, verifies their
recorded hash, recomputes every compatible official CAID2 method with the same
metric implementation, and runs paired statistics against the top methods.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b1_caid3_fair_comparison.build_b1 import (  # noqa: E402
    Prediction,
    align_method,
    atomic_json,
    compute_metrics,
    flatten_pairs,
    make_common_pairs,
    paired_delong,
    paired_protein_bootstrap,
    parse_caid_predictions,
    parse_reference,
    write_csv,
)


DEFAULT_NOX_REFERENCE = (
    PROJECT_ROOT / "data/caid_holdout/references/caid2/disorder_nox.fasta"
)
DEFAULT_PDB_REFERENCE = (
    PROJECT_ROOT / "data/caid_holdout/references/caid2/disorder_pdb.fasta"
)
DEFAULT_PREPARATION_REPORT = (
    PROJECT_ROOT / "data/external/caid2_locked/prepared/b2_caid2_preparation_report.json"
)
DEFAULT_A10_PREDICTIONS = (
    PROJECT_ROOT / "outputs/final/a10/caid2_union_predictions.tsv.gz"
)
DEFAULT_A10_REPORT = PROJECT_ROOT / "outputs/final/a10/caid2_union_report.json"
DEFAULT_OFFICIAL_PREDICTIONS = (
    PROJECT_ROOT
    / "data/external/caid2_locked/official_predictions/caid2_predictions.zip"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/analysis/b2_caid2_fair_comparison"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def parse_a10_tsv(path: Path) -> dict[str, Prediction]:
    residues: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"protein_id", "position", "residue", "idr_probability"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("A10 prediction TSV has the wrong header")
        for line_number, row in enumerate(reader, start=2):
            protein_id = str(row["protein_id"])
            position = int(row["position"])
            residue = str(row["residue"]).upper()
            score = float(row["idr_probability"])
            if not protein_id or len(residue) != 1 or not np.isfinite(score):
                raise ValueError(f"invalid A10 prediction at line {line_number}")
            residues.setdefault(protein_id, [])
            scores.setdefault(protein_id, [])
            expected_position = len(residues[protein_id]) + 1
            if position != expected_position:
                raise ValueError(
                    f"non-contiguous A10 positions for {protein_id}: "
                    f"{position} != {expected_position}"
                )
            residues[protein_id].append(residue)
            scores[protein_id].append(score)
    if not residues:
        raise ValueError("A10 prediction TSV is empty")
    return {
        protein_id: Prediction(
            protein_id=protein_id,
            sequence="".join(residues[protein_id]),
            scores=np.asarray(scores[protein_id], dtype=np.float64),
        )
        for protein_id in residues
    }


def markdown(
    rows: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    inference: dict[str, object],
) -> str:
    lines = [
        "# B2 locked A10 evaluation on CAID2",
        "",
        "A10 was locked before CAID2 label access. This is post-lock external evaluation only.",
        "",
    ]
    for track in ("disorder_nox", "disorder_pdb"):
        lines += [
            f"## {track}",
            "",
            "| Rank | Method | ROC-AUC | AUPRC | APS | F1@0.5 | MCC@0.5 | Fmax | Coverage |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        selected = [row for row in rows if row["track"] == track][:15]
        for row in selected:
            lines.append(
                f"| {row['rank_by_roc_auc']} | {row['method']} | "
                f"{float(row['roc_auc_caid_compatible']):.3f} | "
                f"{float(row['aucpr_trapezoid']):.3f} | "
                f"{float(row['aps']):.3f} | "
                f"{float(row['f1_at_0_5']):.3f} | "
                f"{float(row['mcc_at_0_5']):.3f} | "
                f"{float(row['fmax']):.3f} | "
                f"{100.0 * float(row['protein_coverage']):.1f}% |"
            )
        lines.append("")
    lines += [
        "## Paired statistics",
        "",
        "| Track | First | Second | Common proteins | AUC delta | DeLong p | Protein bootstrap 95% CI | Bootstrap p |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for item in pairwise:
        bootstrap = item["protein_bootstrap"]
        ci = bootstrap.get("percentile_95_ci")
        ci_text = (
            f"[{float(ci[0]):+.4f}, {float(ci[1]):+.4f}]"
            if ci is not None
            else "not run"
        )
        bootstrap_p = bootstrap.get("two_sided_empirical_p_plus_one_correction")
        lines.append(
            f"| {item['track']} | {item['first']} | {item['second']} | "
            f"{item['common_proteins']} | "
            f"{float(item['delong']['delta_first_minus_second']):+.4f} | "
            f"{float(item['delong']['two_sided_p_value']):.4g} | {ci_text} | "
            f"{float(bootstrap_p):.4g} |"
        )
    lines += [
        "",
        "## A10 inference",
        "",
        f"- Proteins: {inference['proteins']}",
        f"- Residues: {inference['residues']}",
        f"- Device: {inference['device_name']}",
        f"- Elapsed seconds: {float(inference['elapsed_seconds']):.3f}",
        f"- Peak CUDA memory GiB: {int(inference['peak_cuda_memory_bytes']) / 1024 ** 3:.3f}",
        "- Output: one classic-IDR probability per residue; no soft-disorder output.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nox-reference", type=Path, default=DEFAULT_NOX_REFERENCE)
    parser.add_argument("--pdb-reference", type=Path, default=DEFAULT_PDB_REFERENCE)
    parser.add_argument(
        "--preparation-report", type=Path, default=DEFAULT_PREPARATION_REPORT
    )
    parser.add_argument("--a10-predictions", type=Path, default=DEFAULT_A10_PREDICTIONS)
    parser.add_argument("--a10-report", type=Path, default=DEFAULT_A10_REPORT)
    parser.add_argument(
        "--official-predictions", type=Path, default=DEFAULT_OFFICIAL_PREDICTIONS
    )
    parser.add_argument("--official-predictions-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-comparators", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_comparators < 1:
        raise ValueError("--top-comparators must be positive")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")

    preparation = json.loads(args.preparation_report.read_text(encoding="utf-8-sig"))
    if preparation.get("status") != "pass":
        raise ValueError("CAID2 preparation report did not pass")
    if preparation.get("model_locked_before_caid2_label_access") is not True:
        raise ValueError("preparation report does not prove the pre-label model lock")
    if preparation.get("caid2_used_for_training_or_tuning") is not False:
        raise ValueError("preparation report permits CAID2 training or tuning")

    reference_paths = {
        "disorder_nox": args.nox_reference,
        "disorder_pdb": args.pdb_reference,
    }
    reference_report_keys = {"disorder_nox": "nox", "disorder_pdb": "pdb"}
    references = {}
    for track, path in reference_paths.items():
        expected_hash = preparation[reference_report_keys[track]]["source_sha256"]
        observed_hash = sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(f"CAID2 reference hash mismatch for {track}")
        references[track] = parse_reference(path.read_text(encoding="utf-8-sig"))

    inference = json.loads(args.a10_report.read_text(encoding="utf-8-sig"))
    if inference.get("status") != "pass":
        raise ValueError("A10 CAID2 inference did not pass")
    if inference.get("locked_members") != 30:
        raise ValueError("A10 CAID2 inference did not use 30 locked members")
    if inference.get("scores_used_for_training_or_tuning") is not False:
        raise ValueError("A10 report permits CAID2 score tuning")
    a10_hash = sha256(args.a10_predictions)
    if a10_hash != inference.get("output_sha256"):
        raise ValueError("A10 CAID2 prediction hash mismatch")
    a10_predictions = parse_a10_tsv(args.a10_predictions)

    official_hash = sha256(args.official_predictions)
    if official_hash != args.official_predictions_sha256.lower():
        raise ValueError("CAID2 official prediction archive hash mismatch")

    prediction_sets: dict[str, dict[str, Prediction]] = {
        "A10-locked": a10_predictions
    }
    skipped_archive_members: list[dict[str, str]] = []
    with zipfile.ZipFile(args.official_predictions) as official_zip:
        members = sorted(
            member
            for member in official_zip.namelist()
            if member.lower().endswith(".caid")
            and "/merged/" in member.replace("\\", "/")
        )
        if not members:
            raise ValueError("official CAID2 prediction archive has no merged .caid files")
        for member in members:
            method = Path(member).stem
            try:
                prediction_sets[method] = parse_caid_predictions(
                    official_zip.read(member).decode("utf-8-sig")
                )
            except Exception as error:  # retain a complete audit of incompatible files
                skipped_archive_members.append(
                    {"member": member, "method": method, "reason": str(error)}
                )

    rows: list[dict[str, object]] = []
    per_protein: dict[
        str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
    ] = {track: {} for track in references}
    skipped_evaluations: list[dict[str, str]] = []
    for track, reference in references.items():
        for method, prediction_set in prediction_sets.items():
            try:
                y_true, y_score, proteins, coverage = align_method(
                    reference, prediction_set
                )
            except Exception as error:
                skipped_evaluations.append(
                    {"track": track, "method": method, "reason": str(error)}
                )
                continue
            per_protein[track][method] = proteins
            rows.append(
                {
                    "track": track,
                    "method": method,
                    "source": (
                        "locked_a10"
                        if method == "A10-locked"
                        else "official_caid2_predictions"
                    ),
                    **coverage,
                    **compute_metrics(y_true, y_score),
                }
            )

    for track in references:
        ranked = sorted(
            [row for row in rows if row["track"] == track],
            key=lambda row: (
                -float(row["roc_auc_caid_compatible"]), str(row["method"])
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank_by_roc_auc"] = rank
    ranked_rows = sorted(
        rows, key=lambda row: (str(row["track"]), int(row["rank_by_roc_auc"]))
    )

    a10_rows = [row for row in ranked_rows if row["method"] == "A10-locked"]
    if len(a10_rows) != 2:
        raise RuntimeError("A10 did not evaluate on both CAID2 tracks")

    pairwise: list[dict[str, object]] = []
    for track_index, track in enumerate(references):
        official_rows = [
            row
            for row in ranked_rows
            if row["track"] == track and row["method"] != "A10-locked"
        ][: args.top_comparators]
        for comparator_index, comparator in enumerate(official_rows):
            method = str(comparator["method"])
            pairs = make_common_pairs(
                per_protein[track]["A10-locked"], per_protein[track][method]
            )
            labels, first_scores, second_scores = flatten_pairs(pairs)
            seed = args.random_seed + 100 * track_index + comparator_index
            pairwise.append(
                {
                    "track": track,
                    "first": "A10-locked",
                    "second": method,
                    "common_proteins": len(pairs),
                    "common_known_residues": int(labels.size),
                    "delong": paired_delong(labels, first_scores, second_scores),
                    "protein_bootstrap": paired_protein_bootstrap(
                        pairs, args.bootstrap_replicates, seed
                    ),
                }
            )

    a10_ranks = {
        str(row["track"]): int(row["rank_by_roc_auc"])
        for row in a10_rows
    }
    report = {
        "schema_version": 1,
        "experiment": "b2_caid2_locked_fair_comparison",
        "status": "pass",
        "objective": "ROC-AUC primary; AUPRC/APS/F1/MCC secondary",
        "a10_locked_before_caid2_label_access": True,
        "caid2_labels_used_for_training_or_tuning": False,
        "caid2_labels_used_for_post_lock_evaluation": True,
        "preparation_report": str(args.preparation_report),
        "preparation_report_sha256": sha256(args.preparation_report),
        "a10_predictions": str(args.a10_predictions),
        "a10_predictions_sha256": a10_hash,
        "a10_inference_report": str(args.a10_report),
        "a10_inference_report_sha256": sha256(args.a10_report),
        "official_predictions": str(args.official_predictions),
        "official_predictions_sha256": official_hash,
        "metrics": ranked_rows,
        "a10_ranks": a10_ranks,
        "pairwise_statistics": pairwise,
        "skipped_archive_members": skipped_archive_members,
        "skipped_evaluations": skipped_evaluations,
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
    atomic_json(args.output_dir / "b2_report.json", report)
    write_csv(args.output_dir / "b2_rankings.csv", ranked_rows)
    write_csv(args.output_dir / "b2_pairwise_statistics.csv", pairwise)
    (args.output_dir / "b2_summary.md").write_text(
        markdown(ranked_rows, pairwise, inference), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "a10_ranks": a10_ranks,
                "official_methods_parsed": len(prediction_sets) - 1,
                "pairwise_comparisons": len(pairwise),
                "bootstrap_replicates_per_comparison": args.bootstrap_replicates,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
