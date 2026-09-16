"""Build a provenance-aware unified comparison from locked B1 and B2 results.

B3 is a post-lock reporting step.  It does not read residue labels, predictions,
or model checkpoints.  Instead, it consumes the already evaluated B1 (CAID3)
and B2 (CAID2) result archives, verifies their SHA256 hashes, and separates
same-reference raw recomputations from literature-only context.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_B1_ARCHIVE = (
    PROJECT_ROOT / "release/b1_caid3_fair_comparison_20260825.tar.gz"
)
DEFAULT_B2_ARCHIVE = PROJECT_ROOT / "release/b2_caid2_final_results_20260825.tar.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/analysis/b3_unified_caid_comparison"

EXPECTED_B1_SHA256 = "5b225f47c99b52904a8ae66ea0a0968da50ec25697c747f0fd91358b4561441c"
EXPECTED_B2_SHA256 = "3a398b74f9887b34a1bc530edabedae81da046efa6473f8a30c9f9ff5235d535"

B1_REPORT_MEMBER = (
    "outputs/analysis/b1_caid3_fair_comparison/b1_report.json"
)
B2_REPORT_MEMBER = (
    "outputs/analysis/b2_caid2_fair_comparison/formal/b2_report.json"
)
B2_INFERENCE_REPORT_MEMBER = "outputs/final/a10/caid2_union_report.json"

FORMAL_COLUMNS = [
    "challenge_round",
    "track",
    "rank_by_roc_auc",
    "method",
    "model_role",
    "evidence_level",
    "same_reference",
    "rank_eligible",
    "source",
    "reference_proteins",
    "predicted_proteins",
    "protein_coverage",
    "reference_residues",
    "predicted_residues",
    "residue_coverage",
    "known_residues",
    "positive_labels",
    "negative_labels",
    "roc_auc",
    "auprc",
    "aps",
    "f1_at_0_5",
    "mcc_at_0_5",
    "fmax",
    "mcc_at_fmax",
    "fmax_threshold",
]

PAIRWISE_COLUMNS = [
    "challenge_round",
    "track",
    "first",
    "second",
    "common_proteins",
    "common_known_residues",
    "first_auc_on_common_set",
    "second_auc_on_common_set",
    "auc_delta_first_minus_second",
    "delong_two_sided_p",
    "bootstrap_replicates",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_probability_delta_positive",
    "bootstrap_two_sided_empirical_p",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_tar_bytes(archive: Path, member: str) -> bytes:
    with tarfile.open(archive, "r:gz") as bundle:
        try:
            extracted = bundle.extractfile(member)
        except KeyError as error:
            raise ValueError(f"missing archive member {member!r} in {archive}") from error
        if extracted is None:
            raise ValueError(f"archive member is not a file: {member!r}")
        return extracted.read()


def read_tar_json(archive: Path, member: str) -> dict[str, Any]:
    value = json.loads(read_tar_bytes(archive, member).decode("utf-8-sig"))
    require(isinstance(value, dict), f"JSON member {member!r} is not an object")
    return value


def model_role(method: str) -> str:
    if method == "A10-locked":
        return "proposed"
    if method in {"PUNCH2", "PUNCH2-Light"}:
        return "punch2"
    if "LoRA" in method:
        return "lora_dr_suite"
    return "official_comparator"


def normalize_formal_metrics(
    challenge_round: str, metrics: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        row = {
            "challenge_round": challenge_round,
            "track": metric["track"],
            "rank_by_roc_auc": int(metric["rank_by_roc_auc"]),
            "method": metric["method"],
            "model_role": model_role(str(metric["method"])),
            "evidence_level": "raw_per_residue_recomputed_same_reference",
            "same_reference": True,
            "rank_eligible": True,
            "source": metric["source"],
            "reference_proteins": int(metric["reference_proteins"]),
            "predicted_proteins": int(metric["predicted_proteins"]),
            "protein_coverage": float(metric["protein_coverage"]),
            "reference_residues": int(metric["reference_residues"]),
            "predicted_residues": int(metric["predicted_residues"]),
            "residue_coverage": float(metric["residue_coverage"]),
            "known_residues": int(metric["known_residues"]),
            "positive_labels": int(metric["positive_labels"]),
            "negative_labels": int(metric["negative_labels"]),
            "roc_auc": float(metric["roc_auc_caid_compatible"]),
            "auprc": float(metric["aucpr_trapezoid"]),
            "aps": float(metric["aps"]),
            "f1_at_0_5": float(metric["f1_at_0_5"]),
            "mcc_at_0_5": float(metric["mcc_at_0_5"]),
            "fmax": float(metric["fmax"]),
            "mcc_at_fmax": float(metric["mcc_at_fmax"]),
            "fmax_threshold": float(metric["fmax_threshold"]),
        }
        rows.append(row)
    return rows


def normalize_pairwise(
    challenge_round: str, rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        delong = row["delong"]
        bootstrap = row["protein_bootstrap"]
        interval = bootstrap["percentile_95_ci"]
        normalized.append(
            {
                "challenge_round": challenge_round,
                "track": row["track"],
                "first": row["first"],
                "second": row["second"],
                "common_proteins": int(row["common_proteins"]),
                "common_known_residues": int(row["common_known_residues"]),
                "first_auc_on_common_set": float(delong["first_auc"]),
                "second_auc_on_common_set": float(delong["second_auc"]),
                "auc_delta_first_minus_second": float(
                    delong["delta_first_minus_second"]
                ),
                "delong_two_sided_p": float(delong["two_sided_p_value"]),
                "bootstrap_replicates": int(bootstrap["replicates"]),
                "bootstrap_ci_low": float(interval[0]),
                "bootstrap_ci_high": float(interval[1]),
                "bootstrap_probability_delta_positive": float(
                    bootstrap["probability_delta_positive"]
                ),
                "bootstrap_two_sided_empirical_p": float(
                    bootstrap["two_sided_empirical_p_plus_one_correction"]
                ),
            }
        )
    return normalized


def normalize_literature_context(b1_report: dict[str, Any]) -> list[dict[str, Any]]:
    section = b1_report["lora_dr_suite_reference_only"]
    require(section["same_reference_as_locked_a10"] is False, "LoRA reference flag changed")
    rows: list[dict[str, Any]] = []
    for value in section["rows"]:
        rows.append(
            {
                "challenge_round": "CAID3",
                "track": section["track"],
                "method": value["method"],
                "model_role": "lora_dr_suite",
                "evidence_level": "literature_only_different_reference",
                "same_reference": False,
                "rank_eligible": False,
                "reference_proteins": 148,
                "roc_auc": float(value["roc_auc"]),
                "auprc": float(value["pr_auc"]),
                "aps": None,
                "f1_at_0_5": float(value["f1"]),
                "mcc_at_0_5": float(value["mcc"]),
                "fmax": float(value["fmax"]),
                "comparison_note": (
                    "Paper-specific 148-sequence CAID3_NOX set; excluded from "
                    "same-reference ranking and paired statistics."
                ),
                "training_note": section["training_note"],
            }
        )
    return rows


def validate_formal_rows(rows: list[dict[str, Any]]) -> None:
    require(rows, "no formal rows")
    identities: set[tuple[str, str, str]] = set()
    a10: list[dict[str, Any]] = []
    reference_sizes: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        identity = (row["challenge_round"], row["track"], row["method"])
        require(identity not in identities, f"duplicate formal row: {identity}")
        identities.add(identity)
        require(row["same_reference"] is True, f"non-comparable formal row: {identity}")
        require(row["rank_eligible"] is True, f"ineligible formal row: {identity}")
        require(0.0 <= row["roc_auc"] <= 1.0, f"invalid ROC-AUC: {identity}")
        require(0.0 <= row["protein_coverage"] <= 1.0, f"invalid coverage: {identity}")
        reference_sizes[(row["challenge_round"], row["track"])].add(
            row["reference_proteins"]
        )
        if row["method"] == "A10-locked":
            a10.append(row)
    require(len(a10) == 4, f"expected four A10 rows, found {len(a10)}")
    require(
        {(row["challenge_round"], row["track"]) for row in a10}
        == {
            ("CAID2", "disorder_nox"),
            ("CAID2", "disorder_pdb"),
            ("CAID3", "disorder_nox"),
            ("CAID3", "disorder_pdb"),
        },
        "A10 does not cover all four round/track combinations",
    )
    require(
        all(math.isclose(row["protein_coverage"], 1.0) for row in a10),
        "A10 must have full protein coverage",
    )
    expected_reference_sizes = {
        ("CAID2", "disorder_nox"): {210},
        ("CAID2", "disorder_pdb"): {348},
        ("CAID3", "disorder_nox"): {204},
        ("CAID3", "disorder_pdb"): {319},
    }
    require(reference_sizes == expected_reference_sizes, "reference sizes changed")
    require(
        not any(row["model_role"] == "lora_dr_suite" for row in rows),
        "literature-only LoRA row leaked into the formal table",
    )


def make_a10_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["challenge_round"], row["track"])].append(row)
    summary: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda x: (int(x[0][-1]), x[1])):
        group = sorted(groups[key], key=lambda row: (row["rank_by_roc_auc"], row["method"]))
        proposed = next(row for row in group if row["method"] == "A10-locked")
        leader = group[0]
        best_other = next(row for row in group if row["method"] != "A10-locked")
        summary.append(
            {
                "challenge_round": key[0],
                "track": key[1],
                "methods_evaluated": len(group),
                "a10_rank": proposed["rank_by_roc_auc"],
                "a10_roc_auc": proposed["roc_auc"],
                "a10_auprc": proposed["auprc"],
                "a10_aps": proposed["aps"],
                "a10_f1_at_0_5": proposed["f1_at_0_5"],
                "a10_mcc_at_0_5": proposed["mcc_at_0_5"],
                "a10_fmax": proposed["fmax"],
                "a10_protein_coverage": proposed["protein_coverage"],
                "leader_method": leader["method"],
                "leader_roc_auc": leader["roc_auc"],
                "a10_delta_vs_leader": proposed["roc_auc"] - leader["roc_auc"],
                "best_non_a10_method": best_other["method"],
                "best_non_a10_roc_auc": best_other["roc_auc"],
                "a10_delta_vs_best_non_a10": proposed["roc_auc"]
                - best_other["roc_auc"],
            }
        )
    return summary


def select_manuscript_rows(
    rows: list[dict[str, Any]], top_methods: int
) -> list[dict[str, Any]]:
    require(top_methods >= 1, "top_methods must be positive")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["challenge_round"], row["track"])].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda x: (int(x[0][-1]), x[1])):
        group = sorted(groups[key], key=lambda row: (row["rank_by_roc_auc"], row["method"]))
        chosen: dict[str, set[str]] = defaultdict(set)
        for row in group[:top_methods]:
            chosen[row["method"]].add(f"top_{top_methods}")
        chosen["A10-locked"].add("proposed")
        for method in ("PUNCH2", "PUNCH2-Light"):
            if any(row["method"] == method for row in group):
                chosen[method].add("named_baseline")
        for row in group:
            if row["method"] in chosen:
                concise = {column: row[column] for column in FORMAL_COLUMNS}
                concise["selection_reason"] = "+".join(sorted(chosen[row["method"]]))
                selected.append(concise)
    return selected


def make_efficiency_rows(
    b1_report: dict[str, Any], b2_report: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for challenge_round, source in (
        ("CAID2", b2_report["a10_efficiency"]),
        ("CAID3", b1_report["a10_efficiency"]),
    ):
        elapsed = float(source["elapsed_seconds"])
        rows.append(
            {
                "challenge_round": challenge_round,
                "proteins": int(source["proteins"]),
                "residues": int(source["residues"]),
                "elapsed_seconds": elapsed,
                "proteins_per_second": int(source["proteins"]) / elapsed,
                "residues_per_second": int(source["residues"]) / elapsed,
                "peak_cuda_memory_gib": int(source["peak_cuda_memory_bytes"])
                / (1024**3),
                "device_name": source["device_name"],
                "precision": source["precision"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def f3(value: float) -> str:
    return f"{value:.3f}"


def signed4(value: float) -> str:
    return f"{value:+.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def build_summary_markdown(
    a10_summary: list[dict[str, Any]],
    manuscript_rows: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    literature: list[dict[str, Any]],
    efficiency: list[dict[str, Any]],
) -> str:
    lines = [
        "# B3 unified CAID2/CAID3 fair comparison",
        "",
        "B3 is a post-lock reporting analysis. It performs no training, tuning, or",
        "new label access. Formal rankings contain only raw per-residue predictions",
        "recomputed on the same reference set with the CAID-compatible metric protocol.",
        "Literature values from a different reference set are reported separately.",
        "",
        "## A10 across challenge rounds",
        "",
    ]
    lines += markdown_table(
        [
            "Round",
            "Track",
            "Rank",
            "Methods",
            "ROC-AUC",
            "AUPRC",
            "APS",
            "F1@0.5",
            "MCC@0.5",
            "Fmax",
            "Leader",
            "Delta vs leader",
        ],
        [
            [
                row["challenge_round"],
                row["track"],
                str(row["a10_rank"]),
                str(row["methods_evaluated"]),
                f3(row["a10_roc_auc"]),
                f3(row["a10_auprc"]),
                f3(row["a10_aps"]),
                f3(row["a10_f1_at_0_5"]),
                f3(row["a10_mcc_at_0_5"]),
                f3(row["a10_fmax"]),
                row["leader_method"],
                signed4(row["a10_delta_vs_leader"]),
            ]
            for row in a10_summary
        ],
    )
    lines += ["", "## Manuscript comparison table", ""]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manuscript_rows:
        grouped[(row["challenge_round"], row["track"])].append(row)
    for key in sorted(grouped, key=lambda x: (int(x[0][-1]), x[1])):
        lines += [f"### {key[0]} {key[1]}", ""]
        lines += markdown_table(
            ["Rank", "Method", "ROC-AUC", "AUPRC", "APS", "F1", "MCC", "Fmax", "Coverage"],
            [
                [
                    str(row["rank_by_roc_auc"]),
                    row["method"],
                    f3(row["roc_auc"]),
                    f3(row["auprc"]),
                    f3(row["aps"]),
                    f3(row["f1_at_0_5"]),
                    f3(row["mcc_at_0_5"]),
                    f3(row["fmax"]),
                    f"{100.0 * row['protein_coverage']:.1f}%",
                ]
                for row in grouped[key]
            ],
        )
        lines.append("")
    lines += ["## Paired AUC statistics", ""]
    lines += markdown_table(
        ["Round", "Track", "First", "Second", "Proteins", "Delta", "Bootstrap 95% CI", "Bootstrap p"],
        [
            [
                row["challenge_round"],
                row["track"],
                row["first"],
                row["second"],
                str(row["common_proteins"]),
                signed4(row["auc_delta_first_minus_second"]),
                f"[{signed4(row['bootstrap_ci_low'])}, {signed4(row['bootstrap_ci_high'])}]",
                f"{row['bootstrap_two_sided_empirical_p']:.4g}",
            ]
            for row in pairwise
        ],
    )
    lines += [
        "",
        "The protein-cluster bootstrap is the primary uncertainty analysis because",
        "residues from the same protein are not independent. DeLong values are retained",
        "in the CSV and JSON outputs as a secondary residue-level analysis.",
        "",
        "## LoRA-DR-Suite literature context (not rank-comparable)",
        "",
        "These published values use a paper-specific 148-sequence CAID3_NOX set,",
        "not the locked 204-protein CAID3 NOX reference. They are excluded from ranks",
        "and paired statistics.",
        "",
    ]
    lines += markdown_table(
        ["Method", "ROC-AUC", "PR-AUC", "F1", "MCC", "Fmax", "Evidence"],
        [
            [
                row["method"],
                f3(row["roc_auc"]),
                f3(row["auprc"]),
                f3(row["f1_at_0_5"]),
                f3(row["mcc_at_0_5"]),
                f3(row["fmax"]),
                "literature/different reference",
            ]
            for row in literature
        ],
    )
    lines += ["", "## A10 efficiency", ""]
    lines += markdown_table(
        ["Round", "Proteins", "Residues", "Seconds", "Residues/s", "Peak GiB", "Device"],
        [
            [
                row["challenge_round"],
                str(row["proteins"]),
                str(row["residues"]),
                f"{row['elapsed_seconds']:.3f}",
                f"{row['residues_per_second']:.1f}",
                f"{row['peak_cuda_memory_gib']:.3f}",
                row["device_name"],
            ]
            for row in efficiency
        ],
    )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- A10 ranks first on CAID2 disorder-PDB.",
        "- A10 ranks second on CAID3 disorder-PDB; its protein-bootstrap difference",
        "  from PUNCH2 includes zero, so the two are statistically competitive here.",
        "- A10 does not lead disorder-NOX (rank 13 on CAID2 and rank 8 on CAID3 raw",
        "  archive recomputation). A universal state-of-the-art claim is not supported.",
        "- Formal claims must use the same-reference tables. LoRA-DR-Suite literature",
        "  values remain contextual until B4 can produce matched raw predictions.",
        "- A10 emits one classic-IDR probability per residue and no soft-disorder output.",
        "",
    ]
    return "\n".join(lines)


def artifact_hashes(output_dir: Path, names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        path = output_dir / name
        rows.append({"artifact": name, "sha256": sha256(path), "bytes": path.stat().st_size})
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    b1_archive = Path(args.b1_archive).resolve()
    b2_archive = Path(args.b2_archive).resolve()
    output_dir = Path(args.output_dir).resolve()
    require(b1_archive.is_file(), f"B1 archive not found: {b1_archive}")
    require(b2_archive.is_file(), f"B2 archive not found: {b2_archive}")

    b1_hash = sha256(b1_archive)
    b2_hash = sha256(b2_archive)
    require(b1_hash == args.b1_sha256.lower(), "B1 archive SHA256 mismatch")
    require(b2_hash == args.b2_sha256.lower(), "B2 archive SHA256 mismatch")

    b1 = read_tar_json(b1_archive, B1_REPORT_MEMBER)
    b2 = read_tar_json(b2_archive, B2_REPORT_MEMBER)
    inference = read_tar_json(b2_archive, B2_INFERENCE_REPORT_MEMBER)
    require(b1.get("status") == "pass", "B1 report did not pass")
    require(b2.get("status") == "pass", "B2 report did not pass")
    require(b1.get("a10_locked_before_caid_label_access") is True, "B1 lock flag failed")
    require(b2.get("a10_locked_before_caid2_label_access") is True, "B2 lock flag failed")
    require(b1.get("caid_labels_used_for_training_or_tuning") is False, "B1 leakage flag failed")
    require(b2.get("caid2_labels_used_for_training_or_tuning") is False, "B2 leakage flag failed")
    require(inference.get("locked_members") == 30, "A10 member count changed")
    require(inference.get("output_heads") == 1, "A10 output head count changed")
    require(inference.get("soft_disorder_output") is False, "soft disorder output enabled")

    formal = normalize_formal_metrics("CAID2", b2["metrics"])
    formal += normalize_formal_metrics("CAID3", b1["metrics"])
    formal.sort(
        key=lambda row: (
            int(row["challenge_round"][-1]),
            row["track"],
            row["rank_by_roc_auc"],
            row["method"],
        )
    )
    validate_formal_rows(formal)
    require(len(formal) == 168, f"expected 168 formal rows, found {len(formal)}")

    pairwise = normalize_pairwise("CAID2", b2["pairwise_statistics"])
    pairwise += normalize_pairwise("CAID3", b1["pairwise_statistics"])
    pairwise.sort(key=lambda row: (int(row["challenge_round"][-1]), row["track"], row["second"]))
    require(len(pairwise) == 11, f"expected 11 pairwise rows, found {len(pairwise)}")
    require(all(row["first"] == "A10-locked" for row in pairwise), "unexpected first model")

    literature = normalize_literature_context(b1)
    require(len(literature) == 4, "expected four LoRA literature rows")
    require(all(not row["rank_eligible"] for row in literature), "LoRA rank leakage")

    a10_summary = make_a10_summary(formal)
    manuscript = select_manuscript_rows(formal, args.top_methods)
    efficiency = make_efficiency_rows(b1, b2)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "b3_unified_rankings.csv", formal, FORMAL_COLUMNS)
    write_csv(
        output_dir / "b3_manuscript_comparison.csv",
        manuscript,
        FORMAL_COLUMNS + ["selection_reason"],
    )
    write_csv(output_dir / "b3_a10_round_summary.csv", a10_summary, list(a10_summary[0]))
    write_csv(output_dir / "b3_pairwise_statistics.csv", pairwise, PAIRWISE_COLUMNS)
    write_csv(
        output_dir / "b3_lora_dr_suite_literature_context.csv",
        literature,
        list(literature[0]),
    )
    write_csv(output_dir / "b3_efficiency.csv", efficiency, list(efficiency[0]))
    summary = build_summary_markdown(a10_summary, manuscript, pairwise, literature, efficiency)
    (output_dir / "b3_summary.md").write_text(summary, encoding="utf-8", newline="\n")

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "b3_unified_caid2_caid3_fair_comparison",
        "status": "pass",
        "analysis_type": "post_lock_reporting_only",
        "primary_metric": "caid_compatible_residue_level_roc_auc",
        "secondary_metrics": ["auprc", "aps", "f1_at_0_5", "mcc_at_0_5", "fmax"],
        "uncertainty_primary": "paired_protein_cluster_bootstrap",
        "metric_implementation": "shared B1 implementation reused by B2",
        "official_caid_revision": b1["official_caid_revision"],
        "input_archives": {
            "b1_caid3": {"path": str(b1_archive), "sha256": b1_hash},
            "b2_caid2": {"path": str(b2_archive), "sha256": b2_hash},
        },
        "source_separation": {
            "formal_evidence": "raw_per_residue_recomputed_same_reference",
            "formal_rows": len(formal),
            "literature_context": "different_reference_not_rank_eligible",
            "literature_rows": len(literature),
            "lora_rows_in_formal_ranking": 0,
        },
        "a10_model_lock": {
            "locked_members": inference["locked_members"],
            "a2_members": inference["a2_members"],
            "a8_members": inference["a8_members"],
            "seeds": inference["seeds"],
            "folds": inference["folds"],
            "uniform_logit_weight": inference["uniform_logit_weight"],
            "output_heads": inference["output_heads"],
            "soft_disorder_output": inference["soft_disorder_output"],
        },
        "a10_round_summary": a10_summary,
        "formal_pairwise_comparisons": len(pairwise),
        "efficiency": efficiency,
        "claim_guardrails": {
            "caid2_pdb_rank_1": True,
            "caid3_pdb_rank_2": True,
            "universal_state_of_the_art_supported": False,
            "caid_labels_used_for_training_or_tuning": False,
            "single_output_score": True,
            "soft_disorder_output": False,
        },
    }
    write_json(output_dir / "b3_report.json", result)
    output_names = [
        "b3_unified_rankings.csv",
        "b3_manuscript_comparison.csv",
        "b3_a10_round_summary.csv",
        "b3_pairwise_statistics.csv",
        "b3_lora_dr_suite_literature_context.csv",
        "b3_efficiency.csv",
        "b3_summary.md",
        "b3_report.json",
    ]
    hashes = artifact_hashes(output_dir, output_names)
    write_csv(output_dir / "b3_artifact_sha256.csv", hashes, ["artifact", "sha256", "bytes"])
    result["artifacts"] = hashes
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1-archive", type=Path, default=DEFAULT_B1_ARCHIVE)
    parser.add_argument("--b1-sha256", default=EXPECTED_B1_SHA256)
    parser.add_argument("--b2-archive", type=Path, default=DEFAULT_B2_ARCHIVE)
    parser.add_argument("--b2-sha256", default=EXPECTED_B2_SHA256)
    parser.add_argument("--top-methods", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    result = build(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "formal_rows": result["source_separation"]["formal_rows"],
                "literature_rows": result["source_separation"]["literature_rows"],
                "pairwise_comparisons": result["formal_pairwise_comparisons"],
                "a10_caid2_pdb_rank": next(
                    row["a10_rank"]
                    for row in result["a10_round_summary"]
                    if row["challenge_round"] == "CAID2" and row["track"] == "disorder_pdb"
                ),
                "a10_caid3_pdb_rank": next(
                    row["a10_rank"]
                    for row in result["a10_round_summary"]
                    if row["challenge_round"] == "CAID3" and row["track"] == "disorder_pdb"
                ),
                "universal_state_of_the_art_supported": False,
                "caid_labels_used_for_training_or_tuning": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
