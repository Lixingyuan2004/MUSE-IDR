from __future__ import annotations

import csv
import hashlib
import json
import statistics
import tarfile
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "a1_a10_20260824"
SEEDS = (17, 29, 43)


ARCHIVES = {
    "a1": PROJECT_ROOT / "experiments/ablations/a1_frozen_esm2/a1_results_20260823.tar.gz",
    "a2": PROJECT_ROOT / "experiments/ablations/a2_multiscale_context/a2_results_20260823.tar.gz",
    "a3": PROJECT_ROOT / "experiments/ablations/a3_auc_ranking_loss/a3_dev_results_20260824.tar.gz",
    "a4": PROJECT_ROOT / "experiments/ablations/a4_esm2_lora_multiscale/a4_dev_results_20260824.tar.gz",
    "a5": PROJECT_ROOT / "a5_results_20260824.tar.gz",
    "a5_bootstrap": PROJECT_ROOT / "a5_results_bootstrap_20260824.tar.gz",
    "a6": PROJECT_ROOT / "experiments/ablations/a6_long_range_dilated_context/a6_dev_results_20260824.tar.gz",
    "a7": PROJECT_ROOT / "experiments/ablations/a7_bidirectional_sequence_context/a7_cv_seed17_results_20260824.tar.gz",
    "a8_a10": PROJECT_ROOT / "experiments/ablations/a10_cross_architecture_ensemble/a8_a9_a10_results_20260824.tar.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(archive: Path, member: str) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as handle:
        extracted = handle.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Missing archive member: {archive}::{member}")
        return json.loads(extracted.read().decode("utf-8"))


def metric_triplet(report: dict[str, Any]) -> dict[str, float]:
    metrics = report["oof_metrics"]
    return {
        "micro_roc_auc": float(metrics["micro_roc_auc"]),
        "micro_pr_auc": float(metrics["micro_pr_auc"]),
        "macro_roc_auc": float(metrics["macro_roc_auc"]),
    }


def load_seed_reports(
    archive_key: str, member_template: str
) -> dict[int, dict[str, Any]]:
    return {
        seed: read_json(ARCHIVES[archive_key], member_template.format(seed=seed))
        for seed in SEEDS
    }


def mean_sd(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)),
    }


def seed_summary(reports: dict[int, dict[str, Any]]) -> dict[str, Any]:
    per_seed = {str(seed): metric_triplet(report) for seed, report in reports.items()}
    summary: dict[str, Any] = {"per_seed": per_seed}
    for metric in ("micro_roc_auc", "micro_pr_auc", "macro_roc_auc"):
        summary[metric] = mean_sd(per_seed[str(seed)][metric] for seed in SEEDS)
    return summary


def fmt(value: float) -> str:
    return f"{value:.6f}"


def fmt_signed(value: float) -> str:
    return f"{value:+.6f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    missing = [str(path) for path in ARCHIVES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required archives: {missing}")

    a1 = load_seed_reports(
        "a1", "outputs/ablations/a1_frozen_esm2/seed_{seed}/oof_metrics.json"
    )
    a2 = load_seed_reports(
        "a2", "outputs/ablations/a2_multiscale_context/seed_{seed}/oof_metrics.json"
    )
    a8 = load_seed_reports(
        "a8_a10",
        "outputs/ablations/a8_multilayer_esm2_fusion/seed_{seed}/oof_metrics.json",
    )
    a7 = read_json(
        ARCHIVES["a7"],
        "outputs/ablations/a7_bidirectional_sequence_context/seed_17/oof_metrics.json",
    )
    a5 = read_json(
        ARCHIVES["a5"],
        "outputs/ablations/a5_seed_ensemble/a5_ensemble_report.json",
    )
    a5_bootstrap = read_json(
        ARCHIVES["a5_bootstrap"],
        "outputs/ablations/a5_seed_ensemble/a5_bootstrap_report.json",
    )
    a9 = read_json(
        ARCHIVES["a8_a10"],
        "outputs/ablations/a9_a8_seed_ensemble/a9_ensemble_report.json",
    )
    a10 = read_json(
        ARCHIVES["a8_a10"],
        "outputs/ablations/a10_cross_architecture_ensemble/a10_ensemble_report.json",
    )
    a10_bootstrap = read_json(
        ARCHIVES["a8_a10"],
        "outputs/ablations/a10_cross_architecture_ensemble/bootstrap/a10_bootstrap_report.json",
    )

    a2_dev = read_json(
        ARCHIVES["a2"],
        "outputs/ablations/a2_multiscale_context/seed_17/dev_fold4_backup/oof_metrics.json",
    )
    a2_dev_auc = float(a2_dev["oof_metrics"]["micro_roc_auc"])
    development_specs = [
        (
            "A2 reference",
            "Multiscale CNN",
            a2_dev,
            "reference",
            int(a2_dev.get("trainable_parameters", 609025)),
        ),
        (
            "A3 rank=0.20",
            "A2 + sampled AUC-ranking loss",
            read_json(
                ARCHIVES["a3"],
                "outputs/ablations/a3_auc_ranking_loss/seed_17/oof_metrics.json",
            ),
            "rejected",
            609025,
        ),
        (
            "A3 rank=0.05",
            "A2 + sampled AUC-ranking loss",
            read_json(
                ARCHIVES["a3"],
                "outputs/ablations/a3_auc_ranking_loss_rank005/seed_17/oof_metrics.json",
            ),
            "rejected",
            609025,
        ),
        (
            "A3 rank=0.10",
            "A2 + sampled AUC-ranking loss",
            read_json(
                ARCHIVES["a3"],
                "outputs/ablations/a3_auc_ranking_loss_rank010/seed_17/oof_metrics.json",
            ),
            "rejected",
            609025,
        ),
        (
            "A3 rank=0.40",
            "A2 + sampled AUC-ranking loss",
            read_json(
                ARCHIVES["a3"],
                "outputs/ablations/a3_auc_ranking_loss_rank040/seed_17/oof_metrics.json",
            ),
            "rejected",
            609025,
        ),
        (
            "A4 frozen control",
            "Online frozen ESM2 + A2 head",
            read_json(
                ARCHIVES["a4"],
                "outputs/ablations/a4_esm2_lora_multiscale/frozen_raw_control/seed_17/oof_metrics.json",
            ),
            "rejected",
            609025,
        ),
        (
            "A4 LoRA q/v r=8",
            "ESM2 LoRA + A2 head",
            read_json(
                ARCHIVES["a4"],
                "outputs/ablations/a4_esm2_lora_multiscale/lora_qv_r8/seed_17/oof_metrics.json",
            ),
            "rejected",
            1960705,
        ),
        (
            "A6 dilated context",
            "A2 + dilated residual stack",
            read_json(
                ARCHIVES["a6"],
                "outputs/ablations/a6_long_range_dilated_context/seed_17/oof_metrics.json",
            ),
            "rejected",
            1144321,
        ),
        (
            "A7 BiGRU",
            "A2 + bidirectional GRU",
            read_json(
                ARCHIVES["a7"],
                "outputs/ablations/a7_bidirectional_sequence_context/development/dev_fold4_seed17_metrics.json",
            ),
            "advanced then rejected by full CV",
            1103105,
        ),
        (
            "A8 last-four-layer fusion",
            "Learned ESM2 layer mix + A2",
            read_json(
                ARCHIVES["a8_a10"],
                "outputs/ablations/a8_multilayer_esm2_fusion/development/dev_fold4_seed17_metrics.json",
            ),
            "advanced",
            609029,
        ),
    ]
    development_rows: list[dict[str, Any]] = []
    for experiment, module, report, decision, parameters in development_specs:
        metrics = report["oof_metrics"]
        development_rows.append(
            {
                "experiment": experiment,
                "module": module,
                "scope": "seed17_fold4_development",
                "micro_roc_auc": float(metrics["micro_roc_auc"]),
                "delta_vs_a2_fold4": float(metrics["micro_roc_auc"]) - a2_dev_auc,
                "micro_pr_auc": float(metrics["micro_pr_auc"]),
                "macro_roc_auc": float(metrics["macro_roc_auc"]),
                "trainable_parameters": parameters,
                "decision": decision,
            }
        )

    architecture_rows = []
    architecture_specs = [
        ("A1", "Frozen ESM2 last layer + linear residue head", a1[17], None, "reference"),
        ("A2", "A1 + gated multiscale CNN", a2[17], 609025, "accepted"),
        ("A7", "A2 + bidirectional GRU", a7, 1103105, "rejected after full CV"),
        ("A8", "Last-four-layer scalar mix + A2", a8[17], 609029, "accepted"),
    ]
    for experiment, model, report, parameters, decision in architecture_specs:
        metrics = report["oof_metrics"]
        architecture_rows.append(
            {
                "experiment": experiment,
                "model": model,
                "scope": "5fold_oof_seed17",
                "micro_roc_auc": float(metrics["micro_roc_auc"]),
                "delta_vs_a2_seed17": float(metrics["micro_roc_auc"])
                - float(a2[17]["oof_metrics"]["micro_roc_auc"]),
                "micro_pr_auc": float(metrics["micro_pr_auc"]),
                "macro_roc_auc": float(metrics["macro_roc_auc"]),
                "trainable_parameters": parameters,
                "decision": decision,
            }
        )

    ensemble_specs = [
        (
            "A5",
            "A2 seeds 17/29/43 equal-logit ensemble",
            a5["method_metrics"]["uniform_logit_mean"]["oof"],
            "accepted baseline ensemble",
        ),
        (
            "A9",
            "A8 seeds 17/29/43 equal-logit ensemble",
            a9["method_metrics"]["uniform_logit_mean"]["oof"],
            "not selected; below A5",
        ),
        (
            "A10",
            "A2+A8 six-model equal-logit ensemble",
            a10["method_metrics"]["six_model_uniform_logit_mean"]["oof"],
            "selected final candidate",
        ),
    ]
    a5_auc = float(ensemble_specs[0][2]["micro_roc_auc"])
    ensemble_rows = [
        {
            "experiment": experiment,
            "model": model,
            "scope": "full_5fold_oof",
            "micro_roc_auc": float(metrics["micro_roc_auc"]),
            "delta_vs_a5": float(metrics["micro_roc_auc"]) - a5_auc,
            "micro_pr_auc": float(metrics["micro_pr_auc"]),
            "macro_roc_auc": float(metrics["macro_roc_auc"]),
            "decision": decision,
        }
        for experiment, model, metrics, decision in ensemble_specs
    ]

    summary = {
        "schema_version": 1,
        "objective": "maximize per-residue classic-IDR micro ROC-AUC",
        "dataset_protocol": {
            "proteins": 1133,
            "known_residues": 202662,
            "positive_residues": 105036,
            "negative_residues": 97626,
            "folds": 5,
            "unknown_label": -1,
            "manifest_sha256": "df6a4deee4a65d009f4faf41c6b8930f9db8600fa28b6711da7ff48ad2c70bf4",
            "esm2_revision": "08e4846e537177426273712802403f7ba8261b6c",
            "caid1_caid2_caid3_used_for_training_or_tuning": False,
            "output_heads": 1,
            "soft_disorder_output": False,
        },
        "archive_sha256": {key: sha256(path) for key, path in ARCHIVES.items()},
        "three_seed_robustness": {
            "A1": seed_summary(a1),
            "A2": seed_summary(a2),
            "A8": seed_summary(a8),
        },
        "formal_architecture_rows": architecture_rows,
        "development_screening_rows": development_rows,
        "formal_ensemble_rows": ensemble_rows,
        "bootstrap": {
            "A5_vs_A2_seed17": a5_bootstrap,
            "A10_vs_A5": a10_bootstrap,
        },
        "selected_model": {
            "name": "A10",
            "definition": "equal-logit mean of A2 and A8, seeds 17/29/43",
            "micro_roc_auc": float(
                a10["method_metrics"]["six_model_uniform_logit_mean"]["oof"][
                    "micro_roc_auc"
                ]
            ),
            "single_classic_idr_output": True,
            "caid_used_for_selection": False,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "a1_a10_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(OUTPUT_DIR / "formal_architecture_seed17.csv", architecture_rows)
    write_csv(OUTPUT_DIR / "development_fold4_screening.csv", development_rows)
    write_csv(OUTPUT_DIR / "formal_ensembles.csv", ensemble_rows)

    robustness = summary["three_seed_robustness"]
    a10_boot = a10_bootstrap["bootstrap"]
    markdown = [
        "# A1-A10 experiment summary",
        "",
        "## Fixed protocol",
        "",
        "- 1,133 proteins; 202,662 known residues (105,036 positive / 97,626 negative).",
        "- Protein-level five-fold OOF evaluation; unknown label `-1` excluded.",
        "- One classic-IDR probability per residue; no soft-disorder output.",
        "- CAID1/2/3 were not used for training, tuning, or model selection.",
        "",
        "## Formal architecture comparison (seed 17, full five-fold OOF)",
        "",
        "| Experiment | Model | ROC-AUC | Delta vs A2 | PR-AUC | Macro ROC-AUC | Trainable params | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in architecture_rows:
        parameters = "not reported" if row["trainable_parameters"] is None else f"{row['trainable_parameters']:,}"
        markdown.append(
            f"| {row['experiment']} | {row['model']} | {fmt(row['micro_roc_auc'])} | "
            f"{fmt_signed(row['delta_vs_a2_seed17'])} | {fmt(row['micro_pr_auc'])} | "
            f"{fmt(row['macro_roc_auc'])} | {parameters} | {row['decision']} |"
        )
    markdown += [
        "",
        "## Three-seed robustness (mean +/- sample SD)",
        "",
        "| Experiment | ROC-AUC | PR-AUC | Macro ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    for name in ("A1", "A2", "A8"):
        item = robustness[name]
        markdown.append(
            f"| {name} | {fmt(item['micro_roc_auc']['mean'])} +/- {fmt(item['micro_roc_auc']['sample_sd'])} | "
            f"{fmt(item['micro_pr_auc']['mean'])} +/- {fmt(item['micro_pr_auc']['sample_sd'])} | "
            f"{fmt(item['macro_roc_auc']['mean'])} +/- {fmt(item['macro_roc_auc']['sample_sd'])} |"
        )
    markdown += [
        "",
        "## Formal ensembles (full five-fold OOF)",
        "",
        "| Experiment | Model | ROC-AUC | Delta vs A5 | PR-AUC | Macro ROC-AUC | Decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in ensemble_rows:
        markdown.append(
            f"| {row['experiment']} | {row['model']} | {fmt(row['micro_roc_auc'])} | "
            f"{fmt_signed(row['delta_vs_a5'])} | {fmt(row['micro_pr_auc'])} | "
            f"{fmt(row['macro_roc_auc'])} | {row['decision']} |"
        )
    markdown += [
        "",
        "## Development screening (seed 17, fold 4 only; not formal OOF)",
        "",
        "| Experiment | Module | ROC-AUC | Delta vs A2 fold 4 | Params | Decision |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in development_rows:
        markdown.append(
            f"| {row['experiment']} | {row['module']} | {fmt(row['micro_roc_auc'])} | "
            f"{fmt_signed(row['delta_vs_a2_fold4'])} | {row['trainable_parameters']:,} | {row['decision']} |"
        )
    markdown += [
        "",
        "## Statistical evidence for A10",
        "",
        f"- A10 minus A5 point delta: {fmt_signed(a10_bootstrap['point_estimate']['paired_delta'])}.",
        f"- Paired protein-cluster bootstrap mean delta: {fmt_signed(a10_boot['paired_delta_mean'])}.",
        f"- 95% percentile CI: [{fmt_signed(a10_boot['paired_delta_95_percentile_ci'][0])}, {fmt_signed(a10_boot['paired_delta_95_percentile_ci'][1])}].",
        f"- Probability delta > 0: {100.0 * a10_boot['probability_delta_positive']:.2f}%.",
        f"- One-sided bootstrap p-value: {a10_boot['one_sided_p_delta_le_zero_plus_one_correction']:.4f}.",
        "",
        "## Selection conclusion",
        "",
        "A10 is the frozen development winner. It averages six scores in logit space: three A2 models and three A8 models. All six weights are fixed at 1/6. The OOF-tuned 0.54/0.46 family weighting is exploratory and is not selected. CAID2/3 remain untouched external tests.",
        "",
        "A7 is the key cautionary result: its fold-4 gain was only +0.000031, but full five-fold OOF fell below A2 by -0.002839. This justifies requiring full OOF confirmation before accepting a module.",
        "",
    ]
    (OUTPUT_DIR / "a1_a10_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "output_dir": str(OUTPUT_DIR),
        "selected_model": summary["selected_model"],
        "a10_bootstrap_ci": a10_boot["paired_delta_95_percentile_ci"],
        "caid_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
