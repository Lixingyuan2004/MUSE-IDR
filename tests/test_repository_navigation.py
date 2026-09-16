"""Structural tests for the reader-facing repository entry points."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reader_facing_directories_exist() -> None:
    required = (
        "data",
        "models",
        "training",
        "inference",
        "evaluation",
        "tests",
        "src/muse_idr",
        "experiments/supplementary/s1_mechanistic_ablation",
        "experiments/supplementary/s2_independent_holdout",
        "results/s1",
        "results/s2",
    )
    assert all((ROOT / name).is_dir() for name in required)


def test_stable_entry_points_target_canonical_implementations() -> None:
    required = (
        "training/run_experiment.py",
        "inference/predict.py",
        "evaluation/evaluate_residue_predictions.py",
        "experiments/final_models/a10_locked_inference/predict_a10.py",
        "experiments/final_models/a10_locked_inference/a10_lock_manifest.json",
        "experiments/supplementary/s1_mechanistic_ablation/analyze_s1.py",
        "experiments/supplementary/s2_independent_holdout/evaluate_s2_three_models.py",
        "results/s1/s1_summary.md",
        "results/s2/three_model_evaluation/s2_three_model_summary.md",
    )
    assert all((ROOT / relative).is_file() for relative in required)


def test_github_community_files_exist() -> None:
    required = (
        ".github/workflows/ci.yml",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "DATA_AND_SOFTWARE_AVAILABILITY.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
    )
    assert all((ROOT / relative).is_file() for relative in required)
