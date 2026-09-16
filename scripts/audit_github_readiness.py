"""Audit the reader-facing layout and GitHub publication support files."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def numbered_directories(parent: Path, prefix: str, start: int, stop: int) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for number in range(start, stop + 1):
        key = f"{prefix}{number}"
        matches = sorted(path for path in parent.glob(f"{key}_*") if path.is_dir())
        if len(matches) == 1:
            resolved[key] = matches[0].relative_to(ROOT).as_posix()
    return resolved


def main() -> None:
    required_directories = (
        "data",
        "models",
        "training",
        "inference",
        "evaluation",
        "tests",
        "src/muse_idr",
        "experiments/ablations",
        "experiments/analysis",
        "experiments/supplementary/s1_mechanistic_ablation",
        "experiments/supplementary/s2_independent_holdout",
        "models/weights/a10_locked",
        "results/s1",
        "results/s2",
    )
    required_files = (
        "README.md",
        "README_CN.md",
        "LICENSE",
        "CITATION.cff",
        "MODEL_CARD.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "CHANGELOG.md",
        "pyproject.toml",
        "requirements-paper.txt",
        "MANIFEST.sha256",
        "RELEASE_CHECKLIST.md",
        "DATA_AND_SOFTWARE_AVAILABILITY.md",
        "data/README.md",
        "models/README.md",
        "models/a2.py",
        "models/a8.py",
        "models/a10.py",
        "training/README.md",
        "training/run_experiment.py",
        "inference/README.md",
        "inference/predict.py",
        "evaluation/README.md",
        "evaluation/evaluate_residue_predictions.py",
        "experiments/supplementary/s1_mechanistic_ablation/analyze_s1.py",
        "experiments/supplementary/s2_independent_holdout/evaluate_s2_three_models.py",
        "results/s1/s1_summary.md",
        "results/s2/three_model_evaluation/s2_three_model_summary.md",
        "results/s2/README.md",
        "docs/REPOSITORY_ENTRYPOINTS.md",
        ".github/workflows/ci.yml",
        ".github/workflows/repository-readiness.yml",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
    )

    missing = [name for name in required_directories if not (ROOT / name).is_dir()]
    missing.extend(name for name in required_files if not (ROOT / name).is_file())

    a_modules = numbered_directories(ROOT / "experiments/ablations", "a", 1, 10)
    b_modules = numbered_directories(ROOT / "experiments/analysis", "b", 1, 7)
    if len(a_modules) != 10:
        missing.append("exactly one A1-A10 directory per experiment")
    if len(b_modules) != 7:
        missing.append("exactly one B1-B7 directory per experiment")

    checkpoints = sorted((ROOT / "models/weights/a10_locked").rglob("*.pt"))
    family_counts = {
        "a2": sum("/a2/" in f"/{path.relative_to(ROOT).as_posix()}/" for path in checkpoints),
        "a8": sum(
            "/a8_a9_a10/" in f"/{path.relative_to(ROOT).as_posix()}/"
            for path in checkpoints
        ),
    }
    if len(checkpoints) != 30 or family_counts != {"a2": 15, "a8": 15}:
        missing.append("30 locked checkpoints (15 A2 + 15 A8)")

    report = {
        "status": "pass" if not missing else "fail",
        "root": str(ROOT),
        "requested_directories": list(required_directories),
        "a1_a10": a_modules,
        "b1_b7": b_modules,
        "locked_checkpoints": len(checkpoints),
        "checkpoint_families": family_counts,
        "github_support_files": [
            name for name in required_files if name.startswith(".github/")
        ],
        "missing": missing,
    }
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
