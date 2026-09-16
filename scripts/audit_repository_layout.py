"""Audit the public MUSE-IDR research-repository layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-local-data",
        action="store_true",
        help="also fail when the ignored local training/evaluation data are absent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    public_required = [
        "README.md",
        "README_CN.md",
        "LICENSE",
        "CITATION.cff",
        "MODEL_CARD.md",
        "pyproject.toml",
        "requirements-paper.txt",
        "MANIFEST.sha256",
        "DATA_AND_SOFTWARE_AVAILABILITY.md",
        "models/a2.py",
        "models/a8.py",
        "models/a10.py",
        "training/run_experiment.py",
        "inference/predict.py",
        "evaluation/evaluate_residue_predictions.py",
        "experiments/final_models/a10_locked_inference/predict_a10.py",
        "experiments/final_models/a10_locked_inference/a10_lock_manifest.json",
        "experiments/supplementary/s1_mechanistic_ablation/analyze_s1.py",
        "experiments/supplementary/s2_independent_holdout/evaluate_s2_three_models.py",
        "results/s1/s1_summary.md",
        "results/s2/three_model_evaluation/s2_three_model_summary.md",
        "src/muse_idr/__init__.py",
        "scripts/verify_release.py",
        "docs/DATA_SPLITS.md",
        "docs/EXTERNAL_DATA.md",
    ]
    public_required += [
        f"experiments/ablations/a{index}_" for index in range(1, 11)
    ]
    public_required += [
        f"experiments/analysis/b{index}_" for index in range(1, 8)
    ]

    missing: list[str] = []
    resolved: dict[str, str] = {}
    for relative in public_required:
        if relative.endswith("_"):
            parent = ROOT / Path(relative).parent
            matches = sorted(parent.glob(Path(relative).name + "*"))
            if len(matches) != 1 or not matches[0].is_dir():
                missing.append(relative + "*")
            else:
                resolved[relative + "*"] = matches[0].relative_to(ROOT).as_posix()
        else:
            path = ROOT / relative
            if not path.is_file():
                missing.append(relative)

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

    local_data = {
        "training_sequences": ROOT
        / "data/processed/classic_idr_2018/a1_sequences.jsonl",
        "training_manifest": ROOT
        / "data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl",
        "five_fold_assignments": ROOT
        / "data/processed/classic_idr_2018/homology_5fold_assignments.jsonl",
        "caid2_nox_reference": ROOT
        / "data/caid_holdout/references/caid2/disorder_nox.fasta",
        "caid2_pdb_reference": ROOT
        / "data/caid_holdout/references/caid2/disorder_pdb.fasta",
    }
    local_status = {name: path.is_file() for name, path in local_data.items()}
    if args.require_local_data:
        missing.extend(name for name, present in local_status.items() if not present)

    report = {
        "status": "pass" if not missing else "fail",
        "root": str(ROOT),
        "ablation_modules": 10,
        "analysis_modules_b1_b7": 7,
        "locked_checkpoints": len(checkpoints),
        "checkpoint_families": family_counts,
        "local_data": local_status,
        "local_data_required": args.require_local_data,
        "resolved_modules": resolved,
        "missing": missing,
    }
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
