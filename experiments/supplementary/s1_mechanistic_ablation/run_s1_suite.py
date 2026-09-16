"""Run every predeclared new S1 variant across three seeds and five folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

FORMAL_VARIANTS = (
    "ungated_add",
    "additive_projection",
    "single_k3",
    "single_k7",
    "single_k15",
    "single_k31",
    "uniform_layer_mix",
)
FORMAL_SEEDS = (17, 29, 43)


def valid_completed_report(path: Path, variant: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        (
            report.get("status") == "pass",
            report.get("variant") == variant,
            int(report.get("seed", -1)) == seed,
            report.get("complete_oof") is True,
            report.get("caid1_caid2_caid3_used_for_training_tuning_or_reselection")
            is False,
            report.get("locked_a10_modified") is False,
        )
    )


def save_driver_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--final-layer-cache", type=Path, required=True)
    parser.add_argument("--multilayer-cache", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/supplementary/s1_mechanistic_ablation"),
    )
    parser.add_argument("--variant", action="append", choices=FORMAL_VARIANTS)
    parser.add_argument("--seed", action="append", type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = tuple(args.variant or FORMAL_VARIANTS)
    seeds = tuple(args.seed or FORMAL_SEEDS)
    train_script = Path(__file__).with_name("train_s1.py")
    started = perf_counter()
    jobs: list[dict[str, Any]] = []
    report_path = args.output_root / "s1_driver_report.json"

    for variant in variants:
        for seed in seeds:
            output_report = (
                args.output_root / variant / f"seed_{seed}" / "oof_metrics.json"
            )
            if output_report.exists() and not args.resume:
                raise FileExistsError(
                    f"S1 output already exists; inspect it or rerun with --resume: "
                    f"{output_report}"
                )
            if args.resume and valid_completed_report(output_report, variant, seed):
                jobs.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "status": "skipped_verified_complete",
                        "report": output_report.as_posix(),
                    }
                )
                print(
                    json.dumps(
                        {"variant": variant, "seed": seed, "status": "SKIP"},
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                continue

            command = [
                sys.executable,
                "-u",
                str(train_script),
                "--manifest",
                str(args.manifest),
                "--final-layer-cache",
                str(args.final_layer_cache),
                "--multilayer-cache",
                str(args.multilayer_cache),
                "--config",
                str(args.config),
                "--output-root",
                str(args.output_root),
                "--variant",
                variant,
                "--seed",
                str(seed),
            ]
            print(
                json.dumps(
                    {"variant": variant, "seed": seed, "status": "START"},
                    separators=(",", ":"),
                ),
                flush=True,
            )
            job_started = perf_counter()
            completed = subprocess.run(command, check=False)
            job = {
                "variant": variant,
                "seed": seed,
                "status": "pass" if completed.returncode == 0 else "fail",
                "returncode": completed.returncode,
                "elapsed_seconds": perf_counter() - job_started,
                "report": output_report.as_posix(),
            }
            jobs.append(job)
            driver_report = {
                "schema_version": 1,
                "experiment": "s1_post_lock_mechanistic_ablation_driver",
                "status": "running" if completed.returncode == 0 else "fail",
                "variants": list(variants),
                "seeds": list(seeds),
                "jobs": jobs,
                "elapsed_seconds": perf_counter() - started,
                "caid_labels_accessed": False,
                "locked_a10_modified": False,
            }
            save_driver_report(report_path, driver_report)
            if completed.returncode != 0:
                raise RuntimeError(f"S1 failed for {variant}, seed {seed}")
            if not valid_completed_report(output_report, variant, seed):
                raise RuntimeError(
                    f"S1 child exited successfully but report is invalid: {output_report}"
                )
            print(
                json.dumps(
                    {"variant": variant, "seed": seed, "status": "PASS"},
                    separators=(",", ":"),
                ),
                flush=True,
            )

    final_report = {
        "schema_version": 1,
        "experiment": "s1_post_lock_mechanistic_ablation_driver",
        "status": "pass",
        "variants": list(variants),
        "seeds": list(seeds),
        "completed_jobs": len(jobs),
        "jobs": jobs,
        "elapsed_seconds": perf_counter() - started,
        "caid_labels_accessed": False,
        "locked_a10_modified": False,
    }
    save_driver_report(report_path, final_report)
    print(json.dumps(final_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
