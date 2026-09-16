"""Run locked A10 prediction on the label-free, frozen S2 FASTA."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
PINNED_FILES = {
    "data/external/s2_database_curated_20260906/sequences_only/s2_sequences.fasta": "d777bfeb9cd8cf82af003c10bcef8b5b3e09dbee276e7e16164f408e336d67df",
    "experiments/final_models/a10_locked_inference/predict_a10.py": "ddadcf6999289a8dda158903698ac1a82242e67f869266ed5cf5048066a37c5d",
    "experiments/final_models/a10_locked_inference/a10_lock_manifest.json": "5547980f456104d756b2e8ca080991b8b82a08b8be08359d084c76ff085dda98",
    "experiments/ablations/a2_multiscale_context/model.py": "cc3566e8c87614355bb7be702630fdf1afd40717325aacd86392b90e4615c453",
    "experiments/ablations/a8_multilayer_esm2_fusion/model.py": "8268bdcda2188699eca7d4403a8df27a2a64828dac3f526368858e6e11bdd54a",
    "experiments/ablations/a8_multilayer_esm2_fusion/cache_multilayer.py": "64400b083038b18af7f19581ca52ca83dd32c0fa7868d966a5eb317266c8e9c6",
    "experiments/ablations/a1_frozen_esm2/data.py": "c8c34b11ceea6d415bf78bc4c4ce9db526a026a990752a40da1a4298e0e1ab57",
}
EXPECTED_PROTEINS = 87
EXPECTED_RESIDUES = 56394


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def verify_pins() -> None:
    for relative, expected in PINNED_FILES.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"pinned input hash mismatch: {relative}: {observed}")


def read_label_free_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    protein_id: str | None = None
    sequence: list[str] = []

    def finish() -> None:
        nonlocal protein_id, sequence
        if protein_id is None:
            return
        value = "".join(sequence).upper()
        if not value:
            raise ValueError(f"empty FASTA sequence: {protein_id}")
        if set(value) - set("ACDEFGHIKLMNPQRSTVWYBXZJUO"):
            raise ValueError(f"invalid residues in FASTA: {protein_id}")
        records.append((protein_id, value))
        protein_id = None
        sequence = []

    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish()
            protein_id = line[1:].split(maxsplit=1)[0]
            if not protein_id or protein_id in seen:
                raise ValueError(f"empty or duplicate FASTA identifier at line {line_number}")
            seen.add(protein_id)
        else:
            if protein_id is None:
                raise ValueError(f"sequence before FASTA header at line {line_number}")
            sequence.append("".join(line.split()))
    finish()
    if len(records) != EXPECTED_PROTEINS:
        raise ValueError(f"expected {EXPECTED_PROTEINS} proteins, got {len(records)}")
    residues = sum(len(sequence) for _, sequence in records)
    if residues != EXPECTED_RESIDUES:
        raise ValueError(f"expected {EXPECTED_RESIDUES} residues, got {residues}")
    return records


def validate_predictions(
    path: Path, records: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    expected = [
        (protein_id, position, residue)
        for protein_id, sequence in records
        for position, residue in enumerate(sequence, start=1)
    ]
    minimum = 1.0
    maximum = 0.0
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["protein_id", "position", "residue", "idr_probability"]:
            raise ValueError(f"unexpected prediction columns: {reader.fieldnames}")
        for row, target in zip(reader, expected, strict=False):
            observed = (row["protein_id"], int(row["position"]), row["residue"])
            if observed != target:
                raise ValueError(f"prediction alignment mismatch at row {count + 1}")
            score = float(row["idr_probability"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"invalid probability at row {count + 1}: {score}")
            minimum = min(minimum, score)
            maximum = max(maximum, score)
            count += 1
        if next(reader, None) is not None:
            raise ValueError("prediction file contains extra rows")
    if count != len(expected):
        raise ValueError(f"expected {len(expected)} prediction rows, got {count}")
    return {"prediction_rows": count, "minimum_probability": minimum, "maximum_probability": maximum}


def validate_inference_report(report: dict[str, Any], fasta: Path, output: Path) -> None:
    required = {
        "status": "pass",
        "locked_members": 30,
        "a2_members": 15,
        "a8_members": 15,
        "proteins": EXPECTED_PROTEINS,
        "residues": EXPECTED_RESIDUES,
        "input_contains_labels": False,
        "all_checkpoint_hashes_match": True,
        "development_checkpoint_excluded": True,
        "caid1_caid2_caid3_labels_accessed": False,
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise ValueError(f"inference report {field}={report.get(field)!r}, expected {expected!r}")
    if report.get("input_fasta_sha256") != sha256(fasta):
        raise ValueError("inference report FASTA hash mismatch")
    if report.get("output_sha256") != sha256(output):
        raise ValueError("inference report output hash mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-root", type=Path, required=True)
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("data/external/s2_database_curated_20260906/sequences_only/s2_sequences.fasta"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/supplementary/s2_independent_holdout/a10_s2_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/supplementary/s2_independent_holdout/a10_prediction_20260907"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_pins()
    fasta = resolve(args.fasta)
    records = read_label_free_fasta(fasta)
    locked_root = resolve(args.locked_root)
    predictor = ROOT / "experiments/final_models/a10_locked_inference/predict_a10.py"
    lock_manifest = ROOT / "experiments/final_models/a10_locked_inference/a10_lock_manifest.json"
    base_command = [
        sys.executable,
        "-u",
        str(predictor),
        "--locked-root",
        str(locked_root),
        "--lock-manifest",
        str(lock_manifest),
        "--device",
        args.device,
    ]
    if args.verify_only:
        subprocess.run([*base_command, "--verify-only"], cwd=ROOT, check=True)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "s2_fasta_sha256": sha256(fasta),
                    "proteins": len(records),
                    "residues": sum(len(sequence) for _, sequence in records),
                    "locked_a10_verified": True,
                    "input_contains_labels": False,
                },
                indent=2,
            )
        )
        return

    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite prediction output: {output_dir}")
    output_dir.mkdir(parents=True)
    cache_dir = resolve(args.cache_dir)
    prediction_path = output_dir / "muse_idr_a10_s2_predictions.tsv.gz"
    inference_report_path = output_dir / "a10_inference_report.json"
    subprocess.run(
        [
            *base_command,
            "--fasta",
            str(fasta),
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(prediction_path),
            "--report",
            str(inference_report_path),
            "--precision",
            args.precision,
            "--max-residues",
            "1022",
            "--window-stride",
            "511",
            "--window-batch-size",
            "1",
        ],
        cwd=ROOT,
        check=True,
    )
    inference_report = json.loads(inference_report_path.read_text(encoding="utf-8"))
    validate_inference_report(inference_report, fasta, prediction_path)
    prediction_validation = validate_predictions(prediction_path, records)

    driver_report = {
        "schema_version": 1,
        "experiment": "s2_locked_a10_prediction_blind",
        "status": "pass",
        "input_fasta": fasta.relative_to(ROOT).as_posix(),
        "input_fasta_sha256": sha256(fasta),
        "input_contains_labels": False,
        "proteins": EXPECTED_PROTEINS,
        "residues": EXPECTED_RESIDUES,
        "pinned_source_hashes": PINNED_FILES,
        "locked_members": 30,
        "ensemble_method": "thirty_model_uniform_logit_mean",
        "precision": args.precision,
        "prediction_validation": prediction_validation,
        "prediction": prediction_path.relative_to(ROOT).as_posix(),
        "prediction_sha256": sha256(prediction_path),
        "inference_report": inference_report_path.relative_to(ROOT).as_posix(),
        "inference_report_sha256": sha256(inference_report_path),
        "s2_reference_or_labels_accessed": False,
        "scores_used_for_training_tuning_or_selection": False,
        "locked_a10_modified": False,
        "ready_for_label_side_evaluation": True,
    }
    driver_report_path = output_dir / "s2_a10_prediction_report.json"
    driver_report_path.write_text(
        json.dumps(driver_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    test_path = Path(__file__).with_name("test_run_s2_a10_prediction.py")
    locked_paths = [
        *(ROOT / relative for relative in PINNED_FILES),
        Path(__file__).resolve(),
        test_path,
        prediction_path,
        inference_report_path,
        driver_report_path,
    ]
    lock_path = output_dir / "s2_a10_prediction_lock.sha256"
    lock_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in locked_paths),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(driver_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
