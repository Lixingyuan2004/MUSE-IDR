"""Run the two frozen S2 baselines on the label-free sequence FASTA.

This driver deliberately has no label/reference argument.  It validates the
frozen protocol, source files, S2 sequences and model provenance before it
starts either PUNCH2-Light Released-13 or LoRA-DR-Suite ESM2-650M DisProt7.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_PROTEINS = 87
EXPECTED_RESIDUES = 56394
S2_FASTA = Path(
    "data/external/s2_database_curated_20260906/sequences_only/s2_sequences.fasta"
)
PROTOCOL = Path(
    "experiments/supplementary/s2_independent_holdout/"
    "s2_baseline_comparison_protocol_20260907.json"
)
PROTOCOL_SHA256 = "a231176522a0bd1bd5a0566c74d9006ab9ea6a65100d58b371b617d3533413e2"
PUNCH_REPO_COMMIT = "6c7935b3597c056d2e6b3845bb54fc101c5bc574"
PROTT5_MODEL_ID = "Rostlab/prot_t5_xl_uniref50"
PROTT5_REVISION = "973be27c52ee6474de9c945952a8008aeb2a1a73"
LORA_MODEL_ID = "CQSB/esm2_650M-LoRA-ID-DisProt7"
LORA_REVISION = "4ee4e4d404a224d76cf1c59fb99aebd7b52f7915"
LORA_BASE_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
LORA_BASE_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
LORA_OFFLINE_WRAPPER = Path(
    "experiments/supplementary/s2_independent_holdout/"
    "predict_lora_official_offline.py"
)
LORA_ADAPTER_CONFIG_SHA256 = (
    "c04ea976b27dca7c75a524695ade8d3f1410572c2ebe743626854ee84694b40d"
)
LORA_ADAPTER_WEIGHTS_SHA256 = (
    "02f1e288f36f369786db68df5270affa693d2359a5badb0275c28e81977448fe"
)

PINNED_FILES = {
    S2_FASTA.as_posix(): "d777bfeb9cd8cf82af003c10bcef8b5b3e09dbee276e7e16164f408e336d67df",
    PROTOCOL.as_posix(): PROTOCOL_SHA256,
    "experiments/analysis/b4_baseline_reproduction/cache_punch2_light_features.py": "83f4a3eb3257423930d69dcf3943dffd769948228ab362c0e880cf3bcf2b7b18",
    "experiments/analysis/b4_baseline_reproduction/predict_punch2_light_official.py": "12c40576757be6542b191201b449b838a6821ad1ab46d9a14c129f81fbe07d90",
    "experiments/analysis/b4_baseline_reproduction/punch2_light_common.py": "27372a078873cce20477a9dd87053713f9d3b4f97d6239e5861579ece99a0034",
    "experiments/analysis/b4_baseline_reproduction/predict_lora_official.py": "1fc9db12bda7539b2b8d3545c8194b27fec21384effd4871a8326fb07fedae3b",
    "experiments/analysis/b4_baseline_reproduction/run_lora_official_smoke.py": "09ad1dc290629c6132fa9960e3f9b2706102290afe4de5d9dadd638812bac063",
    LORA_OFFLINE_WRAPPER.as_posix(): "57653fdd66be807849ebdbe4a4e552118fa5585e1f24d6226798c0fd40951817",
}

PUNCH_REQUIRED_FILES = {
    "model/cnn2_L12.py": "be4b0b7eacf4185cdc71d839acfde8155b6df8a803f8f80fe34e029e630128df",
    "model/cnn2_L3_100_50.py": "a3da860ed0afe1491cce1b755cf63bc07e201f75ed339c7104684502f0df5059",
    "predictor/onehot/cnn2_L12_withFullyDisorder78.pth_f1": "0907a9da88e8e3e5bbf06793360630523514665ed6e467f56812ca54132afc4c",
    "predictor/onehot/cnn2_L12_withFullyDisorder78.pth_f2": "e86c53e22a8b62cb67457ce9537499786693c86c7f44415cf8a853d8151bc3f0",
    "predictor/onehot/cnn2_L12_withFullyDisorder78.pth_f3": "9c29c59f65dd1e6a413fb78ea86fec0d59e7fc1fb7206d021ddcdca38f50e7ab",
    "predictor/protTrans/cnn2_L3_100_50_withFullyDisorder78.pth_f1": "098e3e0187dedcebd503d157ff1430e60800c0fbed6ed984f723607caa71a3a7",
    "predictor/protTrans/cnn2_L3_100_50_withFullyDisorder78.pth_f2": "31600b220e6d431e37dd392b2a798f91b26ae2b60268fe4cffad5be3a80b1b10",
    "predictor/protTrans/cnn2_L3_100_50_withFullyDisorder78.pth_f3": "2765a9489486ae1d97aa10a237fdaab4ba418067499c11b88d5b11b446880647",
    "predictor/protTrans/cnn2_L3_100_50_withFullyDisorder78.pth_f4": "fabbcf25e680c7a26a36528a7e89be0366bf9edba2361380994feec454a0a41d",
    "predictor/protTrans/cnn2_L3_100_50_withFullyDisorder78.pth_f5": "856529e06ffe021dea7eb9ff8c41d1c0168c7b26c510a51b9fdf0a9d3aa6ff62",
    "predictor/protTrans/cnn2_L12_withFullyDisorder78.pth_f1": "0bf83f2590e4800cd378b04dc188c633fe1cac3ac05d9158c7041322648a0f54",
    "predictor/protTrans/cnn2_L12_withFullyDisorder78.pth_f2": "1ac2e0703c9e0a700dd90097e88d8ef4cb6ca50926d74db12bf5f189627b1925",
    "predictor/protTrans/cnn2_L12_withFullyDisorder78.pth_f3": "b03960c2a729eb7a249afdab43aadaefff99e66986d636a6624b3cf9145fab64",
    "predictor/protTrans/cnn2_L12_withFullyDisorder78.pth_f4": "ac92bbb02b80f5d5446805f0b9ebeb46e4c5fe15eae141db3da1687bb0f55602",
    "predictor/protTrans/cnn2_L12_withFullyDisorder78.pth_f5": "f94ac7faa059ee5f5337f3b4a6953450c88be51d125e7174d50644210d6f82c7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text_sha256(path: Path) -> str:
    """Hash source text after canonical LF conversion for cross-platform checks."""
    text = path.read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def verify_hashes(root: Path, expected: dict[str, str]) -> None:
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != digest:
            raise ValueError(f"hash mismatch: {relative}: {observed} != {digest}")


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
        if not value or set(value) - set("ACDEFGHIKLMNPQRSTVWYBXZJUO"):
            raise ValueError(f"invalid sequence-only FASTA record: {protein_id}")
        records.append((protein_id, value))
        protein_id = None
        sequence = []

    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish()
            protein_id = line[1:].split(maxsplit=1)[0]
            if not protein_id or protein_id in seen:
                raise ValueError(f"empty or duplicate FASTA identifier at line {number}")
            seen.add(protein_id)
        else:
            if protein_id is None:
                raise ValueError(f"sequence before FASTA header at line {number}")
            sequence.append("".join(line.split()))
    finish()
    if len(records) != EXPECTED_PROTEINS:
        raise ValueError(f"expected {EXPECTED_PROTEINS} proteins, got {len(records)}")
    residues = sum(len(value) for _, value in records)
    if residues != EXPECTED_RESIDUES:
        raise ValueError(f"expected {EXPECTED_RESIDUES} residues, got {residues}")
    return records


def verify_protocol() -> dict[str, Any]:
    path = ROOT / PROTOCOL
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError("S2 baseline protocol hash mismatch")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_status") != "frozen_before_s2_baseline_prediction":
        raise ValueError("S2 baseline protocol is not frozen")
    names = [row["name"] for row in protocol.get("formal_comparators", [])]
    if names != [
        "PUNCH2-Light Released-13",
        "LoRA-DR-Suite ESM2-650M DisProt7",
    ]:
        raise ValueError(f"unexpected formal comparator list: {names}")
    return protocol


def verify_punch_repository(repo: Path) -> None:
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"PUNCH2-Light git checkout is required: {repo}")
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PUNCH_REPO_COMMIT:
        raise ValueError(f"PUNCH2-Light commit mismatch: {head}")
    for relative, expected in PUNCH_REQUIRED_FILES.items():
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = normalized_text_sha256(path) if path.suffix == ".py" else sha256(path)
        if observed != expected:
            raise ValueError(
                f"PUNCH2-Light artifact hash mismatch: {relative}: "
                f"{observed} != {expected}"
            )


def dependency_versions(baseline: str) -> dict[str, str]:
    packages = ["numpy", "torch", "transformers", "huggingface-hub"]
    if baseline == "punch2_light":
        packages.append("sentencepiece")
    else:
        packages.append("peft")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required package is missing: {package}") from error
    return versions


def validate_prediction_tsv(
    path: Path,
    records: Sequence[tuple[str, str]],
    position_column: str,
) -> dict[str, Any]:
    expected = (
        (protein_id, position, residue)
        for protein_id, sequence in records
        for position, residue in enumerate(sequence, start=1)
    )
    expected_header = ["protein_id", position_column, "residue", "idr_probability"]
    count = 0
    minimum = 1.0
    maximum = 0.0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected_header:
            raise ValueError(f"unexpected prediction header: {reader.fieldnames}")
        for target in expected:
            row = next(reader, None)
            if row is None:
                raise ValueError(f"prediction ended early after {count} rows")
            observed = (
                row["protein_id"],
                int(row[position_column]),
                row["residue"],
            )
            if observed != target:
                raise ValueError(f"prediction alignment mismatch at row {count + 1}")
            probability = float(row["idr_probability"])
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"invalid probability at row {count + 1}")
            minimum = min(minimum, probability)
            maximum = max(maximum, probability)
            count += 1
        if next(reader, None) is not None:
            raise ValueError("prediction contains extra rows")
    expected_rows = sum(len(sequence) for _, sequence in records)
    if count != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {count}")
    return {
        "prediction_rows": count,
        "minimum_probability": minimum,
        "maximum_probability": maximum,
        "complete_coverage": True,
    }


def require_fields(report: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"report {key}={report.get(key)!r}, expected {value!r}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def lock_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_result_lock(path: Path, files: Sequence[Path]) -> None:
    unique = sorted({item.resolve() for item in files}, key=lambda item: lock_label(item))
    path.write_text(
        "".join(f"{sha256(item)}  {lock_label(item)}\n" for item in unique),
        encoding="utf-8",
        newline="\n",
    )


def run_punch(
    fasta: Path,
    records: Sequence[tuple[str, str]],
    repo: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    feature_dir = output_dir / "features_fp32"
    feature_report = output_dir / "feature_report_fp32.json"
    prediction_dir = output_dir / "predictions"
    inference_report = output_dir / "inference_report.json"
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "experiments/analysis/b4_baseline_reproduction/cache_punch2_light_features.py"),
            "--fasta", str(fasta),
            "--output-dir", str(feature_dir),
            "--report", str(feature_report),
            "--model-id", PROTT5_MODEL_ID,
            "--revision", PROTT5_REVISION,
            "--device", device,
            "--precision", "fp32",
        ],
        cwd=ROOT,
        check=True,
    )
    feature = json.loads(feature_report.read_text(encoding="utf-8"))
    require_fields(
        feature,
        {
            "status": "pass",
            "input_fasta_sha256": sha256(fasta),
            "input_contains_labels": False,
            "proteins": EXPECTED_PROTEINS,
            "residues": EXPECTED_RESIDUES,
            "prott5_model_id": PROTT5_MODEL_ID,
            "prott5_revision": PROTT5_REVISION,
            "inference_precision": "float32",
            "storage_dtype": "float32",
            "caid_labels_accessed": False,
        },
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "experiments/analysis/b4_baseline_reproduction/predict_punch2_light_official.py"),
            "--fasta", str(fasta),
            "--feature-dir", str(feature_dir),
            "--feature-report", str(feature_report),
            "--repo", str(repo),
            "--variants", "released13",
            "--output-dir", str(prediction_dir),
            "--report", str(inference_report),
            "--device", device,
        ],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(inference_report.read_text(encoding="utf-8"))
    require_fields(
        report,
        {
            "status": "pass",
            "source_revision": PUNCH_REPO_COMMIT,
            "input_fasta_sha256": sha256(fasta),
            "input_contains_labels": False,
            "proteins": EXPECTED_PROTEINS,
            "residues": EXPECTED_RESIDUES,
            "released_members_loaded": 13,
            "prediction_alignment": True,
            "caid_labels_accessed": False,
            "scores_used_for_training_or_tuning": False,
        },
    )
    if set(report.get("variants", {})) != {"released13"}:
        raise ValueError("PUNCH2-Light run must contain Released-13 only")
    prediction = prediction_dir / "punch2_light_released13_predictions.tsv.gz"
    caid_output = prediction_dir / "punch2_light_released13.caid"
    validation = validate_prediction_tsv(prediction, records, "residue_index")
    return {
        "baseline": "PUNCH2-Light Released-13",
        "prediction": prediction,
        "caid_output": caid_output,
        "feature_report": feature_report,
        "inference_report": inference_report,
        "validation": validation,
        "model_provenance": {
            "source_revision": PUNCH_REPO_COMMIT,
            "members": 13,
            "prot_t5_model_id": PROTT5_MODEL_ID,
            "prot_t5_revision": PROTT5_REVISION,
        },
    }


def run_lora(
    fasta: Path,
    records: Sequence[tuple[str, str]],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    prediction = output_dir / "lora_dr_suite_650m_disprot7_predictions.tsv.gz"
    caid_output = output_dir / "lora_dr_suite_650m_disprot7.caid"
    inference_report = output_dir / "inference_report.json"
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / LORA_OFFLINE_WRAPPER),
            "--fasta", str(fasta),
            "--model-id", LORA_MODEL_ID,
            "--revision", LORA_REVISION,
            "--output", str(prediction),
            "--caid-output", str(caid_output),
            "--report", str(inference_report),
            "--device", device,
            "--precision", "bf16",
            "--max-residues", "1022",
            "--window-stride", "511",
            "--window-batch-size", "1",
        ],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(inference_report.read_text(encoding="utf-8"))
    require_fields(
        report,
        {
            "status": "pass",
            "model_id": LORA_MODEL_ID,
            "model_revision": LORA_REVISION,
            "base_model_id": LORA_BASE_MODEL_ID,
            "base_model_revision": LORA_BASE_REVISION,
            "adapter_config_sha256": LORA_ADAPTER_CONFIG_SHA256,
            "adapter_weights_sha256": LORA_ADAPTER_WEIGHTS_SHA256,
            "offline_cached_loading": True,
            "network_requests_required": False,
            "training_regime": "DisProt_7_only",
            "input_fasta_sha256": sha256(fasta),
            "input_contains_labels": False,
            "proteins": EXPECTED_PROTEINS,
            "residues": EXPECTED_RESIDUES,
            "max_residues": 1022,
            "window_stride": 511,
            "window_batch_size": 1,
            "prediction_alignment": True,
            "caid_labels_accessed": False,
            "scores_used_for_training_or_tuning": False,
        },
    )
    validation = validate_prediction_tsv(prediction, records, "position")
    return {
        "baseline": "LoRA-DR-Suite ESM2-650M DisProt7",
        "prediction": prediction,
        "caid_output": caid_output,
        "inference_report": inference_report,
        "validation": validation,
        "model_provenance": {
            "model_id": LORA_MODEL_ID,
            "model_revision": LORA_REVISION,
            "base_model_id": LORA_BASE_MODEL_ID,
            "base_model_revision": LORA_BASE_REVISION,
            "adapter_config_sha256": LORA_ADAPTER_CONFIG_SHA256,
            "adapter_weights_sha256": LORA_ADAPTER_WEIGHTS_SHA256,
            "offline_cached_loading": True,
            "network_requests_required": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        choices=("punch2_light", "lora_dr_suite"),
    )
    parser.add_argument("--fasta", type=Path, default=S2_FASTA)
    parser.add_argument(
        "--punch-repo",
        type=Path,
        default=Path("third_party/baselines/punch2_light"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_hashes(ROOT, PINNED_FILES)
    protocol = verify_protocol()
    fasta = resolve(args.fasta)
    records = read_label_free_fasta(fasta)
    versions = dependency_versions(args.baseline)
    punch_repo = resolve(args.punch_repo)
    if args.baseline == "punch2_light":
        verify_punch_repository(punch_repo)

    preflight = {
        "schema_version": 1,
        "experiment": "s2_locked_baseline_label_blind_prediction",
        "status": "pass",
        "mode": "preflight" if args.verify_only else "prediction",
        "baseline": args.baseline,
        "protocol": PROTOCOL.as_posix(),
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_status": protocol["protocol_status"],
        "input_fasta": lock_label(fasta),
        "input_fasta_sha256": sha256(fasta),
        "input_contains_labels": False,
        "proteins": len(records),
        "residues": sum(len(sequence) for _, sequence in records),
        "dependency_versions": versions,
        "s2_reference_or_labels_accessed": False,
        "locked_a10_modified": False,
    }
    if args.verify_only:
        if args.baseline == "punch2_light":
            preflight["punch_repository_revision"] = PUNCH_REPO_COMMIT
            preflight["punch_required_files_verified"] = len(PUNCH_REQUIRED_FILES)
        else:
            preflight["cached_model_to_verify_during_load"] = LORA_MODEL_ID
            preflight["cached_model_revision"] = LORA_REVISION
            preflight["cached_base_model_revision"] = LORA_BASE_REVISION
            preflight["network_requests_required"] = False
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return

    default_name = (
        "punch2_light_released13_prediction_20260907"
        if args.baseline == "punch2_light"
        else "lora_dr_suite_650m_disprot7_offline_prediction_20260907"
    )
    output_dir = resolve(
        args.output_dir
        or Path("outputs/supplementary/s2_independent_holdout") / default_name
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite baseline output: {output_dir}")
    output_dir.mkdir(parents=True)

    if args.baseline == "punch2_light":
        result = run_punch(fasta, records, punch_repo, output_dir, args.device)
    else:
        result = run_lora(fasta, records, output_dir, args.device)

    driver_report = {
        **preflight,
        "baseline_name": result["baseline"],
        "prediction": lock_label(result["prediction"]),
        "prediction_sha256": sha256(result["prediction"]),
        "prediction_validation": result["validation"],
        "model_provenance": result["model_provenance"],
        "scores_used_for_training_tuning_calibration_or_selection": False,
        "ready_for_label_side_evaluation": True,
    }
    driver_report_path = output_dir / "s2_baseline_prediction_report.json"
    write_json(driver_report_path, driver_report)
    locked_files = [
        *(ROOT / relative for relative in PINNED_FILES),
        Path(__file__).resolve(),
        result["prediction"],
        result["caid_output"],
        result["inference_report"],
        driver_report_path,
    ]
    if args.baseline == "punch2_light":
        locked_files.extend(punch_repo / relative for relative in PUNCH_REQUIRED_FILES)
        locked_files.append(result["feature_report"])
    lock_path = output_dir / "s2_baseline_prediction_lock.sha256"
    write_result_lock(lock_path, locked_files)
    driver_report["result_lock"] = lock_label(lock_path)
    driver_report["result_lock_sha256"] = sha256(lock_path)
    print(json.dumps(driver_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
