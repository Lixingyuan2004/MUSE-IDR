"""Build the deterministic, label-free S2 two-baseline inference package."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from run_s2_baseline_predictions import (
    EXPECTED_PROTEINS,
    EXPECTED_RESIDUES,
    PINNED_FILES,
    PROTOCOL,
    S2_FASTA,
    read_label_free_fasta,
    sha256,
)


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_NAME = "s2_two_baseline_inference_upload_20260907_r2"
FASTA_SOURCE = Path(
    "outputs/supplementary/s2_independent_holdout/"
    "final_database_curated_20260906/s2_sequences.fasta"
)
CODE_FILES = [
    Path("experiments/analysis/b4_baseline_reproduction/cache_punch2_light_features.py"),
    Path("experiments/analysis/b4_baseline_reproduction/predict_punch2_light_official.py"),
    Path("experiments/analysis/b4_baseline_reproduction/punch2_light_common.py"),
    Path("experiments/analysis/b4_baseline_reproduction/predict_lora_official.py"),
    Path("experiments/analysis/b4_baseline_reproduction/run_lora_official_smoke.py"),
    PROTOCOL,
    Path(
        "experiments/supplementary/s2_independent_holdout/"
        "predict_lora_official_offline.py"
    ),
    Path("experiments/supplementary/s2_independent_holdout/run_s2_baseline_predictions.py"),
    Path("experiments/supplementary/s2_independent_holdout/test_run_s2_baseline_predictions.py"),
]
FORBIDDEN_PAYLOAD_FRAGMENTS = (
    "s2_reference",
    "s2_labels",
    "manual_review",
    "provisional_reference",
    "evaluation_20260907",
    "a10_prediction_20260907",
)


def assert_no_forbidden_payload(paths: list[Path]) -> None:
    for path in paths:
        normalized = path.as_posix().casefold()
        if any(fragment in normalized for fragment in FORBIDDEN_PAYLOAD_FRAGMENTS):
            raise ValueError(f"forbidden label/result-side artifact in upload: {path}")


def write_deterministic_tar_gz(source: Path, destination: Path) -> None:
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(item for item in source.rglob("*") if item.is_file()):
                    relative = path.relative_to(source).as_posix()
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=Path("release"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = args.release_dir if args.release_dir.is_absolute() else ROOT / args.release_dir
    stage = release_dir / PACKAGE_NAME
    archive = release_dir / f"{PACKAGE_NAME}.tar.gz"
    if stage.exists() or archive.exists():
        raise FileExistsError(f"refusing to overwrite package: {stage} or {archive}")

    source_fasta = ROOT / FASTA_SOURCE
    if sha256(source_fasta) != PINNED_FILES[S2_FASTA.as_posix()]:
        raise ValueError("S2 sequence FASTA does not match its frozen hash")
    records = read_label_free_fasta(source_fasta)
    if len(records) != EXPECTED_PROTEINS:
        raise ValueError("S2 protein count mismatch")
    if sum(len(sequence) for _, sequence in records) != EXPECTED_RESIDUES:
        raise ValueError("S2 residue count mismatch")

    copied = [S2_FASTA, *CODE_FILES]
    assert_no_forbidden_payload(copied)
    stage.mkdir(parents=True)
    destination_fasta = stage / S2_FASTA
    destination_fasta.parent.mkdir(parents=True)
    shutil.copyfile(source_fasta, destination_fasta)
    for relative in CODE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    metadata_dir = stage / "release_metadata" / PACKAGE_NAME
    metadata_dir.mkdir(parents=True)
    sequence_manifest_path = metadata_dir / "sequence_manifest.json"
    sequence_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "S2 database-curated temporal holdout",
                "fasta": S2_FASTA.as_posix(),
                "fasta_sha256": sha256(destination_fasta),
                "proteins": EXPECTED_PROTEINS,
                "residues": EXPECTED_RESIDUES,
                "input_contains_labels": False,
                "reference_or_label_files_included": False,
                "a10_or_baseline_predictions_included": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    instructions_path = metadata_dir / "RUN_AUTODL.md"
    instructions_path.write_text(
        """# S2 two-baseline blind inference on AutoDL

This package contains the frozen comparison protocol, the same 87 sequence-only
S2 FASTA records, and drivers for exactly two formal baselines:
PUNCH2-Light Released-13 and LoRA-DR-Suite ESM2-650M DisProt7.  It contains no
S2 labels, MUSE-IDR scores, S2 evaluation results, or baseline predictions.

The AutoDL project must already contain the official PUNCH2-Light checkout at
`third_party/baselines/punch2_light`.  The preflight verifies commit
`6c7935b3597c056d2e6b3845bb54fc101c5bc574` plus all 13 weights and the two
model source files.  LoRA uses the exact adapter and ESM-2 revisions already
present in the configured Hugging Face cache; both are loaded with
`local_files_only=True`, and the adapter hashes are checked after loading.
No third-party weights are redistributed.

## 1. Extract and test

Run from the repository root after checking the archive SHA256 printed by the
local build report:

```bash
tar -xzf s2_two_baseline_inference_upload_20260907_r2.tar.gz
source .venv/bin/activate

python -m py_compile \\
  experiments/supplementary/s2_independent_holdout/predict_lora_official_offline.py \\
  experiments/supplementary/s2_independent_holdout/run_s2_baseline_predictions.py \\
  experiments/supplementary/s2_independent_holdout/test_run_s2_baseline_predictions.py

python experiments/supplementary/s2_independent_holdout/test_run_s2_baseline_predictions.py
```

If LoRA preflight reports that `peft` is missing, install the already validated
version into this virtual environment:

```bash
python -m pip install peft==0.20.0
```

If PUNCH2-Light preflight reports that `sentencepiece` is missing, install it
before feature extraction:

```bash
python -m pip install sentencepiece
```

## 2. Preflight both frozen baselines

```bash
python -u experiments/supplementary/s2_independent_holdout/run_s2_baseline_predictions.py \\
  --baseline punch2_light --device cuda --verify-only

python -u experiments/supplementary/s2_independent_holdout/run_s2_baseline_predictions.py \\
  --baseline lora_dr_suite --device cuda --verify-only
```

Do not start prediction unless both preflights report `status: pass`.

## 3. Run PUNCH2-Light Released-13 first

ProtT5 is pinned and should reuse `cache/huggingface`.  The formal
feature extraction precision is FP32.

```bash
mkdir -p logs
screen -dmS s2punchlight bash -lc '
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
export HF_HOME="$(pwd)/cache/huggingface"
export HF_HUB_OFFLINE=1
exec > logs/s2_punch2_light_released13_20260907.log 2>&1
python -u experiments/supplementary/s2_independent_holdout/run_s2_baseline_predictions.py \\
  --baseline punch2_light --device cuda
'
```

## 4. Run LoRA-DR-Suite second

The wrapper preserves the official model, residue windowing and output logic,
but resolves the already cached adapter and base model by their fixed commit
hashes.  It makes no Hugging Face network request.

```bash
screen -dmS s2lora bash -lc '
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
export HF_HOME="$(pwd)/cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec > logs/s2_lora_dr_suite_650m_disprot7_20260907.log 2>&1
python -u experiments/supplementary/s2_independent_holdout/run_s2_baseline_predictions.py \\
  --baseline lora_dr_suite --device cuda
'
```

Each driver refuses to overwrite an existing output directory, validates every
residue identifier/position/amino acid/probability, and writes a SHA256 result
lock.  Only the locked prediction outputs are downloaded for local label-side
evaluation.
""",
        encoding="utf-8",
        newline="\n",
    )

    report_path = metadata_dir / "upload_report.json"
    report = {
        "schema_version": 1,
        "experiment": "s2_two_baseline_label_blind_inference_upload",
        "status": "pass",
        "package_name": PACKAGE_NAME,
        "formal_baselines": [
            "PUNCH2-Light Released-13",
            "LoRA-DR-Suite ESM2-650M DisProt7",
        ],
        "protocol_sha256": sha256(stage / PROTOCOL),
        "fasta_sha256": sha256(destination_fasta),
        "proteins": EXPECTED_PROTEINS,
        "residues": EXPECTED_RESIDUES,
        "input_contains_labels": False,
        "reference_or_label_files_included": False,
        "muse_idr_predictions_or_s2_results_included": False,
        "third_party_weights_included": False,
        "punch2_light_checkout_expected_at": "third_party/baselines/punch2_light",
        "lora_weights_loaded_from_pinned_offline_cache": True,
        "lora_network_requests_required": False,
        "locked_a10_modified": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest_path = metadata_dir / "UPLOAD_MANIFEST.sha256"
    payload = sorted(path for path in stage.rglob("*") if path.is_file() and path != manifest_path)
    assert_no_forbidden_payload([path.relative_to(stage) for path in payload])
    manifest_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(stage).as_posix()}\n" for path in payload),
        encoding="utf-8",
        newline="\n",
    )
    write_deterministic_tar_gz(stage, archive)
    print(
        json.dumps(
            {
                **report,
                "stage": str(stage),
                "archive": str(archive),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": sha256(archive),
                "manifest_entries": len(payload),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
