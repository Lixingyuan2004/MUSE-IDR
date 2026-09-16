"""Build a deterministic, label-free S2 A10 inference upload archive."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from run_s2_a10_prediction import (
    EXPECTED_PROTEINS,
    EXPECTED_RESIDUES,
    PINNED_FILES,
    read_label_free_fasta,
    sha256,
)


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_NAME = "s2_a10_inference_upload_20260907"
FASTA_SOURCE = Path(
    "outputs/supplementary/s2_independent_holdout/"
    "final_database_curated_20260906/s2_sequences.fasta"
)
FASTA_DESTINATION = Path(
    "data/external/s2_database_curated_20260906/"
    "sequences_only/s2_sequences.fasta"
)
CODE_FILES = [
    Path("experiments/ablations/a1_frozen_esm2/__init__.py"),
    Path("experiments/ablations/a1_frozen_esm2/data.py"),
    Path("experiments/ablations/a2_multiscale_context/__init__.py"),
    Path("experiments/ablations/a2_multiscale_context/model.py"),
    Path("experiments/ablations/a8_multilayer_esm2_fusion/__init__.py"),
    Path("experiments/ablations/a8_multilayer_esm2_fusion/cache_multilayer.py"),
    Path("experiments/ablations/a8_multilayer_esm2_fusion/model.py"),
    Path("experiments/final_models/__init__.py"),
    Path("experiments/final_models/a10_locked_inference/__init__.py"),
    Path("experiments/final_models/a10_locked_inference/a10_lock_manifest.json"),
    Path("experiments/final_models/a10_locked_inference/predict_a10.py"),
    Path("experiments/supplementary/s2_independent_holdout/run_s2_a10_prediction.py"),
    Path("experiments/supplementary/s2_independent_holdout/test_run_s2_a10_prediction.py"),
]
FORBIDDEN_PAYLOAD_FRAGMENTS = (
    "s2_reference",
    "s2_labels",
    "s2_cohort",
    "manual_review",
)


def assert_no_forbidden_payload(paths: list[Path]) -> None:
    for path in paths:
        normalized = path.as_posix().casefold()
        if any(fragment in normalized for fragment in FORBIDDEN_PAYLOAD_FRAGMENTS):
            raise ValueError(f"forbidden label-side artifact in upload: {path}")


def write_deterministic_tar_gz(source: Path, destination: Path) -> None:
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted((item for item in source.rglob("*") if item.is_file())):
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
    if sha256(source_fasta) != PINNED_FILES[FASTA_DESTINATION.as_posix()]:
        raise ValueError("S2 FASTA does not match its frozen hash")
    records = read_label_free_fasta(source_fasta)
    if len(records) != EXPECTED_PROTEINS or sum(len(sequence) for _, sequence in records) != EXPECTED_RESIDUES:
        raise ValueError("S2 FASTA statistics do not match the frozen cohort")

    assert_no_forbidden_payload([FASTA_DESTINATION, *CODE_FILES])
    stage.mkdir(parents=True)
    destination_fasta = stage / FASTA_DESTINATION
    destination_fasta.parent.mkdir(parents=True)
    shutil.copyfile(source_fasta, destination_fasta)
    for relative in CODE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    sequence_manifest_path = stage / FASTA_DESTINATION.parent / "s2_sequence_manifest.json"
    sequence_manifest = {
        "schema_version": 1,
        "dataset": "S2 database-curated temporal holdout",
        "fasta": FASTA_DESTINATION.as_posix(),
        "fasta_sha256": sha256(destination_fasta),
        "proteins": EXPECTED_PROTEINS,
        "residues": EXPECTED_RESIDUES,
        "input_contains_labels": False,
        "reference_or_label_files_included": False,
        "model_predictions_accessed_before_freeze": False,
    }
    sequence_manifest_path.write_text(
        json.dumps(sequence_manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    metadata_dir = stage / "release_metadata" / PACKAGE_NAME
    metadata_dir.mkdir(parents=True)
    instructions_path = metadata_dir / "RUN_AUTODL.md"
    instructions_path.write_text(
        """# Run S2 locked A10 inference on AutoDL

This upload contains the sequence-only S2 FASTA and the exact v1.0.0-paper
A10 inference sources. It contains no S2 reference labels.

From the repository root, verify the archive hash shown by the local build
report, extract it, activate `.venv`, and run:

```bash
python -m py_compile \\
  experiments/supplementary/s2_independent_holdout/run_s2_a10_prediction.py \\
  experiments/supplementary/s2_independent_holdout/test_run_s2_a10_prediction.py

python experiments/supplementary/s2_independent_holdout/test_run_s2_a10_prediction.py

export HF_HOME="$(pwd)/cache/huggingface"
export HF_HUB_OFFLINE=1

python -u experiments/supplementary/s2_independent_holdout/run_s2_a10_prediction.py \\
  --locked-root models/weights/a10_locked \\
  --device cuda \\
  --verify-only
```

Only after verification passes, start the prediction in `screen` with the
same driver, `--precision bf16`, and no `--verify-only`. The driver refuses
to overwrite an existing result directory and creates a prediction SHA256
lock before label-side evaluation is allowed.
""",
        encoding="utf-8",
        newline="\n",
    )

    copied = [FASTA_DESTINATION, *CODE_FILES]
    report_path = metadata_dir / "upload_report.json"
    report = {
        "schema_version": 1,
        "experiment": "s2_a10_label_blind_inference_upload",
        "status": "pass",
        "package_name": PACKAGE_NAME,
        "fasta_sha256": sha256(destination_fasta),
        "proteins": EXPECTED_PROTEINS,
        "residues": EXPECTED_RESIDUES,
        "payload_files_before_metadata": len(copied),
        "input_contains_labels": False,
        "reference_or_label_files_included": False,
        "locked_weights_included": False,
        "locked_weights_expected_at": "models/weights/a10_locked",
        "a10_sources_match_v1.0.0-paper": True,
        "model_predictions_accessed": False,
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
