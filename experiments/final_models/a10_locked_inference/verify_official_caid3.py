"""Verify official CAID3 ROC-AUC results against the locked internal report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


OFFICIAL_CAID_REPOSITORY = "https://github.com/BioComputingUP/CAID"
OFFICIAL_CAID_REVISION = "7bab7e8880d7f949e480fdb377c5a5c8546ae336"
OFFICIAL_SOURCE_ARCHIVE_SHA256 = (
    "10e1da1a8124de1cb2a299096537278039170c724b4552f7ede53a58238f6b34"
)
TRACKS = ("disorder_nox", "disorder_pdb")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_official_aucroc(path: Path, model_name: str) -> float:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "aucroc" not in reader.fieldnames:
            raise ValueError(f"official metrics file has no aucroc column: {path}")
        model_column = reader.fieldnames[0]
        matches = [row for row in reader if row.get(model_column) == model_name]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {model_name!r} row in {path}, found {len(matches)}"
        )
    value = float(matches[0]["aucroc"])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"invalid official aucroc in {path}: {value}")
    return value


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def build_consistency_report(
    *,
    official_metrics: dict[str, Path],
    internal_report_path: Path,
    prediction_path: Path,
    source_archive_path: Path,
    requirements_path: Path,
    model_name: str = "a10_locked",
    tolerance: float = 1e-12,
    expected_source_sha256: str = OFFICIAL_SOURCE_ARCHIVE_SHA256,
) -> dict[str, object]:
    if set(official_metrics) != set(TRACKS):
        raise ValueError(f"official metrics tracks must be exactly {list(TRACKS)}")
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")

    internal = json.loads(internal_report_path.read_text(encoding="utf-8-sig"))
    if internal.get("status") != "pass":
        raise ValueError("internal CAID3 evaluation report did not pass")
    if internal.get("official_caid_revision") != OFFICIAL_CAID_REVISION:
        raise ValueError("internal report used a different official CAID revision")
    if internal.get("caid_labels_used_for_training_or_tuning") is not False:
        raise ValueError("internal report does not prove label-free model selection")
    if internal.get("official_caid_prediction_sha256") != file_sha256(prediction_path):
        raise ValueError("official prediction hash does not match the internal report")

    source_sha256 = file_sha256(source_archive_path)
    if source_sha256 != expected_source_sha256:
        raise ValueError("official CAID source archive hash mismatch")

    track_reports: dict[str, dict[str, object]] = {}
    for track in TRACKS:
        official_auc = read_official_aucroc(official_metrics[track], model_name)
        internal_auc = float(internal["tracks"][track]["auc_roc_caid_compatible"])
        delta = official_auc - internal_auc
        if abs(delta) > tolerance:
            raise ValueError(
                f"official/internal CAID-compatible ROC-AUC mismatch for {track}: {delta}"
            )
        track_reports[track] = {
            "official_aucroc": official_auc,
            "internal_caid_compatible_aucroc": internal_auc,
            "delta": delta,
            "official_metrics": str(official_metrics[track]),
            "official_metrics_sha256": file_sha256(official_metrics[track]),
        }

    return {
        "schema_version": 1,
        "experiment": "a10_locked_official_caid3_consistency",
        "status": "pass",
        "model_name": model_name,
        "official_caid_repository": OFFICIAL_CAID_REPOSITORY,
        "official_caid_revision": OFFICIAL_CAID_REVISION,
        "official_source_archive": str(source_archive_path),
        "official_source_archive_sha256": source_sha256,
        "official_requirements": str(requirements_path),
        "official_requirements_sha256": file_sha256(requirements_path),
        "prediction": str(prediction_path),
        "prediction_sha256": file_sha256(prediction_path),
        "internal_report": str(internal_report_path),
        "internal_report_sha256": file_sha256(internal_report_path),
        "comparison_tolerance": tolerance,
        "tracks": track_reports,
        "model_selection_completed_before_label_access": internal.get(
            "model_selection_completed_before_label_access"
        )
        is True,
        "caid_labels_used_for_training_or_tuning": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-nox-metrics", type=Path, required=True)
    parser.add_argument("--official-pdb-metrics", type=Path, required=True)
    parser.add_argument("--internal-report", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--official-source-archive", type=Path, required=True)
    parser.add_argument("--official-requirements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="a10_locked")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_consistency_report(
        official_metrics={
            "disorder_nox": args.official_nox_metrics,
            "disorder_pdb": args.official_pdb_metrics,
        },
        internal_report_path=args.internal_report,
        prediction_path=args.prediction,
        source_archive_path=args.official_source_archive,
        requirements_path=args.official_requirements,
        model_name=args.model_name,
        tolerance=args.tolerance,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
