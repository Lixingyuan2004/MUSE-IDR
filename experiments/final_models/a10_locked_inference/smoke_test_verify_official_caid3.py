from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.final_models.a10_locked_inference.verify_official_caid3 import (  # noqa: E402
    OFFICIAL_CAID_REVISION,
    build_consistency_report,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_metrics(path: Path, aucroc: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", "aucroc", "aucpr"])
        writer.writerow(["a10_locked", aucroc, 0.8])


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        nox_metrics = root / "nox.csv"
        pdb_metrics = root / "pdb.csv"
        write_metrics(nox_metrics, 0.852)
        write_metrics(pdb_metrics, 0.954)
        prediction = root / "a10_locked.caid"
        prediction.write_text(">p1\n1\tA\t0.5\n", encoding="utf-8")
        source_archive = root / "CAID.tar.gz"
        source_archive.write_bytes(b"fixed official source")
        requirements = root / "requirements.txt"
        requirements.write_text("numpy==2.3.2\n", encoding="utf-8")
        internal_report = root / "internal.json"
        internal_report.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "official_caid_revision": OFFICIAL_CAID_REVISION,
                    "official_caid_prediction_sha256": sha256(prediction),
                    "model_selection_completed_before_label_access": True,
                    "caid_labels_used_for_training_or_tuning": False,
                    "tracks": {
                        "disorder_nox": {"auc_roc_caid_compatible": 0.852},
                        "disorder_pdb": {"auc_roc_caid_compatible": 0.954},
                    },
                }
            ),
            encoding="utf-8",
        )
        common = {
            "official_metrics": {
                "disorder_nox": nox_metrics,
                "disorder_pdb": pdb_metrics,
            },
            "internal_report_path": internal_report,
            "prediction_path": prediction,
            "source_archive_path": source_archive,
            "requirements_path": requirements,
            "expected_source_sha256": sha256(source_archive),
        }
        report = build_consistency_report(**common)
        mismatch_rejected = False
        write_metrics(pdb_metrics, 0.900)
        try:
            build_consistency_report(**common)
        except ValueError:
            mismatch_rejected = True
        result = {
            "status": "pass",
            "official_csv_parsed": report["tracks"]["disorder_nox"][
                "official_aucroc"
            ]
            == 0.852,
            "zero_delta_required": report["tracks"]["disorder_pdb"]["delta"]
            == 0.0,
            "prediction_hash_verified": bool(report["prediction_sha256"]),
            "source_archive_hash_verified": bool(
                report["official_source_archive_sha256"]
            ),
            "mismatch_rejected": mismatch_rejected,
            "caid_labels_used_for_training_or_tuning": report[
                "caid_labels_used_for_training_or_tuning"
            ],
        }
        if (
            not all(
                value
                for key, value in result.items()
                if key
                not in {"status", "caid_labels_used_for_training_or_tuning"}
            )
            or result["caid_labels_used_for_training_or_tuning"]
        ):
            result["status"] = "fail"
            print(json.dumps(result, indent=2))
            raise AssertionError("official CAID3 verification smoke test failed")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
