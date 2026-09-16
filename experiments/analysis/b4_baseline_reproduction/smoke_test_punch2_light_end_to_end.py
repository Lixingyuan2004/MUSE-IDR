from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from punch2_light_common import (
    DEFAULT_PUNCH2_LIGHT_REPO,
    author_onehot,
    sha256,
)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    predictor = script_dir / "predict_punch2_light_official.py"
    sequences = {
        "smoke_a": "ACDUZOBJX",
        "smoke_b": "MSTNPKPQRKTKRNTNRRPQ",
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fasta = root / "sequences.fasta"
        fasta.write_text(
            "".join(f">{protein_id}\n{sequence}\n" for protein_id, sequence in sequences.items()),
            encoding="utf-8",
        )
        feature_dir = root / "features"
        (feature_dir / "onehot").mkdir(parents=True)
        (feature_dir / "protTrans").mkdir(parents=True)
        generator = np.random.default_rng(20260826)
        for protein_id, sequence in sequences.items():
            np.save(
                feature_dir / "onehot" / f"{protein_id}.npy",
                author_onehot(sequence),
                allow_pickle=False,
            )
            np.save(
                feature_dir / "protTrans" / f"{protein_id}.npy",
                generator.normal(0, 0.1, size=(1, len(sequence), 1024)).astype(np.float32),
                allow_pickle=False,
            )
        feature_report = root / "feature_report.json"
        feature_report.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "input_fasta_sha256": sha256(fasta),
                    "input_contains_labels": False,
                    "caid_labels_accessed": False,
                }
            ),
            encoding="utf-8",
        )
        output_dir = root / "outputs"
        report_path = root / "inference_report.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(predictor),
                "--fasta",
                str(fasta),
                "--feature-dir",
                str(feature_dir),
                "--feature-report",
                str(feature_report),
                "--repo",
                str(DEFAULT_PUNCH2_LIGHT_REPO),
                "--variants",
                "paper8",
                "released13",
                "--output-dir",
                str(output_dir),
                "--report",
                str(report_path),
                "--device",
                "cpu",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        total_residues = sum(map(len, sequences.values()))
        if report["status"] != "pass" or not report["prediction_alignment"]:
            raise AssertionError("end-to-end inference report did not pass")
        if report["released_members_loaded"] != 13:
            raise AssertionError("released member count mismatch")
        expected_members = {"paper8": 8, "released13": 13}
        for variant, members in expected_members.items():
            variant_report = report["variants"][variant]
            if variant_report["ensemble_members"] != members:
                raise AssertionError(f"{variant}: ensemble member count mismatch")
            if variant_report["output_scores"] != total_residues:
                raise AssertionError(f"{variant}: output score count mismatch")
            tsv_path = Path(variant_report["output"])
            caid_path = Path(variant_report["caid_output"])
            if sha256(tsv_path) != variant_report["output_sha256"]:
                raise AssertionError(f"{variant}: TSV hash mismatch")
            if sha256(caid_path) != variant_report["caid_output_sha256"]:
                raise AssertionError(f"{variant}: CAID hash mismatch")
            with gzip.open(tsv_path, "rt", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            if len(rows) != total_residues:
                raise AssertionError(f"{variant}: TSV alignment mismatch")
            probabilities = [float(row["idr_probability"]) for row in rows]
            if not all(0 <= probability <= 1 for probability in probabilities):
                raise AssertionError(f"{variant}: invalid probabilities")
            caid_lines = caid_path.read_text(encoding="utf-8").splitlines()
            if sum(not line.startswith(">") for line in caid_lines) != total_residues:
                raise AssertionError(f"{variant}: CAID alignment mismatch")

    print(
        json.dumps(
            {
                "status": "pass",
                "sequence_only_fasta": True,
                "feature_cache_join": True,
                "official_thirteen_checkpoints_loaded": True,
                "paper8_output_written": True,
                "released13_output_written": True,
                "prediction_alignment": True,
                "one_score_per_residue": True,
                "output_hashes_verified": True,
                "soft_disorder_output": False,
                "caid_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
