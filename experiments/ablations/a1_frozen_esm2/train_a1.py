"""Supported command-line entry point for the A1 ablation.

The training/evaluation functions live in ``run.py``. This small entry point
keeps experiment orchestration explicit and is the command referenced by the
project hand-off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from data import ProteinRecord, load_manifest
from run import metrics_from_records, run_fold, save_json, save_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A1 ablation: frozen ESM2-650M plus one residue-level linear head"
    )
    parser.add_argument("--manifest", required=True, help="Five-fold JSONL manifest")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument(
        "--fold",
        action="append",
        default=None,
        help="Validation fold to run; repeat for multiple folds. Default: all folds.",
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    records = load_manifest(args.manifest)
    available_folds = sorted({record.fold for record in records})
    selected_folds = args.fold or available_folds
    missing = set(selected_folds) - set(available_folds)
    if missing:
        raise ValueError(f"unknown folds requested: {sorted(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = str(config["model"]["pretrained_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    output_root = Path(config["experiment"]["output_dir"])
    all_oof_records: list[ProteinRecord] = []
    all_oof_predictions: dict[str, np.ndarray] = {}
    fold_reports = []

    for fold in selected_folds:
        report, fold_records, fold_predictions = run_fold(
            fold, records, config, tokenizer, device
        )
        fold_reports.append(report)
        for record in fold_records:
            if record.protein_id in all_oof_predictions:
                raise RuntimeError(f"duplicate OOF prediction: {record.protein_id}")
            all_oof_records.append(record)
            all_oof_predictions[record.protein_id] = fold_predictions[record.protein_id]

    oof_metrics = metrics_from_records(all_oof_records, all_oof_predictions)
    summary = {
        "experiment": config["experiment"]["name"],
        "model": model_id,
        "device": str(device),
        "folds": selected_folds,
        "fold_metrics": [report["metrics"] for report in fold_reports],
        "oof_metrics": oof_metrics,
        "caid_labels_accessed": False,
    }
    save_json(output_root / "oof_metrics.json", summary)
    save_predictions(
        output_root / "oof_predictions.csv", all_oof_records, all_oof_predictions
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
