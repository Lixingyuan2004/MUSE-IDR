"""Train A6: A2 local context plus gated long-range dilated context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a1_frozen_esm2.data import (  # noqa: E402
    ProteinRecord,
    count_known_labels,
    load_manifest,
)
from experiments.ablations.a1_frozen_esm2.train_cached_a1 import (  # noqa: E402
    CacheBundle,
    file_sha256,
    load_verified_cache,
    metrics_from_records,
    save_json,
    save_predictions,
    set_seed,
)
from experiments.ablations.a2_multiscale_context.train_a2 import (  # noqa: E402
    make_loader,
    predict_records,
    train_epoch,
)
from experiments.ablations.a6_long_range_dilated_context.model import (  # noqa: E402
    LongRangeDilatedContextIdrHead,
)


def build_model(
    config: dict[str, Any], cache: CacheBundle
) -> LongRangeDilatedContextIdrHead:
    model_cfg = config["model"]
    configured_input = int(model_cfg["input_size"])
    if configured_input != cache.hidden_size:
        raise ValueError(
            f"configured input size {configured_input} != cache hidden size {cache.hidden_size}"
        )
    return LongRangeDilatedContextIdrHead(
        input_size=configured_input,
        hidden_size=int(model_cfg["hidden_size"]),
        kernels=tuple(int(value) for value in model_cfg["kernels"]),
        dilations=tuple(int(value) for value in model_cfg["dilations"]),
        dropout=float(model_cfg["dropout"]),
        dilated_dropout=float(model_cfg["dilated_dropout"]),
        residual_gate_init=float(model_cfg["residual_gate_init"]),
    )


def run_fold(
    fold: str,
    records: Sequence[ProteinRecord],
    cache: CacheBundle,
    config: dict[str, Any],
    seed: int,
    output_root: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[ProteinRecord], dict[str, np.ndarray]]:
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    set_seed(seed)
    train_records = [record for record in records if record.fold != fold]
    val_records = [record for record in records if record.fold == fold]
    if not train_records or not val_records:
        raise ValueError(f"invalid split for fold {fold}")

    positives, negatives = count_known_labels(train_records)
    if positives == 0 or negatives == 0:
        raise ValueError(f"fold {fold} training data must contain both classes")
    pos_weight = negatives / positives if bool(train_cfg["class_balance"]) else 1.0
    unknown_label = int(eval_cfg["unknown_label"])
    batch_size = int(train_cfg["protein_batch_size"])
    num_workers = int(train_cfg["num_workers"])
    train_loader = make_loader(
        train_records,
        cache,
        batch_size,
        num_workers,
        unknown_label,
        True,
        seed,
    )
    val_loader = make_loader(
        val_records,
        cache,
        batch_size,
        num_workers,
        unknown_label,
        False,
        seed,
    )

    model = build_model(config, cache).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    fold_dir = output_root / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / "best_head.pt"
    history: list[dict[str, Any]] = []
    best_auc = -float("inf")
    epochs_without_improvement = 0
    started = perf_counter()

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pos_weight,
            unknown_label,
            float(train_cfg["gradient_clip_norm"]),
        )
        val_predictions = predict_records(model, val_loader, device)
        val_metrics = metrics_from_records(val_records, val_predictions)
        val_auc = val_metrics["micro_roc_auc"]
        if val_auc is None:
            raise RuntimeError(f"fold {fold} validation ROC-AUC is undefined")
        epoch_row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(epoch_row)
        print(json.dumps({"fold": fold, **epoch_row}, ensure_ascii=False), flush=True)

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "seed": seed,
                    "model_revision": cache.model_revision,
                    "cache_index_sha256": cache.index_sha256,
                    "model_state_dict": model.state_dict(),
                    "best_micro_roc_auc": best_auc,
                    "model_config": config["model"],
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(train_cfg["patience"]):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_predictions = predict_records(model, val_loader, device)
    best_metrics = metrics_from_records(val_records, best_predictions)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    report = {
        "experiment": config["experiment"]["name"],
        "training_mode": "verified_frozen_embedding_cache",
        "context_module": "a2_local_plus_gated_dilated_residual_stack",
        "context_kernels": list(model.kernels),
        "context_dilations": list(model.dilations),
        "dilated_stack_receptive_field": model.dilated_stack_receptive_field,
        "maximum_input_receptive_field": model.maximum_input_receptive_field,
        "output_heads": model.output_heads,
        "output_definition": "classic_idr_probability_per_residue",
        "fold": fold,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "pos_weight": pos_weight,
        "train_proteins": len(train_records),
        "validation_proteins": len(val_records),
        "trainable_parameters": trainable_parameters,
        "metrics": best_metrics,
        "history": history,
        "elapsed_seconds": perf_counter() - started,
        "backbone_forward_passes_during_training": 0,
        "cache_index_sha256": cache.index_sha256,
        "caid2_caid3_labels_accessed": False,
    }
    save_json(fold_dir / "metrics.json", report)
    save_predictions(fold_dir / "predictions.csv", val_records, best_predictions)
    return report, val_records, best_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument("--fold", action="append", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    records = load_manifest(args.manifest)
    cache = load_verified_cache(
        args.cache_dir, records, str(config["model"]["revision"])
    )
    available_folds = sorted({record.fold for record in records})
    selected_folds = args.fold or available_folds
    missing = set(selected_folds) - set(available_folds)
    if missing:
        raise ValueError(f"unknown folds requested: {sorted(missing)}")

    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(config["experiment"]["output_dir"]) / f"seed_{seed}"
    all_records: list[ProteinRecord] = []
    all_predictions: dict[str, np.ndarray] = {}
    fold_reports: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for fold in selected_folds:
        report, fold_records, fold_predictions = run_fold(
            fold, records, cache, config, seed, output_root, device
        )
        fold_reports.append(report)
        for record in fold_records:
            if record.protein_id in all_predictions:
                raise RuntimeError(f"duplicate OOF prediction: {record.protein_id}")
            all_records.append(record)
            all_predictions[record.protein_id] = fold_predictions[record.protein_id]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "schema_version": 1,
        "experiment": config["experiment"]["name"],
        "training_mode": "verified_frozen_embedding_cache",
        "context_module": "a2_local_plus_gated_dilated_residual_stack",
        "context_kernels": [int(value) for value in config["model"]["kernels"]],
        "context_dilations": [
            int(value) for value in config["model"]["dilations"]
        ],
        "dilated_stack_receptive_field": fold_reports[0][
            "dilated_stack_receptive_field"
        ],
        "maximum_input_receptive_field": fold_reports[0][
            "maximum_input_receptive_field"
        ],
        "output_heads": 1,
        "output_definition": "classic_idr_probability_per_residue",
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "folds": selected_folds,
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": file_sha256(args.manifest),
        "cache_index_sha256": cache.index_sha256,
        "model_revision": cache.model_revision,
        "hidden_size": cache.hidden_size,
        "trainable_parameters": fold_reports[0]["trainable_parameters"],
        "fold_metrics": [report["metrics"] for report in fold_reports],
        "oof_metrics": metrics_from_records(all_records, all_predictions),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "backbone_forward_passes_during_training": 0,
        "input_cache_contains_residue_labels": False,
        "caid1_caid2_caid3_used_for_training_or_tuning": False,
        "caid2_caid3_labels_accessed": False,
    }
    save_json(output_root / "oof_metrics.json", summary)
    save_predictions(output_root / "oof_predictions.csv", all_records, all_predictions)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
