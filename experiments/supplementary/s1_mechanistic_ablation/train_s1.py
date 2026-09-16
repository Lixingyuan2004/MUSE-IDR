"""Train predeclared S1 mechanism controls without changing locked A10."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

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
from experiments.ablations.a2_multiscale_context.model import (  # noqa: E402
    MultiscaleContextIdrHead,
)
from experiments.ablations.a2_multiscale_context.train_a2 import (  # noqa: E402
    make_loader as make_final_layer_loader,
    predict_records,
    train_epoch,
)
from experiments.ablations.a8_multilayer_esm2_fusion.data import (  # noqa: E402
    MultiLayerCacheBundle,
    load_verified_multilayer_cache,
)
from experiments.ablations.a8_multilayer_esm2_fusion.train_a8 import (  # noqa: E402
    make_loader as make_multilayer_loader,
)
from experiments.supplementary.s1_mechanistic_ablation.model import (  # noqa: E402
    ContextFusionAblationIdrHead,
    UniformLayerEsm2FusionIdrModel,
)

Cache = CacheBundle | MultiLayerCacheBundle
LoaderFactory = Callable[
    [Sequence[ProteinRecord], Cache, int, int, int, bool, int], DataLoader
]


def build_model(
    config: dict[str, Any], variant: str, cache: Cache
) -> tuple[nn.Module, dict[str, Any]]:
    model_cfg = config["model"]
    try:
        variant_cfg = dict(config["variants"][variant])
    except KeyError as error:
        choices = sorted(config.get("variants", {}))
        raise ValueError(f"unknown S1 variant {variant!r}; expected one of {choices}") from error

    input_size = int(model_cfg["input_size"])
    hidden_size = int(model_cfg["hidden_size"])
    dropout = float(model_cfg["dropout"])
    kernels = tuple(int(value) for value in variant_cfg["kernels"])
    if input_size != cache.hidden_size:
        raise ValueError(
            f"configured input size {input_size} != cache hidden size {cache.hidden_size}"
        )

    architecture = str(variant_cfg["architecture"])
    if architecture == "gated_context":
        model: nn.Module = MultiscaleContextIdrHead(
            input_size=input_size,
            hidden_size=hidden_size,
            kernels=kernels,
            dropout=dropout,
        )
    elif architecture in ContextFusionAblationIdrHead.ALLOWED_FUSION_MODES:
        model = ContextFusionAblationIdrHead(
            input_size=input_size,
            hidden_size=hidden_size,
            kernels=kernels,
            dropout=dropout,
            fusion_mode=architecture,
        )
    elif architecture == "uniform_layer_mix":
        if not isinstance(cache, MultiLayerCacheBundle):
            raise TypeError("uniform layer fusion requires the verified multilayer cache")
        layer_indices = tuple(int(value) for value in model_cfg["layer_indices"])
        if layer_indices != cache.layer_indices:
            raise ValueError(
                f"configured layers {layer_indices} != cache layers {cache.layer_indices}"
            )
        model = UniformLayerEsm2FusionIdrModel(
            input_size=input_size,
            hidden_size=hidden_size,
            layer_indices=layer_indices,
            kernels=kernels,
            dropout=dropout,
        )
    else:
        raise ValueError(f"unsupported S1 architecture: {architecture}")

    metadata = {
        "variant": variant,
        "cache_kind": str(variant_cfg["cache_kind"]),
        "architecture": architecture,
        "kernels": list(kernels),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "dropout": dropout,
    }
    if isinstance(model, UniformLayerEsm2FusionIdrModel):
        metadata.update(
            {
                "layer_indices": list(model.layer_indices),
                "fixed_layer_weights": [
                    float(value) for value in model.layer_weights.tolist()
                ],
                "layer_weights_trainable": False,
            }
        )
    return model, metadata


def load_cache_and_loader(
    args: argparse.Namespace,
    config: dict[str, Any],
    records: Sequence[ProteinRecord],
    variant: str,
) -> tuple[Cache, LoaderFactory]:
    variant_cfg = config["variants"][variant]
    cache_kind = str(variant_cfg["cache_kind"])
    revision = str(config["model"]["revision"])
    if cache_kind == "final_layer":
        if args.final_layer_cache is None:
            raise ValueError(f"--final-layer-cache is required for {variant}")
        cache = load_verified_cache(args.final_layer_cache, records, revision)
        return cache, make_final_layer_loader  # type: ignore[return-value]
    if cache_kind == "multilayer":
        if args.multilayer_cache is None:
            raise ValueError(f"--multilayer-cache is required for {variant}")
        layers = tuple(int(value) for value in config["model"]["layer_indices"])
        cache = load_verified_multilayer_cache(
            args.multilayer_cache,
            records,
            expected_revision=revision,
            expected_layers=layers,
        )
        return cache, make_multilayer_loader  # type: ignore[return-value]
    raise ValueError(f"unsupported cache kind for {variant}: {cache_kind}")


def run_fold(
    fold: str,
    records: Sequence[ProteinRecord],
    cache: Cache,
    loader_factory: LoaderFactory,
    config: dict[str, Any],
    variant: str,
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
    loader_args = (
        int(train_cfg["protein_batch_size"]),
        int(train_cfg["num_workers"]),
        unknown_label,
    )
    train_loader = loader_factory(
        train_records, cache, *loader_args, True, seed
    )
    val_loader = loader_factory(
        val_records, cache, *loader_args, False, seed
    )

    model, model_metadata = build_model(config, variant, cache)
    model = model.to(device)
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
            model,  # type: ignore[arg-type]
            train_loader,
            optimizer,
            device,
            pos_weight,
            unknown_label,
            float(train_cfg["gradient_clip_norm"]),
        )
        val_predictions = predict_records(
            model, val_loader, device  # type: ignore[arg-type]
        )
        val_metrics = metrics_from_records(val_records, val_predictions)
        val_auc = val_metrics["micro_roc_auc"]
        if val_auc is None:
            raise RuntimeError(f"fold {fold} validation ROC-AUC is undefined")
        epoch_row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(epoch_row)
        print(
            json.dumps(
                {"variant": variant, "fold": fold, **epoch_row},
                ensure_ascii=False,
            ),
            flush=True,
        )

        if val_auc > best_auc:
            best_auc = float(val_auc)
            epochs_without_improvement = 0
            torch.save(
                {
                    "experiment": config["experiment"]["name"],
                    "purpose": config["experiment"]["purpose"],
                    "variant": variant,
                    "fold": fold,
                    "epoch": epoch,
                    "seed": seed,
                    "model_revision": cache.model_revision,
                    "cache_index_sha256": cache.index_sha256,
                    "model_state_dict": model.state_dict(),
                    "best_micro_roc_auc": best_auc,
                    "model_config": model_metadata,
                    "caid_labels_accessed": False,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(train_cfg["patience"]):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    best_predictions = predict_records(
        model, val_loader, device  # type: ignore[arg-type]
    )
    best_metrics = metrics_from_records(val_records, best_predictions)
    report = {
        "schema_version": 1,
        "experiment": config["experiment"]["name"],
        "purpose": config["experiment"]["purpose"],
        **model_metadata,
        "training_mode": "verified_frozen_embedding_cache",
        "fold": fold,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "pos_weight": pos_weight,
        "train_proteins": len(train_records),
        "validation_proteins": len(val_records),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "metrics": best_metrics,
        "history": history,
        "elapsed_seconds": perf_counter() - started,
        "output_heads": int(getattr(model, "output_heads")),
        "output_definition": "classic_idr_probability_per_residue",
        "backbone_forward_passes_during_training": 0,
        "cache_index_sha256": cache.index_sha256,
        "caid1_caid2_caid3_used_for_training_tuning_or_reselection": False,
        "locked_a10_modified": False,
    }
    save_json(fold_dir / "metrics.json", report)
    save_predictions(fold_dir / "predictions.csv", val_records, best_predictions)
    return report, val_records, best_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--final-layer-cache", type=Path)
    parser.add_argument("--multilayer-cache", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fold", action="append", default=None)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    if args.variant not in config["variants"]:
        raise ValueError(
            f"unknown S1 variant {args.variant!r}; "
            f"expected one of {sorted(config['variants'])}"
        )
    formal_seeds = tuple(int(value) for value in config["experiment"]["formal_seeds"])
    if args.seed not in formal_seeds:
        raise ValueError(f"seed {args.seed} is not predeclared in {formal_seeds}")

    records = load_manifest(args.manifest)
    cache, loader_factory = load_cache_and_loader(
        args, config, records, args.variant
    )
    available_folds = sorted({record.fold for record in records})
    selected_folds = args.fold or available_folds
    missing = set(selected_folds) - set(available_folds)
    if missing:
        raise ValueError(f"unknown folds requested: {sorted(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configured_root = Path(config["experiment"]["output_dir"])
    output_root = (
        args.output_root if args.output_root is not None else configured_root
    ) / args.variant / f"seed_{args.seed}"
    all_records: list[ProteinRecord] = []
    all_predictions: dict[str, np.ndarray] = {}
    fold_reports: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for fold in selected_folds:
        report, fold_records, fold_predictions = run_fold(
            fold,
            records,
            cache,
            loader_factory,
            config,
            args.variant,
            args.seed,
            output_root,
            device,
        )
        fold_reports.append(report)
        for record in fold_records:
            if record.protein_id in all_predictions:
                raise RuntimeError(f"duplicate OOF prediction: {record.protein_id}")
            all_records.append(record)
            all_predictions[record.protein_id] = fold_predictions[record.protein_id]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_metrics = metrics_from_records(all_records, all_predictions)
    complete_oof = selected_folds == available_folds
    _, model_metadata = build_model(config, args.variant, cache)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "experiment": config["experiment"]["name"],
        "purpose": config["experiment"]["purpose"],
        **model_metadata,
        "seed": args.seed,
        "formal_seeds": list(formal_seeds),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "selected_folds": selected_folds,
        "available_folds": available_folds,
        "complete_oof": complete_oof,
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": file_sha256(args.manifest),
        "config_sha256": file_sha256(args.config),
        "cache_index_sha256": cache.index_sha256,
        "model_revision": cache.model_revision,
        "fold_metrics": [report["metrics"] for report in fold_reports],
        "selected_fold_metrics": selected_metrics,
        "oof_metrics": selected_metrics if complete_oof else None,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "backbone_forward_passes_during_training": 0,
        "input_cache_contains_residue_labels": False,
        "caid1_caid2_caid3_used_for_training_tuning_or_reselection": False,
        "locked_a10_modified": False,
    }
    save_json(output_root / "oof_metrics.json", summary)
    save_predictions(
        output_root / "oof_predictions.csv", all_records, all_predictions
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
