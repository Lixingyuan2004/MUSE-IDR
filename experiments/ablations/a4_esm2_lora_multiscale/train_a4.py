from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a4_esm2_lora_multiscale.data import (
    UNKNOWN_LABEL,
    EsmWindowCollator,
    ProteinRecord,
    ProteinWindowDataset,
    label_counts,
    load_manifest,
)
from experiments.ablations.a4_esm2_lora_multiscale.model import (
    Esm2LoraMultiscaleIdrModel,
)


ModelBuilder = Callable[[Mapping[str, Any], torch.device], Esm2LoraMultiscaleIdrModel]


@dataclass
class Metrics:
    micro_roc_auc: float
    micro_pr_auc: float
    macro_roc_auc: float
    known_residues: int
    positive_residues: int
    negative_residues: int
    proteins: int
    macro_auc_proteins: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def _finite_metric(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def compute_metrics(predictions: Sequence[dict[str, Any]]) -> Metrics:
    known_labels: list[np.ndarray] = []
    known_scores: list[np.ndarray] = []
    macro_auc: list[float] = []
    positive = 0
    negative = 0
    for prediction in predictions:
        labels = np.asarray(prediction["labels"], dtype=np.int64)
        scores = np.asarray(prediction["scores"], dtype=np.float64)
        known = np.isin(labels, [0, 1])
        y_true = labels[known]
        y_score = scores[known]
        positive += int(np.sum(y_true == 1))
        negative += int(np.sum(y_true == 0))
        if y_true.size:
            known_labels.append(y_true)
            known_scores.append(y_score)
        if np.unique(y_true).size == 2:
            macro_auc.append(float(roc_auc_score(y_true, y_score)))
    if positive == 0 or negative == 0:
        raise ValueError("Micro ROC-AUC requires both positive and negative known residues")
    labels = np.concatenate(known_labels)
    scores = np.concatenate(known_scores)
    return Metrics(
        micro_roc_auc=_finite_metric(roc_auc_score(labels, scores)),
        micro_pr_auc=_finite_metric(average_precision_score(labels, scores)),
        macro_roc_auc=_finite_metric(np.mean(macro_auc) if macro_auc else float("nan")),
        known_residues=int(labels.size),
        positive_residues=positive,
        negative_residues=negative,
        proteins=len(predictions),
        macro_auc_proteins=len(macro_auc),
    )


def make_loader(
    records: Sequence[ProteinRecord],
    tokenizer: Any,
    config: Mapping[str, Any],
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    data_cfg = config["data"]
    dataset = ProteinWindowDataset(
        records,
        max_residues=int(data_cfg["max_residues"]),
        stride=int(data_cfg["stride"]),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=bool(torch.cuda.is_available()),
        collate_fn=EsmWindowCollator(tokenizer),
        generator=generator,
        drop_last=False,
    )


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key in (
        "input_ids",
        "attention_mask",
        "residue_mask",
        "labels",
        "loss_weight",
        "residue_position",
        "record_index",
        "start",
    ):
        result[key] = batch[key].to(device, non_blocking=True)
    return result


def weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    residue_mask: torch.Tensor,
    loss_weight: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    known = residue_mask.bool() & ((labels == 0) | (labels == 1))
    known_count = int(known.sum().item())
    if known_count == 0:
        return logits.sum() * 0.0, 0
    raw = F.binary_cross_entropy_with_logits(
        logits,
        labels.clamp_min(0).float(),
        reduction="none",
        pos_weight=pos_weight,
    )
    weights = loss_weight * known.float()
    return (raw * weights).sum() / weights.sum().clamp_min(1e-12), known_count


def _lr_schedule(total_steps: int, warmup_ratio: float):
    warmup_steps = int(round(total_steps * warmup_ratio))

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / remaining, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule


def evaluate(
    model: Esm2LoraMultiscaleIdrModel,
    loader: DataLoader,
    records: Sequence[ProteinRecord],
    device: torch.device,
) -> tuple[Metrics, list[dict[str, Any]], int]:
    model.eval()
    logit_sum = [np.zeros(len(record.sequence), dtype=np.float64) for record in records]
    logit_count = [np.zeros(len(record.sequence), dtype=np.int64) for record in records]
    forward_passes = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            logits = model(
                batch["input_ids"], batch["attention_mask"], batch["residue_mask"]
            ).float()
            forward_passes += 1
            for row in range(logits.shape[0]):
                record_index = int(batch["record_index"][row].item())
                start = int(batch["start"][row].item())
                mask = batch["residue_mask"][row]
                local_positions = batch["residue_position"][row, mask].detach().cpu().numpy()
                values = logits[row, mask].detach().cpu().numpy().astype(np.float64)
                global_positions = start + local_positions
                logit_sum[record_index][global_positions] += values
                logit_count[record_index][global_positions] += 1

    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if np.any(logit_count[index] == 0):
            raise RuntimeError(f"Evaluation left unpredicted residues in {record.protein_id}")
        mean_logit = logit_sum[index] / logit_count[index]
        scores = 1.0 / (1.0 + np.exp(-np.clip(mean_logit, -60.0, 60.0)))
        predictions.append(
            {
                "protein_id": record.protein_id,
                "fold": record.fold,
                "labels": record.labels.copy(),
                "scores": scores.astype(np.float32),
            }
        )
    return compute_metrics(predictions), predictions, forward_passes


def write_predictions(path: str | Path, predictions: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            row = {
                "protein_id": prediction["protein_id"],
                "fold": prediction["fold"],
                "scores": [round(float(value), 8) for value in prediction["scores"]],
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def build_model(
    config: Mapping[str, Any], device: torch.device
) -> Esm2LoraMultiscaleIdrModel:
    return Esm2LoraMultiscaleIdrModel.from_pretrained_config(config, device)


def run_fold(
    *,
    records: Sequence[ProteinRecord],
    tokenizer: Any,
    config: Mapping[str, Any],
    fold: str,
    seed: int,
    output_dir: str | Path,
    device: torch.device,
    model_builder: ModelBuilder = build_model,
) -> dict[str, Any]:
    fold = str(fold)
    # Match A1/A2 exactly: the declared experiment seed is reused for every fold.
    fold_seed = int(seed)
    set_seed(fold_seed)
    train_records = [record for record in records if record.fold != fold]
    validation_records = [record for record in records if record.fold == fold]
    if not train_records or not validation_records:
        raise ValueError(f"Fold {fold} has an empty train or validation split")

    train_loader = make_loader(
        train_records, tokenizer, config, shuffle=True, seed=fold_seed
    )
    validation_loader = make_loader(
        validation_records, tokenizer, config, shuffle=False, seed=fold_seed
    )
    model = model_builder(config, device)
    model.assert_trainable_contract()
    report = model.parameter_report()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    positive, negative = label_counts(train_records)
    if positive == 0 or negative == 0:
        raise ValueError("Training split requires positive and negative known residues")
    pos_weight = torch.tensor(negative / positive, dtype=torch.float32, device=device)

    training_cfg = config["training"]
    parameter_groups: list[dict[str, Any]] = []
    lora_parameters = [parameter for _, parameter in model.lora_named_parameters()]
    if lora_parameters:
        parameter_groups.append(
            {"params": lora_parameters, "lr": float(training_cfg["lora_learning_rate"])}
        )
    parameter_groups.append(
        {
            "params": [parameter for _, parameter in model.head_named_parameters()],
            "lr": float(training_cfg["head_learning_rate"]),
        }
    )
    optimizer = AdamW(
        parameter_groups,
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(training_cfg["epochs"])
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = max(updates_per_epoch * epochs, 1)
    scheduler = LambdaLR(
        optimizer,
        _lr_schedule(total_updates, float(training_cfg.get("warmup_ratio", 0.0))),
    )

    fold_dir = Path(output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / "best_compact_checkpoint.pt"
    patience = int(training_cfg.get("patience", epochs))
    best_auc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    training_forward_passes = 0
    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_numerator = 0.0
        epoch_known = 0
        current_group_size = accumulation
        for batch_index, raw_batch in enumerate(train_loader):
            if batch_index % accumulation == 0:
                current_group_size = min(accumulation, len(train_loader) - batch_index)
            batch = _to_device(raw_batch, device)
            logits = model(
                batch["input_ids"], batch["attention_mask"], batch["residue_mask"]
            )
            training_forward_passes += 1
            loss, known = weighted_bce_loss(
                logits,
                batch["labels"],
                batch["residue_mask"],
                batch["loss_weight"],
                pos_weight,
            )
            (loss / current_group_size).backward()
            epoch_loss_numerator += float(loss.detach().item()) * known
            epoch_known += known
            is_update = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if is_update:
                clip_grad_norm_(
                    [parameter for group in parameter_groups for parameter in group["params"]],
                    float(training_cfg.get("max_grad_norm", 1.0)),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation_metrics, _, _ = evaluate(
            model, validation_loader, validation_records, device
        )
        epoch_row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": epoch_loss_numerator / max(epoch_known, 1),
            **asdict(validation_metrics),
        }
        print(json.dumps(epoch_row), flush=True)
        if validation_metrics.micro_roc_auc > best_auc:
            best_auc = validation_metrics.micro_roc_auc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": 1,
                    "fold": fold,
                    "seed": seed,
                    "epoch": epoch,
                    "selection_metric": "micro_roc_auc",
                    "selection_value": best_auc,
                    "model": model.compact_state_dict(),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_compact_state_dict(checkpoint["model"])
    final_metrics, predictions, evaluation_forward_passes = evaluate(
        model, validation_loader, validation_records, device
    )
    write_predictions(fold_dir / "predictions.jsonl.gz", predictions)
    elapsed_seconds = time.perf_counter() - start_time
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    fold_report = {
        "schema_version": 1,
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "lora_enabled": model.lora_enabled,
        **report,
        **asdict(final_metrics),
        "train_proteins": len(train_records),
        "validation_proteins": len(validation_records),
        "training_backbone_forward_passes": training_forward_passes,
        "final_evaluation_backbone_forward_passes": evaluation_forward_passes,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": peak_memory,
        "output_heads": model.output_heads,
        "caid2_caid3_labels_accessed": False,
    }
    with (fold_dir / "fold_report.json").open("w", encoding="utf-8") as handle:
        json.dump(fold_report, handle, indent=2)
    return {
        "metrics": final_metrics,
        "predictions": predictions,
        "report": fold_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", action="append", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    records = load_manifest(args.manifest)
    available_folds = sorted({record.fold for record in records})
    folds = [str(fold) for fold in args.fold] if args.fold else available_folds
    invalid = sorted(set(folds) - set(available_folds))
    if invalid:
        raise ValueError(f"Unknown folds: {invalid}; available folds: {available_folds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["name"]), revision=str(model_cfg["revision"]), use_fast=True
    )
    variant = str(config.get("variant", "lora"))
    output_dir = Path(
        args.output_dir
        or Path("outputs/ablations/a4_esm2_lora_multiscale")
        / variant
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    oof_predictions: list[dict[str, Any]] = []
    for fold in folds:
        result = run_fold(
            records=records,
            tokenizer=tokenizer,
            config=config,
            fold=fold,
            seed=args.seed,
            output_dir=output_dir,
            device=device,
        )
        fold_results.append(result["report"])
        oof_predictions.extend(result["predictions"])
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_metrics = compute_metrics(oof_predictions)
    summary = {
        "schema_version": 1,
        "experiment": "a4_esm2_lora_multiscale",
        "variant": variant,
        "training_mode": "online_esm2_lora" if config["lora"]["enabled"] else "online_frozen_esm2_control",
        "single_output_head": True,
        "output_definition": "classic_idr_probability_per_residue",
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "folds": folds,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "model_name": str(model_cfg["name"]),
        "model_revision": str(model_cfg["revision"]),
        "fold_metrics": [
            {
                key: row[key]
                for key in (
                    "fold",
                    "micro_roc_auc",
                    "micro_pr_auc",
                    "macro_roc_auc",
                    "known_residues",
                    "positive_residues",
                    "negative_residues",
                    "proteins",
                    "macro_auc_proteins",
                )
            }
            for row in fold_results
        ],
        "oof_metrics": asdict(oof_metrics),
        "fold_reports": fold_results,
        "caid1_caid2_caid3_used_for_training_or_tuning": False,
        "caid2_caid3_labels_accessed": False,
    }
    with (output_dir / "oof_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
