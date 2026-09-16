from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import (
    EsmWindowCollator,
    ProteinRecord,
    ProteinWindowDataset,
    count_known_labels,
    load_manifest,
)
from model import FrozenEsmResidueClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def metrics_from_records(
    records: Sequence[ProteinRecord], predictions: dict[str, np.ndarray]
) -> dict[str, Any]:
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    per_protein_auc: list[float] = []

    for record in records:
        score = predictions[record.protein_id]
        if score.shape != record.labels.shape:
            raise ValueError(f"prediction shape mismatch for {record.protein_id}")
        known = record.labels >= 0
        labels = record.labels[known]
        scores = score[known]
        pooled_labels.append(labels)
        pooled_scores.append(scores)
        auc = safe_auc(labels, scores)
        if auc is not None:
            per_protein_auc.append(auc)

    y_true = np.concatenate(pooled_labels)
    y_score = np.concatenate(pooled_scores)
    return {
        "micro_roc_auc": safe_auc(y_true, y_score),
        "micro_pr_auc": float(average_precision_score(y_true, y_score)),
        "macro_roc_auc": float(np.mean(per_protein_auc)) if per_protein_auc else None,
        "known_residues": int(y_true.size),
        "positive_residues": int(np.sum(y_true == 1)),
        "negative_residues": int(np.sum(y_true == 0)),
        "proteins": len(records),
        "macro_auc_proteins": len(per_protein_auc),
    }


def train_epoch(
    model: FrozenEsmResidueClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: float,
    unknown_label: int,
) -> float:
    model.train()
    total_weighted_loss = 0.0
    total_weight = 0.0
    positive_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss_weight = batch["loss_weight"].to(device)

        logits = model(input_ids, attention_mask)
        known = labels != float(unknown_label)
        if not torch.any(known):
            continue

        per_token_loss = F.binary_cross_entropy_with_logits(
            logits[known],
            labels[known],
            pos_weight=positive_weight,
            reduction="none",
        )
        weights = loss_weight[known]
        loss = torch.sum(per_token_loss * weights) / torch.sum(weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_weight = float(torch.sum(weights).detach().cpu())
        total_weighted_loss += float(loss.detach().cpu()) * batch_weight
        total_weight += batch_weight

    if total_weight == 0:
        raise RuntimeError("training fold contains no known labels")
    return total_weighted_loss / total_weight


@torch.inference_mode()
def predict_records(
    model: FrozenEsmResidueClassifier,
    records: Sequence[ProteinRecord],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    logit_sums = {
        record.protein_id: np.zeros(len(record.sequence), dtype=np.float64)
        for record in records
    }
    weight_sums = {
        record.protein_id: np.zeros(len(record.sequence), dtype=np.float64)
        for record in records
    }

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        residue_mask = batch["residue_mask"]
        logits = model(input_ids, attention_mask).cpu()

        for batch_index, metadata in enumerate(batch["metadata"]):
            residue_logits = logits[batch_index][residue_mask[batch_index]].numpy()
            start = int(metadata["start"])
            end = int(metadata["end"])
            expected = end - start
            if residue_logits.size != expected:
                raise RuntimeError(
                    f"output alignment failure for {metadata['protein_id']}: "
                    f"{residue_logits.size} != {expected}"
                )
            merge_weight = metadata["merge_weight"].astype(np.float64, copy=False)
            protein_id = metadata["protein_id"]
            logit_sums[protein_id][start:end] += residue_logits * merge_weight
            weight_sums[protein_id][start:end] += merge_weight

    predictions: dict[str, np.ndarray] = {}
    for record in records:
        denominator = weight_sums[record.protein_id]
        if np.any(denominator <= 0):
            raise RuntimeError(f"uncovered prediction residues for {record.protein_id}")
        merged_logits = logit_sums[record.protein_id] / denominator
        predictions[record.protein_id] = (
            1.0 / (1.0 + np.exp(-merged_logits))
        ).astype(np.float32)
    return predictions


def save_predictions(
    path: Path,
    records: Sequence[ProteinRecord],
    predictions: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "position", "label", "probability"])
        for record in records:
            score = predictions[record.protein_id]
            for position, (label, probability) in enumerate(
                zip(record.labels, score), start=1
            ):
                writer.writerow(
                    [record.protein_id, position, int(label), f"{float(probability):.9g}"]
                )


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def make_loader(
    records: Sequence[ProteinRecord],
    tokenizer,
    max_residues: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    unknown_label: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ProteinWindowDataset(records, max_residues=max_residues, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=EsmWindowCollator(tokenizer, unknown_label=unknown_label),
        pin_memory=torch.cuda.is_available(),
    )


def run_fold(
    fold: str,
    records: Sequence[ProteinRecord],
    config: dict[str, Any],
    tokenizer,
    device: torch.device,
) -> tuple[dict[str, Any], list[ProteinRecord], dict[str, np.ndarray]]:
    experiment_cfg = config["experiment"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    seed = int(experiment_cfg["seed"])
    set_seed(seed)

    train_records = [record for record in records if record.fold != fold]
    val_records = [record for record in records if record.fold == fold]
    if not train_records or not val_records:
        raise ValueError(f"invalid split for fold {fold}")

    train_loader = make_loader(
        train_records,
        tokenizer,
        int(model_cfg["max_residues"]),
        int(model_cfg["window_stride"]),
        int(train_cfg["batch_size"]),
        int(train_cfg["num_workers"]),
        int(eval_cfg["unknown_label"]),
        shuffle=True,
    )
    val_loader = make_loader(
        val_records,
        tokenizer,
        int(model_cfg["max_residues"]),
        int(model_cfg["window_stride"]),
        int(train_cfg["batch_size"]),
        int(train_cfg["num_workers"]),
        int(eval_cfg["unknown_label"]),
        shuffle=False,
    )

    positives, negatives = count_known_labels(train_records)
    if positives == 0 or negatives == 0:
        raise ValueError(f"fold {fold} training data must contain both classes")
    pos_weight = negatives / positives if bool(train_cfg["class_balance"]) else 1.0

    model = FrozenEsmResidueClassifier(
        pretrained_model=str(model_cfg["pretrained_model"]),
        dropout=float(model_cfg["dropout"]),
        precision=str(model_cfg["precision"]),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    output_root = Path(experiment_cfg["output_dir"])
    fold_dir = output_root / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / "best_head.pt"
    history: list[dict[str, Any]] = []
    best_auc = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pos_weight,
            int(eval_cfg["unknown_label"]),
        )
        val_predictions = predict_records(model, val_records, val_loader, device)
        val_metrics = metrics_from_records(val_records, val_predictions)
        val_auc = val_metrics["micro_roc_auc"]
        if val_auc is None:
            raise RuntimeError(f"fold {fold} validation ROC-AUC is undefined")

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(epoch_row)
        print(json.dumps({"fold": fold, **epoch_row}, ensure_ascii=False))

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "seed": seed,
                    "model_id": model_cfg["pretrained_model"],
                    "classifier_state_dict": model.classifier.state_dict(),
                    "best_micro_roc_auc": best_auc,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(train_cfg["patience"]):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    best_predictions = predict_records(model, val_records, val_loader, device)
    best_metrics = metrics_from_records(val_records, best_predictions)
    fold_report = {
        "experiment": experiment_cfg["name"],
        "fold": fold,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "pos_weight": pos_weight,
        "train_proteins": len(train_records),
        "validation_proteins": len(val_records),
        "metrics": best_metrics,
        "history": history,
    }
    save_json(fold_dir / "metrics.json", fold_report)
    save_predictions(fold_dir / "predictions.csv", val_records, best_predictions)
    return fold_report, val_records, best_predictions


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
    fold_reports: list[dict[str, Any]] = []

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
        "caid_labels_accessed": false,
    }
    save_json(output_root / "oof_metrics.json", summary)
    save_predictions(
        output_root / "oof_predictions.csv", all_oof_records, all_oof_predictions
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
