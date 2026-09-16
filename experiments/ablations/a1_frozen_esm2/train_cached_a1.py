"""Train the A1 single-output IDR head from a verified frozen ESM2 cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:  # Package import during tests.
    from .data import ProteinRecord, count_known_labels, load_manifest
except ImportError:  # Direct script execution on AutoDL.
    from data import ProteinRecord, count_known_labels, load_manifest


@dataclass(frozen=True)
class CacheBundle:
    cache_dir: Path
    entries: dict[str, dict[str, Any]]
    hidden_size: int
    model_revision: str
    index_sha256: str
    verification: dict[str, Any]


class CachedIdrHead(nn.Module):
    """One scalar classic-IDR logit per cached residue representation."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features)).squeeze(-1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _embedding_path(cache_dir: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe cache embedding path: {relative_path}")
    return cache_dir / relative


def load_verified_cache(
    cache_dir: Path,
    records: Sequence[ProteinRecord],
    expected_revision: str | None = None,
) -> CacheBundle:
    """Join a verified label-free cache to the labeled five-fold manifest."""

    verification_path = cache_dir / "cache_verification_report.json"
    index_path = cache_dir / "cache_index.jsonl"
    if not verification_path.is_file():
        raise FileNotFoundError(
            f"missing independent cache verification report: {verification_path}"
        )
    if not index_path.is_file():
        raise FileNotFoundError(f"missing cache index: {index_path}")

    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    required_checks = {
        "status": "pass",
        "all_finite": True,
        "all_file_hashes_match": True,
        "input_contains_residue_labels": False,
        "caid2_caid3_labels_accessed": False,
    }
    for key, expected in required_checks.items():
        if verification.get(key) != expected:
            raise ValueError(
                f"cache verification failed requirement {key}: "
                f"{verification.get(key)!r} != {expected!r}"
            )

    observed_index_sha256 = file_sha256(index_path)
    if verification.get("cache_index_sha256") != observed_index_sha256:
        raise ValueError("cache index changed after independent verification")

    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    entries = {str(row["protein_id"]): row for row in rows}
    if len(entries) != len(rows):
        raise ValueError("cache index contains duplicate protein IDs")

    manifest_ids = {record.protein_id for record in records}
    if set(entries) != manifest_ids:
        missing = sorted(manifest_ids - set(entries))
        extra = sorted(set(entries) - manifest_ids)
        raise ValueError(
            f"labeled manifest/cache ID mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    hidden_sizes: set[int] = set()
    revisions: set[str] = set()
    for record in records:
        entry = entries[record.protein_id]
        if entry.get("sequence_sha256") != sequence_sha256(record.sequence):
            raise ValueError(f"sequence/cache mismatch for {record.protein_id}")
        if int(entry.get("sequence_length", -1)) != len(record.sequence):
            raise ValueError(f"sequence length/cache mismatch for {record.protein_id}")
        if str(entry.get("fold")) != record.fold:
            raise ValueError(f"fold/cache mismatch for {record.protein_id}")

        hidden_size = int(entry["hidden_size"])
        embedding = np.load(
            _embedding_path(cache_dir, str(entry["embedding_file"])),
            mmap_mode="r",
            allow_pickle=False,
        )
        if embedding.shape != (len(record.sequence), hidden_size):
            raise ValueError(f"embedding shape/cache mismatch for {record.protein_id}")
        if embedding.dtype != np.float16:
            raise ValueError(f"embedding dtype must be float16 for {record.protein_id}")
        hidden_sizes.add(hidden_size)
        revisions.add(str(entry["model_revision"]))

    if len(hidden_sizes) != 1 or len(revisions) != 1:
        raise ValueError("cache mixes hidden sizes or model revisions")
    model_revision = next(iter(revisions))
    if expected_revision is not None and model_revision != expected_revision:
        raise ValueError(
            f"cache model revision {model_revision} != configured revision {expected_revision}"
        )

    return CacheBundle(
        cache_dir=cache_dir,
        entries=entries,
        hidden_size=next(iter(hidden_sizes)),
        model_revision=model_revision,
        index_sha256=observed_index_sha256,
        verification=verification,
    )


def load_known_residue_tensors(
    records: Sequence[ProteinRecord], cache: CacheBundle
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize only labeled training residues; unknown labels never enter loss."""

    known_count = sum(int(np.sum(record.labels >= 0)) for record in records)
    if known_count == 0:
        raise ValueError("training split contains no known residue labels")
    features = np.empty((known_count, cache.hidden_size), dtype=np.float16)
    labels = np.empty(known_count, dtype=np.float32)
    cursor = 0
    for record in records:
        known = record.labels >= 0
        count = int(np.sum(known))
        if count == 0:
            continue
        entry = cache.entries[record.protein_id]
        embedding = np.load(
            _embedding_path(cache.cache_dir, str(entry["embedding_file"])),
            mmap_mode="r",
            allow_pickle=False,
        )
        features[cursor : cursor + count] = embedding[known]
        labels[cursor : cursor + count] = record.labels[known]
        cursor += count
    if cursor != known_count:
        raise RuntimeError(f"known-residue materialization mismatch: {cursor} != {known_count}")
    return torch.from_numpy(features), torch.from_numpy(labels)


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
        scores = predictions[record.protein_id]
        if scores.shape != record.labels.shape:
            raise ValueError(f"prediction shape mismatch for {record.protein_id}")
        known = record.labels >= 0
        labels = record.labels[known]
        known_scores = scores[known]
        pooled_labels.append(labels)
        pooled_scores.append(known_scores)
        auc = safe_auc(labels, known_scores)
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


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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
            for position, (label, probability) in enumerate(
                zip(record.labels, predictions[record.protein_id]), start=1
            ):
                writer.writerow(
                    [record.protein_id, position, int(label), f"{float(probability):.9g}"]
                )


@torch.inference_mode()
def predict_records(
    model: CachedIdrHead,
    records: Sequence[ProteinRecord],
    cache: CacheBundle,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    predictions: dict[str, np.ndarray] = {}
    for record in records:
        entry = cache.entries[record.protein_id]
        embedding = np.load(
            _embedding_path(cache.cache_dir, str(entry["embedding_file"])),
            mmap_mode="r",
            allow_pickle=False,
        )
        protein_scores = np.empty(len(record.sequence), dtype=np.float32)
        for start in range(0, len(record.sequence), batch_size):
            end = min(start + batch_size, len(record.sequence))
            feature_batch = torch.from_numpy(
                np.array(embedding[start:end], dtype=np.float32, copy=True)
            ).to(device, non_blocking=True)
            protein_scores[start:end] = torch.sigmoid(model(feature_batch)).cpu().numpy()
        predictions[record.protein_id] = protein_scores
    return predictions


def train_epoch(
    model: CachedIdrHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: float,
) -> float:
    model.train()
    positive_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    total_loss = 0.0
    total_residues = 0
    for feature_batch, label_batch in loader:
        features = feature_batch.to(device, dtype=torch.float32, non_blocking=True)
        labels = label_batch.to(device, non_blocking=True)
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=positive_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_residues = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * batch_residues
        total_residues += batch_residues
    return total_loss / total_residues


def run_fold(
    fold: str,
    records: Sequence[ProteinRecord],
    cache: CacheBundle,
    config: dict[str, Any],
    seed: int,
    output_root: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[ProteinRecord], dict[str, np.ndarray]]:
    model_cfg = config["model"]
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

    train_features, train_labels = load_known_residue_tensors(train_records, cache)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=int(train_cfg["residue_batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = CachedIdrHead(cache.hidden_size, float(model_cfg["dropout"])).to(device)
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
        loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_predictions = predict_records(
            model,
            val_records,
            cache,
            device,
            int(eval_cfg["inference_residue_batch_size"]),
        )
        val_metrics = metrics_from_records(val_records, val_predictions)
        val_auc = val_metrics["micro_roc_auc"]
        if val_auc is None:
            raise RuntimeError(f"fold {fold} validation ROC-AUC is undefined")
        epoch_row = {"epoch": epoch, "train_loss": loss, **val_metrics}
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
                    "hidden_size": cache.hidden_size,
                    "model_revision": cache.model_revision,
                    "cache_index_sha256": cache.index_sha256,
                    "classifier_state_dict": model.state_dict(),
                    "best_micro_roc_auc": best_auc,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(train_cfg["patience"]):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["classifier_state_dict"])
    best_predictions = predict_records(
        model,
        val_records,
        cache,
        device,
        int(eval_cfg["inference_residue_batch_size"]),
    )
    best_metrics = metrics_from_records(val_records, best_predictions)
    report = {
        "experiment": config["experiment"]["name"],
        "training_mode": "verified_frozen_embedding_cache",
        "output_heads": 1,
        "output_definition": "classic_idr_probability_per_residue",
        "fold": fold,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "pos_weight": pos_weight,
        "train_proteins": len(train_records),
        "train_known_residues": positives + negatives,
        "validation_proteins": len(val_records),
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
    expected_revision = str(config["model"]["revision"])
    cache = load_verified_cache(args.cache_dir, records, expected_revision)

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
        "fold_metrics": [report["metrics"] for report in fold_reports],
        "oof_metrics": metrics_from_records(all_records, all_predictions),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "backbone_forward_passes_during_training": 0,
        "input_cache_contains_residue_labels": False,
        "caid2_caid3_labels_accessed": False,
    }
    save_json(output_root / "oof_metrics.json", summary)
    save_predictions(output_root / "oof_predictions.csv", all_records, all_predictions)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
