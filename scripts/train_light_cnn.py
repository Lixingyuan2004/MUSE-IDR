"""Train and evaluate the lightweight one-head residue CNN baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from muse_idr.data.caid import file_sha256
from muse_idr.evaluation import ResiduePrediction, evaluate_predictions
from muse_idr.models import LightIDRCNN
from muse_idr.training.data import (
    VOCAB_SIZE,
    collate_proteins,
    load_training_examples,
    make_token_budget_batches,
)
from muse_idr.training.engine import (
    make_cosine_with_warmup_scheduler,
    predict_residues,
    set_reproducible_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train/smoke_light_cnn.json")
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_predictions(path: Path, predictions: list[ResiduePrediction]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            row = {"protein_id": prediction.protein_id, "scores": list(prediction.scores)}
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _write_references(path: Path, predictions: list[ResiduePrediction]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            row = {
                "protein_id": prediction.protein_id,
                "sequence": prediction.sequence,
                "labels": list(prediction.labels),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _optional_positive_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive or null")
    return parsed


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config_path = _resolve(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    training_config = config["training"]
    model_config = config["model"]
    dataset_path = _resolve(root, data_config["dataset_jsonl"])
    assignments_path = _resolve(root, data_config["fold_assignments_jsonl"])
    readiness_path = _resolve(root, data_config["readiness_manifest"])
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if readiness.get("training_readiness") != "ready_for_baseline_training":
        raise ValueError("Training data readiness manifest does not permit baseline training")
    if readiness.get("audit", {}).get("homology_clusters_split_across_folds") != 0:
        raise ValueError("Training readiness manifest reports a split homology cluster")

    output_dir = _resolve(root, config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_training_examples(dataset_path, assignments_path)
    validation_fold = int(data_config["validation_fold"])
    train_indices = [
        index for index, example in enumerate(examples) if example.fold != validation_fold
    ]
    validation_indices = [
        index for index, example in enumerate(examples) if example.fold == validation_fold
    ]
    if not train_indices or not validation_indices:
        raise ValueError("Training and validation partitions must both be non-empty")
    seed = int(training_config["seed"])
    set_reproducible_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(training_config["mixed_precision"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = LightIDRCNN(
        vocab_size=VOCAB_SIZE,
        hidden_size=int(model_config["hidden_size"]),
        kernel_size=int(model_config["kernel_size"]),
        dilations=tuple(int(value) for value in model_config["dilations"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    positives = sum(examples[index].labels.count(1) for index in train_indices)
    negatives = sum(examples[index].labels.count(0) for index in train_indices)
    if positives == 0 or negatives == 0:
        raise ValueError("Training partition must contain positive and negative residues")
    pos_weight_value = negatives / positives
    pos_weight = torch.tensor(pos_weight_value, device=device)

    epochs = int(training_config["epochs"])
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    max_train_batches = _optional_positive_int(
        training_config.get("max_train_batches"), name="max_train_batches"
    )
    train_batch_plans = [
        make_token_budget_batches(
            examples,
            train_indices,
            token_budget=int(training_config["token_budget"]),
            max_batch_size=int(training_config["max_batch_size"]),
            shuffle=True,
            seed=seed + epoch,
            bucket_size=int(training_config["bucket_size"]),
        )
        for epoch in range(epochs)
    ]
    planned_optimization_steps = sum(
        min(len(plan), max_train_batches) if max_train_batches is not None else len(plan)
        for plan in train_batch_plans
    )
    scheduler = make_cosine_with_warmup_scheduler(
        optimizer,
        total_steps=planned_optimization_steps,
        warmup_ratio=float(training_config.get("warmup_ratio", 0.0)),
        min_factor=float(training_config.get("min_learning_rate_factor", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    validation_batches = make_token_budget_batches(
        examples,
        validation_indices,
        token_budget=int(training_config["token_budget"]),
        max_batch_size=int(training_config["max_batch_size"]),
        shuffle=False,
        seed=seed,
        bucket_size=int(training_config["bucket_size"]),
    )
    validation_loader = DataLoader(
        examples,
        batch_sampler=validation_batches,
        collate_fn=collate_proteins,
        num_workers=int(training_config["num_workers"]),
    )

    history: list[dict[str, object]] = []
    best_auc = -1.0
    best_metrics: dict[str, object] | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    early_stopping_patience = _optional_positive_int(
        training_config.get("early_stopping_patience"), name="early_stopping_patience"
    )
    minimum_auc_improvement = float(training_config.get("minimum_auc_improvement", 0.0))
    stopped_early = False
    for epoch, train_batches in enumerate(train_batch_plans):
        epoch_started = perf_counter()
        train_loader = DataLoader(
            examples,
            batch_sampler=train_batches,
            collate_fn=collate_proteins,
            num_workers=int(training_config["num_workers"]),
        )
        train_summary = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            pos_weight=pos_weight,
            use_amp=use_amp,
            max_batches=max_train_batches,
            scaler=scaler,
            scheduler=scheduler,
        )
        predictions = predict_residues(
            model, validation_loader, device=device, use_amp=use_amp
        )
        metrics = evaluate_predictions(predictions, threshold=0.5, bootstrap_replicates=0)
        auc = float(metrics["metrics"]["residue_micro_roc_auc"])
        epoch_seconds = perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_summary,
                "validation_auc": auc,
                "elapsed_seconds": epoch_seconds,
            }
        )
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_summary["loss"],
                    "validation_auc": auc,
                    "learning_rate": train_summary["learning_rate_end"],
                    "elapsed_seconds": round(epoch_seconds, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if auc > best_auc + minimum_auc_improvement:
            best_auc = auc
            best_metrics = metrics
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "model_config": model_config,
                    "validation_fold": validation_fold,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "validation_auc": auc,
                },
                output_dir / "best_checkpoint.pt",
            )
            _write_predictions(output_dir / "validation_predictions.jsonl", predictions)
            _write_references(output_dir / "validation_reference.jsonl", predictions)
            _write_json(output_dir / "best_metrics.json", metrics)
        else:
            epochs_without_improvement += 1
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            break
    if best_metrics is None:
        raise RuntimeError("Training produced no validation metrics")

    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    )
    run_manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_stage": config["experiment_stage"],
        "warning": config.get("warning"),
        "task": "single_output_classic_idr",
        "output_heads": 1,
        "config": config,
        "config_sha256": file_sha256(config_path),
        "data_sha256": file_sha256(dataset_path),
        "fold_assignments_sha256": file_sha256(assignments_path),
        "readiness_manifest_sha256": file_sha256(readiness_path),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "peak_memory_bytes": peak_memory,
        },
        "partitions": {
            "training_proteins": len(train_indices),
            "validation_proteins": len(validation_indices),
            "validation_fold": validation_fold,
            "training_positive_residues": positives,
            "training_negative_residues": negatives,
            "pos_weight": pos_weight_value,
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "optimization": {
            "planned_steps": planned_optimization_steps,
            "executed_steps": sum(int(row["train"]["batches"]) for row in history),
            "epochs_planned": epochs,
            "epochs_executed": len(history),
            "stopped_early": stopped_early,
            "early_stopping_patience": early_stopping_patience,
            "best_epoch": best_epoch,
        },
        "best_validation_residue_micro_roc_auc": best_auc,
        "caid2_caid3_labels_accessed": False,
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "stage": config["experiment_stage"],
                "device": device.type,
                "validation_fold": validation_fold,
                "validation_proteins": len(validation_indices),
                "residue_micro_roc_auc": best_auc,
                "output_dir": output_dir.relative_to(root).as_posix(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
