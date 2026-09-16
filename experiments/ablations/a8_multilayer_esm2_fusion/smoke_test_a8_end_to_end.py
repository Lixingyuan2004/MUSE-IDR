"""Run a tiny end-to-end A8 fold using an independently verified cache."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a2_multiscale_context.smoke_test_a2_end_to_end import (
    make_records,
)
from experiments.ablations.a8_multilayer_esm2_fusion.cache_multilayer import (
    artifact_stem,
    file_sha256,
)
from experiments.ablations.a8_multilayer_esm2_fusion.data import (
    load_verified_multilayer_cache,
)
from experiments.ablations.a8_multilayer_esm2_fusion.train_a8 import run_fold
from experiments.ablations.a8_multilayer_esm2_fusion.verify_multilayer_cache import (
    verify_cache,
)


def write_multilayer_cache(
    cache_dir: Path,
    manifest: Path,
    records,
    hidden_size: int,
    layer_indices: tuple[int, ...],
) -> None:
    embedding_dir = cache_dir / "embeddings"
    embedding_dir.mkdir(parents=True)
    manifest_rows = []
    index_rows = []
    revision = "synthetic-locked-revision"
    for index, record in enumerate(records):
        manifest_rows.append(
            {"id": record.protein_id, "sequence": record.sequence, "fold": record.fold}
        )
        generator = np.random.default_rng(index + 1)
        embedding = generator.normal(
            size=(len(record.sequence), len(layer_indices), hidden_size)
        ).astype(np.float16)
        stem = artifact_stem(record.protein_id)
        relative = f"embeddings/{stem}.npy"
        embedding_path = cache_dir / relative
        np.save(embedding_path, embedding, allow_pickle=False)
        index_rows.append(
            {
                "protein_id": record.protein_id,
                "sequence_sha256": __import__("hashlib")
                .sha256(record.sequence.encode("ascii"))
                .hexdigest(),
                "sequence_length": len(record.sequence),
                "fold": record.fold,
                "model_id": "synthetic-esm2",
                "model_revision": revision,
                "layer_indices": list(layer_indices),
                "num_layers": len(layer_indices),
                "max_residues": 16,
                "window_stride": 8,
                "dtype": "float16",
                "hidden_size": hidden_size,
                "embedding_file": relative,
                "embedding_nbytes": int(embedding.nbytes),
                "embedding_sha256": file_sha256(embedding_path),
                "finite": True,
                "caid2_caid3_labels_accessed": False,
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    (cache_dir / "cache_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows), encoding="utf-8"
    )
    verification = verify_cache(manifest, cache_dir)
    if verification["input_contains_residue_labels"] is not False:
        raise RuntimeError("synthetic A8 verification failed the label-free audit")


def main() -> None:
    records = make_records()
    layers = (30, 31, 32, 33)
    config = {
        "experiment": {"name": "a8_synthetic_smoke"},
        "model": {
            "revision": "synthetic-locked-revision",
            "layer_indices": list(layers),
            "input_size": 16,
            "hidden_size": 8,
            "kernels": [3, 5],
            "dropout": 0.0,
            "initial_last_layer_weight": 0.7,
        },
        "training": {
            "epochs": 2,
            "patience": 2,
            "protein_batch_size": 2,
            "learning_rate": 0.01,
            "layer_mix_learning_rate": 0.01,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "num_workers": 0,
            "class_balance": True,
        },
        "evaluation": {"unknown_label": -1},
    }

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cache_dir = root / "cache"
        cache_dir.mkdir()
        manifest = root / "label_free_sequences.jsonl"
        write_multilayer_cache(cache_dir, manifest, records, 16, layers)
        cache = load_verified_multilayer_cache(
            cache_dir,
            records,
            expected_revision="synthetic-locked-revision",
            expected_layers=layers,
        )
        report, validation_records, predictions = run_fold(
            "1",
            records,
            cache,
            config,
            seed=17,
            output_root=root / "outputs",
            device=torch.device("cpu"),
        )
        if report["output_heads"] != 1:
            raise RuntimeError("A8 end-to-end smoke created more than one output head")
        if report["metrics"]["micro_roc_auc"] is None:
            raise RuntimeError("A8 end-to-end smoke did not produce ROC-AUC")
        if len(report["learned_layer_weights"]) != len(layers):
            raise RuntimeError("A8 did not report one learned weight per cached layer")
        if not np.isclose(sum(report["learned_layer_weights"]), 1.0):
            raise RuntimeError("A8 learned layer weights do not sum to one")
        for record in validation_records:
            if predictions[record.protein_id].shape != record.labels.shape:
                raise RuntimeError(f"prediction shape mismatch for {record.protein_id}")
        required = [
            root / "outputs" / "fold_1" / "best_head.pt",
            root / "outputs" / "fold_1" / "metrics.json",
            root / "outputs" / "fold_1" / "predictions.csv",
        ]
        if not all(path.is_file() for path in required):
            raise RuntimeError("A8 end-to-end smoke did not write all fold artifacts")

    print(
        json.dumps(
            {
                "status": "pass",
                "independently_verified_multilayer_cache": True,
                "cache_input_contains_labels": False,
                "fold_training": True,
                "layer_weights_learned": True,
                "checkpoint_written": True,
                "prediction_alignment": True,
                "micro_roc_auc_defined": True,
                "output_heads": 1,
                "caid2_caid3_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
