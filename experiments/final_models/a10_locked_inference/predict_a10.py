"""Run the locked 30-member A10 ensemble on label-free FASTA sequences."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence, TextIO

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablations.a2_multiscale_context.model import (  # noqa: E402
    MultiscaleContextIdrHead,
)
from experiments.ablations.a8_multilayer_esm2_fusion.cache_multilayer import (  # noqa: E402
    SequenceRecord as CacheSequenceRecord,
    atomic_save_numpy,
    atomic_write_json,
    encode_record,
)
from experiments.ablations.a8_multilayer_esm2_fusion.model import (  # noqa: E402
    MultilayerEsm2FusionIdrModel,
)


MODEL_ID = "facebook/esm2_t33_650M_UR50D"
MODEL_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
LAYER_INDICES = (30, 31, 32, 33)
SEEDS = (17, 29, 43)
FOLDS = (0, 1, 2, 3, 4)
ALLOWED_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


@dataclass(frozen=True)
class FastaRecord:
    protein_id: str
    sequence: str

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest()


@dataclass
class EnsembleMember:
    family: str
    seed: int
    fold: int
    relative_path: str
    sha256: str
    model: nn.Module


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def read_fasta(path: str | Path) -> list[FastaRecord]:
    """Read a sequence-only FASTA and reject ambiguous record structure."""

    path = Path(path)
    records: list[FastaRecord] = []
    seen: set[str] = set()
    header: str | None = None
    sequence_lines: list[str] = []

    def finish_record() -> None:
        nonlocal header, sequence_lines
        if header is None:
            return
        protein_id = header.split(maxsplit=1)[0]
        sequence = "".join(sequence_lines).replace(" ", "").upper()
        if not protein_id:
            raise ValueError("FASTA contains an empty protein identifier")
        if protein_id in seen:
            raise ValueError(f"duplicate FASTA protein identifier: {protein_id}")
        if not sequence:
            raise ValueError(f"empty FASTA sequence: {protein_id}")
        invalid = sorted(set(sequence) - ALLOWED_RESIDUES)
        if invalid:
            raise ValueError(
                f"unsupported residues in {protein_id}: {invalid}; aligned FASTA is not accepted"
            )
        seen.add(protein_id)
        records.append(FastaRecord(protein_id, sequence))
        header = None
        sequence_lines = []

    with _open_text(path, "r") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"empty FASTA header at line {line_number}")
            else:
                if header is None:
                    raise ValueError(
                        f"sequence appears before the first FASTA header at line {line_number}"
                    )
                sequence_lines.append("".join(line.split()))
    finish_record()
    if not records:
        raise ValueError(f"FASTA contains no sequences: {path}")
    return records


def load_lock_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported A10 lock manifest schema")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("lock manifest uses the wrong ESM2 revision")
    if tuple(manifest.get("layer_indices", [])) != LAYER_INDICES:
        raise ValueError("lock manifest uses the wrong ESM2 layers")
    if tuple(manifest.get("seeds", [])) != SEEDS:
        raise ValueError("lock manifest does not contain the locked seeds")
    if tuple(manifest.get("folds", [])) != FOLDS:
        raise ValueError("lock manifest does not contain the locked folds")
    checkpoints = manifest.get("checkpoints", [])
    expected_keys = {
        (family, seed, fold)
        for family in ("a2", "a8")
        for seed in SEEDS
        for fold in FOLDS
    }
    observed_keys = {
        (str(row["family"]), int(row["seed"]), int(row["fold"]))
        for row in checkpoints
    }
    if len(checkpoints) != 30 or observed_keys != expected_keys:
        raise ValueError("lock manifest must contain exactly the formal 30 A10 members")
    return manifest


def _validate_model_config(family: str, config: dict[str, Any]) -> None:
    expected_common: dict[str, Any] = {
        "revision": MODEL_REVISION,
        "input_size": 1280,
        "hidden_size": 256,
        "kernels": [3, 7, 15, 31],
        "dropout": 0.2,
    }
    for key, expected in expected_common.items():
        observed = config.get(key)
        if observed != expected:
            raise ValueError(
                f"{family} checkpoint model_config[{key!r}]={observed!r}, expected {expected!r}"
            )
    if family == "a8":
        if config.get("layer_indices") != list(LAYER_INDICES):
            raise ValueError("A8 checkpoint has the wrong ESM2 layer indices")
        if float(config.get("initial_last_layer_weight", -1.0)) != 0.7:
            raise ValueError("A8 checkpoint has the wrong initial layer mixture")


def _build_member_model(family: str, config: dict[str, Any]) -> nn.Module:
    if family == "a2":
        return MultiscaleContextIdrHead(
            input_size=int(config["input_size"]),
            hidden_size=int(config["hidden_size"]),
            kernels=tuple(int(value) for value in config["kernels"]),
            dropout=float(config["dropout"]),
        )
    if family == "a8":
        return MultilayerEsm2FusionIdrModel(
            input_size=int(config["input_size"]),
            hidden_size=int(config["hidden_size"]),
            layer_indices=tuple(int(value) for value in config["layer_indices"]),
            kernels=tuple(int(value) for value in config["kernels"]),
            dropout=float(config["dropout"]),
            initial_last_layer_weight=float(config["initial_last_layer_weight"]),
        )
    raise ValueError(f"unknown A10 family: {family}")


def verify_and_load_members(
    locked_root: str | Path,
    lock_manifest: dict[str, Any],
    device: torch.device,
) -> list[EnsembleMember]:
    locked_root = Path(locked_root).resolve()
    members: list[EnsembleMember] = []
    for row in lock_manifest["checkpoints"]:
        relative_path = Path(str(row["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe checkpoint path: {relative_path}")
        checkpoint_path = locked_root / relative_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing locked checkpoint: {checkpoint_path}")
        observed_hash = file_sha256(checkpoint_path)
        expected_hash = str(row["sha256"]).lower()
        if observed_hash != expected_hash:
            raise ValueError(f"checkpoint hash mismatch: {relative_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        family = str(row["family"])
        seed = int(row["seed"])
        fold = int(row["fold"])
        if int(checkpoint.get("seed", -1)) != seed:
            raise ValueError(f"checkpoint seed mismatch: {relative_path}")
        if int(checkpoint.get("fold", -1)) != fold:
            raise ValueError(f"checkpoint fold mismatch: {relative_path}")
        if checkpoint.get("model_revision") != MODEL_REVISION:
            raise ValueError(f"checkpoint ESM2 revision mismatch: {relative_path}")
        model_config = dict(checkpoint["model_config"])
        _validate_model_config(family, model_config)
        model = _build_member_model(family, model_config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.requires_grad_(False)
        model.eval().to(device)
        members.append(
            EnsembleMember(
                family=family,
                seed=seed,
                fold=fold,
                relative_path=relative_path.as_posix(),
                sha256=observed_hash,
                model=model,
            )
        )
    if len(members) != 30:
        raise RuntimeError("A10 did not load exactly 30 formal members")
    return members


def probability_from_member_logits(member_logits: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(member_logits) != 30:
        raise ValueError("A10 external inference requires exactly 30 member logits")
    reference_shape = member_logits[0].shape
    if any(logits.shape != reference_shape for logits in member_logits):
        raise ValueError("A10 member logits are not aligned")
    return torch.sigmoid(torch.stack(list(member_logits), dim=0).mean(dim=0))


@torch.inference_mode()
def predict_representation(
    representation: np.ndarray,
    members: Sequence[EnsembleMember],
    device: torch.device,
) -> np.ndarray:
    expected_shape = (len(representation), len(LAYER_INDICES), 1280)
    if representation.shape != expected_shape:
        raise ValueError(
            f"multilayer representation shape {representation.shape} != {expected_shape}"
        )
    features = torch.from_numpy(
        np.asarray(representation, dtype=np.float32)
    ).unsqueeze(0).to(device)
    residue_mask = torch.ones((1, len(representation)), dtype=torch.bool, device=device)
    final_layer = features[:, :, LAYER_INDICES.index(33), :]
    logits: list[torch.Tensor] = []
    for member in members:
        if member.family == "a2":
            logits.append(member.model(final_layer, residue_mask))
        else:
            logits.append(member.model(features, residue_mask))
    probability = probability_from_member_logits(logits)[0]
    result = probability.float().cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (len(representation),) or not np.all(np.isfinite(result)):
        raise RuntimeError("A10 produced invalid residue probabilities")
    return result


def _artifact_stem(record: FastaRecord) -> str:
    return hashlib.sha256(record.protein_id.encode("utf-8")).hexdigest()[:24]


def _load_valid_cache(
    record: FastaRecord,
    embedding_path: Path,
    metadata_path: Path,
    expected: dict[str, Any],
) -> np.ndarray | None:
    if not embedding_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        representation = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        if representation.shape != (len(record.sequence), len(LAYER_INDICES), 1280):
            return None
        if representation.dtype != np.float16:
            return None
        if file_sha256(embedding_path) != metadata.get("embedding_sha256"):
            return None
        return representation
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def load_or_encode_representation(
    record: FastaRecord,
    cache_dir: Path,
    tokenizer,
    backbone,
    device: torch.device,
    precision: str,
    max_residues: int,
    stride: int,
    window_batch_size: int,
) -> tuple[np.ndarray, bool]:
    embedding_dir = cache_dir / "embeddings"
    metadata_dir = cache_dir / "metadata"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(record)
    embedding_path = embedding_dir / f"{stem}.npy"
    metadata_path = metadata_dir / f"{stem}.json"
    expected = {
        "protein_id": record.protein_id,
        "sequence_sha256": record.sequence_sha256,
        "sequence_length": len(record.sequence),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer_indices": list(LAYER_INDICES),
        "max_residues": max_residues,
        "window_stride": stride,
        "dtype": "float16",
    }
    cached = _load_valid_cache(record, embedding_path, metadata_path, expected)
    if cached is not None:
        return cached, True

    cache_record = CacheSequenceRecord(record.protein_id, record.sequence, "external")
    generated = encode_record(
        cache_record,
        tokenizer,
        backbone,
        LAYER_INDICES,
        device,
        precision,
        max_residues,
        stride,
        window_batch_size,
    ).astype(np.float16, copy=False)
    atomic_save_numpy(embedding_path, generated)
    metadata = {
        **expected,
        "num_layers": len(LAYER_INDICES),
        "hidden_size": 1280,
        "embedding_file": embedding_path.relative_to(cache_dir).as_posix(),
        "embedding_nbytes": int(generated.nbytes),
        "embedding_sha256": file_sha256(embedding_path),
        "finite": bool(np.all(np.isfinite(generated))),
        "input_contains_residue_labels": False,
        "caid1_caid2_caid3_labels_accessed": False,
    }
    atomic_write_json(metadata_path, metadata)
    return generated, False


def _model_dtype(device: torch.device, precision: str) -> torch.dtype:
    if device.type != "cuda" or precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"unsupported precision: {precision}")


def load_backbone(device: torch.device, precision: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    backbone = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=_model_dtype(device, precision),
        add_pooling_layer=False,
    ).to(device)
    backbone.requires_grad_(False)
    backbone.eval()
    resolved_revision = getattr(backbone.config, "_commit_hash", None)
    if resolved_revision != MODEL_REVISION:
        raise RuntimeError(
            f"resolved ESM2 revision {resolved_revision!r} != locked revision {MODEL_REVISION!r}"
        )
    return tokenizer, backbone


def write_predictions(
    path: str | Path,
    records: Sequence[FastaRecord],
    predictions: dict[str, np.ndarray],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path, "w") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["protein_id", "position", "residue", "idr_probability"])
        for record in records:
            scores = predictions[record.protein_id]
            if scores.shape != (len(record.sequence),):
                raise ValueError(f"prediction length mismatch: {record.protein_id}")
            for position, (residue, score) in enumerate(
                zip(record.sequence, scores), start=1
            ):
                writer.writerow(
                    [record.protein_id, position, residue, f"{float(score):.9g}"]
                )


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--locked-root", type=Path, required=True)
    parser.add_argument(
        "--lock-manifest",
        type=Path,
        default=Path(__file__).with_name("a10_lock_manifest.json"),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-residues", type=int, default=1022)
    parser.add_argument("--window-stride", type=int, default=511)
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    lock_manifest = load_lock_manifest(args.lock_manifest)
    started = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    members = verify_and_load_members(args.locked_root, lock_manifest, device)
    verification = {
        "status": "pass",
        "locked_members": len(members),
        "a2_members": sum(member.family == "a2" for member in members),
        "a8_members": sum(member.family == "a8" for member in members),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "uniform_logit_weight": 1.0 / len(members),
        "all_checkpoint_hashes_match": True,
        "development_checkpoint_excluded": True,
        "output_heads": 1,
        "soft_disorder_output": False,
        "caid1_caid2_caid3_labels_accessed": False,
    }
    if args.verify_only:
        print(json.dumps(verification, indent=2, ensure_ascii=False))
        return
    if args.fasta is None or args.cache_dir is None or args.output is None:
        raise ValueError("--fasta, --cache-dir and --output are required for prediction")
    if args.max_residues < 1 or args.window_stride < 1:
        raise ValueError("window sizes must be positive")
    if args.window_stride > args.max_residues:
        raise ValueError("window stride cannot exceed max residues")
    if args.window_batch_size < 1:
        raise ValueError("window batch size must be positive")

    records = read_fasta(args.fasta)
    tokenizer, backbone = load_backbone(device, args.precision)
    predictions: dict[str, np.ndarray] = {}
    cached_count = 0
    generated_count = 0
    for index, record in enumerate(records, start=1):
        representation, reused = load_or_encode_representation(
            record,
            args.cache_dir,
            tokenizer,
            backbone,
            device,
            args.precision,
            args.max_residues,
            args.window_stride,
            args.window_batch_size,
        )
        if reused:
            cached_count += 1
        else:
            generated_count += 1
        predictions[record.protein_id] = predict_representation(
            representation, members, device
        )
        print(
            json.dumps(
                {
                    "protein": index,
                    "total": len(records),
                    "protein_id": record.protein_id,
                    "length": len(record.sequence),
                    "representation": "cache" if reused else "generated",
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    write_predictions(args.output, records, predictions)
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.json")
    report = {
        "schema_version": 1,
        "experiment": "a10_locked_external_inference",
        **verification,
        "ensemble_method": "thirty_model_uniform_logit_mean",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer_indices": list(LAYER_INDICES),
        "backbone_extractions_shared_by_all_members": True,
        "input_fasta": str(args.fasta),
        "input_fasta_sha256": file_sha256(args.fasta),
        "input_contains_labels": False,
        "proteins": len(records),
        "residues": sum(len(record.sequence) for record in records),
        "cached_representations": cached_count,
        "generated_representations": generated_count,
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "precision": args.precision,
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "scores_used_for_training_or_tuning": False,
    }
    save_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
