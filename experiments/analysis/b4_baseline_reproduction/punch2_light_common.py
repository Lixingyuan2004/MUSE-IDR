from __future__ import annotations

import gzip
import hashlib
import importlib.util
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PUNCH2_LIGHT_REPO = PROJECT_ROOT / "third_party" / "baselines" / "punch2_light"
EXPECTED_PUNCH2_LIGHT_COMMIT = "6c7935b3597c056d2e6b3845bb54fc101c5bc574"
DEFAULT_PROTT5_MODEL_ID = "Rostlab/prot_t5_xl_uniref50"

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWYX")
AA_TO_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
AUTHOR_AMBIGUOUS_PATTERN = re.compile(r"[UZOB]")


@dataclass(frozen=True)
class FastaRecord:
    protein_id: str
    sequence: str


@dataclass(frozen=True)
class MemberSpec:
    member_id: str
    feature_type: str
    architecture: str
    fold: int
    checkpoint: Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("rt", encoding="utf-8-sig")


def read_sequence_only_fasta(path: str | Path) -> list[FastaRecord]:
    path = Path(path)
    records: list[FastaRecord] = []
    current_id: str | None = None
    sequence_parts: list[str] = []

    def finish() -> None:
        nonlocal current_id, sequence_parts
        if current_id is None:
            return
        sequence = "".join(sequence_parts).replace(" ", "").upper()
        if not sequence or not sequence.isalpha():
            raise ValueError(
                f"{current_id}: input must contain amino-acid sequences only; "
                "CAID reference/label files are forbidden during inference"
            )
        records.append(FastaRecord(current_id, sequence))
        current_id = None
        sequence_parts = []

    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                current_id = line[1:].strip().split()[0]
                if not current_id:
                    raise ValueError("empty FASTA identifier")
            elif current_id is None:
                raise ValueError("FASTA sequence encountered before identifier")
            else:
                sequence_parts.append(line)
    finish()

    if not records:
        raise ValueError("no FASTA records found")
    identifiers = [record.protein_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate FASTA identifiers are not allowed")
    return records


def author_sequence_mapping(sequence: str) -> str:
    """Match deemeng/embedding: only U, Z, O and B are mapped to X."""
    return AUTHOR_AMBIGUOUS_PATTERN.sub("X", sequence.upper())


def author_onehot(sequence: str) -> np.ndarray:
    mapped = author_sequence_mapping(sequence)
    encoded = np.zeros((1, len(mapped), len(AMINO_ACIDS)), dtype=np.float32)
    for position, residue in enumerate(mapped):
        index = AA_TO_INDEX.get(residue)
        if index is not None:
            encoded[0, position, index] = 1.0
    return encoded


def repository_head(repo: str | Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(repo)), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def verify_repository(repo: str | Path) -> str:
    repo = Path(repo)
    observed = repository_head(repo)
    if observed != EXPECTED_PUNCH2_LIGHT_COMMIT:
        raise ValueError(
            f"PUNCH2-Light source revision mismatch: {observed} != "
            f"{EXPECTED_PUNCH2_LIGHT_COMMIT}"
        )
    return observed


def released_member_specs(repo: str | Path) -> list[MemberSpec]:
    repo = Path(repo)
    specs: list[MemberSpec] = []
    for fold in range(1, 4):
        specs.append(
            MemberSpec(
                member_id=f"onehot_l12_f{fold}",
                feature_type="onehot",
                architecture="cnn2_L12",
                fold=fold,
                checkpoint=repo / "predictor" / "onehot"
                / f"cnn2_L12_withFullyDisorder78.pth_f{fold}",
            )
        )
    for fold in range(1, 6):
        specs.append(
            MemberSpec(
                member_id=f"prottrans_l3_f{fold}",
                feature_type="protTrans",
                architecture="cnn2_L3_100_50",
                fold=fold,
                checkpoint=repo / "predictor" / "protTrans"
                / f"cnn2_L3_100_50_withFullyDisorder78.pth_f{fold}",
            )
        )
    for fold in range(1, 6):
        specs.append(
            MemberSpec(
                member_id=f"prottrans_l12_f{fold}",
                feature_type="protTrans",
                architecture="cnn2_L12",
                fold=fold,
                checkpoint=repo / "predictor" / "protTrans"
                / f"cnn2_L12_withFullyDisorder78.pth_f{fold}",
            )
        )
    missing = [str(spec.checkpoint) for spec in specs if not spec.checkpoint.is_file()]
    if missing:
        raise FileNotFoundError(f"missing PUNCH2-Light checkpoints: {missing}")
    return specs


def paper8_member_ids() -> set[str]:
    return {
        *(f"onehot_l12_f{fold}" for fold in range(1, 4)),
        *(f"prottrans_l12_f{fold}" for fold in range(1, 6)),
    }


def _load_architecture(repo: Path, architecture: str):
    module_path = repo / "model" / f"{architecture}.py"
    module_name = f"b4_punch2_light_{architecture}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official architecture from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_members(
    repo: str | Path,
    device: torch.device,
) -> tuple[list[MemberSpec], dict[str, torch.nn.Module], dict[str, str]]:
    repo = Path(repo)
    verify_repository(repo)
    specs = released_member_specs(repo)
    modules = {
        architecture: _load_architecture(repo, architecture)
        for architecture in sorted({spec.architecture for spec in specs})
    }
    models: dict[str, torch.nn.Module] = {}
    hashes: dict[str, str] = {}
    for spec in specs:
        in_features = 21 if spec.feature_type == "onehot" else 1024
        model = modules[spec.architecture].Net(
            in_features=in_features,
            dropout=0,
        ).to(device)
        try:
            checkpoint = torch.load(
                spec.checkpoint,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(spec.checkpoint, map_location=device)
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"{spec.checkpoint}: model_state_dict is missing")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        models[spec.member_id] = model
        hashes[spec.member_id] = sha256(spec.checkpoint)
    return specs, models, hashes


def validate_feature_array(
    array: np.ndarray,
    length: int,
    width: int,
    feature_type: str,
) -> np.ndarray:
    array = np.asarray(array)
    expected = (1, length, width)
    if array.shape != expected:
        raise ValueError(f"{feature_type} shape mismatch: {array.shape} != {expected}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{feature_type} must use a floating dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{feature_type} contains non-finite values")
    return np.asarray(array, dtype=np.float32)


def predict_member_probabilities(
    specs: list[MemberSpec],
    models: dict[str, torch.nn.Module],
    onehot: np.ndarray,
    prottrans: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    features = {"onehot": onehot, "protTrans": prottrans}
    predictions: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for spec in specs:
            array = features[spec.feature_type]
            tensor = torch.from_numpy(array[0].T.copy()).unsqueeze(0).to(device)
            prediction = models[spec.member_id](tensor)[0]
            values = prediction.detach().cpu().numpy().astype(np.float64, copy=False)
            if values.shape != (array.shape[1],):
                raise ValueError(
                    f"{spec.member_id}: prediction shape {values.shape} does not "
                    f"match residue count {array.shape[1]}"
                )
            if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
                raise ValueError(f"{spec.member_id}: invalid probabilities")
            predictions[spec.member_id] = values
    return predictions


def ensemble_probabilities(
    member_predictions: dict[str, np.ndarray],
    variant: str,
) -> np.ndarray:
    if variant == "paper8":
        selected = paper8_member_ids()
    elif variant == "released13":
        selected = set(member_predictions)
    else:
        raise ValueError(f"unknown PUNCH2-Light variant: {variant}")
    missing = selected - set(member_predictions)
    if missing:
        raise ValueError(f"missing ensemble members: {sorted(missing)}")
    ordered = [member_predictions[key] for key in sorted(selected)]
    return np.stack(ordered, axis=0).mean(axis=0)
