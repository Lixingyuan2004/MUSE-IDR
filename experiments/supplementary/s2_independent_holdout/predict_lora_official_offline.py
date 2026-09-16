"""Offline loader for the pinned official LoRA-DR-Suite inference code.

All sequence parsing, windowing, logit fusion and output writing are delegated
to the already locked B4 official predictor.  Only model discovery is replaced:
the exact adapter and base revisions must already exist in the Hugging Face
cache, and no network request is made.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.b4_baseline_reproduction import predict_lora_official as official


MODEL_ID = "CQSB/esm2_650M-LoRA-ID-DisProt7"
MODEL_REVISION = "4ee4e4d404a224d76cf1c59fb99aebd7b52f7915"
BASE_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
BASE_MODEL_REVISION = "08e4846e537177426273712802403f7ba8261b6c"


def cached_snapshot(model_id: str, revision: str) -> Path:
    """Resolve an exact cached snapshot without contacting Hugging Face."""
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    except Exception as error:
        raise FileNotFoundError(
            f"required offline Hugging Face snapshot is unavailable: "
            f"{model_id}@{revision}; HF_HOME={os.environ.get('HF_HOME')!r}"
        ) from error
    snapshot = Path(path).resolve()
    if snapshot.name != revision:
        raise RuntimeError(f"cached snapshot resolved to {snapshot.name}, expected {revision}")
    return snapshot


def load_model_offline(model_id: str, revision: str | None, device, dtype):
    from peft import PeftConfig, PeftModel
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    if model_id != MODEL_ID:
        raise ValueError(f"offline loader only permits {MODEL_ID}")
    if revision != MODEL_REVISION:
        raise ValueError(f"offline loader requires adapter revision {MODEL_REVISION}")

    adapter_snapshot = cached_snapshot(MODEL_ID, MODEL_REVISION)
    base_snapshot = cached_snapshot(BASE_MODEL_ID, BASE_MODEL_REVISION)
    adapter_config_path = adapter_snapshot / "adapter_config.json"
    adapter_weights_path = adapter_snapshot / "adapter_model.safetensors"
    for required in (adapter_config_path, adapter_weights_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    peft_config = PeftConfig.from_pretrained(
        str(adapter_snapshot),
        local_files_only=True,
    )
    if peft_config.base_model_name_or_path != BASE_MODEL_ID:
        raise RuntimeError(
            "cached adapter declares unexpected base model: "
            f"{peft_config.base_model_name_or_path!r}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_snapshot),
        local_files_only=True,
    )
    base_config = AutoConfig.from_pretrained(
        str(base_snapshot),
        local_files_only=True,
    )
    if base_config.model_type != "esm":
        raise RuntimeError(f"expected ESM base model, got {base_config.model_type!r}")
    base_config.num_labels = 2
    base_model = AutoModelForTokenClassification.from_pretrained(
        str(base_snapshot),
        config=base_config,
        dtype=dtype,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_snapshot),
        local_files_only=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    adapter_verification = official.verify_loaded_adapter(model, peft_config)

    provenance = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "adapter_config_sha256": official.file_sha256(adapter_config_path),
        "adapter_weights_sha256": official.file_sha256(adapter_weights_path),
        "offline_cached_loading": True,
        "network_requests_required": False,
        **adapter_verification,
    }
    return tokenizer, model, provenance


def main() -> None:
    official.load_model = load_model_offline
    official.main()


if __name__ == "__main__":
    main()
