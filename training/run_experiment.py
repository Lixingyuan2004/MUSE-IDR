"""Dispatch to a canonical MUSE-IDR training experiment without duplicating it."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPTS = {
    "a1": ROOT / "experiments/ablations/a1_frozen_esm2/train_a1.py",
    "a2": ROOT / "experiments/ablations/a2_multiscale_context/train_a2.py",
    "a3": ROOT / "experiments/ablations/a3_auc_ranking_loss/train_a3.py",
    "a4": ROOT / "experiments/ablations/a4_esm2_lora_multiscale/train_a4.py",
    "a6": ROOT / "experiments/ablations/a6_long_range_dilated_context/train_a6.py",
    "a7": ROOT / "experiments/ablations/a7_bidirectional_sequence_context/train_a7.py",
    "a8": ROOT / "experiments/ablations/a8_multilayer_esm2_fusion/train_a8.py",
}


def print_available() -> None:
    """Print the stable experiment-to-script mapping."""
    for name, path in TRAINING_SCRIPTS.items():
        print(f"{name}\t{path.relative_to(ROOT).as_posix()}")


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] in {"--list", "-l"}:
        print_available()
        return

    experiment = sys.argv[1].lower()
    script = TRAINING_SCRIPTS.get(experiment)
    if script is None:
        available = ", ".join(TRAINING_SCRIPTS)
        raise SystemExit(f"unknown training experiment {experiment!r}; choose one of: {available}")
    if not script.is_file():
        raise SystemExit(f"canonical training script is missing: {script}")

    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
