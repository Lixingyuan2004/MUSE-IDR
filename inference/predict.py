"""Stable repository-local command for locked MUSE-IDR inference."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.final_models.a10_locked_inference.predict_a10 import main  # noqa: E402


if __name__ == "__main__":
    main()
