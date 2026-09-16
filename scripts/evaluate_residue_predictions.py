"""Evaluate canonical residue-level references and universal IDR predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from muse_idr.evaluation import evaluate_predictions, load_aligned_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Canonical reference JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Metrics JSON")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_aligned_jsonl(args.reference, args.predictions)
    result = evaluate_predictions(
        records,
        threshold=args.threshold,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
