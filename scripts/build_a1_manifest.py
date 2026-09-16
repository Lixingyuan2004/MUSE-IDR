"""Build the A1 ablation manifest from the frozen classic-IDR dataset and folds."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from muse_idr.data.caid import file_sha256
from muse_idr.training.data import load_training_examples


def build_manifest(
    dataset_path: Path,
    assignments_path: Path,
    output_path: Path,
) -> dict[str, object]:
    examples = load_training_examples(dataset_path, assignments_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fold_counts: Counter[int] = Counter()
    positives = 0
    negatives = 0
    unknowns = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            fold_counts[example.fold] += 1
            positives += example.labels.count(1)
            negatives += example.labels.count(0)
            unknowns += example.labels.count(-1)
            row = {
                "id": example.protein_id,
                "sequence": example.sequence,
                "labels": list(example.labels),
                "fold": str(example.fold),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    report = {
        "schema_version": 1,
        "purpose": "a1_frozen_esm2_internal_ablation",
        "source_dataset": dataset_path.as_posix(),
        "source_dataset_sha256": file_sha256(dataset_path),
        "source_fold_assignments": assignments_path.as_posix(),
        "source_fold_assignments_sha256": file_sha256(assignments_path),
        "output_manifest": output_path.as_posix(),
        "output_manifest_sha256": file_sha256(output_path),
        "proteins": len(examples),
        "fold_proteins": {str(fold): fold_counts[fold] for fold in range(5)},
        "positive_residues": positives,
        "negative_residues": negatives,
        "unknown_residues": unknowns,
        "caid2_caid3_labels_accessed": False,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/classic_idr_2018/trainable_caid23_filtered.jsonl"),
    )
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path("data/processed/classic_idr_2018/homology_5fold_assignments.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/ablations/a1_frozen_esm2/data_manifest_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_manifest(args.dataset, args.fold_assignments, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
