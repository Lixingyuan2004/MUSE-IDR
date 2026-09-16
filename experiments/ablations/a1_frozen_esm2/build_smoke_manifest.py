"""Select a tiny deterministic internal-data manifest for pipeline smoke testing."""

from __future__ import annotations

import json
from pathlib import Path


def has_both_classes(rows: list[dict[str, object]]) -> bool:
    labels = [label for row in rows for label in row["labels"]]
    return 0 in labels and 1 in labels


def select_rows(rows: list[dict[str, object]], fold: str, maximum: int = 8):
    candidates = sorted(
        (row for row in rows if str(row["fold"]) == fold),
        key=lambda row: (len(row["sequence"]), row["id"]),
    )
    selected = []
    for row in candidates:
        if len(row["sequence"]) > 256:
            continue
        selected.append(row)
        if len(selected) >= 2 and has_both_classes(selected):
            return selected
        if len(selected) >= maximum:
            break
    raise RuntimeError(f"could not select a two-class smoke subset for fold {fold}")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    source = root / "data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl"
    output = root / "data/processed/classic_idr_2018/a1_frozen_esm2_smoke_manifest.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]

    # Fold 4 is the pre-declared development validation fold. Fold 0 supplies
    # the tiny training side. No CAID2/3 test records are read.
    selected = select_rows(rows, "0") + select_rows(rows, "4")
    output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "output": output.relative_to(root).as_posix(),
        "proteins": len(selected),
        "fold_0_proteins": sum(str(row["fold"]) == "0" for row in selected),
        "fold_4_proteins": sum(str(row["fold"]) == "4" for row in selected),
        "maximum_sequence_length": max(len(row["sequence"]) for row in selected),
        "caid2_caid3_labels_accessed": False,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
