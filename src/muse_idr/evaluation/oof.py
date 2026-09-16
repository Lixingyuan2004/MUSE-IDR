"""Fail-closed assembly of out-of-fold residue predictions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .records import ResiduePrediction


def load_fold_assignments(path: Path) -> dict[str, int]:
    assignments: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            protein_id = row.get("protein_id")
            fold = row.get("fold")
            if not isinstance(protein_id, str) or not protein_id.strip():
                raise ValueError(f"Invalid protein_id at {path}:{line_number}")
            protein_id = protein_id.strip()
            if protein_id in assignments:
                raise ValueError(f"Duplicate fold assignment for {protein_id}")
            if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
                raise ValueError(f"Invalid fold for {protein_id} at {path}:{line_number}")
            assignments[protein_id] = fold
    if not assignments:
        raise ValueError(f"No fold assignments found in {path}")
    return assignments


def validate_and_order_oof_records(
    records_by_fold: dict[int, Sequence[ResiduePrediction]],
    assignments: dict[str, int],
) -> list[ResiduePrediction]:
    """Require exact fold/ID coverage and return assignment-file order."""

    expected_folds = set(assignments.values())
    if set(records_by_fold) != expected_folds:
        raise ValueError(
            f"OOF fold mismatch: expected={sorted(expected_folds)}, "
            f"observed={sorted(records_by_fold)}"
        )
    observed: dict[str, ResiduePrediction] = {}
    for fold, records in records_by_fold.items():
        for record in records:
            if record.protein_id in observed:
                raise ValueError(f"Duplicate OOF prediction for {record.protein_id}")
            assigned_fold = assignments.get(record.protein_id)
            if assigned_fold is None:
                raise ValueError(f"OOF prediction has no fold assignment: {record.protein_id}")
            if assigned_fold != fold:
                raise ValueError(
                    f"OOF prediction for {record.protein_id} came from fold {fold}, "
                    f"but assignment requires fold {assigned_fold}"
                )
            observed[record.protein_id] = record
    missing = sorted(set(assignments) - set(observed))
    if missing:
        raise ValueError(f"Missing OOF predictions: {missing[:10]}")
    return [observed[protein_id] for protein_id in assignments]
