"""Canonical JSONL readers with fail-closed reference/prediction alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .records import ResiduePrediction


def _jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield line_number, row


def _require_list(row: dict[str, object], key: str, path: Path, line_number: int) -> list[object]:
    value = row.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Expected list field {key!r} at {path}:{line_number}")
    return value


def _require_string(row: dict[str, object], key: str, path: Path, line_number: int) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string field {key!r} at {path}:{line_number}")
    return value


def load_aligned_jsonl(reference_path: Path, prediction_path: Path) -> list[ResiduePrediction]:
    """Load separate references and predictions, requiring exact ID coverage."""

    references: dict[str, tuple[str, list[object]]] = {}
    reference_order: list[str] = []
    for line_number, row in _jsonl_rows(reference_path):
        protein_id = _require_string(row, "protein_id", reference_path, line_number).strip()
        sequence = _require_string(row, "sequence", reference_path, line_number)
        labels = _require_list(row, "labels", reference_path, line_number)
        if not protein_id:
            raise ValueError(f"Missing protein_id at {reference_path}:{line_number}")
        if protein_id in references:
            raise ValueError(f"Duplicate reference protein_id: {protein_id}")
        references[protein_id] = (sequence, labels)
        reference_order.append(protein_id)

    predictions: dict[str, list[object]] = {}
    for line_number, row in _jsonl_rows(prediction_path):
        protein_id = _require_string(row, "protein_id", prediction_path, line_number).strip()
        scores = _require_list(row, "scores", prediction_path, line_number)
        if not protein_id:
            raise ValueError(f"Missing protein_id at {prediction_path}:{line_number}")
        if protein_id in predictions:
            raise ValueError(f"Duplicate prediction protein_id: {protein_id}")
        predictions[protein_id] = scores

    if not references:
        raise ValueError(f"No reference records found in {reference_path}")
    missing = sorted(set(references) - set(predictions))
    extra = sorted(set(predictions) - set(references))
    if missing or extra:
        raise ValueError(
            "Reference/prediction ID mismatch: "
            f"missing_predictions={missing[:10]}, extra_predictions={extra[:10]}"
        )

    records: list[ResiduePrediction] = []
    for protein_id in reference_order:
        sequence, labels = references[protein_id]
        records.append(
            ResiduePrediction(
                protein_id=protein_id,
                sequence=sequence,
                labels=tuple(labels),
                scores=tuple(predictions[protein_id]),
            )
        )
    return records
