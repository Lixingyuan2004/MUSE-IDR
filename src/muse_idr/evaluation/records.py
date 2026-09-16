"""Validated records used by every model evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

from muse_idr.data.caid import normalize_sequence


@dataclass(frozen=True)
class ResiduePrediction:
    """One protein with canonical labels and one universal IDR score vector."""

    protein_id: str
    sequence: str
    labels: tuple[int, ...]
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.protein_id, str) or not isinstance(self.sequence, str):
            raise ValueError("protein_id and sequence must be strings")
        protein_id = self.protein_id.strip()
        if not protein_id:
            raise ValueError("protein_id is empty")
        sequence = normalize_sequence(self.sequence)
        labels = tuple(self.labels)
        scores = tuple(self.scores)
        if len(sequence) != len(labels) or len(sequence) != len(scores):
            raise ValueError(
                f"Length mismatch for {protein_id}: sequence={len(sequence)}, "
                f"labels={len(labels)}, scores={len(scores)}"
            )

        normalized_labels: list[int] = []
        for index, label in enumerate(labels, start=1):
            if (
                isinstance(label, bool)
                or not isinstance(label, Integral)
                or int(label) not in {-1, 0, 1}
            ):
                raise ValueError(f"Invalid label for {protein_id} at residue {index}: {label!r}")
            normalized_labels.append(int(label))

        normalized_scores: list[float] = []
        for index, score in enumerate(scores, start=1):
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError(
                    f"Non-numeric score for {protein_id} at residue {index}: {score!r}"
                )
            value = float(score)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Score outside [0, 1] for {protein_id} at residue {index}: {score!r}"
                )
            normalized_scores.append(value)

        object.__setattr__(self, "protein_id", protein_id)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "labels", tuple(normalized_labels))
        object.__setattr__(self, "scores", tuple(normalized_scores))
