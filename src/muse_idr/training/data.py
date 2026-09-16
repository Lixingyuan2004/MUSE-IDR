"""Strict fold-aware loading and length-efficient batching for residue models."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_TO_TOKEN = {amino_acid: index + 1 for index, amino_acid in enumerate(STANDARD_AMINO_ACIDS)}
UNKNOWN_TOKEN = len(AMINO_TO_TOKEN) + 1
VOCAB_SIZE = UNKNOWN_TOKEN + 1


@dataclass(frozen=True)
class ProteinExample:
    protein_id: str
    sequence: str
    labels: tuple[int, ...]
    fold: int

    @property
    def length(self) -> int:
        return len(self.sequence)


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def encode_sequence(sequence: str) -> tuple[int, ...]:
    return tuple(AMINO_TO_TOKEN.get(amino_acid, UNKNOWN_TOKEN) for amino_acid in sequence)


def load_training_examples(dataset_path: Path, assignments_path: Path) -> list[ProteinExample]:
    """Load the frozen dataset and require an exact one-to-one fold assignment."""

    records = _jsonl(dataset_path)
    assignments = _jsonl(assignments_path)
    assignment_by_id: dict[str, dict[str, object]] = {}
    for assignment in assignments:
        identifier = str(assignment.get("protein_id", ""))
        if not identifier or identifier in assignment_by_id:
            raise ValueError("Fold assignment protein IDs must be non-empty and unique")
        assignment_by_id[identifier] = assignment

    examples: list[ProteinExample] = []
    seen: set[str] = set()
    for record in records:
        identifier = str(record.get("protein_id", ""))
        sequence = str(record.get("sequence", ""))
        labels = tuple(record.get("labels", ()))
        if not identifier or identifier in seen or identifier not in assignment_by_id:
            raise ValueError(f"Missing, duplicate or unmatched training ID: {identifier!r}")
        seen.add(identifier)
        assignment = assignment_by_id[identifier]
        fold = int(assignment["fold"])
        if fold not in {0, 1, 2, 3, 4}:
            raise ValueError(f"Invalid fold for {identifier}: {fold}")
        if len(sequence) != len(labels) or not sequence:
            raise ValueError(f"Sequence/label length mismatch for {identifier}")
        if not set(labels) <= {-1, 0, 1}:
            raise ValueError(f"Unexpected residue label for {identifier}")
        if int(assignment["sequence_length"]) != len(sequence):
            raise ValueError(f"Assignment sequence length changed for {identifier}")
        observed_counts = {
            "positive_residues": labels.count(1),
            "negative_residues": labels.count(0),
            "unknown_residues": labels.count(-1),
        }
        if any(int(assignment[name]) != value for name, value in observed_counts.items()):
            raise ValueError(f"Assignment label counts changed for {identifier}")
        examples.append(ProteinExample(identifier, sequence, labels, fold))
    if seen != set(assignment_by_id):
        raise ValueError("Fold assignments contain proteins absent from the training dataset")
    return examples


def collate_proteins(examples: list[ProteinExample]) -> dict[str, object]:
    if not examples:
        raise ValueError("Cannot collate an empty protein batch")
    maximum_length = max(example.length for example in examples)
    tokens = torch.zeros((len(examples), maximum_length), dtype=torch.long)
    labels = torch.full((len(examples), maximum_length), -1, dtype=torch.int8)
    lengths = torch.tensor([example.length for example in examples], dtype=torch.long)
    for index, example in enumerate(examples):
        tokens[index, : example.length] = torch.tensor(encode_sequence(example.sequence))
        labels[index, : example.length] = torch.tensor(example.labels, dtype=torch.int8)
    return {
        "protein_ids": [example.protein_id for example in examples],
        "sequences": [example.sequence for example in examples],
        "tokens": tokens,
        "labels": labels,
        "lengths": lengths,
    }


def make_token_budget_batches(
    examples: list[ProteinExample],
    indices: Iterable[int],
    *,
    token_budget: int,
    max_batch_size: int,
    shuffle: bool,
    seed: int,
    bucket_size: int = 64,
) -> list[list[int]]:
    """Pack similar lengths while bounding padded residues per batch."""

    if token_budget < 1 or max_batch_size < 1 or bucket_size < 1:
        raise ValueError("Batching limits must be positive")
    ordered = list(indices)
    if len(ordered) != len(set(ordered)) or any(not 0 <= index < len(examples) for index in ordered):
        raise ValueError("Batch indices must be unique and within the dataset")
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(ordered)
        buckets = [ordered[start : start + bucket_size] for start in range(0, len(ordered), bucket_size)]
        for bucket in buckets:
            bucket.sort(key=lambda index: examples[index].length)
        rng.shuffle(buckets)
        ordered = [index for bucket in buckets for index in bucket]
    else:
        ordered.sort(key=lambda index: examples[index].length)

    batches: list[list[int]] = []
    current: list[int] = []
    current_maximum = 0
    for index in ordered:
        proposed_maximum = max(current_maximum, examples[index].length)
        proposed_size = len(current) + 1
        if current and (
            proposed_size > max_batch_size or proposed_maximum * proposed_size > token_budget
        ):
            batches.append(current)
            current = []
            current_maximum = 0
        current.append(index)
        current_maximum = max(current_maximum, examples[index].length)
    if current:
        batches.append(current)
    if sorted(index for batch in batches for index in batch) != sorted(ordered):
        raise ValueError("Token-budget batching lost or duplicated examples")
    return batches
