from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from muse_idr.models import LightIDRCNN
from muse_idr.training import (
    collate_proteins,
    cosine_with_warmup_factor,
    load_training_examples,
    make_token_budget_batches,
    masked_bce_with_logits,
)
from muse_idr.training.data import ProteinExample, VOCAB_SIZE


class TrainingPipelineTests(unittest.TestCase):
    def test_strict_fold_loading_and_token_budget_batching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            assignments = root / "assignments.jsonl"
            records = [
                {"protein_id": "p1", "sequence": "ACD", "labels": [1, -1, 0]},
                {"protein_id": "p2", "sequence": "GGGGG", "labels": [0, 0, 1, 1, -1]},
            ]
            fold_rows = [
                {
                    "protein_id": "p1",
                    "fold": 0,
                    "sequence_length": 3,
                    "positive_residues": 1,
                    "negative_residues": 1,
                    "unknown_residues": 1,
                },
                {
                    "protein_id": "p2",
                    "fold": 1,
                    "sequence_length": 5,
                    "positive_residues": 2,
                    "negative_residues": 2,
                    "unknown_residues": 1,
                },
            ]
            dataset.write_text("".join(json.dumps(row) + "\n" for row in records))
            assignments.write_text("".join(json.dumps(row) + "\n" for row in fold_rows))
            examples = load_training_examples(dataset, assignments)
        batches = make_token_budget_batches(
            examples,
            [0, 1],
            token_budget=5,
            max_batch_size=2,
            shuffle=False,
            seed=17,
        )
        self.assertEqual(batches, [[0], [1]])

    def test_unknown_and_padding_are_excluded_from_loss(self) -> None:
        batch = collate_proteins(
            [
                ProteinExample("p1", "AC", (1, -1), 0),
                ProteinExample("p2", "D", (0,), 1),
            ]
        )
        logits = torch.zeros_like(batch["labels"], dtype=torch.float32)
        loss, known = masked_bce_with_logits(logits, batch["labels"], torch.tensor(1.0))
        self.assertEqual(known, 2)
        self.assertAlmostEqual(float(loss), 0.693147, places=5)

    def test_model_has_one_logit_per_residue(self) -> None:
        model = LightIDRCNN(vocab_size=VOCAB_SIZE, hidden_size=16, dilations=(1, 2))
        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(model.idr_head.out_features, 1)

    def test_cosine_schedule_warms_up_and_decays_to_floor(self) -> None:
        self.assertAlmostEqual(
            cosine_with_warmup_factor(
                0, total_steps=100, warmup_steps=10, min_factor=0.05
            ),
            0.1,
        )
        self.assertAlmostEqual(
            cosine_with_warmup_factor(
                10, total_steps=100, warmup_steps=10, min_factor=0.05
            ),
            1.0,
        )
        self.assertAlmostEqual(
            cosine_with_warmup_factor(
                100, total_steps=100, warmup_steps=10, min_factor=0.05
            ),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
