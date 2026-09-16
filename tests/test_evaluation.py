from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from muse_idr.evaluation import (
    ResiduePrediction,
    evaluate_predictions,
    load_aligned_jsonl,
    roc_auc_score,
    validate_and_order_oof_records,
)


def brute_force_auc(labels: list[int], scores: list[float]) -> float:
    positives = [scores[index] for index, label in enumerate(labels) if label == 1]
    negatives = [scores[index] for index, label in enumerate(labels) if label == 0]
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += float(positive > negative) + 0.5 * float(positive == negative)
    return favorable / (len(positives) * len(negatives))


class EvaluationTests(unittest.TestCase):
    def test_auc_perfect_reversed_and_ties(self) -> None:
        self.assertEqual(roc_auc_score([0, 1], [0.1, 0.9]), 1.0)
        self.assertEqual(roc_auc_score([0, 1], [0.9, 0.1]), 0.0)
        self.assertEqual(roc_auc_score([0, 1], [0.5, 0.5]), 0.5)

    def test_auc_matches_pairwise_definition_with_ties(self) -> None:
        labels = [1, 0, 1, 0, 1, 0]
        scores = [0.8, 0.8, 0.2, 0.1, 0.5, 0.5]
        self.assertAlmostEqual(roc_auc_score(labels, scores), brute_force_auc(labels, scores))

    def test_auc_matches_pairwise_definition_on_random_small_cases(self) -> None:
        rng = random.Random(1729)
        for _ in range(50):
            labels = [0, 1] + [rng.randrange(2) for _ in range(8)]
            rng.shuffle(labels)
            scores = [rng.randrange(5) / 4.0 for _ in labels]
            self.assertAlmostEqual(
                roc_auc_score(labels, scores),
                brute_force_auc(labels, scores),
            )

    def test_unknown_mask_and_macro_exclusion(self) -> None:
        records = [
            ResiduePrediction("P1", "ACDE", (0, 1, -1, 1), (0.1, 0.9, 0.4, 0.8)),
            ResiduePrediction("P2", "FG", (0, 0), (0.2, 0.3)),
        ]
        result = evaluate_predictions(records, threshold=0.5)
        self.assertEqual(result["counts"]["residues_labeled"], 5)
        self.assertEqual(result["counts"]["residues_unknown_masked"], 1)
        self.assertEqual(result["metrics"]["residue_micro_roc_auc"], 1.0)
        macro = result["metrics"]["protein_macro_roc_auc"]
        self.assertEqual(macro["value"], 1.0)
        self.assertEqual(macro["proteins_excluded_single_class"], 1)

    def test_threshold_metrics_are_predeclared_not_optimized(self) -> None:
        record = ResiduePrediction("P1", "ACDE", (0, 1, 0, 1), (0.2, 0.9, 0.6, 0.4))
        result = evaluate_predictions([record], threshold=0.5)
        threshold = result["metrics"]["threshold_metrics"]
        self.assertEqual(threshold["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(threshold["f1"], 0.5)
        self.assertEqual(threshold["mcc"], 0.0)

    def test_oof_assembly_requires_exact_assigned_fold(self) -> None:
        first = ResiduePrediction("p1", "AC", (1, 0), (0.9, 0.1))
        second = ResiduePrediction("p2", "GG", (0, 1), (0.2, 0.8))
        ordered = validate_and_order_oof_records(
            {0: [second], 1: [first]}, {"p1": 1, "p2": 0}
        )
        self.assertEqual([record.protein_id for record in ordered], ["p1", "p2"])
        with self.assertRaisesRegex(ValueError, "came from fold"):
            validate_and_order_oof_records(
                {0: [first], 1: [second]}, {"p1": 1, "p2": 0}
            )

    def test_protein_bootstrap_is_deterministic(self) -> None:
        records = [
            ResiduePrediction("P1", "AC", (0, 1), (0.1, 0.9)),
            ResiduePrediction("P2", "DE", (0, 1), (0.2, 0.8)),
            ResiduePrediction("P3", "FG", (-1, -1), (0.4, 0.6)),
        ]
        first = evaluate_predictions(records, bootstrap_replicates=20, bootstrap_seed=7)
        second = evaluate_predictions(records, bootstrap_replicates=20, bootstrap_seed=7)
        self.assertEqual(first["uncertainty"], second["uncertainty"])
        interval = first["uncertainty"]["residue_micro_roc_auc"]
        self.assertEqual(interval["replicates_valid"], 20)
        self.assertEqual(interval["lower"], 1.0)
        self.assertEqual(interval["upper"], 1.0)

    def test_reject_single_class_auc_and_nonfinite_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive and one negative"):
            roc_auc_score([1, 1], [0.2, 0.8])
        with self.assertRaisesRegex(ValueError, "outside"):
            ResiduePrediction("P1", "AC", (0, 1), (0.1, math.nan))

    def test_jsonl_alignment_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            prediction = root / "prediction.jsonl"
            reference.write_text(
                json.dumps({"protein_id": "P1", "sequence": "AC", "labels": [0, 1]}) + "\n",
                encoding="utf-8",
            )
            prediction.write_text(
                json.dumps({"protein_id": "P2", "scores": [0.1, 0.9]}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ID mismatch"):
                load_aligned_jsonl(reference, prediction)


if __name__ == "__main__":
    unittest.main()
