"""Safety and alignment tests for the label-blind S2 baseline driver."""
from __future__ import annotations

import csv
import gzip
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from run_s2_baseline_predictions import (
    LORA_ADAPTER_WEIGHTS_SHA256,
    LORA_MODEL_ID,
    LORA_OFFLINE_WRAPPER,
    PINNED_FILES,
    PROTOCOL,
    PROTOCOL_SHA256,
    ROOT,
    S2_FASTA,
    read_label_free_fasta,
    sha256,
    validate_prediction_tsv,
    verify_protocol,
)


class S2BaselinePredictionTests(unittest.TestCase):
    def test_frozen_inputs_match_hashes_and_counts(self) -> None:
        self.assertEqual(sha256(ROOT / PROTOCOL), PROTOCOL_SHA256)
        self.assertEqual(
            sha256(ROOT / S2_FASTA),
            PINNED_FILES[S2_FASTA.as_posix()],
        )
        records = read_label_free_fasta(ROOT / S2_FASTA)
        self.assertEqual(len(records), 87)
        self.assertEqual(sum(len(sequence) for _, sequence in records), 56394)

    def test_only_two_formal_comparators_are_frozen(self) -> None:
        protocol = verify_protocol()
        self.assertEqual(
            [row["name"] for row in protocol["formal_comparators"]],
            [
                "PUNCH2-Light Released-13",
                "LoRA-DR-Suite ESM2-650M DisProt7",
            ],
        )
        self.assertEqual(LORA_MODEL_ID, "CQSB/esm2_650M-LoRA-ID-DisProt7")

    def test_lora_loader_is_pinned_and_network_free(self) -> None:
        source = (ROOT / LORA_OFFLINE_WRAPPER).read_text(encoding="utf-8")
        self.assertIn("local_files_only=True", source)
        self.assertNotIn("HfApi", source)
        self.assertNotIn("model_info(", source)
        self.assertEqual(
            LORA_ADAPTER_WEIGHTS_SHA256,
            "02f1e288f36f369786db68df5270affa693d2359a5badb0275c28e81977448fe",
        )

    def test_driver_exposes_no_label_or_reference_argument(self) -> None:
        module = __import__("run_s2_baseline_predictions")
        source = inspect.getsource(module.parse_args)
        self.assertNotIn("--labels", source)
        self.assertNotIn("--reference", source)

    def test_prediction_alignment_and_range(self) -> None:
        records = [("p1", "AC"), ("p2", "G")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.tsv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["protein_id", "position", "residue", "idr_probability"])
                writer.writerow(["p1", 1, "A", "0.1"])
                writer.writerow(["p1", 2, "C", "0.5"])
                writer.writerow(["p2", 1, "G", "0.9"])
            report = validate_prediction_tsv(path, records, "position")
            self.assertEqual(report["prediction_rows"], 3)
            self.assertTrue(report["complete_coverage"])

    def test_tampered_alignment_is_rejected(self) -> None:
        records = [("p1", "AC")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.tsv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["protein_id", "residue_index", "residue", "idr_probability"])
                writer.writerow(["p1", 1, "A", "0.1"])
                writer.writerow(["p1", 2, "G", "0.5"])
            with self.assertRaises(ValueError):
                validate_prediction_tsv(path, records, "residue_index")

    def test_protocol_guardrails_are_false_for_forbidden_actions(self) -> None:
        protocol = json.loads((ROOT / PROTOCOL).read_text(encoding="utf-8"))
        guardrails = protocol["guardrails"]
        self.assertFalse(guardrails["s2_labels_sent_to_inference_host"])
        self.assertFalse(guardrails["baseline_training_or_fine_tuning_on_s2"])
        self.assertFalse(guardrails["a10_reselection_or_modification"])


if __name__ == "__main__":
    unittest.main()
