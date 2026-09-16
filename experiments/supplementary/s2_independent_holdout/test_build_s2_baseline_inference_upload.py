"""Tests for the deterministic, label-free S2 baseline upload builder."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_s2_baseline_inference_upload import (
    assert_no_forbidden_payload,
    sha256,
    write_deterministic_tar_gz,
)


class BuildS2BaselineUploadTests(unittest.TestCase):
    def test_label_file_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_forbidden_payload([Path("data/s2_labels.tsv.gz")])

    def test_previous_a10_result_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_forbidden_payload([Path("outputs/a10_prediction_20260907/x.tsv.gz")])

    def test_sequence_only_fasta_is_allowed(self) -> None:
        assert_no_forbidden_payload([Path("data/sequences_only/s2_sequences.fasta")])

    def test_archive_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "fixed.txt").write_text("fixed\n", encoding="utf-8")
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"
            write_deterministic_tar_gz(source, first)
            write_deterministic_tar_gz(source, second)
            self.assertEqual(sha256(first), sha256(second))


if __name__ == "__main__":
    unittest.main()
