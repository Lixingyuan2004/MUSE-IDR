import unittest

import numpy as np

from data import ProteinRecord, ProteinWindowDataset, window_starts


class WindowingTests(unittest.TestCase):
    def test_boundary_lengths(self):
        self.assertEqual(window_starts(1022, 1022, 511), [0])
        self.assertEqual(window_starts(1023, 1022, 511), [0, 1])
        self.assertEqual(window_starts(1500, 1022, 511), [0, 478])

    def test_every_residue_has_total_loss_weight_one(self):
        length = 1500
        record = ProteinRecord(
            protein_id="long",
            sequence="A" * length,
            labels=np.zeros(length, dtype=np.int8),
            fold="A",
        )
        dataset = ProteinWindowDataset([record], max_residues=1022, stride=511)
        accumulated = np.zeros(length, dtype=np.float32)
        for example in dataset:
            accumulated[example["start"] : example["end"]] += example["loss_weight"]
            self.assertTrue(np.all(example["merge_weight"] > 0))
        np.testing.assert_allclose(accumulated, np.ones(length), rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
