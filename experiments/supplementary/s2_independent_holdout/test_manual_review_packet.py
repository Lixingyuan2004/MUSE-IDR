"""Unit tests for the S2 manual positive-region review packet."""
from __future__ import annotations

import json
import unittest

from prepare_manual_review_packet import (
    compact_statements,
    compact_text,
    evidence_fingerprint,
    parse_bool,
)


class ManualReviewPacketTests(unittest.TestCase):
    def test_parse_bool_is_strict(self) -> None:
        self.assertTrue(parse_bool("True"))
        self.assertFalse(parse_bool("false"))
        with self.assertRaises(ValueError):
            parse_bool("maybe")

    def test_compact_text_removes_layout_whitespace(self) -> None:
        self.assertEqual(compact_text("  direct\n  evidence\t here "), "direct evidence here")

    def test_statements_are_compact_json(self) -> None:
        encoded = compact_statements([{"type": "Results", "text": "a\n b"}])
        self.assertEqual(json.loads(encoded), [{"type": "Results", "text": "a b"}])

    def test_fingerprint_is_order_independent(self) -> None:
        self.assertEqual(evidence_fingerprint({"a": 1, "b": 2}), evidence_fingerprint({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
