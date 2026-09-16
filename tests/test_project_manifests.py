from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectManifestTests(unittest.TestCase):
    def test_external_test_sentinel_protects_only_caid2_and_caid3(self) -> None:
        path = PROJECT_ROOT / "data/manifests/holdouts/caid23_test_sentinel.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertIs(manifest["labels_included"], False)
        self.assertEqual(manifest["protected_editions"], ["caid2", "caid3"])
        self.assertEqual(manifest["summary"]["unique_sequences"], 666)
        for target in manifest["targets"]:
            editions = {source.split(":", 1)[0] for source in target["sources"]}
            self.assertTrue(editions <= {"caid2", "caid3"})
            self.assertNotIn("labels", target)

    def test_training_config_uses_external_test_sentinel(self) -> None:
        config = (PROJECT_ROOT / "configs/data/classic_idr.yaml").read_text(encoding="utf-8")
        self.assertIn("caid23_test_sentinel.json", config)
        self.assertNotIn("caid_all_rounds_sentinel.json", config)

    def test_historical_training_manifest_is_filtered_but_not_training_ready(self) -> None:
        path = (
            PROJECT_ROOT
            / "data/manifests/training/historical_disprot_2018_positive.json"
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        summary = manifest["summary"]
        self.assertEqual(summary["sequences_after_direct_filter"], 1346)
        self.assertEqual(summary["positive_residues_after_direct_filter"], 109364)
        self.assertEqual(summary["negative_residues_after_direct_filter"], 0)
        direct_filter = manifest["external_test_direct_filter"]
        self.assertEqual(len(direct_filter["findings"]), 7)
        self.assertEqual(direct_filter["post_filter_findings"], [])
        self.assertTrue(manifest["training_readiness"].startswith("blocked_"))

    def test_pdb_merged_manifest_has_balanced_labels_and_remains_blocked(self) -> None:
        path = PROJECT_ROOT / "data/manifests/training/classic_idr_2018_pdb_merged.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        summary = manifest["summary"]
        self.assertEqual(summary["trainable_sequences"], 1183)
        self.assertEqual(summary["positive_residues"], 109364)
        self.assertEqual(summary["negative_residues"], 100114)
        self.assertEqual(summary["post_merge_direct_leakage_findings"], 0)
        self.assertTrue(manifest["training_readiness"].startswith("blocked_"))


if __name__ == "__main__":
    unittest.main()
