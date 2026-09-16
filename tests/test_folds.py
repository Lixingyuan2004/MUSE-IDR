from __future__ import annotations

import unittest

from muse_idr.data.folds import (
    HomologyCluster,
    assign_balanced_folds,
    build_homology_clusters,
    summarize_folds,
)
from muse_idr.data.leakage import HomologyHit


def record(identifier: str, labels: list[int]) -> dict[str, object]:
    return {"protein_id": identifier, "labels": labels}


class HomologyFoldTests(unittest.TestCase):
    def test_connected_components_are_transitive_and_identity_is_strict(self) -> None:
        records = [record("a", [1]), record("b", [0]), record("c", [1]), record("d", [0])]
        hits = [
            HomologyHit("a", "b", 0.31, 8, 10, 10, 0.8, 0.8),
            HomologyHit("b", "c", 0.40, 9, 10, 10, 0.9, 0.9),
            HomologyHit("c", "d", 0.30, 10, 10, 10, 1.0, 1.0),
        ]
        clusters, edges = build_homology_clusters(records, hits)
        members = {cluster.members for cluster in clusters}
        self.assertEqual(members, {("a", "b", "c"), ("d",)})
        self.assertEqual(edges, {("a", "b"), ("b", "c")})

    def test_balanced_assignment_is_deterministic_and_never_splits_cluster(self) -> None:
        clusters = [
            HomologyCluster(f"HC{index:04d}", (f"p{index}",), 10 + index, 20 - index, 5)
            for index in range(1, 11)
        ]
        first = assign_balanced_folds(clusters, folds=5, seed=17)
        second = assign_balanced_folds(clusters, folds=5, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {cluster.cluster_id for cluster in clusters})
        summaries = summarize_folds(clusters, first, folds=5)
        self.assertTrue(all(summary["positive_residues"] > 0 for summary in summaries))
        self.assertTrue(all(summary["negative_residues"] > 0 for summary in summaries))


if __name__ == "__main__":
    unittest.main()
