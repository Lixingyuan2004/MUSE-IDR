"""Homology-connected components and deterministic balanced fold assignment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from .leakage import HomologyHit, is_forbidden_homolog


@dataclass(frozen=True)
class HomologyCluster:
    cluster_id: str
    members: tuple[str, ...]
    positive_residues: int
    negative_residues: int
    unknown_residues: int

    @property
    def sequences(self) -> int:
        return len(self.members)

    @property
    def total_residues(self) -> int:
        return self.positive_residues + self.negative_residues + self.unknown_residues


def build_homology_clusters(
    records: Iterable[dict[str, object]], hits: Iterable[HomologyHit]
) -> tuple[list[HomologyCluster], set[tuple[str, str]]]:
    """Build connected components from all forbidden, undirected homology edges."""

    rows = list(records)
    identifiers = [str(row["protein_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Training protein identifiers must be unique")
    known = set(identifiers)
    parent = {identifier: identifier for identifier in identifiers}

    def root(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(left: str, right: str) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    edges: set[tuple[str, str]] = set()
    for hit in hits:
        if not is_forbidden_homolog(hit) or hit.query == hit.target:
            continue
        if hit.query not in known or hit.target not in known:
            raise ValueError(f"Homology edge references an unknown protein: {hit.query}, {hit.target}")
        edge = tuple(sorted((hit.query, hit.target)))
        edges.add(edge)
        union(*edge)

    components: dict[str, list[str]] = {}
    for identifier in identifiers:
        components.setdefault(root(identifier), []).append(identifier)
    member_groups = sorted(tuple(sorted(members)) for members in components.values())
    rows_by_id = {str(row["protein_id"]): row for row in rows}
    clusters: list[HomologyCluster] = []
    for index, members in enumerate(member_groups, start=1):
        labels = [label for member in members for label in rows_by_id[member]["labels"]]
        clusters.append(
            HomologyCluster(
                cluster_id=f"HC{index:04d}",
                members=members,
                positive_residues=labels.count(1),
                negative_residues=labels.count(0),
                unknown_residues=labels.count(-1),
            )
        )
    return clusters, edges


def assign_balanced_folds(
    clusters: Iterable[HomologyCluster], folds: int = 5, seed: int = 17
) -> dict[str, int]:
    """Greedily balance class residues, protein count and compute load without splitting clusters."""

    groups = list(clusters)
    if folds < 2 or len(groups) < folds:
        raise ValueError("Need at least one homology cluster per fold")
    totals = {
        "positive": sum(group.positive_residues for group in groups),
        "negative": sum(group.negative_residues for group in groups),
        "sequences": sum(group.sequences for group in groups),
        "residues": sum(group.total_residues for group in groups),
    }
    if totals["positive"] == 0 or totals["negative"] == 0:
        raise ValueError("Every fold split requires both positive and negative training labels")
    targets = {name: value / folds for name, value in totals.items()}
    weights = {"positive": 1.0, "negative": 1.0, "sequences": 0.2, "residues": 0.2}
    loads = [dict.fromkeys(totals, 0) for _ in range(folds)]
    rng = random.Random(seed)
    random_ties = {group.cluster_id: rng.random() for group in groups}

    def values(group: HomologyCluster) -> dict[str, int]:
        return {
            "positive": group.positive_residues,
            "negative": group.negative_residues,
            "sequences": group.sequences,
            "residues": group.total_residues,
        }

    def size_key(group: HomologyCluster) -> tuple[float, int, float, str]:
        group_values = values(group)
        relative_load = max(group_values[name] / targets[name] for name in targets)
        return (-relative_load, -group.total_residues, random_ties[group.cluster_id], group.cluster_id)

    def global_cost(candidate: HomologyCluster, candidate_fold: int) -> float:
        candidate_values = values(candidate)
        cost = 0.0
        for fold_index, fold_load in enumerate(loads):
            for name, target in targets.items():
                value = fold_load[name]
                if fold_index == candidate_fold:
                    value += candidate_values[name]
                cost += weights[name] * ((value - target) / target) ** 2
        return cost

    assignments: dict[str, int] = {}
    for group in sorted(groups, key=size_key):
        fold_ties = list(range(folds))
        rng.shuffle(fold_ties)
        tie_rank = {fold: rank for rank, fold in enumerate(fold_ties)}
        chosen = min(
            range(folds),
            key=lambda fold: (
                global_cost(group, fold),
                loads[fold]["sequences"],
                tie_rank[fold],
            ),
        )
        assignments[group.cluster_id] = chosen
        group_values = values(group)
        for name in loads[chosen]:
            loads[chosen][name] += group_values[name]
    return assignments


def summarize_folds(
    clusters: Iterable[HomologyCluster], assignments: dict[str, int], folds: int
) -> list[dict[str, int]]:
    """Return deterministic per-fold counts for manifests and validation."""

    summaries = [
        {
            "fold": fold,
            "clusters": 0,
            "sequences": 0,
            "positive_residues": 0,
            "negative_residues": 0,
            "unknown_residues": 0,
            "total_residues": 0,
        }
        for fold in range(folds)
    ]
    for cluster in clusters:
        fold = assignments[cluster.cluster_id]
        summary = summaries[fold]
        summary["clusters"] += 1
        summary["sequences"] += cluster.sequences
        summary["positive_residues"] += cluster.positive_residues
        summary["negative_residues"] += cluster.negative_residues
        summary["unknown_residues"] += cluster.unknown_residues
        summary["total_residues"] += cluster.total_residues
    return summaries
