"""Exact, identifier and homology leakage checks for candidate training data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .caid import SequenceRecord, parse_fasta, sequence_sha256

ISOFORM_SUFFIX_RE = re.compile(r"-\d+$")
TOKEN_SPLIT_RE = re.compile(r"[|\s,;]+")


@dataclass(frozen=True)
class HomologyHit:
    """One candidate-to-holdout alignment in an explicit fraction format."""

    query: str
    target: str
    fraction_identity: float
    alignment_length: int
    query_length: int
    target_length: int
    explicit_query_coverage: float | None = None
    explicit_target_coverage: float | None = None

    @property
    def query_coverage(self) -> float:
        if self.explicit_query_coverage is not None:
            return self.explicit_query_coverage
        return self.alignment_length / self.query_length

    @property
    def target_coverage(self) -> float:
        if self.explicit_target_coverage is not None:
            return self.explicit_target_coverage
        return self.alignment_length / self.target_length


def canonical_isoform(identifier: str) -> str:
    """Collapse an accession isoform suffix such as P12345-2 to P12345."""

    return ISOFORM_SUFFIX_RE.sub("", identifier.strip().upper())


def identifier_aliases(header: str) -> set[str]:
    """Create conservative exact-token and isoform-parent identifier aliases."""

    tokens = {token.upper() for token in TOKEN_SPLIT_RE.split(header.strip()) if token}
    aliases = set(tokens)
    aliases.update(canonical_isoform(token) for token in tokens)
    return aliases


def sentinel_sets(manifest_path: Path) -> tuple[set[str], set[str]]:
    """Load only hashes and identifier aliases from a label-free manifest."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("labels_included") is not False:
        raise ValueError("Refusing a holdout manifest that does not explicitly omit labels")
    hashes: set[str] = set()
    aliases: set[str] = set()
    for target in manifest.get("targets", []):
        hashes.add(str(target["sequence_sha256"]).lower())
        for identifier in target.get("reference_ids", []):
            aliases.update(identifier_aliases(str(identifier)))
        for accession in target.get("uniprot_accessions", []):
            aliases.update(identifier_aliases(str(accession)))
    return hashes, aliases


def audit_records(
    records: Iterable[SequenceRecord], manifest_path: Path
) -> list[dict[str, object]]:
    """Report exact-sequence and identifier leaks without silently removing them."""

    holdout_hashes, holdout_aliases = sentinel_sets(manifest_path)
    findings: list[dict[str, object]] = []
    for record in records:
        digest = sequence_sha256(record.sequence)
        reasons: list[str] = []
        if digest in holdout_hashes:
            reasons.append("exact_sequence_sha256")
        overlap = sorted(identifier_aliases(record.header) & holdout_aliases)
        if overlap:
            reasons.append("identifier_or_isoform")
        if reasons:
            findings.append(
                {
                    "candidate_id": record.identifier,
                    "sequence_sha256": digest,
                    "reasons": reasons,
                    "matched_identifier_aliases": overlap,
                }
            )
    return findings


def parse_homology_tsv(path: Path) -> list[HomologyHit]:
    """Parse MMseqs homology output with explicit bidirectional coverage.

    The canonical eight columns are
    query,target,fident,alnlen,qlen,tlen,qcov,tcov. Both identities and
    coverages are fractions in [0, 1]. Legacy six-column, gap-free fixtures
    remain supported, but formal MMseqs audits must include qcov and tcov
    because alnlen counts alignment columns and can include gaps.
    """

    hits: list[HomologyHit] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) not in {6, 8}:
                raise ValueError(f"Expected 6 or 8 TSV columns at {path}:{line_number}")
            explicit_query_coverage = float(fields[6]) if len(fields) == 8 else None
            explicit_target_coverage = float(fields[7]) if len(fields) == 8 else None
            hit = HomologyHit(
                query=fields[0],
                target=fields[1],
                fraction_identity=float(fields[2]),
                alignment_length=int(fields[3]),
                query_length=int(fields[4]),
                target_length=int(fields[5]),
                explicit_query_coverage=explicit_query_coverage,
                explicit_target_coverage=explicit_target_coverage,
            )
            if not 0.0 <= hit.fraction_identity <= 1.0:
                raise ValueError(f"fident must be a fraction at {path}:{line_number}")
            if min(hit.alignment_length, hit.query_length, hit.target_length) <= 0:
                raise ValueError(f"Lengths must be positive at {path}:{line_number}")
            if len(fields) == 6 and (
                hit.alignment_length > hit.query_length
                or hit.alignment_length > hit.target_length
            ):
                raise ValueError(
                    "Six-column coverage is invalid for a gapped alignment at "
                    f"{path}:{line_number}; add qcov and tcov"
                )
            if not 0.0 <= hit.query_coverage <= 1.0:
                raise ValueError(f"qcov must be a fraction at {path}:{line_number}")
            if not 0.0 <= hit.target_coverage <= 1.0:
                raise ValueError(f"tcov must be a fraction at {path}:{line_number}")
            hits.append(hit)
    return hits


def is_forbidden_homolog(
    hit: HomologyHit,
    identity_threshold: float = 0.30,
    query_coverage_threshold: float = 0.80,
    target_coverage_threshold: float = 0.80,
) -> bool:
    """Apply the predeclared strict identity and bidirectional coverage rule."""

    return (
        hit.fraction_identity > identity_threshold
        and hit.query_coverage >= query_coverage_threshold
        and hit.target_coverage >= target_coverage_threshold
    )
