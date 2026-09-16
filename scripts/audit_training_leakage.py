"""Audit candidate training FASTA against a label-free external-test sentinel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from muse_idr.data.caid import parse_fasta
from muse_idr.data.leakage import (
    audit_records,
    is_forbidden_homolog,
    parse_homology_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_fasta", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/holdouts/caid23_test_sentinel.json"),
    )
    parser.add_argument(
        "--homology-tsv",
        type=Path,
        help="Optional query,target,fident,alnlen,qlen,tlen,qcov,tcov TSV",
    )
    parser.add_argument(
        "--allow-exact-only",
        action="store_true",
        help="Allow a non-final smoke audit without homology search",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    findings = audit_records(parse_fasta(args.candidate_fasta), args.manifest)
    homologs: list[dict[str, object]] = []
    if args.homology_tsv:
        for hit in parse_homology_tsv(args.homology_tsv):
            if is_forbidden_homolog(hit):
                homologs.append(
                    {
                        "query": hit.query,
                        "target": hit.target,
                        "fraction_identity": hit.fraction_identity,
                        "query_coverage": hit.query_coverage,
                        "target_coverage": hit.target_coverage,
                    }
                )
    homology_missing = args.homology_tsv is None and not args.allow_exact_only
    report = {
        "status": "fail" if findings or homologs or homology_missing else "pass",
        "homology_audit": (
            "complete"
            if args.homology_tsv
            else "skipped_for_smoke"
            if args.allow_exact_only
            else "missing_required_input"
        ),
        "exact_or_identifier_findings": findings,
        "forbidden_homology_hits": homologs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "findings": len(findings) + len(homologs)}))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
