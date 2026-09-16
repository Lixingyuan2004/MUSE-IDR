"""Audit S2 source dates and sequence novelty before residue-label preparation.

This is a feasibility audit, not a final test-set builder or model evaluation.
DisProt regions/consensus are not inspected. All thresholds are sequence-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
RELEASES = ("2023_06", "2025_06", "2026_06")
CUTOFFS = ("2023_06", "2025_06")
PINS = {
    "data/caid_holdout/references/caid1/rebuild_sources/disprot-2018-11.json": "2062da33ae925e64daa2a0bb899ad6e911020d8caa55121fcf1cbed9e18db0e0",
    "data/raw/disprot/2023_06/disprot.json": "cbe1f6ff22545e0a7a14abf7d184218fbe830358aa03da2b4ca0e96df51cfe74",
    "data/raw/disprot/2025_06/disprot.json": "c2a259fb1fd211ba357ccff7775d362fd49557a8dd6b50a52a89416196cc7ef0",
    "data/raw/disprot/2026_06/disprot.json": "afed1664507df330c22ad5ddd59796af49851e1f9df3f88a7ec0fb87f37d0d9b",
    "data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl": "df6a4deee4a65d009f4faf41c6b8930f9db8600fa28b6711da7ff48ad2c70bf4",
    "data/processed/classic_idr_2018/trainable_caid23_filtered.jsonl": "124d05084015d659b4e0a0ff4d5a7d32ab1a8394ab4b51bf79d954c770f58be2",
    "data/manifests/holdouts/caid23_test_sentinel.json": "e149d1dea349953480d6d4944dd68c34075ccf9a8fc58abb9d8db52a1e3b066c",
    "data/caid_holdout/derived/caid23/targets.fasta": "bd8c40e75d1ce4ffd0c9d3950f351dad888d8418e648550355cb13bf06e75744",
}
MMSEQS_VERSION = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def accession_parent(value):
    return re.sub(r"-\d+$", "", value.strip().upper())


def metadata(entry):
    """Explicit allowlist: do not read regions, consensus or disorder_content."""
    identifier = str(entry["disprot_id"]).strip().upper()
    if not re.fullmatch(r"DP\d+", identifier):
        raise ValueError(f"Invalid DisProt ID: {identifier}")
    sequence = "".join(str(entry.get("sequence", "")).split()).upper()
    accession = str(entry.get("acc") or "").strip().upper()
    issues = []
    if not sequence or set(sequence) - ALPHABET:
        issues.append("invalid_or_empty_sequence")
    if len(sequence) != entry.get("length"):
        issues.append("declared_length_mismatch")
    if not accession:
        issues.append("missing_accession")
    for flag in ("obsolete", "ambiguous", "is_obsolete", "is_ambiguous"):
        if entry.get(flag) not in (None, False, 0, "false", "False", ""):
            issues.append(flag)
    return {
        "protein_id": identifier, "accession": accession,
        "sequence": sequence, "length": len(sequence),
        "sequence_sha256": sequence_hash(sequence) if sequence.isascii() else None,
        "entry_released": entry.get("released"),
        "entry_date": entry.get("date"),
        "name": str(entry.get("name", "")),
        "issues": issues,
    }


def read_snapshot(path):
    payload = read_json(path)
    entries = payload["data"]
    if len(entries) != payload["size"]:
        raise ValueError(f"Snapshot size mismatch: {path}")
    rows = [metadata(entry) for entry in entries]
    if len({row["protein_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate DisProt IDs: {path}")
    return rows


def index(rows):
    return {
        "ids": {r["protein_id"] for r in rows},
        "accessions": {accession_parent(r["accession"]) for r in rows if r["accession"]},
        "hashes": {r["sequence_sha256"] for r in rows if r["sequence_sha256"]},
    }


def match_reasons(row, reference):
    reasons = []
    if row["protein_id"] in reference["ids"]:
        reasons.append("identifier")
    if row["accession"] and accession_parent(row["accession"]) in reference["accessions"]:
        reasons.append("accession_or_isoform_parent")
    if row["sequence_sha256"] in reference["hashes"]:
        reasons.append("exact_sequence")
    return reasons


def screen(rows, older, training, caid, cutoff):
    records, seen_hashes, seen_accessions = [], set(), set()
    counts = Counter(entry_release_in_interval=0, direct_screen_pass_records=0,
                     unique_sequence_candidates=0)
    for row in sorted(rows, key=lambda r: r["protein_id"]):
        release = str(row["entry_released"] or "")
        if not re.fullmatch(r"\d{4}_(0[1-9]|1[0-2])", release) or release <= cutoff or release > "2026_06":
            continue
        counts["entry_release_in_interval"] += 1
        reasons = list(row["issues"])
        if "polyprotein" in row["name"].lower():
            reasons.append("polyprotein")
        for name, reference in (("older_snapshot", older), ("training", training), ("caid23", caid)):
            reasons.extend(f"{name}:{item}" for item in match_reasons(row, reference))
        if not reasons:
            counts["direct_screen_pass_records"] += 1
            parent = accession_parent(row["accession"])
            if row["sequence_sha256"] in seen_hashes or parent in seen_accessions:
                reasons.append("within_candidate_duplicate_sequence_or_accession")
            else:
                seen_hashes.add(row["sequence_sha256"])
                seen_accessions.add(parent)
                counts["unique_sequence_candidates"] += 1
        records.append({**row, "cutoff_release": cutoff, "exclusion_reasons": reasons})
    counts["excluded_records"] = len(records) - counts["unique_sequence_candidates"]
    return records, dict(counts)


def write_fasta(path, rows):
    Path(path).write_text("".join(f'>{row["protein_id"]}\n{row["sequence"]}\n' for row in rows), encoding="ascii", newline="\n")


def read_fasta(path):
    rows, identifier, parts = [], None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if identifier is not None:
                rows.append({"protein_id": identifier, "sequence": "".join(parts)})
            identifier, parts = line[1:].split()[0], []
        elif line.strip():
            parts.append(line.strip().upper())
    if identifier is not None:
        rows.append({"protein_id": identifier, "sequence": "".join(parts)})
    if len({r["protein_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate FASTA identifiers")
    return rows


def forbidden_hit(identity, query_coverage, target_coverage):
    if any(not 0 <= v <= 1 for v in (identity, query_coverage, target_coverage)):
        raise ValueError("Identity and coverage must be fractions in [0, 1]")
    return identity > 0.30 and query_coverage >= 0.80 and target_coverage >= 0.80


def search_homologs(executable, out, queries, targets):
    executable = Path(executable).resolve()
    version = subprocess.check_output([str(executable), "version"], text=True).strip()
    if version != MMSEQS_VERSION:
        raise ValueError(f"Unexpected MMseqs version: {version}")
    target_file = out / "training_and_caid23_sequences.fasta"
    query_file = out / "sequence_candidates.fasta"
    write_fasta(query_file, queries)
    write_fasta(target_file, targets)
    result = out / "candidate_vs_training_and_caid23.tsv"
    command = [str(executable), "easy-search", str(query_file), str(target_file), str(result),
               str(out / "mmseqs_tmp"), "--prefilter-mode", "2", "--max-seqs", str(len(targets)),
               "--alignment-mode", "3", "--min-seq-id", "0.30", "-c", "0.80", "--cov-mode", "0",
               "-e", "1000000", "--mask", "0", "--comp-bias-corr", "0", "--threads", "8",
               "--format-output", "query,target,fident,alnlen,qlen,tlen,qcov,tcov"]
    env = os.environ.copy()
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    write_json(out / "homology_command.json", {"command": command, "version": version, "executable_sha256": sha(executable)})
    with (out / "homology_search.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    hits = []
    query_lengths = {r["protein_id"]: len(r["sequence"]) for r in queries}
    target_lengths = {r["protein_id"]: len(r["sequence"]) for r in targets}
    for line in result.read_text(encoding="utf-8").splitlines():
        q, t, ident, aln, qlen, tlen, qcov, tcov = line.split("\t")
        if int(qlen) != query_lengths[q] or int(tlen) != target_lengths[t]:
            raise ValueError("Alignment lengths differ from source sequences")
        ident, qcov, tcov = map(float, (ident, qcov, tcov))
        if forbidden_hit(ident, qcov, tcov):
            hits.append({"query": q, "target": t, "identity": ident, "qcov": qcov, "tcov": tcov})
    write_json(out / "forbidden_homology_hits.json", hits)
    return {hit["query"] for hit in hits}, {
        "status": "pass", "version": version, "queries": len(queries), "targets": len(targets),
        "forbidden_pairs": len(hits), "queries_with_forbidden_hits": len({h["query"] for h in hits}),
        "identity_strictly_greater_than": 0.30, "bidirectional_coverage_at_least": 0.80,
        "tsv_sha256": sha(result), "short_local_homology_not_excluded_by_this_rule": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new audit directory: {out}")
    sources = []
    for relative, expected in PINS.items():
        path = ROOT / relative
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"Input hash mismatch: {relative}")
        sources.append({"path": relative, "sha256": actual, "bytes": path.stat().st_size})
    print(json.dumps({"verified_source_files": len(sources)}), flush=True)
    manifest = [json.loads(line) for line in (ROOT / "data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl").read_text().splitlines()]
    detailed = [json.loads(line) for line in (ROOT / "data/processed/classic_idr_2018/trainable_caid23_filtered.jsonl").read_text().splitlines()]
    by_id = {r["protein_id"]: r for r in detailed}
    if len(manifest) != 1133 or len(by_id) != 1133 or {r['id'] for r in manifest} != set(by_id):
        raise ValueError("Training protein set mismatch")
    labels = Counter()
    train_index = {"ids": set(), "accessions": set(), "hashes": set()}
    for record in manifest:
        original = by_id[record["id"]]
        if original["sequence"] != record["sequence"] or original["labels"] != record["labels"]:
            raise ValueError("Training manifest and provenance disagree")
        if len(record["sequence"]) != len(record["labels"]):
            raise ValueError("Training label length mismatch")
        labels.update(record["labels"])
        train_index["ids"].update(original["disprot_ids"])
        train_index["accessions"].update(accession_parent(a) for a in original["uniprot_accessions"])
        train_index["hashes"].add(sequence_hash(record["sequence"]))
    if dict(labels) != {1: 105036, 0: 97626, -1: 370759}:
        raise ValueError("Training label totals mismatch")
    sentinel = read_json(ROOT / "data/manifests/holdouts/caid23_test_sentinel.json")
    if sentinel["labels_included"] is not False:
        raise ValueError("Expected a label-free CAID sentinel")
    caid_index = {"ids": set(), "accessions": set(), "hashes": set()}
    for target in sentinel["targets"]:
        caid_index["ids"].update(target["reference_ids"])
        caid_index["accessions"].update(accession_parent(a) for a in target["uniprot_accessions"])
        caid_index["hashes"].add(target["sequence_sha256"])
    historical = read_snapshot(ROOT / "data/caid_holdout/references/caid1/rebuild_sources/disprot-2018-11.json")
    snapshots = {v: read_snapshot(ROOT / f"data/raw/disprot/{v}/disprot.json") for v in RELEASES}
    out.mkdir(parents=True)
    cohorts, records_by_cutoff = {}, {}
    for cutoff in CUTOFFS:
        older = historical + [r for v, rows in snapshots.items() if v <= cutoff for r in rows]
        screened, counts = screen(snapshots["2026_06"], index(older), train_index, caid_index, cutoff)
        records_by_cutoff[cutoff] = screened
        cohorts[cutoff] = counts
    union = {r["protein_id"]: r for rows in records_by_cutoff.values() for r in rows if not r["exclusion_reasons"]}
    queries = sorted(union.values(), key=lambda r: r["protein_id"])
    write_fasta(out / "sequence_candidates.fasta", queries)
    homology = {"status": "not_run"}
    forbidden = set()
    if args.mmseqs and queries:
        caid_fasta = read_fasta(ROOT / "data/caid_holdout/derived/caid23/targets.fasta")
        if {sequence_hash(r['sequence']) for r in caid_fasta} != caid_index["hashes"]:
            raise ValueError("CAID sentinel and label-free FASTA differ")
        targets = [{"protein_id": f'train__{r["id"]}', "sequence": r["sequence"]} for r in manifest]
        targets.extend({"protein_id": f'caid23__{r["protein_id"]}', "sequence": r["sequence"]} for r in caid_fasta)
        print(json.dumps({"homology_queries": len(queries), "homology_targets": len(targets)}), flush=True)
        forbidden, homology = search_homologs(args.mmseqs, out, queries, targets)
    for cutoff, rows in records_by_cutoff.items():
        for row in rows:
            if not row["exclusion_reasons"] and row["protein_id"] in forbidden:
                row["exclusion_reasons"].append("homology_to_training_or_caid23")
        survivors = [r for r in rows if not r["exclusion_reasons"]]
        cohorts[cutoff]["after_homology_sequences"] = len(survivors) if homology["status"] == "pass" else None
        cohorts[cutoff]["candidate_residues"] = sum(r["length"] for r in survivors)
        cohorts[cutoff]["longest_candidate"] = max((r["length"] for r in survivors), default=0)
        cohorts[cutoff]["exclusion_reason_counts"] = dict(Counter(reason for r in rows for reason in r["exclusion_reasons"]))
        write_fasta(out / f"after_{cutoff}_sequence_candidates.fasta", survivors)
        with (out / f"after_{cutoff}_audit.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps({k: v for k, v in row.items() if k != "sequence"}, ensure_ascii=False) + "\n")
    inventory = []
    for version, rows in snapshots.items():
        older_version = "2023_06" if version == "2025_06" else "2025_06" if version == "2026_06" else None
        ids = {r["protein_id"] for r in rows}
        earlier_ids = {r["protein_id"] for r in snapshots[older_version]} if older_version else set()
        inventory.append({"release": version, "records": len(rows),
                          "entry_release_distribution": dict(sorted(Counter(str(r['entry_released']) for r in rows).items())),
                          "metadata_issue_records": sum(bool(r['issues']) for r in rows),
                          "previous_local_release": older_version,
                          "ids_absent_from_previous_local_snapshot": len(ids - earlier_ids) if older_version else None,
                          "previous_ids_absent_from_this_snapshot": len(earlier_ids - ids) if older_version else None})
    report = {
        "schema_version": 1, "status": "audit_complete_test_set_not_ready",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "s2_metadata_and_sequence_feasibility_audit",
        "script_sha256": sha(Path(__file__)), "input_sources": sources,
        "training": {"proteins": len(manifest), "known_residues": labels[0] + labels[1], "positive": labels[1], "negative": labels[0], "unknown": labels[-1], "positive_label_release": "2018_11", "pdb_release_cutoff": "2018-11-30", "later_mapping_resources": ["DisProt 2026_06 sequences/accessions", "SIFTS 2026-08-17 mapping"]},
        "snapshot_inventory": inventory, "candidate_cohorts": cohorts, "homology": homology,
        "temporal_interpretation": "Retrospective release-defined candidate audit, not prospective post-model-lock evaluation. Entry released/date are metadata, not proof of first experimental publication.",
        "caid_reference_labels_read": False, "disprot_region_fields_used": False,
        "training_labels_read_for_integrity_checks": True,
        "model_predictions_read_or_generated": False, "locked_model_modified": False,
        "unresolved": ["Check entry history and migration/obsolete/ambiguous evidence before final eligibility", "Map direct positive evidence without treating unannotated residues as negatives", "Obtain reliable observed-structure negative labels with sequence-coordinate checks", "Define evaluation metrics, baseline cohort and test lock before predictions", "Audit baseline training overlap; frozen ESM2 pretraining novelty is not established", "Long-fragment/domain homology and test-set internal clustering need a separate sensitivity audit"],
        "official_documentation": ["https://disprot.org/release-notes", "https://www.disprot.org/history/DP00609"],
    }
    write_json(out / "s2_source_audit.json", report)
    lines = ["# S2 数据来源与序列候选审计", "", "审计已完成；候选序列尚不是最终测试集。", "",
             "训练集核验：1133 条蛋白，202662 个已知标签残基；正标签版本 2018_11，PDB 发布截止 2018-11-30。",
             "训练数据构建使用过 2026_06 的序列/accession 和 2026 年 SIFTS 映射，不能描述为所有资源均冻结于 2018 年。", "",
             "| 本地快照 | 条目数 |", "| --- | ---: |"]
    lines += [f'| {r["release"]} | {r["records"]} |' for r in inventory]
    lines += ["", "条目数减少不能解读为负的新增量；候选池通过元数据和序列逐条比较。", "",
              "| 时间区间（至 2026_06） | 条目 released 字段在区间内 | 去旧条目、训练/CAID 重叠及重复后 | 同源过滤后 |",
              "| --- | ---: | ---: | ---: |"]
    for cutoff, c in cohorts.items():
        lines.append(f'| 晚于 {cutoff} | {c.get("entry_release_in_interval", 0)} | {c.get("unique_sequence_candidates", 0)} | {c["after_homology_sequences"]} |')
    lines += ["", "时间解释：这些快照在模型锁定前已经下载。本次可支持回顾性版本划分的可行性分析，不能称为锁模后前瞻性验证。released/date 字段不等于首次实验发表日期。", "",
              "同源规则：identity > 0.30 且双向 coverage >= 0.80；此规则不排除短片段/结构域同源，也不证明未进入 ESM-2 的预训练语料。", "",
              "本次未提取 DisProt 区域标签、未读取 CAID 参考标签、未生成预测。已知标签计数只用于核验现有训练集。", "",
              "## 后续必须完成", ""]
    lines += [f"- {item}" for item in report['unresolved']]
    lines += ["", "版本与修订依据：[DisProt 发布记录](https://disprot.org/release-notes)；[条目历史示例](https://www.disprot.org/history/DP00609)。", ""]
    (out / "s2_source_audit.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    locked_files = [p for p in out.iterdir() if p.is_file()] + [Path(__file__)]
    (out / "s2_audit_lock.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in sorted(locked_files)), encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "cohorts": cohorts, "homology": homology}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
