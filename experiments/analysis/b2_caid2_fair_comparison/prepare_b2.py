"""Prepare label-free CAID2 inputs for the already locked A10 model.

This program may read CAID2 reference labels only to validate their alignment
and summarize the external test set.  The FASTA files written for inference
contain protein identifiers and sequences only.  No model training, model
selection, threshold selection, or ensemble weighting is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.final_models.a10_locked_inference.prepare_caid_references import (  # noqa: E402
    ReferenceRecord,
    dataset_summary,
    file_sha256,
    parse_reference,
    read_training_sequences,
    write_fasta,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nox-reference", type=Path, required=True)
    parser.add_argument("--pdb-reference", type=Path, required=True)
    parser.add_argument("--training-sequences", type=Path, required=True)
    parser.add_argument("--homology-audit", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def main() -> None:
    args = parse_args()
    nox = parse_reference(args.nox_reference)
    pdb = parse_reference(args.pdb_reference)
    training = read_training_sequences(args.training_sequences)
    homology = json.loads(args.homology_audit.read_text(encoding="utf-8-sig"))
    model_lock = json.loads(args.model_lock.read_text(encoding="utf-8-sig"))

    if homology.get("status") != "pass" or homology.get("homology_audit") != "complete":
        raise ValueError("CAID2/3 homology audit is not complete and passing")
    if homology.get("exact_or_identifier_findings"):
        raise ValueError("homology audit reports exact or identifier findings")
    if homology.get("forbidden_homology_hits"):
        raise ValueError("homology audit reports forbidden homology hits")
    if model_lock.get("lock_status") != "frozen_before_caid_external_evaluation":
        raise ValueError("A10 was not locked before CAID external evaluation")
    if model_lock.get("data_policy", {}).get(
        "caid1_caid2_caid3_used_for_training_or_tuning"
    ) is not False:
        raise ValueError("model lock does not exclude CAID from training and tuning")

    union_by_id: dict[str, ReferenceRecord] = {}
    for record in [*nox, *pdb]:
        previous = union_by_id.get(record.protein_id)
        if previous is not None and previous.sequence != record.sequence:
            raise ValueError(f"NOX/PDB sequence disagreement: {record.protein_id}")
        union_by_id.setdefault(record.protein_id, record)
    union = list(union_by_id.values())

    training_hash_to_ids: dict[str, list[str]] = {}
    for protein_id, sequence in training.items():
        training_hash_to_ids.setdefault(sequence_sha256(sequence), []).append(protein_id)
    exact_sequence_overlaps = [
        {
            "caid2_id": record.protein_id,
            "training_ids": training_hash_to_ids[sequence_sha256(record.sequence)],
        }
        for record in union
        if sequence_sha256(record.sequence) in training_hash_to_ids
    ]
    exact_id_overlaps = sorted(set(union_by_id) & set(training))
    if exact_id_overlaps or exact_sequence_overlaps:
        raise ValueError("locked A10 training data overlaps CAID2")

    sequence_dir = args.output_dir / "sequences_only"
    nox_output = sequence_dir / "caid2_disorder_nox_sequences.fasta"
    pdb_output = sequence_dir / "caid2_disorder_pdb_sequences.fasta"
    union_output = sequence_dir / "caid2_disorder_union_sequences.fasta"
    write_fasta(nox_output, nox)
    write_fasta(pdb_output, pdb)
    write_fasta(union_output, union)

    report = {
        "schema_version": 1,
        "experiment": "b2_caid2_locked_input_preparation",
        "status": "pass",
        "purpose": "locked_external_test_input_preparation",
        "nox": dataset_summary(args.nox_reference, nox),
        "pdb": dataset_summary(args.pdb_reference, pdb),
        "nox_pdb_shared_ids": len(
            {record.protein_id for record in nox}
            & {record.protein_id for record in pdb}
        ),
        "union_proteins": len(union),
        "union_residues": sum(len(record.sequence) for record in union),
        "sequence_outputs": {
            "nox": str(nox_output),
            "nox_sha256": file_sha256(nox_output),
            "pdb": str(pdb_output),
            "pdb_sha256": file_sha256(pdb_output),
            "union": str(union_output),
            "union_sha256": file_sha256(union_output),
        },
        "training_sequence_manifest": str(args.training_sequences),
        "training_sequence_manifest_sha256": file_sha256(args.training_sequences),
        "training_proteins": len(training),
        "exact_id_overlaps": exact_id_overlaps,
        "exact_sequence_overlaps": exact_sequence_overlaps,
        "homology_audit": str(args.homology_audit),
        "homology_audit_sha256": file_sha256(args.homology_audit),
        "homology_audit_status": "pass",
        "model_lock": str(args.model_lock),
        "model_lock_sha256": file_sha256(args.model_lock),
        "model_locked_before_caid2_label_access": True,
        "caid2_used_for_training_or_tuning": False,
        "inference_inputs_contain_labels": False,
    }
    report_path = args.output_dir / "b2_caid2_preparation_report.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
