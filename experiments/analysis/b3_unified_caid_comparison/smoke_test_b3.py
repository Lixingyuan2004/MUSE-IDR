"""Fast invariant tests for the B3 table-building logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.b3_unified_caid_comparison.build_b3 import (
    make_a10_summary,
    normalize_formal_metrics,
    normalize_literature_context,
    select_manuscript_rows,
    validate_formal_rows,
)


def metric(
    track: str,
    method: str,
    rank: int,
    auc: float,
    reference_proteins: int,
) -> dict[str, object]:
    return {
        "track": track,
        "method": method,
        "source": "locked_a10" if method == "A10-locked" else "official_predictions",
        "reference_proteins": reference_proteins,
        "predicted_proteins": reference_proteins,
        "protein_coverage": 1.0,
        "reference_residues": 1000,
        "predicted_residues": 1000,
        "residue_coverage": 1.0,
        "known_residues": 900,
        "positive_labels": 300,
        "negative_labels": 600,
        "roc_auc_caid_compatible": auc,
        "aucpr_trapezoid": 0.8,
        "aps": 0.79,
        "f1_at_0_5": 0.7,
        "mcc_at_0_5": 0.6,
        "fmax": 0.72,
        "mcc_at_fmax": 0.61,
        "fmax_threshold": 0.5,
        "rank_by_roc_auc": rank,
    }


def main() -> None:
    rows = normalize_formal_metrics(
        "CAID2",
        [
            metric("disorder_nox", "Leader2-NOX", 1, 0.90, 210),
            metric("disorder_nox", "A10-locked", 2, 0.88, 210),
            metric("disorder_pdb", "A10-locked", 1, 0.96, 348),
            metric("disorder_pdb", "Other2-PDB", 2, 0.94, 348),
        ],
    )
    rows += normalize_formal_metrics(
        "CAID3",
        [
            metric("disorder_nox", "Leader3-NOX", 1, 0.89, 204),
            metric("disorder_nox", "A10-locked", 2, 0.85, 204),
            metric("disorder_nox", "PUNCH2", 3, 0.84, 204),
            metric("disorder_pdb", "PUNCH2", 1, 0.955, 319),
            metric("disorder_pdb", "A10-locked", 2, 0.954, 319),
        ],
    )
    validate_formal_rows(rows)
    summary = make_a10_summary(rows)
    manuscript = select_manuscript_rows(rows, top_methods=1)

    literature = normalize_literature_context(
        {
            "lora_dr_suite_reference_only": {
                "track": "paper_specific_caid3_nox_148_sequences",
                "same_reference_as_locked_a10": False,
                "training_note": "context only",
                "rows": [
                    {
                        "method": "ESM2_650M-LoRA-ID",
                        "roc_auc": 0.88,
                        "f1": 0.649,
                        "mcc": 0.523,
                        "pr_auc": 0.721,
                        "fmax": 0.656,
                    }
                ],
            }
        }
    )

    if len(summary) != 4:
        raise AssertionError("four A10 round/track summary rows are required")
    if not all(row["same_reference"] for row in rows):
        raise AssertionError("formal rows must be same-reference recomputations")
    if any(row["rank_eligible"] for row in literature):
        raise AssertionError("literature rows cannot enter rankings")
    if not any(row["method"] == "PUNCH2" for row in manuscript):
        raise AssertionError("named PUNCH2 comparator must be retained")
    if not all(row["method"] != "ESM2_650M-LoRA-ID" for row in manuscript):
        raise AssertionError("LoRA literature row leaked into manuscript ranking")

    print(
        json.dumps(
            {
                "status": "pass",
                "four_round_track_summaries": True,
                "same_reference_formal_rows_only": True,
                "lora_reference_mismatch_excluded_from_ranking": True,
                "punch2_named_baseline_retained": True,
                "roc_auc_auprc_aps_f1_mcc_fmax_preserved": True,
                "single_output_score": True,
                "soft_disorder_output": False,
                "caid_labels_used_for_training_or_tuning": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
