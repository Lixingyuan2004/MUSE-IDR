# B3: unified CAID2/CAID3 comparison

B3 combines the already locked B1 (CAID3) and B2 (CAID2) result archives.
It does not train a model, access raw CAID labels, or tune A10.

## Evidence levels

1. **Formal same-reference comparison**: raw per-residue predictions were
   recomputed on the same locked reference with the shared CAID-compatible
   metric implementation. These rows may be ranked and used for paired tests.
2. **Literature-only context**: published metrics use a different reference
   set. These rows are excluded from ranks and paired statistics. The current
   LoRA-DR-Suite rows are in this category because its paper reports a
   148-sequence CAID3_NOX set rather than the locked 204-protein reference.

## Inputs

- `release/b1_caid3_fair_comparison_20260825.tar.gz`
- `release/b2_caid2_final_results_20260825.tar.gz`

Both archives are protected by fixed SHA256 checks.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  experiments\analysis\b3_unified_caid_comparison\build_b3.py `
  experiments\analysis\b3_unified_caid_comparison\smoke_test_b3.py

.\.venv\Scripts\python.exe `
  experiments\analysis\b3_unified_caid_comparison\smoke_test_b3.py

.\.venv\Scripts\python.exe `
  experiments\analysis\b3_unified_caid_comparison\build_b3.py
```

Outputs are written to `outputs/analysis/b3_unified_caid_comparison/`.

## Outputs

- `b3_unified_rankings.csv`: all formal CAID2/CAID3 rows.
- `b3_manuscript_comparison.csv`: top methods plus A10 and named PUNCH2 rows.
- `b3_a10_round_summary.csv`: four A10 round/track results.
- `b3_pairwise_statistics.csv`: normalized paired DeLong and protein bootstrap.
- `b3_lora_dr_suite_literature_context.csv`: explicitly non-rankable context.
- `b3_efficiency.csv`: A10 runtime and peak GPU memory.
- `b3_summary.md`: manuscript-oriented readable summary.
- `b3_report.json`: structured provenance and claim guardrails.
- `b3_artifact_sha256.csv`: output artifact checksums.
