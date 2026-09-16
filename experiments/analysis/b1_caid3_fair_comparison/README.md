# B1: CAID3 fair comparison

This analysis compares the already locked A10 model with public CAID3
per-residue predictions. It does not train or tune any model.

## Inputs

- `release/a10_caid3_final_results_20260825.tar.gz`
- `data/external/caid3_locked/official_predictions/caid3_predictions.zip`
- `published_baselines.json`

Both binary archives are protected by fixed SHA256 checks in `build_b1.py`.

## Metrics

- Primary: CAID-compatible residue-level ROC-AUC. A full-precision ROC-AUC is
  retained as a diagnostic value.
- Secondary: CAID-compatible trapezoidal PR-AUC, APS, F1 and MCC at 0.5,
  Fmax and MCC at the Fmax threshold. Full-precision variants are retained in
  b1_report.json as diagnostic values only.
- Statistics: paired residue-level DeLong tests and paired protein-cluster
  bootstrap (2,000 replicates per selected comparison). The cluster bootstrap
  is the main uncertainty analysis because residues within a protein are not
  independent.

Coverage is reported explicitly. Comparisons with an incomplete method are
paired only on the common proteins. LoRA-DR-Suite values are retained as
literature context because its paper uses a 148-sequence CAID3_NOX set rather
than the locked 204-sequence official NOX reference used for A10.

## Run

```powershell
.\.venv\Scripts\python.exe -m experiments.analysis.b1_caid3_fair_comparison.smoke_test_b1

.\.venv\Scripts\python.exe experiments\analysis\b1_caid3_fair_comparison\build_b1.py `
  --bootstrap-replicates 2000
```

Outputs are written to `outputs/analysis/b1_caid3_fair_comparison/`.
