# B7 error, robustness, and complementarity analysis

B7 is a post-lock descriptive analysis of A10-locked, LoRA-DR-Suite 650M,
PUNCH2-Light Paper-8, and PUNCH2-Light Released-13 on CAID2 and CAID3.

It answers three questions without changing the locked model:

1. Which protein and residue regimes are difficult for each model?
2. Are the observed strengths stable across CAID2 and CAID3?
3. Do A10 and the baselines make complementary errors?

## Predeclared descriptive strata

- Protein length: `1-200`, `201-500`, `501-1000`, `1001+` residues.
- Known-label disorder fraction: `0-0.10`, `(0.10,0.30]`,
  `(0.30,0.60]`, `(0.60,1.00]`.
- Distance to the nearest observed 0/1 transition: `0-2`, `3-5`,
  `6-15`, `16+`, or `no-transition`.
- True ordered/disordered run length: `1-15`, `16-30`, `31-100`,
  or `101+` residues.

All thresholded diagnostics use the already declared probability threshold
of 0.5. No subgroup, threshold, ensemble weight, or model is selected from
CAID2/CAID3 labels. B7 is descriptive; B6 remains the confirmatory analysis.

## Smoke test

```bash
python experiments/analysis/b7_error_complementarity/smoke_test_b7.py
```

## Formal run

```bash
python -u experiments/analysis/b7_error_complementarity/build_b7.py \
  --output-dir outputs/analysis/b7_error_complementarity/formal
```

The formal run writes point metrics, per-protein metrics, subgroup metrics,
segment and boundary diagnostics, pairwise complementarity, a robustness
table, a machine-readable report, and a manuscript-ready Markdown summary.
