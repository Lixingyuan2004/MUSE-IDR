# B6 paired model statistics

B6 performs post-lock paired comparisons of A10-locked, LoRA-DR-Suite 650M,
PUNCH2-Light Paper-8, and PUNCH2-Light Released-13 on CAID2 and CAID3.

## Statistical protocol

- The sampling unit is the complete protein, preserving residue dependence.
- One bootstrap resample is shared by all four models within a track.
- Six full-precision metrics are bootstrapped: ROC-AUC, trapezoidal AUPRC,
  average precision, F1@0.5, MCC@0.5, and Fmax.
- Paired DeLong is additionally reported for full-precision ROC-AUC.
- Holm correction is applied to 48 primary hypotheses: A10 versus LoRA and
  A10 versus Released-13 across four tracks and six metrics.
- A10 versus Paper-8 is supplementary and corrected as a separate 24-test
  family.
- Every reference and prediction file must match the locked SHA256 manifest.
- CAID2/CAID3 labels are post-lock evaluation-only and cannot be used to
  retune A10.

## Smoke test

```bash
python experiments/analysis/b6_paired_model_statistics/smoke_test_b6.py
```

## Formal run

```bash
python -u experiments/analysis/b6_paired_model_statistics/build_b6.py \
  --replicates 2000 \
  --workers 8 \
  --output-dir outputs/analysis/b6_paired_model_statistics/formal
```

The formal run writes point metrics, pairwise statistics, bootstrap samples,
a machine-readable report, and a manuscript-ready Markdown summary.
