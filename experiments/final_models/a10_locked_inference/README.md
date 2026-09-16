# Locked A10 external inference

This package predicts one classic-IDR probability per residue. It never reads
residue labels and has no soft-disorder output.

The production ensemble is fixed to A2 and A8, seeds 17/29/43, folds 0--4:
30 heads with equal raw-logit weights. ESM2 layers 30--33 are extracted once
per sequence window and shared by every head. The A2 members consume layer 33;
the A8 members learn their own frozen four-layer scalar mixtures.

## Local checks (no ESM2 download)

```powershell
..\.venv\Scripts\python.exe experiments\final_models\a10_locked_inference\smoke_test_a10_inference.py

..\.venv\Scripts\python.exe experiments\final_models\a10_locked_inference\predict_a10.py `
  --locked-root artifacts\a10_locked_20260824 `
  --device cpu `
  --verify-only
```

## GPU prediction

```bash
python -u experiments/final_models/a10_locked_inference/predict_a10.py \
  --fasta data/external/caid_sequences.fasta \
  --locked-root models/weights/a10_locked \
  --cache-dir cache/a10_external \
  --output outputs/final/a10/caid_predictions.tsv.gz \
  --device cuda \
  --precision bf16
```

The output positions are one-based. Do not use a CAID label file as input to
this command. A benchmark-specific format converter and evaluator should be
kept separate from inference.
