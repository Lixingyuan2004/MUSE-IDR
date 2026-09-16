# A5: A2 three-seed ensemble

A5 applies the PUNCH2 ensemble lesson to the three already trained A2 models
(seeds 17, 29 and 43). It emits one classic-IDR score per residue and never
loads CAID data.

The script separates two claims:

- uniform probability/logit averaging and the median use no labels to set
  weights, so each reported OOF score is a fair deterministic ensemble result;
- optimized weights use classic-dataset OOF labels. They are valid development
  choices for a later untouched CAID evaluation, but their score on those same
  OOF labels is not presented as an unbiased external estimate.

The formal A5 predictor is the equal-weight logit mean. The OOF-tuned logit
weights have a slightly higher pooled development AUC but lose to equal weights
in four of five folds, so they remain exploratory rather than the registered
external-test predictor.

Run locally:

```powershell
.\.venv\Scripts\python.exe experiments\ablations\a5_seed_ensemble\smoke_test_a5.py

.\.venv\Scripts\python.exe experiments\ablations\a5_seed_ensemble\ensemble_a2.py `
  --archive experiments\ablations\a2_multiscale_context\a2_results_20260823.tar.gz `
  --manifest data\processed\classic_idr_2018\a1_frozen_esm2_manifest.jsonl `
  --output-dir outputs\ablations\a5_seed_ensemble

.\.venv\Scripts\python.exe experiments\ablations\a5_seed_ensemble\bootstrap_a5.py `
  --predictions outputs\ablations\a5_seed_ensemble\a5_oof_predictions.csv.gz `
  --ensemble-report outputs\ablations\a5_seed_ensemble\a5_ensemble_report.json `
  --output-dir outputs\ablations\a5_seed_ensemble `
  --replicates 2000 --random-seed 20260824 --workers 8
```

The bootstrap resamples whole proteins with replacement, stratified by the
fixed five folds. It never treats residues from one protein as independent
bootstrap units.
