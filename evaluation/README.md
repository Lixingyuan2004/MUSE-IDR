# Evaluation

Evaluation is kept separate from prediction so external labels cannot affect
model selection or inference. The repository provides three levels:

1. `evaluation/evaluate_residue_predictions.py` forwards to the generic
   residue-level evaluator in `scripts/`;
2. `experiments/final_models/a10_locked_inference/` contains CAID preparation,
   alignment, official-format, and verification tools;
3. `experiments/analysis/b1_*` through `b7_*` contain fair comparison, baseline
   reproduction, paired statistics, robustness, and complementarity analyses.

Generic JSONL evaluation example:

```bash
python evaluation/evaluate_residue_predictions.py \
  --reference path/to/reference.jsonl \
  --predictions path/to/predictions.jsonl \
  --output results/metrics.json \
  --bootstrap-replicates 2000
```

Reported metrics include ROC-AUC, average precision/AUPRC, F1, MCC, and related
diagnostics as defined by the selected evaluation script. Masked labels are
excluded. B6 is the confirmatory whole-protein paired analysis; B7 is
descriptive post-lock analysis.
