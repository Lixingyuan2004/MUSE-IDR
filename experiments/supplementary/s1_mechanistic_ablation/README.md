# S1 post-lock mechanistic ablation

S1 is an explanation-only supplementary experiment. It must not retrain,
replace, reweight, or reselect the locked A10 model. CAID1/2/3 labels are not
permitted as training, tuning, early-stopping, or model-selection inputs.

## Questions tested

1. Does the gated four-scale A2 head outperform direct ungated fusion?
2. Is sigmoid multiplicative gating better than a parameter-matched learned
   additive projection?
3. Does the four-scale `(3, 7, 15, 31)` head outperform each single scale?
4. Does learned A8 layer mixing outperform a fixed uniform mean of ESM2 layers
   30, 31, 32, and 33?

The existing formal A2 and A8 OOF predictions are the references. S1 adds only
the seven missing controls. The predeclared comparison family is corrected
together with Holm's method after all three seeds and five folds finish.

## Variants

| Variant | Representation | Context fusion / scale |
| --- | --- | --- |
| `ungated_add` | ESM2 layer 33 | direct four-scale residual addition |
| `additive_projection` | ESM2 layer 33 | parameter-matched learned additive projection |
| `single_k3` | ESM2 layer 33 | gated kernel 3 only |
| `single_k7` | ESM2 layer 33 | gated kernel 7 only |
| `single_k15` | ESM2 layer 33 | gated kernel 15 only |
| `single_k31` | ESM2 layer 33 | gated kernel 31 only |
| `uniform_layer_mix` | uniform layers 30/31/32/33 | exact gated four-scale A2 head |

## Local smoke test

```bash
python -m py_compile \
  experiments/supplementary/s1_mechanistic_ablation/model.py \
  experiments/supplementary/s1_mechanistic_ablation/train_s1.py \
  experiments/supplementary/s1_mechanistic_ablation/run_s1_suite.py \
  experiments/supplementary/s1_mechanistic_ablation/smoke_test_s1.py

python experiments/supplementary/s1_mechanistic_ablation/smoke_test_s1.py
```

## Formal AutoDL run

Run the suite in a detached `screen` session. The final-layer and multilayer
caches must already have passed their independent verification reports.

```bash
screen -dmS s1formal bash -lc '
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
set -euo pipefail

python -u \
  experiments/supplementary/s1_mechanistic_ablation/run_s1_suite.py \
  --manifest data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl \
  --final-layer-cache data/cache/a1_esm2_650m \
  --multilayer-cache data/cache/a8_esm2_last4 \
  > logs/s1_mechanistic_ablation.log 2>&1
'
```

Progress can be inspected with:

```bash
screen -ls
tail -n 100 -f logs/s1_mechanistic_ablation.log
```

Use `--resume` only after inspecting a stopped run. It skips a job only when
the saved report declares a complete five-fold OOF result, no CAID label use,
and no modification of locked A10.

Outputs are written under
`outputs/supplementary/s1_mechanistic_ablation/<variant>/seed_<seed>/`.

## Locked statistical analysis

After downloading and SHA256-verifying the formal S1 result archive, run the
numerical smoke test and then the predeclared paired analysis:

```bash
python experiments/supplementary/s1_mechanistic_ablation/smoke_test_analyze_s1.py

python -u experiments/supplementary/s1_mechanistic_ablation/analyze_s1.py \
  --replicates 2000 \
  --workers 8 \
  --seed 20260904 \
  --progress-every 100 \
  --output-dir outputs/supplementary/s1_mechanistic_analysis/formal
```

The analysis reconstructs fixed equal-logit three-seed ensembles, resamples
whole proteins within each of the five OOF folds, and applies one Holm
correction across the seven predeclared ROC-AUC comparisons. AUPRC and macro
ROC-AUC are reported descriptively. Positive paired deltas mean that the
matching A2 or A8 reference outperformed the control. S1 remains explanatory
and cannot modify or reselect the locked A10 model.
