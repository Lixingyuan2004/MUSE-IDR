# Training entry points

MUSE-IDR preserves training code inside the experiment that produced each
result. This directory provides a stable launcher and a map to those canonical
files; it does not duplicate or rewrite the historical implementations.

## Trainable experiments

| Experiment | Main training implementation | Purpose |
| --- | --- | --- |
| A1 | `experiments/ablations/a1_frozen_esm2/train_a1.py` | frozen ESM-2 baseline |
| A2 | `experiments/ablations/a2_multiscale_context/train_a2.py` | multiscale context head |
| A3 | `experiments/ablations/a3_auc_ranking_loss/train_a3.py` | ranking-loss ablation |
| A4 | `experiments/ablations/a4_esm2_lora_multiscale/train_a4.py` | LoRA ablation |
| A6 | `experiments/ablations/a6_long_range_dilated_context/train_a6.py` | dilated-context ablation |
| A7 | `experiments/ablations/a7_bidirectional_sequence_context/train_a7.py` | bidirectional sequence ablation |
| A8 | `experiments/ablations/a8_multilayer_esm2_fusion/train_a8.py` | multilayer fusion family |

A5 and A9 aggregate already trained seeds; A10 combines the locked A2 and A8
families. Their code is under the corresponding experiment directories and is
not a separate parameter-training stage.

List the launchable experiments:

```bash
python training/run_experiment.py --list
```

Pass all remaining arguments directly to the canonical script, for example:

```bash
python training/run_experiment.py a2 \
  --manifest data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl \
  --cache-dir data/cache/a1_esm2_650m \
  --config experiments/ablations/a2_multiscale_context/config.yaml \
  --fold 0 --seed 17
```

Long GPU jobs should be launched through a persistent session such as `screen`
on the training server. CAID2 and CAID3 labels are external-test-only and must
not be passed to any training command.
