# A4: online ESM2 LoRA + A2 multiscale context

A4 keeps the successful A2 gated multiscale context head and replaces the
frozen embedding cache with an online, parameter-efficient ESM2 adaptation.
The model still emits exactly one classic-IDR probability for each residue.

## Scientific controls

- `config_frozen_raw_control.yaml` uses the identical raw-sequence windowing,
  tokenizer, online ESM2 forward pass, overlap merge, optimizer and A2 head,
  but disables LoRA.  It isolates the adapter effect from pipeline changes.
- The ESM2 revision, 5-fold assignment and labels stay fixed.
- Unknown residue labels are excluded from both loss and metrics.
- No CAID1/2/3 labels or sequences are loaded by this code.
- Checkpoints contain only LoRA adapter tensors plus the A2 head, never the
  full ESM2 backbone.

## Local verification

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  experiments\ablations\a4_esm2_lora_multiscale\data.py `
  experiments\ablations\a4_esm2_lora_multiscale\model.py `
  experiments\ablations\a4_esm2_lora_multiscale\train_a4.py `
  experiments\ablations\a4_esm2_lora_multiscale\smoke_test_a4.py `
  experiments\ablations\a4_esm2_lora_multiscale\smoke_test_a4_end_to_end.py

.\.venv\Scripts\python.exe experiments\ablations\a4_esm2_lora_multiscale\smoke_test_a4.py
.\.venv\Scripts\python.exe experiments\ablations\a4_esm2_lora_multiscale\smoke_test_a4_end_to_end.py
```

## First 4090 development gate

Run only fold 4 / seed 17 for both variants.  Select neither hyperparameters
nor checkpoints using CAID.

```bash
python -u experiments/ablations/a4_esm2_lora_multiscale/train_a4.py \
  --manifest data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl \
  --config experiments/ablations/a4_esm2_lora_multiscale/config_frozen_raw_control.yaml \
  --fold 4 --seed 17

python -u experiments/ablations/a4_esm2_lora_multiscale/train_a4.py \
  --manifest data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl \
  --config experiments/ablations/a4_esm2_lora_multiscale/config.yaml \
  --fold 4 --seed 17
```

The primary gate is paired fold-4 Micro ROC-AUC: LoRA must beat the raw frozen
control, and should also beat the existing A2 fold-4 result
`0.967695178971393` before a full 5-fold x 3-seed run is justified.
