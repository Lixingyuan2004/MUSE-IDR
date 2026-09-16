# B4: official baseline reproduction

B4 reproduces the original PUNCH2/PUNCH2-Light and LoRA-DR-Suite intrinsic-
disorder predictors. It does not modify or retune locked A10. It also excludes
all LoRA soft-disorder checkpoints.

## Why B4 starts with an audit

The reviewed paper protocol and the currently released PUNCH entry points are
not identical:

- PUNCH2 paper: 3 One-Hot + 5 ProtTrans + 5 MSA-Transformer = 13 members.
- PUNCH2 released `main.py`: the above plus 5 ProtTrans CNN_L3 members = 18.
- PUNCH2-Light paper: 3 One-Hot + 5 ProtTrans = 8 members.
- PUNCH2-Light released `main.py`: the above plus 5 ProtTrans CNN_L3 members = 13.
- The PUNCH2 paper threshold is 0.378, while the released code uses 0.39.

Therefore, B4 must report paper-faithful and released-entry-point results as
different variants. Merging them under one name would not be reproducible.

LoRA-DR-Suite also has two distinct intrinsic-disorder training regimes:

- `CQSB/esm2_650M-LoRA-ID-DisProt7`: DisProt 7.0 only; eligible for the
  CAID-label-free comparison on CAID2 and CAID3.
- `CQSB/esm2_650M-LoRA-ID`: DisProt 7.0 + CAID1 + CAID2; eligible for the
  original paper's CAID3 protocol, but not for CAID2 and not training-data-
  matched to A10.

## Source locks

- PUNCH2: `815a1040e38bf8c6930c11f51067506932db6d94`
- PUNCH2-Light: `6c7935b3597c056d2e6b3845bb54fc101c5bc574`
- LoRA-DR-Suite code: `d8893fc4121d979c0389a8944ff0531f22563e90`

The PUNCH repositories include trained CNN checkpoints but no license file was
found. The reviewed LoRA adapter metadata also does not declare an adapter
license. This does not prevent internal scientific evaluation, but permission
must be clarified before redistributing third-party code or weights with a
paper artifact.

## First execution target

Run the LoRA-DR-Suite ESM2-650M DisProt7 checkpoint first. It provides official
adapter weights and direct Transformers loading, whereas PUNCH2 requires
precomputed ProtTrans embeddings and the full variant additionally requires
MSA-Transformer embeddings.

The first execution is only a short-sequence inference smoke test. Formal CAID
evaluation follows only after the author implementation's long-sequence policy
has been verified.

## Offline source audit

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  experiments\analysis\b4_baseline_reproduction\smoke_test_b4_sources.py

.\.venv\Scripts\python.exe `
  experiments\analysis\b4_baseline_reproduction\smoke_test_b4_sources.py
```

## Planned B4 execution order

1. LoRA-DR-Suite DisProt7 short-sequence smoke test.
2. Verify author long-sequence inference behavior; lock it before CAID runs.
3. Evaluate LoRA DisProt7 on CAID2/CAID3 NOX without tuning.
4. Separately reproduce the paper-protocol LoRA-ID result on CAID3 NOX.
5. Run PUNCH2-Light paper-faithful 8-member and released 13-member variants.
6. Run PUNCH2 paper-faithful 13-member and released 18-member variants.
7. Feed all same-reference predictions into the existing B1/B2 metric and
   paired-statistics pipeline.
## B4.2 official LoRA batch inference

`predict_lora_official.py` runs only the ID/DisProt7 LoRA checkpoints. It loads
the ESM2 base model and PEFT adapter explicitly, verifies active query/value
LoRA tensors and the saved token-classification head, and emits one classic-IDR
probability per residue. Long proteins use fixed overlapping windows and a
label-free uniform mean of the disorder-vs-order logit margin.

`evaluate_lora_caid.py` runs only after predictions are locked. It rejects any
report that does not prove DisProt7-only inference, official adapter/head
loading, complete residue alignment, and zero CAID label use during inference
or tuning. Metrics share the already verified B1 implementation.

```bash
python experiments/analysis/b4_baseline_reproduction/smoke_test_lora_batch.py
```
