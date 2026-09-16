# Model code and locked weights

This directory is the stable, reader-facing index for the MUSE-IDR model
implementations. The original experiment implementations remain canonical and
are imported by the small compatibility modules in this directory:

- `a2.py`: the multiscale-context residue-level prediction head;
- `a8.py`: the learnable ESM-2 layer-fusion model;
- `a10.py`: the locked 30-member A2/A8 ensemble utilities.

The only copy of the 30 locked paper checkpoints is stored under
`weights/a10_locked/`. Moving the immutable files into this canonical directory
preserves every recorded SHA256 and avoids a second 70 MiB copy. The lock
contains 15 A2 heads and 15 A8 heads: three seeds (`17`, `29`, and `43`) times
five folds for each family.

Use the top-level inference command for prediction:

```bash
python predict_muse_idr.py --help
```

Use the following command to verify all locked members without loading ESM-2:

```bash
python experiments/final_models/a10_locked_inference/predict_a10.py \
  --locked-root models/weights/a10_locked \
  --device cpu \
  --verify-only
```
