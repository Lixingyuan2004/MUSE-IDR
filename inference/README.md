# Inference

The paper model is the immutable A10 ensemble: 15 A2 heads plus 15 A8 heads,
combined by an equal raw-logit mean. Use either of these equivalent entry
points:

```bash
python predict_muse_idr.py --help
python inference/predict.py --help
```

The implementation is preserved at
`experiments/final_models/a10_locked_inference/predict_a10.py`; the locked
checkpoints are at `models/weights/a10_locked/`. Inference consumes sequence-only
FASTA input and writes exactly one classic-IDR probability per residue.

Do not place CAID labels in the inference input. Evaluation is deliberately a
separate stage under `evaluation/`.
