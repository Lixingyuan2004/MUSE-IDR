# MUSE-IDR model card

## Summary

MUSE-IDR predicts one classical intrinsic-disorder probability per residue. It
uses a frozen ESM-2 650M backbone and a locked 30-member ensemble: 15 A2
final-layer multiscale heads and 15 A8 last-four-layer fusion heads, spanning
three seeds and five folds per family. All members have fixed weight 1/30 in
logit space.

## Intended use

The model is intended for research screening and comparative computational
analysis. Predictions are hypotheses and are not substitutes for experimental,
clinical, or functional evidence. The model does not separately predict soft
disorder, binding regions, phase separation, or disease causality.

## Training and model selection

The development population contains 1,133 proteins and 202,662 known labeled
residues (105,036 positive and 97,626 negative); 370,759 unknown residues are
masked. Internal evaluation uses fixed homology-aware five-fold out-of-fold
predictions. CAID2/CAID3 labels were not used for training, calibration, model
selection, or ensemble weighting.

## External performance

| Dataset | Track | ROC-AUC | AUPRC | F1@0.5 | MCC@0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| CAID2 | disorder_nox | 0.780767 | 0.360582 | 0.514474 | 0.408428 |
| CAID2 | disorder_pdb | 0.957568 | 0.933521 | 0.866783 | 0.813985 |
| CAID3 | disorder_nox | 0.852106 | 0.605800 | 0.645557 | 0.518414 |
| CAID3 | disorder_pdb | 0.953584 | 0.930965 | 0.845464 | 0.771543 |

On the supplementary S2 temporal holdout, MUSE-IDR obtained ROC-AUC 0.977827,
AUPRC 0.847950, and MCC@0.5 0.732873. The paired ROC-AUC advantage over the
reproduced LoRA-DR-Suite 650M comparator was supported after Holm correction;
the interval versus PUNCH2-Light Released-13 included zero.

## Limitations

- Performance differs substantially between PDB and NOX disorder definitions.
- The ESM-2 650M backbone is large and must be downloaded separately.
- Thirty lightweight heads improve robustness but add inference overhead.
- S2 inherits the limitations of database annotations and is supplementary to CAID.
- Upstream sequence, structure, and disorder annotations may contain temporal and mapping uncertainty.

## Reproducibility

```bash
python scripts/verify_release.py
python inference/predict.py \
  --locked-root models/weights/a10_locked \
  --device cpu \
  --verify-only
```

Project-authored code is Apache-2.0. Upstream resources retain their own terms;
see `THIRD_PARTY_NOTICES.md`.
