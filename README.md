# MUSE-IDR

MUSE-IDR is a residue-level predictor of classical intrinsic disorder. It
combines frozen ESM-2 650M representations, parallel multiscale sequence
context, learned fusion of the final four ESM-2 layers, and a locked
cross-architecture ensemble.

This repository is the reproducibility package for the Bioinformatics
submission candidate. The scientific model remains the previously locked
`A10-locked` ensemble; repository reorganization does not retrain, recalibrate,
reweight, or otherwise modify it.

## Model at a glance

- Frozen backbone: `facebook/esm2_t33_650M_UR50D`, revision
  `08e4846e537177426273712802403f7ba8261b6c`.
- A2 family: 15 gated multiscale-context heads using ESM-2 layer 33.
- A8 family: 15 heads using a learned scalar fusion of layers 30–33 followed
  by the same multiscale head.
- Replication: three seeds (17, 29, and 43) across five fixed folds.
- Ensemble: uniform mean of all 30 logits, followed by a sigmoid.
- Output: one classical-IDR probability for every input residue.

## Quick start

Python 3.12 was used for the locked analyses. Install a platform-appropriate
PyTorch build, then install the paper environment:

```bash
python -m pip install -r requirements-paper.txt
```

Verify the repository and the 30 locked checkpoints:

```bash
python scripts/audit_repository_layout.py
python scripts/verify_release.py
python inference/predict.py \
  --locked-root models/weights/a10_locked \
  --device cpu \
  --verify-only
```

Predict an example FASTA file:

```bash
python inference/predict.py \
  --fasta examples/example.fasta \
  --locked-root models/weights/a10_locked \
  --cache-dir cache/muse_idr \
  --output predictions.tsv.gz \
  --report prediction_report.json \
  --device cuda \
  --precision bf16
```

The first full prediction run downloads the locked ESM-2 backbone revision.
The output table contains the protein identifier, one-based residue position,
amino-acid code, and IDR probability.

## Repository map

| Path | Contents |
| --- | --- |
| `models/` | Public A2/A8/A10 model interfaces and 30 locked head checkpoints |
| `training/` | Stable training dispatcher and experiment guide |
| `inference/` | Stable label-free FASTA inference entry point |
| `evaluation/` | Generic residue-level evaluation entry point |
| `experiments/ablations/` | A1–A10 development experiments |
| `experiments/analysis/` | B1–B7 external comparison and statistical analyses |
| `experiments/supplementary/` | S1 mechanistic ablations and S2 temporal holdout workflow |
| `results/` | Compact locked results used in the manuscript and supplement |
| `data/manifests/` | Public provenance, split, and leakage-audit metadata |
| `docs/` | Data, evaluation, experiment, and reproduction documentation |

See `docs/PROJECT_LAYOUT.md` for the full layout and publication policy.

## Main results

### CAID external evaluation

| Dataset | Track | ROC-AUC (full precision) | AUPRC (full precision) | Rank context |
| --- | --- | ---: | ---: | --- |
| CAID2 | disorder_nox | 0.780767 | 0.360582 | 13/72 |
| CAID2 | disorder_pdb | 0.957568 | 0.933521 | 1/72 |
| CAID3 | disorder_nox | 0.852106 | 0.605800 | 8/13 |
| CAID3 | disorder_pdb | 0.953584 | 0.930965 | 2/11 |

The values above are full-precision recomputations. CAID-compatible score-rounded
metrics used to reproduce the leaderboard comparisons are retained in
`results/b6/b6_point_metrics.csv`. Rank context is based on the same-reference
ROC-AUC comparison and is not inferred from literature values evaluated on
different reference sets.

MUSE-IDR is strongest on PDB-defined disorder. Its performance is not uniformly
leading on the broader NOX definition; the repository therefore does not make
a universal state-of-the-art claim.

### S1 mechanistic analysis

S1 contains 21 complete five-fold OOF runs covering seven predeclared control
variants and three seeds. The results support a contribution from multiscale
context relative to the smallest single-scale control and from the reference
fusion organization relative to the parameter-matched additive projection.
They do not establish an independent benefit for every component. See
`results/s1/s1_summary.md`.

### S2 supplementary temporal holdout

S2 contains 87 proteins and 30,728 known labeled residues after fixed source,
homology, observability, and internal-redundancy filters. On the same S2
reference, MUSE-IDR obtained ROC-AUC 0.977827 and AUPRC 0.847950. The paired
whole-protein bootstrap ROC-AUC difference was +0.015691 versus PUNCH2-Light
Released-13 (95% CI -0.000226 to +0.035732; Holm-adjusted p=0.055972) and
+0.034935 versus LoRA-DR-Suite 650M (95% CI +0.015783 to +0.059568;
Holm-adjusted p=0.001999). See `results/s2/README.md`.

S2 is a strictly filtered, database-annotation-based supplementary temporal
holdout. It does not replace CAID2/CAID3 and is not described as a prospective
or manually reannotated gold standard.

## Data and redistribution boundary

The local workstation checkout may contain processed training data and CAID
references, but these paths are ignored by Git. The public repository contains
provenance, fixed split definitions, reconstruction code, and hashes. It does
not relicense CAID labels, third-party model weights, or third-party
residue-level prediction exports. See `data/README.md`,
`DATA_AND_SOFTWARE_AVAILABILITY.md`, and `THIRD_PARTY_NOTICES.md`.

CAID2 and CAID3 labels were not used for training, tuning, calibration, model
selection, or ensemble weighting. They were accessed only after model lock for
external evaluation.

## Citation and license

Project-authored code is released under Apache-2.0. Upstream resources retain
their original terms. Citation metadata are provided in `CITATION.cff`. The
version DOI will be added after the submission release is archived in Zenodo.
