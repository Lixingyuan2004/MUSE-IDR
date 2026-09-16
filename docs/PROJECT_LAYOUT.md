# Project layout

The repository separates stable user entry points, experimental provenance,
locked artifacts, compact manuscript results, and locally held biological data.

| Path | Purpose | Public release policy |
| --- | --- | --- |
| `models/a2.py`, `models/a8.py`, `models/a10.py` | Public model interfaces | Tracked |
| `models/weights/a10_locked/` | 30 SHA256-locked A2/A8 head checkpoints | Tracked |
| `training/` | Stable training dispatcher | Tracked |
| `inference/` | Stable label-free FASTA inference entry point | Tracked |
| `evaluation/` | Stable residue-evaluation entry point | Tracked |
| `src/muse_idr/` | Shared data, evaluation, model, and training utilities | Tracked |
| `experiments/ablations/a1_*` … `a10_*` | A1–A10 development experiments | Tracked |
| `experiments/analysis/b1_*` … `b7_*` | External comparison and statistics | Tracked |
| `experiments/supplementary/s1_*` | Post-lock mechanistic analysis | Tracked |
| `experiments/supplementary/s2_*` | Supplementary temporal holdout workflow | Tracked |
| `results/b6/`, `results/b7/` | CAID statistical and diagnostic outputs | Tracked |
| `results/s1/`, `results/s2/` | Compact S1/S2 manuscript outputs | Tracked |
| `data/manifests/` | Provenance, fixed splits, and leakage sentinels | Tracked |
| `data/processed/`, `data/caid_holdout/`, `data/external/` | Local biological data | Ignored by Git |
| `cache/`, `logs/`, `third_party/`, `tools/` | Generated or externally obtained assets | Ignored by Git |

## Canonical commands

- End-user prediction: `python inference/predict.py ...`
- Training dispatcher: `python training/run_experiment.py --list`
- Generic evaluation: `python evaluation/evaluate_residue_predictions.py ...`
- Repository audit: `python scripts/audit_repository_layout.py`
- Release verification: `python scripts/verify_release.py`

The old singular `model/` and `experiments/final_model/` paths are not part of
the submission layout. Their contents were either moved into the canonical
locations above or retained only in the separate pre-reorganization backup.
