# Submission asset inventory

This inventory describes the reorganized Bioinformatics submission candidate.
The reorganization changes repository navigation only; it does not change the
locked A10 model.

| Asset | Public location | Status |
| --- | --- | --- |
| A1–A10 code | `experiments/ablations/` | Included |
| B1–B7 code | `experiments/analysis/` | Included |
| S1/S2 code | `experiments/supplementary/` | Included |
| A10 checkpoints | `models/weights/a10_locked/` | 30 files, SHA256 locked |
| Stable user commands | `training/`, `inference/`, `evaluation/` | Included |
| Shared utilities | `src/muse_idr/` | Included |
| B6/B7/S1/S2 compact results | `results/` | Included |
| Training/split provenance | `data/manifests/` | Included |
| Example FASTA | `examples/example.fasta` | Included |

## Local-only assets

Processed biological datasets, CAID references, generated representation
caches, raw third-party prediction exports, downloaded third-party repositories,
logs, and intermediate experiment products remain locally available when
present but are excluded by `.gitignore`.

The private working directory was backed up before reorganization. That backup
is not part of the public repository or software release.
