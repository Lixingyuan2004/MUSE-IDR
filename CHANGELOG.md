# Changelog

## 1.1.0 - 2026-09-15

- Reorganized stable model, training, inference, and evaluation entry points.
- Moved the 30 locked checkpoints to `models/weights/a10_locked/` without changing file contents.
- Renamed the shared Python package from the early working name to `muse_idr`.
- Added complete S1 mechanistic-analysis and S2 supplementary-holdout code.
- Added compact formal S1/S2 results while excluding third-party raw predictions and S2 labels.
- Updated documentation and validation rules for a clean public repository and Zenodo archive.

## Historical locked model

The internal scientific identifier remains `A10-locked`. Repository versioning
does not imply retraining or reselection of the model.
