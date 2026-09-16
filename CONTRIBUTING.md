# Contributing

Thank you for considering a contribution to MUSE-IDR.

1. Open an issue describing the bug, reproducibility problem, or proposed
   change before making a large modification.
2. Create a focused branch and avoid committing datasets, caches, external
   predictor exports, or unapproved model weights.
3. Preserve the locked `v1.1.0-bioinformatics-submission` model. Any change to architecture,
   checkpoints, calibration, or ensemble weights must use a new model/version
   name and must not be reported as `A10-locked`.
4. Run `python scripts/verify_release.py`, `python
   scripts/audit_repository_layout.py`, and the relevant smoke tests.
5. Add provenance and SHA256 information for any new externally sourced
   artifact.

CAID2/CAID3 labels are evaluation-only. Contributions must not use them to tune
the locked model or a result presented as a prospective locked evaluation.
