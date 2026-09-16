## Summary

Describe the scientific or engineering change and its intended scope.

## Verification

- [ ] `python scripts/audit_repository_layout.py` passes.
- [ ] `python scripts/verify_release.py` passes.
- [ ] Relevant smoke/unit tests pass.
- [ ] No CAID labels were used for training, tuning, recalibration, or model selection.
- [ ] Locked checkpoints, prediction files, and result hashes were not changed unintentionally.
- [ ] No local-only data, third-party weights, credentials, caches, or logs are included.

## Reproducibility impact

List any changed configuration, input hash, random seed, model member, or result
artifact. If none changed, state that explicitly.
