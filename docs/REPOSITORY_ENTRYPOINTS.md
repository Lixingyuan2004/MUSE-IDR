# Repository entry points

| Task | Stable entry point | Canonical implementation or asset |
| --- | --- | --- |
| Predict FASTA | `inference/predict.py` | `experiments/final_models/a10_locked_inference/predict_a10.py` |
| Inspect model code | `models/a2.py`, `models/a8.py`, `models/a10.py` | A2/A8 experiment model definitions and A10 loader |
| Verify weights | `scripts/verify_release.py` | `models/weights/a10_locked/` and the A10 lock manifest |
| Train experiments | `training/run_experiment.py` | A1, A2, A3, A4, A6, A7, and A8 trainers |
| Evaluate predictions | `evaluation/evaluate_residue_predictions.py` | Shared metric implementation under `src/muse_idr/` |
| Reproduce A experiments | `experiments/ablations/` | Exactly one directory for each A1–A10 stage |
| Reproduce B analyses | `experiments/analysis/` | Exactly one directory for each B1–B7 stage |
| Reproduce S1/S2 | `experiments/supplementary/` | Mechanistic controls and temporal holdout workflow |
| Read paper results | `results/` | B6, B7, S1, and S2 compact outputs |

Run these checks before committing a release:

```bash
python scripts/audit_github_readiness.py
python scripts/audit_repository_layout.py
python scripts/verify_release.py
python -m compileall -q predict_muse_idr.py models training inference evaluation src scripts experiments tests
```
