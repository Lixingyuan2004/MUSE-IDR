# S1 mechanistic analysis

S1 is a post-lock, explanation-only analysis. It does not modify, reweight, or
reselect A10 and does not use CAID1/2/3 labels.

The experiment contains seven predeclared control variants, three random seeds,
and complete five-fold OOF evaluation: 21 jobs and 105 trained folds. The public
directory contains the compact formal analysis outputs:

- `s1_seed_metrics.csv` and `s1_seed_summary.csv`;
- `s1_ensemble_metrics.csv`;
- `s1_pairwise_bootstrap.csv`;
- `s1_bootstrap_auc_samples.npz`;
- `s1_report.json` and `s1_summary.md`; and
- SHA256 manifests for the analysis bundle.

The approximately 880 MiB of intermediate per-fold and per-residue predictions
remain in the research archive and are not duplicated in Git. They can be
regenerated with `experiments/supplementary/s1_mechanistic_ablation/`.
