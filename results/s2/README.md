# S2 supplementary temporal holdout

S2 is a strictly filtered, database-annotation-based supplementary temporal
holdout. It supplements but does not replace CAID2/CAID3 and is not presented as
a prospective or manually reannotated gold standard.

## Cohort and evaluation

- 87 proteins and 56,394 total residues;
- 30,728 known labels: 2,999 positive and 27,729 negative residues;
- 25,666 unknown residues excluded from metric calculations;
- 22 proteins contain both classes for per-protein ROC-AUC;
- 2,000 whole-protein bootstrap replicates, seed 20260907; and
- two primary ROC-AUC contrasts corrected together with Holm's method.

| Model | ROC-AUC | AUPRC | F1@0.5 | MCC@0.5 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| MUSE-IDR | 0.977827 | 0.847950 | 0.747267 | 0.732873 | 0.050362 |
| PUNCH2-Light Released-13 | 0.962136 | 0.806241 | 0.721660 | 0.692009 | 0.040160 |
| LoRA-DR-Suite 650M | 0.942891 | 0.602560 | 0.646298 | 0.612520 | 0.063172 |

The paired ROC-AUC difference was +0.015691 versus PUNCH2-Light Released-13
(95% CI -0.000226 to +0.035732; Holm-adjusted p=0.055972) and +0.034935 versus
LoRA-DR-Suite 650M (95% CI +0.015783 to +0.059568; Holm-adjusted p=0.001999).
Thus S2 supports an advantage over the reproduced LoRA comparator but does not
establish a conventional 0.05-level difference from PUNCH2-Light.

## Included material

- `three_model_evaluation/`: point estimates, intervals, paired statistics,
  per-protein metrics, bootstrap samples, and the formal report;
- `source_audit/` and `local_homology_audit/`: global and local homology checks;
- `finalization/`: cohort identifiers and finalization reports, without labels;
- `a10_prediction/`: MUSE-IDR label-blind predictions and inference reports; and
- `a10_evaluation/`: single-model S2 evaluation outputs.

Third-party residue-level prediction exports and S2 reference labels are not
redistributed here. The construction and evaluation code is available under
`experiments/supplementary/s2_independent_holdout/`.
