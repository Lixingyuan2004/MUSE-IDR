# S2 database-curated temporal holdout evaluation

All prediction and reference hashes were verified before semantic label access. A10 remained locked; S2 was not used for training, tuning, recalibration, weighting, or model selection.

## Cohort

- Proteins: 87
- Residues: 56394
- Known labels: 30728 (54.49%)
- Positive / negative / unknown: 2999 / 27729 / 25666
- Positive prevalence among known labels: 9.76%
- Reference policy: database-curated DisProt annotations; no partial manual relabeling was applied.

## Locked MUSE-IDR A10 results

| Metric | Point estimate | Protein-bootstrap 95% CI |
| --- | ---: | ---: |
| ROC-AUC | 0.977827 | [0.964305, 0.989485] |
| AUPRC | 0.847950 | [0.654202, 0.939162] |
| APS | 0.847986 | [0.654644, 0.939182] |
| F1 at 0.5 | 0.747267 | [0.580088, 0.856131] |
| MCC at 0.5 | 0.732873 | [0.591298, 0.836352] |
| Fmax (descriptive) | 0.817666 | [0.699105, 0.897336] |
| Brier score | 0.050362 | [0.030800, 0.069598] |
| Macro ROC-AUC (two-class proteins) | 0.895952 | [0.828303, 0.953254] |

## Interpretation constraints

- Full-precision scores are primary; unknown labels (-1) are excluded.
- The fixed 0.5 threshold is locked. Fmax is descriptive and is not used to retune A10.
- Macro ROC-AUC includes only proteins containing both known classes.
- S2 supplements, but does not replace, the CAID2/CAID3 benchmarks.
- No S2 superiority claim is supported until comparator predictions are evaluated on this exact reference.
