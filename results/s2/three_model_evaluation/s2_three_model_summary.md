# S2 three-model locked comparison

All protocol, reference, prediction, and provenance hashes were verified before semantic label access. All three methods were evaluated on the same 87 proteins and the same 30,728 known residues. Unknown labels (-1) were excluded before every metric.

## Point estimates and paired protein-bootstrap intervals

| Model | ROC-AUC | AUPRC | APS | F1@0.5 | MCC@0.5 | Fmax | Brier | Macro ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MUSE-IDR A10 | 0.977827 [0.964305, 0.989485] | 0.847950 [0.654202, 0.939162] | 0.847986 [0.654644, 0.939182] | 0.747267 [0.580088, 0.856131] | 0.732873 [0.591298, 0.836352] | 0.817666 [0.699105, 0.897336] | 0.050362 [0.030800, 0.069598] | 0.895952 [0.828303, 0.953254] |
| PUNCH2-Light Released-13 | 0.962136 [0.936804, 0.982734] | 0.806241 [0.622133, 0.907682] | 0.806272 [0.622283, 0.907713] | 0.721660 [0.561132, 0.825979] | 0.692009 [0.529099, 0.802088] | 0.732162 [0.582020, 0.841119] | 0.040160 [0.025921, 0.058342] | 0.893298 [0.833843, 0.945977] |
| LoRA-DR-Suite ESM2-650M DisProt7 | 0.942891 [0.920963, 0.961401] | 0.602560 [0.432914, 0.759959] | 0.602789 [0.433413, 0.760096] | 0.646298 [0.477109, 0.765336] | 0.612520 [0.460522, 0.726373] | 0.647277 [0.502302, 0.776353] | 0.063172 [0.048699, 0.080630] | 0.894022 [0.833187, 0.943871] |

The brackets are 95% percentile intervals from 2,000 paired whole-protein bootstrap replicates. Lower Brier score is better; higher values are better for the other metrics.

## Predeclared primary ROC-AUC contrasts

All deltas are MUSE-IDR A10 minus comparator. Holm correction covers exactly the two predeclared primary ROC-AUC contrasts.

| Comparator | ROC-AUC delta | 95% CI | P(delta > 0) | Raw two-sided p | Holm p | Interpretation |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| PUNCH2-Light Released-13 | +0.015691 | [-0.000226, +0.035732] | 0.9725 | 0.055972 | 0.055972 | 95% CI includes zero |
| LoRA-DR-Suite ESM2-650M DisProt7 | +0.034935 | [+0.015783, +0.059568] | 1.0000 | 0.0009995 | 0.001999 | supports higher MUSE-IDR ROC-AUC |

## Key secondary AUPRC contrasts

AUPRC is the predeclared key secondary metric. Its paired deltas and intervals are reported descriptively; no additional post hoc hypothesis-test family was introduced.

| Comparator | AUPRC delta | 95% CI | P(delta > 0) |
| --- | ---: | --- | ---: |
| PUNCH2-Light Released-13 | +0.041709 | [-0.031352, +0.107095] | 0.8785 |
| LoRA-DR-Suite ESM2-650M DisProt7 | +0.245391 | [+0.098098, +0.390268] | 0.9995 |

## Threshold and interpretation constraints

- F1 and MCC in the main table use the same fixed threshold of 0.5 for all three models and are descriptive.
- At PUNCH2-Light's separately declared official threshold of 0.35, its F1 is 0.691856 and MCC is 0.658812; these values are not mixed with the common-threshold comparison.
- Macro ROC-AUC averages only the 22 proteins containing both known classes.
- No S2 labels or results were used for training, fine-tuning, calibration, threshold tuning, ensemble reweighting, or A10 reselection.
- This independent temporal holdout supplements the locked CAID2/CAID3 results; it does not justify a universal state-of-the-art claim.
