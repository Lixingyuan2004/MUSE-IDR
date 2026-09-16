# B6 paired protein-cluster statistics

All predictions and references were SHA256-verified before label access. Bootstrap resampling uses whole proteins, and all deltas are A10 minus comparator.

## ROC-AUC comparisons

| Dataset | Track | Comparator | Full-precision ROC delta | 95% CI | Bootstrap p (Holm) | DeLong p (Holm) |
| --- | --- | --- | ---: | --- | ---: | ---: |
| CAID2 | disorder_nox | LoRA-DR-Suite 650M | -0.054077 | [-0.097333, -0.015963] | 0.17991 | 0 |
| CAID2 | disorder_nox | PUNCH2-Light Released-13 | +0.001314 | [-0.013908, +0.015095] | 1 | 0.103776 |
| CAID2 | disorder_nox | PUNCH2-Light Paper-8 | +0.003228 | [-0.011305, +0.017164] | 1 | 5.22981e-06 |
| CAID2 | disorder_pdb | LoRA-DR-Suite 650M | +0.037389 | [+0.022748, +0.053415] | 0.047976 | 0 |
| CAID2 | disorder_pdb | PUNCH2-Light Released-13 | +0.007777 | [-0.000755, +0.016671] | 1 | 8.08721e-49 |
| CAID2 | disorder_pdb | PUNCH2-Light Paper-8 | +0.009042 | [+0.000996, +0.017557] | 0.511744 | 3.03583e-67 |
| CAID3 | disorder_nox | LoRA-DR-Suite 650M | -0.007888 | [-0.047929, +0.032786] | 1 | 6.19167e-12 |
| CAID3 | disorder_nox | PUNCH2-Light Released-13 | +0.012170 | [-0.003445, +0.026805] | 1 | 1.84339e-62 |
| CAID3 | disorder_nox | PUNCH2-Light Paper-8 | +0.026090 | [+0.002379, +0.054622] | 0.509745 | 3.3487e-245 |
| CAID3 | disorder_pdb | LoRA-DR-Suite 650M | +0.033447 | [+0.019244, +0.048374] | 0.047976 | 0 |
| CAID3 | disorder_pdb | PUNCH2-Light Released-13 | +0.001094 | [-0.008496, +0.011684] | 1 | 0.103776 |
| CAID3 | disorder_pdb | PUNCH2-Light Paper-8 | +0.002662 | [-0.007192, +0.013378] | 1 | 7.03376e-06 |

## Interpretation policy

- A positive delta with a 95% CI entirely above zero supports A10 superiority.
- A CI crossing zero is reported as comparable or as a higher point estimate, not as significant superiority.
- Primary Holm correction covers A10 vs LoRA and A10 vs Released-13 across all four tracks and six metrics.
- Paper-8 comparisons are supplementary and corrected in a separate family.
- CAID2/CAID3 labels remain evaluation-only and must not be used to retune A10.
