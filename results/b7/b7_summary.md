# B7 locked-model error, robustness, and complementarity analysis

B7 is descriptive and post-lock. B6 remains the confirmatory statistical analysis.
No CAID labels were used to train, tune, select, or recalibrate A10.

## Overall full-precision metrics

| Dataset | Track | Model | ROC-AUC | AUPRC | APS | F1@0.5 | MCC@0.5 | Brier |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CAID2 | disorder_nox | A10-locked | 0.780767 | 0.360582 | 0.360621 | 0.514474 | 0.408428 | 0.302231 |
| CAID2 | disorder_nox | LoRA-DR-Suite 650M | 0.834845 | 0.588356 | 0.588138 | 0.524082 | 0.392729 | 0.162292 |
| CAID2 | disorder_nox | PUNCH2-Light Paper-8 | 0.777540 | 0.366707 | 0.366736 | 0.514613 | 0.388974 | 0.209265 |
| CAID2 | disorder_nox | PUNCH2-Light Released-13 | 0.779453 | 0.374469 | 0.374507 | 0.515857 | 0.393901 | 0.225344 |
| CAID2 | disorder_pdb | A10-locked | 0.957568 | 0.933521 | 0.933522 | 0.866783 | 0.813985 | 0.060855 |
| CAID2 | disorder_pdb | LoRA-DR-Suite 650M | 0.920179 | 0.840198 | 0.840114 | 0.734498 | 0.647185 | 0.096925 |
| CAID2 | disorder_pdb | PUNCH2-Light Paper-8 | 0.948526 | 0.907807 | 0.907808 | 0.829713 | 0.776109 | 0.075609 |
| CAID2 | disorder_pdb | PUNCH2-Light Released-13 | 0.949791 | 0.911856 | 0.911857 | 0.840483 | 0.786826 | 0.069286 |
| CAID3 | disorder_nox | A10-locked | 0.852106 | 0.605800 | 0.605838 | 0.645557 | 0.518414 | 0.238952 |
| CAID3 | disorder_nox | LoRA-DR-Suite 650M | 0.859995 | 0.695579 | 0.695515 | 0.645162 | 0.502331 | 0.159647 |
| CAID3 | disorder_nox | PUNCH2-Light Paper-8 | 0.826017 | 0.541880 | 0.541924 | 0.644691 | 0.502622 | 0.175619 |
| CAID3 | disorder_nox | PUNCH2-Light Released-13 | 0.839937 | 0.579245 | 0.579286 | 0.645728 | 0.505286 | 0.177571 |
| CAID3 | disorder_pdb | A10-locked | 0.953584 | 0.930965 | 0.930966 | 0.845464 | 0.771543 | 0.077020 |
| CAID3 | disorder_pdb | LoRA-DR-Suite 650M | 0.920137 | 0.866839 | 0.866743 | 0.773770 | 0.679397 | 0.098962 |
| CAID3 | disorder_pdb | PUNCH2-Light Paper-8 | 0.950922 | 0.922517 | 0.922518 | 0.837295 | 0.782478 | 0.080337 |
| CAID3 | disorder_pdb | PUNCH2-Light Released-13 | 0.952490 | 0.925861 | 0.925862 | 0.847906 | 0.792997 | 0.073060 |

## Overall threshold-error complementarity

| Dataset | Track | Comparator | Disagreement | A10-only correct | Comparator-only correct | Either correct | Score Spearman |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| CAID2 | disorder_nox | LoRA-DR-Suite 650M | 0.2050 | 0.0575 | 0.1474 | 0.8139 | 0.7766 |
| CAID2 | disorder_nox | PUNCH2-Light Paper-8 | 0.1180 | 0.0390 | 0.0790 | 0.7454 | 0.8984 |
| CAID2 | disorder_nox | PUNCH2-Light Released-13 | 0.1098 | 0.0383 | 0.0715 | 0.7379 | 0.9000 |
| CAID2 | disorder_pdb | LoRA-DR-Suite 650M | 0.1138 | 0.0880 | 0.0257 | 0.9501 | 0.8151 |
| CAID2 | disorder_pdb | PUNCH2-Light Paper-8 | 0.0623 | 0.0376 | 0.0247 | 0.9491 | 0.8293 |
| CAID2 | disorder_pdb | PUNCH2-Light Released-13 | 0.0560 | 0.0324 | 0.0235 | 0.9479 | 0.8276 |
| CAID3 | disorder_nox | LoRA-DR-Suite 650M | 0.1805 | 0.0672 | 0.1132 | 0.8452 | 0.8161 |
| CAID3 | disorder_nox | PUNCH2-Light Paper-8 | 0.1049 | 0.0369 | 0.0680 | 0.7999 | 0.8934 |
| CAID3 | disorder_nox | PUNCH2-Light Released-13 | 0.0956 | 0.0341 | 0.0614 | 0.7934 | 0.9000 |
| CAID3 | disorder_pdb | LoRA-DR-Suite 650M | 0.1206 | 0.0778 | 0.0428 | 0.9421 | 0.7920 |
| CAID3 | disorder_pdb | PUNCH2-Light Paper-8 | 0.0912 | 0.0416 | 0.0495 | 0.9488 | 0.8351 |
| CAID3 | disorder_pdb | PUNCH2-Light Released-13 | 0.0842 | 0.0358 | 0.0484 | 0.9476 | 0.8385 |

## Interpretation constraints

- Subgroup and complementarity results are descriptive, not new confirmatory tests.
- The fixed 0.5 threshold is used only for error diagnostics.
- Fmax is reported for characterization and is not used to modify the locked model.
- Masked CAID labels are excluded; complete proteins remain the reporting unit where applicable.
- Any future model change must be named and evaluated as a new model, never as A10-locked.
