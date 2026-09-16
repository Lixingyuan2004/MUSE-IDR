# S1 post-lock mechanistic ablation

S1 is explanation-only. It does not modify, reweight, or reselect the locked A10 model.
All 358 S1 result files and all locked A2/A8 references were SHA256-verified before label access.

## Three-seed robustness

| Model | ROC-AUC mean +/- SD | PR-AUC mean +/- SD | Macro ROC-AUC mean +/- SD |
| --- | ---: | ---: | ---: |
| A2 gated four-scale | 0.956764 +/- 0.002944 | 0.964313 +/- 0.003424 | 0.872645 +/- 0.009816 |
| A8 learned layer mix | 0.960049 +/- 0.000958 | 0.967033 +/- 0.000692 | 0.857863 +/- 0.004292 |
| Direct ungated addition | 0.956736 +/- 0.003687 | 0.964043 +/- 0.003819 | 0.873748 +/- 0.007631 |
| Parameter-matched additive projection | 0.954376 +/- 0.006633 | 0.961361 +/- 0.007883 | 0.871880 +/- 0.007939 |
| Single scale k=3 | 0.955618 +/- 0.001125 | 0.963455 +/- 0.002081 | 0.867642 +/- 0.001342 |
| Single scale k=7 | 0.955744 +/- 0.000905 | 0.963540 +/- 0.001765 | 0.868078 +/- 0.003502 |
| Single scale k=15 | 0.955570 +/- 0.001524 | 0.963674 +/- 0.002222 | 0.873763 +/- 0.004571 |
| Single scale k=31 | 0.956239 +/- 0.002236 | 0.963992 +/- 0.002983 | 0.869943 +/- 0.005818 |
| Fixed uniform layer mix | 0.959869 +/- 0.000725 | 0.966799 +/- 0.000662 | 0.859556 +/- 0.003281 |

## Fixed equal-logit three-seed ensembles

| Model | ROC-AUC | PR-AUC | Macro ROC-AUC |
| --- | ---: | ---: | ---: |
| A2 gated four-scale | 0.963497 | 0.970121 | 0.879980 |
| A8 learned layer mix | 0.963342 | 0.969753 | 0.861402 |
| Direct ungated addition | 0.963687 | 0.970260 | 0.881559 |
| Parameter-matched additive projection | 0.961914 | 0.968890 | 0.880017 |
| Single scale k=3 | 0.960439 | 0.967424 | 0.873618 |
| Single scale k=7 | 0.961202 | 0.968296 | 0.874288 |
| Single scale k=15 | 0.961317 | 0.968654 | 0.880473 |
| Single scale k=31 | 0.963691 | 0.970079 | 0.876531 |
| Fixed uniform layer mix | 0.963573 | 0.969809 | 0.862424 |

## Paired fold-stratified whole-protein bootstrap

Positive deltas favor the reference module. The seven predeclared ROC-AUC comparisons share one Holm family.

| Comparison | Reference - control ROC delta | 95% CI | Raw p | Holm p | Interpretation |
| --- | ---: | --- | ---: | ---: | --- |
| sigmoid gate versus direct residual addition | -0.000190 | [-0.000612, +0.000234] | 0.378811 | 1 | no_holm_significant_difference |
| sigmoid gate versus a parameter-matched additive projection | +0.001583 | [+0.000602, +0.002685] | 0.001999 | 0.013993 | supports_reference_module |
| four-scale context versus kernel 3 only | +0.003058 | [+0.001349, +0.004931] | 0.0029985 | 0.017991 | supports_reference_module |
| four-scale context versus kernel 7 only | +0.002295 | [+0.000023, +0.004640] | 0.0509745 | 0.254873 | no_holm_significant_difference |
| four-scale context versus kernel 15 only | +0.002179 | [-0.000138, +0.004656] | 0.0709645 | 0.283858 | no_holm_significant_difference |
| four-scale context versus kernel 31 only | -0.000194 | [-0.001939, +0.001503] | 0.829585 | 1 | no_holm_significant_difference |
| learned last-four-layer mix versus a fixed uniform mean | -0.000231 | [-0.001454, +0.000992] | 0.743628 | 1 | no_holm_significant_difference |

## Guardrails

- Bootstrap resampling uses whole proteins and preserves the five OOF fold sizes.
- Equal-logit ensemble weights are fixed at 1/3 and are not fitted to labels.
- CAID1/2/3 labels are not used in S1 training, tuning, weighting, or model selection.
- S1 conclusions explain components; they do not change the locked A10 model or its external results.
