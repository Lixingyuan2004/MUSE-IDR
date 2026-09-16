# A10: cross-architecture A2 + A8 ensemble

A10 combines the three A2 last-layer models and three A8 learned-multilayer
models using one fixed equal-logit average over all six predictions. The formal
weights are all `1/6` and do not use OOF or CAID labels.

The experiment reproduces all six source metrics, enforces exact residue and
label alignment, writes one classic-IDR score per residue, and compares the
formal six-model candidate with the existing A5/A2 equal-logit ensemble. A
separate paired, fold-stratified protein-cluster bootstrap quantifies the AUC
difference. Any OOF-tuned family weight is exploratory only.
