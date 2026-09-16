# Data and split protocol

## Training and internal validation

The formal training/validation population is the 1,133-protein filtered Classic
IDR 2018 set. In a complete local checkout it is stored in:

- `data/processed/classic_idr_2018/a1_sequences.jsonl`;
- `data/processed/classic_idr_2018/a1_frozen_esm2_manifest.jsonl`; and
- `data/processed/classic_idr_2018/homology_5fold_assignments.jsonl`.

Labels use `1` for disorder, `0` for reliable ordered residues, and `-1` for
unknown/masked residues. Unknown residues do not contribute to training loss or
metrics.

There is no single permanent validation file. Internal validation uses fixed
five-fold, homology-aware cross-validation. All proteins in a homology-connected
component remain in the same fold. Seeds 17, 29, and 43 share exactly the same
fold assignment. Machine-readable provenance is under `data/manifests/training/`
and the split configuration is `configs/data/homology_cluster_5fold.json`.

## External tests

CAID2 and CAID3 are external post-lock test sets, not training or internal
validation data. Their sequence inputs may be prepared locally, but their labels
must remain quarantined until inference outputs are locked. The labels were not
used to choose A10, fit ensemble weights, calibrate scores, or select a threshold.

See `docs/EXTERNAL_DATA.md` for expected paths, upstream sources, and hashes.

## GitHub data boundary

The repository tracks provenance and split manifests. Processed training files,
CAID references, and generated features are copied into the organized local
checkout but ignored by Git by default. Before redistributing any dataset, check
its upstream license and citation requirements independently of the Apache-2.0
software license.
