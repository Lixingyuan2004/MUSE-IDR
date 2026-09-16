# Data organization and publication boundary

This directory separates locally available research data from the manifests
that may be published safely.

| Path | Role | GitHub policy |
| --- | --- | --- |
| `processed/classic_idr_2018/` | 1,133-protein training population, labels, cached-manifest references, and fixed homology-aware folds | local only |
| `manifests/training/` | source provenance, split definitions, and homology/leakage audits | tracked |
| `manifests/holdouts/` | external-test sentinels used to detect forbidden overlap/access | tracked |
| `caid_holdout/` | locally obtained CAID reference labels | local only |
| `external/` | prepared sequence-only external inputs and local reports | local only |
| `cache/` | generated ESM-2 representations | local only; regenerate as needed |

Internal validation uses the fixed five-fold assignments in
`processed/classic_idr_2018/homology_5fold_assignments.jsonl`. CAID2 and CAID3
are post-lock external tests, not validation sets and not tuning data.

See `docs/DATA_PROTOCOL.md`, `docs/DATA_SPLITS.md`, and
`docs/EXTERNAL_DATA.md` for provenance and reconstruction instructions.
