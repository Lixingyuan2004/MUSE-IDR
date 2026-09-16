# Data and software availability

The project-authored MUSE-IDR source code, locked inference configuration, 30
formal prediction-head checkpoints, example input, tests, and compact analysis
outputs are released under Apache-2.0 at
`https://github.com/Lixingyuan2004/MUSE-IDR`. The version corresponding to the
Bioinformatics submission is identified by the immutable tag
`v1.1.0-bioinformatics-submission` and archived in Zenodo under the
version-specific DOI `10.5281/zenodo.22784471`. The concept DOI
`10.5281/zenodo.22784470` resolves to the latest archived release.

The repository includes complete A1–A10, B1–B7, S1, and S2 analysis code. It
also includes public provenance manifests, fixed split definitions, hashes,
and reconstruction instructions. Biological source databases, CAID reference
labels, third-party model weights, and third-party residue-level prediction
exports are not relicensed or redistributed by this project. They must be
obtained from their original providers under the applicable terms. Expected
identifiers, revisions, paths, and hashes are documented in
`docs/EXTERNAL_DATA.md` and `THIRD_PARTY_NOTICES.md`.

CAID2 and CAID3 labels were accessed only after A10 was locked and were not
used for training, tuning, calibration, model selection, or ensemble weighting.
S1 is explanation-only and does not modify A10. S2 is a supplementary,
database-annotation-based temporal holdout constructed under fixed filtering
rules and does not replace the official CAID external benchmarks.
