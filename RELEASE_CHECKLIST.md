# Bioinformatics submission release checklist

Target tag: `v1.1.0-bioinformatics-submission`

## Repository content

- [x] Canonical `models`, `training`, `inference`, and `evaluation` entry points.
- [x] Complete A1–A10 and B1–B7 code.
- [x] S1 and S2 code plus compact formal results.
- [x] Exactly 30 locked checkpoints: 15 A2 and 15 A8.
- [x] Example FASTA and documented inference command.
- [x] Apache-2.0 license for project-authored code.
- [x] Third-party notices and redistribution boundary.
- [x] Fresh-clone CPU verification.
- [x] Fresh-clone short GPU inference.
- [x] Secret, absolute-path, and large-file audit.
- [x] Final SHA256 manifest regenerated and verified.
- [x] Nested B6, B7, and S1 SHA256 result locks verified from the repository root.
- [ ] GitHub Actions pass on the clean-history submission commit.

## Publication

- [x] Review the complete staged diff.
- [x] Preserve the former private repository separately; do not publish its historical objects or tags.
- [x] Start the publication repository from the verified tree with a single clean root commit.
- [ ] Push the verified clean-history submission commit to the new repository.
- [ ] Make the repository public only after the publication audit passes.
- [ ] Connect `Lixingyuan2004/MUSE-IDR` to Zenodo.
- [ ] Create and publish the annotated tag `v1.1.0-bioinformatics-submission`.
- [ ] Confirm that Zenodo archived the exact tag and assigned a version DOI.
- [ ] Add the version DOI to the manuscript and repository landing page.
- [ ] Archive the manuscript Supplementary Material and record its stable link.
