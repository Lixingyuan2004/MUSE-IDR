# GitHub publication checklist

The organized repository includes the files normally required for a scientific
software release:

- [x] `README.md` and `README_CN.md`
- [x] Apache-2.0 `LICENSE`
- [x] `CITATION.cff` with author, ORCID, repository, and version
- [x] `pyproject.toml` and `requirements-paper.txt`
- [x] `MODEL_CARD.md`
- [x] `CHANGELOG.md`
- [x] `THIRD_PARTY_NOTICES.md`
- [x] `CONTRIBUTING.md` and `SECURITY.md`
- [x] `.gitignore` and `.gitattributes`
- [x] GitHub Actions verification workflow
- [x] issue template
- [x] 30 locked weights plus SHA256 lock manifest
- [x] A1-A10, B1-B7, C1/C2, inference, and evaluation source
- [x] machine-readable training/split/leakage manifests
- [x] nested B6, B7, and S1 SHA256 result locks
- [x] single-root-commit publication history without historical third-party prediction exports
- [ ] assign the Zenodo software DOI and add it to the release metadata
- [ ] independently decide whether a results-archive DOI is needed
- [ ] make the repository public only after the data/third-party review

Before every push, run:

```bash
python scripts/audit_repository_layout.py
python scripts/verify_release.py
git status --short
```

Only files shown by `git status`/`git ls-files` are candidates for GitHub. The
local training data and CAID references should appear as ignored, not untracked.
