# S2 local-homology sensitivity audit

The original global exclusion required identity >0.30 and at least 0.80
coverage on both sequences. This audit uses a permissive search prefilter and
reports several post-hoc local-similarity flags. A flag is not an automatic
exclusion because low-complexity IDR composition and short alignments require
biological review.

## Against training and CAID2/3 sequences

- Raw alignments: 396.
- Candidates flagged by at least one sensitivity rule: 21 / 128.
- Original global-rule hits among the already screened candidates: 0.

## Within the S2 candidate cohort

- Non-self raw alignments: 86.
- Candidates flagged by at least one sensitivity rule: 23 / 128.
- Global-rule candidate redundancy: 15 candidates.

No model scores or CAID labels were read. Decisions based on these flags must be
made before S2 predictions are generated.
