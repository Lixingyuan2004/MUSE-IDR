# A9: three-seed A8 ensemble

This experiment aligns the full OOF residue predictions from A8 seeds 17, 29,
and 43 and produces one classic-IDR probability per residue.

The formal method is an equal-weight mean in logit space. Its weights are fixed
without using OOF or CAID labels, so it can later be applied unchanged to CAID2
and CAID3 predictions. Probability averaging and the median are label-free
diagnostics. OOF-tuned weights are reported only as exploratory development
results and must not be called an unbiased test result.

The script verifies exact row/label alignment, reproduces every source metric,
records SHA-256 hashes, excludes unknown labels from metrics, and never accesses
CAID labels.
