# External data required for analysis reproduction

CAID references are intentionally excluded from this software package. Obtain
the official CAID2 and CAID3 reference files, place them at the paths below,
and confirm their SHA256 values before running B6 or B7.

| Dataset | Track | Expected relative path | SHA256 |
| --- | --- | --- | --- |
| CAID2 | disorder_nox | `data/caid_holdout/references/caid2/disorder_nox.fasta` | `4e7cf7e2aca696dd0c16f7626574ba667770589d634178c603b36b382eadb99d` |
| CAID2 | disorder_pdb | `data/caid_holdout/references/caid2/disorder_pdb.fasta` | `559a70179aecaee5a7178acd2e16de4e5adbcca951d41ec5543a4270452d896f` |
| CAID3 | disorder_nox | `data/external/caid3_locked/reference_quarantine/caid3_disorder_nox_reference.fasta` | `78a43f01b26387236ceb363f368b87f1bdf3ec7502a80ec5c86df63ef3487d41` |
| CAID3 | disorder_pdb | `data/external/caid3_locked/reference_quarantine/caid3_disorder_pdb_reference.fasta` | `b65848395557c751a818dc38049c89d95534c05fbe6f14651e9f5bfd2bb60469` |

Official CAID challenge resources are available from
`https://caid.idpcentral.org/`. Users are responsible for complying with the
upstream data terms. Do not use these labels to alter the locked MUSE-IDR model.

## Third-party comparator predictions

Raw residue-level outputs from LoRA-DR-Suite and PUNCH2-Light are intentionally
excluded from the public software package because their redistribution terms
are not explicit. Regenerate them from the upstream model or source at the
locked revisions below, place them at the expected paths, and verify their
SHA256 values before rerunning B6 or B7. The B6 and B7 lock manifests contain
the same paths and hashes as machine-readable provenance.

| Dataset | Comparator | Upstream identifier or revision | Expected relative path | SHA256 |
| --- | --- | --- | --- | --- |
| CAID2 | LoRA-DR-Suite 650M | `CQSB/esm2_650M-LoRA-ID-DisProt7` | `outputs/analysis/b4_baseline_reproduction/caid2/lora_650m_disprot7_predictions.tsv.gz` | `2ef8fc3718a6719de75b404c78868669384782a2ca2f6adc01ede90bdde1ddb4` |
| CAID3 | LoRA-DR-Suite 650M | `CQSB/esm2_650M-LoRA-ID-DisProt7` | `outputs/analysis/b4_baseline_reproduction/caid3/lora_650m_disprot7_predictions.tsv.gz` | `f0833791d681c9d03c433384e417e7bbff70ca550b74076b72bef9e311ba03bc` |
| CAID2 | PUNCH2-Light Paper-8 | `deemeng/punch2_light@6c7935b3597c056d2e6b3845bb54fc101c5bc574` | `outputs/analysis/b4_baseline_reproduction/punch2_light/caid2/predictions/punch2_light_paper8_predictions.tsv.gz` | `1c45095d0efdd31decdf95fc90c1c98523485f65909affcc9b3b86931a569f9b` |
| CAID2 | PUNCH2-Light Released-13 | `deemeng/punch2_light@6c7935b3597c056d2e6b3845bb54fc101c5bc574` | `outputs/analysis/b4_baseline_reproduction/punch2_light/caid2/predictions/punch2_light_released13_predictions.tsv.gz` | `6ab6032502c59c807f379ede2ec4acabf5516946fd7510a93859cdfc19ce7c2e` |
| CAID3 | PUNCH2-Light Paper-8 | `deemeng/punch2_light@6c7935b3597c056d2e6b3845bb54fc101c5bc574` | `outputs/analysis/b4_baseline_reproduction/punch2_light/caid3/predictions/punch2_light_paper8_predictions.tsv.gz` | `631ca06bf52d4149c1006d09f6d4f01eacaf20983599a55b1299f18c7372722c` |
| CAID3 | PUNCH2-Light Released-13 | `deemeng/punch2_light@6c7935b3597c056d2e6b3845bb54fc101c5bc574` | `outputs/analysis/b4_baseline_reproduction/punch2_light/caid3/predictions/punch2_light_released13_predictions.tsv.gz` | `c67052389772846d41c7123b0d771bb8bf2659a3731cb62e6d9bd14b86776dc7` |
