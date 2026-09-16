# B4 PUNCH2-Light reproduction

Two variants are kept separate. `paper8` uses three One-Hot CNN-L12 and five
ProtTrans CNN-L12 members. `released13` additionally uses the five ProtTrans
CNN-L3 members present in the locked repository entry point. Both variants use
the author's uniform mean of member probabilities. The 0.35 threshold affects
only the auxiliary binary CAID column; ranking metrics use continuous scores.

Features follow `deemeng/embedding` revision
`28fe5c371999d27e67563b6cddad4e669653f22d`. One-Hot uses 21 channels;
ProtTrans uses `Rostlab/prot_t5_xl_uniref50` with 1024 channels. The author's
U/Z/O/B-to-X mapping and no-truncation policy are preserved. Input must be a
sequence-only FASTA, and predictions must be locked before CAID labels are
read.

Run local tests with:

```bash
python experiments/analysis/b4_baseline_reproduction/smoke_test_punch2_light.py
python experiments/analysis/b4_baseline_reproduction/smoke_test_punch2_light_end_to_end.py
```
