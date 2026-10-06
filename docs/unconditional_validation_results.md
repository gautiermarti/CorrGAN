# Full unconditional reference-run validation — 2026-10-05

The full reference training budget completed but does not reproduce realistic archive distributions.

The 50-epoch legacy-archive run completed 9,800 optimizer steps. Validation selected epoch 45; its native structural score 3.21389055 reproduces exactly. The trainer and evaluator never open test data or use regime labels to condition the generator.

| Method | Structural SWD ↓ | Portfolio SWD ↓ | Precision | Recall | C2ST AUC (.5 ideal) |
|---|---:|---:|---:|---:|---:|
| CorrGAN selected (epoch 45) | 2.4875 ± 0.0151 | 2.9734 ± 0.1471 | 0.0000 | 0.0000 | 1.0000 |
| CorrGAN final (epoch 50) | 3.4176 ± 0.0828 | 3.6298 ± 0.1997 | 0.0000 | 0.0000 | 1.0000 |
| Training replay | 0.1143 ± 0.0222 | 0.1392 ± 0.0333 | 0.9080 | 0.9067 | 0.5230 |

All 3,000 generated matrices fail raw validity after symmetry and unit-diagonal preparation. Converged float64 nearest-correlation repair makes all valid, with minimum eigenvalues near the configured 1e-6 floor. Selected draws need at most 60 iterations per batch; final draws need at most 237. Repair resolves validity, not the distribution mismatch measured above.

Scores use the same 300 uniform validation rows. Feature and portfolio scales come only from training matrices. Five generation seeds describe draw variability for one fitted model. The summary uses projection seed 2027; native checkpoint selection used 44301 for both draws and projections, so its absolute score is not the summary score.

The recovered native generator separately matches the author-released TensorFlow checkpoint to maximum error 7.153e-7. That inference result is unchanged. The current discriminator, optimizer and training recipe are explicit reference choices, not verified original training code. This result therefore diagnoses this reconstruction; it does not invalidate the published CorrGAN method or certify its reproduction.

Validation has selected the checkpoint and the legacy archive is a random matrix split without dates/constituent identities. No untouched temporal generalization, conditional generation or five-training-seed claim follows. Preserve the failure while investigating the source training recipe and matched data/preprocessing.
