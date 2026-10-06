# Unconditional reference-run validation — 2026-10-05

Before inspecting the queued 50-epoch reference reconstruction, fix this
validation-only diagnostic. Keep the legacy S&P archive and frozen configuration,
one training seed 1701, all 50 epochs, and the native minimum structural-SWD
checkpoint. The original discriminator/optimizer recipe remains unverified.
No test split is loaded by this study.

First audit complete epochs/steps, source/config/code hashes and the selected
checkpoint against the full history. Reproduce its native validation score
using the same 300 uniformly selected validation rows, seed 44301 and sampling
batch 64 on the original CUDA device. Preserve float64 repaired matrices for
this native diagnostic; this is separate from common float32 categorical scores.

Evaluate selected and final weights at generation seeds 2027–2031 against those
same 300 validation rows. Compare with unconditional uniform training replay,
using 300 draws at each seed. Report the frozen full 21-feature SWD, manifold
precision/recall, two-sample classifier AUC and portfolio-functional distance.
Every feature/function scale is fitted on training matrices only. Report raw
validity after upper-triangle symmetry/unit diagonal and repaired validity.
Use the existing converged float64 Higham repair with tolerance 1e-6, cap 1,000
and strict-PD floor 1e-6. It is part of the sampled method, not a hidden fix.

Do not stratify or interpret generated outputs by the archive's regime labels:
this generator never receives a condition. Five generation seeds describe
simulation variation for one fitted model, not five training replications.
Validation has already selected weights, so these are diagnostic scores and
cannot establish held-out generalization or faithful original training.
Preserve all declared methods/draws and unfavorable results; do not extend or
shorten training based on their scores.
