# cCorrGAN fidelity diagnosis — 2026-10-05

## Completed milestone

The original public categorical generator now runs in PyTorch, with numerical
parity against TensorFlow. Its architecture has 4,712,705 parameters. The existing
categorical generator can express the checkpoint exactly; the conversion must
reorder the dense projection from NHWC to NCHW as well as the convolution weights.
This establishes generator inference compatibility. It does not establish the
original discriminator, optimizer schedule, training membership or model selection.

Source chain:

- [Author's public notebook link](https://marti.ai/ml/2021/02/05/conditional-corrgan-colab-example.html).
- [Public Colab](https://colab.research.google.com/drive/1S9qIFOGQZwHAR5-Jqm_u_EuQ2ebRidi-).
- [Original generator](https://sp500-histo.s3-ap-southeast-1.amazonaws.com/cCorrGAN_generator.h5),
  SHA-256 `5a7f0fa2ad35e31e1b71621f7ee4a93f4c01100de74e1608b2054d0decb97984`.

The notebook copies the upper triangle, sets the diagonal to one, projects to a
correlation matrix, then applies Ward optimal leaf ordering using `1-C`.
Our ordering and upper-triangle conventions agree. Our converged float64 repair
adds a strict-PD floor of 1e-6, unlike the notebook's statsmodels default. All
scores below use the frozen CorrBench repair, not a claim of notebook-score parity.
Final ordering does not change the permutation-invariant structural features.

TensorFlow 2.21 needs a compatibility wrapper that drops the obsolete `groups=1`
keyword from Conv2DTranspose. It changes no operation or weight. On 30 fixed
noise/label pairs covering all three regimes, all 192,000 outputs agree within
atol=rtol=2e-5; maximum absolute difference is 2.6226044e-6. The script and report
are `examples/verify_published_ccorrgan.py` and
[published_checkpoint_parity.json](published_checkpoint_parity.json).
Weights are not bundled or redistributed by this package; their separate
distribution terms still need recording before a public checkpoint release.
Validation also passed 50 installed-wheel tests across all three packages,
CorrGAN wheel/sdist construction, lint and the installed sampling CLI. Scientific
dependencies were reused from the existing research environment; this is not a
clean-machine full training reproduction.

## What the existing training actually did

Reevaluated the validation-selected and epoch-200 checkpoints of all ten frozen
runs, using the same fixed 300 validation conditions/noise and converged repair.
All ten selected scores match their saved training histories exactly. No test
NPZ was opened. The data hashes, checkpoint hashes, selected validation rows,
per-feature errors and history checks are in
[fidelity_audit_20261005.json](fidelity_audit_20261005.json).

| Taiwan validation diagnostic | Selected epoch 1, five seeds | Epoch 200, same five seeds |
| --- | --- | --- |
| Frozen full feature SWD, mean ± sample s.d. | 4.586 ± 0.679 | 5.778 ± 0.313 |
| Entry-summary SWD, mean ± sample s.d. | 1.537 ± 0.223 | 1.058 ± 0.214 |
| Fraction of repaired eigenvalues below 1e-5 | 3.585% | 10.926% |

The entry-summary score improves in **all five** Taiwan runs, while the original
full score worsens in all five. The model is learning some entry properties;
epoch-1 selection is not proof that training immediately stopped learning.
Late checkpoints have more eigenvalues close to the repair floor. The full score
includes log determinant and log condition number, which expose this problem.
Small entrywise repair changes can coexist with large inverse/spectral changes.
Those properties matter for inverse-correlation downstream tasks; do not remove
them merely to improve a GAN's score. The subset metrics are exploratory
diagnostics and do not replace the frozen selection/evaluation rule.

The disjoint-row validation-vs-validation diagnostic is 0.067 on Taiwan; train
replay versus validation is 0.959. These are descriptive controls, not independent
time episodes: the existing windows overlap. Fixed-condition noise variation is
nonzero and generally larger at epoch 200; complete latent collapse is not
supported by this check. It does not rule out partial mode/coverage failures.

The public pretrained generator also shows the spectral problem on the legacy
S&P validation sample: full SWD 3.476, entry-summary SWD 0.279, 16.525% of repaired
eigenvalues below 1e-5. Its average correlation and largest eigenvalue are close
to the sampled empirical means, while log determinant and conditioning differ
strongly. Our selected S&P runs have full SWD 3.264 ± 0.137 and entry-summary SWD
0.676 ± 0.205. Thus neither agreement on one aggregate nor a low entry-summary
error establishes full distributional reproduction. The public model's training
membership is unknown and may overlap the legacy archive. Its Taiwan results
are cross-market diagnostics with different regime definitions. Neither is a
new held-out comparison.

## Reproduce

Obtain the checkpoint separately from the source above. Install
`.[published-conditional]` with the pinned CorrBench dependency. From this repo:

```bash
# Run this command in a TensorFlow environment, on CPU:
python examples/verify_published_ccorrgan.py --framework tensorflow \
  --model /path/to/cCorrGAN_generator.h5 --reference /new/path/reference.npz
# Run this command in the CorrGAN environment:
python examples/verify_published_ccorrgan.py --framework torch \
  --model /path/to/cCorrGAN_generator.h5 --reference /new/path/reference.npz \
  --report /new/path/parity.json
python research/audit_fidelity.py --research-root /path/to/corrflow-research \
  --published-model /path/to/cCorrGAN_generator.h5 \
  --output /new/path/audit.json --device cuda
```

The audit uses only the frozen `train.npz` and `validation.npz`, ten run
directories, and original pretrained generator. History checks in the committed
report independently compare selected rows against `history.json`. The research
audit is source-repo tooling and is not a core installed CLI.

## Next experiments

1. Recover and version the original cGAN discriminator/training recipe; the public
   inference notebook alone cannot certify it. Keep the continuous-score extension
   explicitly separate from original categorical cCorrGAN.
2. Predeclare a small diagnostic training comparison after inspecting that recipe.
   Report entry, spectrum, condition-wise coverage, repair and utility together.
   Preserve current scores and checkpoints; numerical repair is part of the model.
3. Validate faithful unconditional CorrGAN training separately.
4. Complete a categorical DiffeoCFM/CorrFlow/cCorrGAN comparison with identical data
   information, and include DiffeoCFM's class-wise OAS Gaussian prior on its own.
   A continuous-score comparison is a separately declared extension.
