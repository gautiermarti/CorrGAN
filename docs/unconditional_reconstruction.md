# Unconditional CorrGAN: recovered generator and reference training

The author's [pretrained-model release](https://marti.ai/qfin/2020/08/11/corrgan-pretrained-models.html)
provides an 80-dimensional TensorFlow SavedModel. Its serialized Keras metadata
and named weights recover the generator exactly enough to test inference across
frameworks. The [CorrGAN paper](https://arxiv.org/abs/1910.09504) describes a DCGAN,
hierarchical ordering and nearest-correlation projection. The available evidence
does not recover the exact original discriminator and optimizer training recipe.

## What is verified

The published generator uses 100 latent coordinates, a bias-free dense layer to
20 x 20 x 256, batch normalization and LeakyReLU(.3), then three bias-free 5 x 5
transposed convolutions. Their output channels are 128, 64, 1 and strides are
1, 2, 2; the first two use batch normalization and LeakyReLU, the final uses tanh.
Batch-normalization epsilon is .001 and TensorFlow momentum is .99. The native
implementation preserves flattened NHWC dense ordering and the asymmetric
TensorFlow SAME crop for stride-two odd-kernel deconvolutions.

[The parity report](original_checkpoint_parity.json) compares 30 fixed latent
draws, 192,000 raw output entries, against TensorFlow 2.21. The maximum absolute
difference is 7.152557373046875e-7. Source SavedModel hashes, runtime versions,
conversion-script hash and reference-array hash are in that report. Tolerance is
2e-5 absolute and relative. This is an inference check for one released model,
not a check of training dynamics, training membership, or held-out fidelity.

The original weights and TensorFlow reference arrays remain local artifacts.
They are not included in the software release; their redistribution permission
has not been established. To repeat the conversion with a separately obtained
checkpoint, use two environments:

```bash
# Environment with TensorFlow; no GPU is needed for this check.
CUDA_VISIBLE_DEVICES='' python examples/verify_original_corrgan.py \
  --framework tensorflow --source /path/to/generator_80d \
  --reference /tmp/original_corrgan_reference.npz
# Environment with this package and PyTorch.
python examples/verify_original_corrgan.py --framework torch \
  --reference /tmp/original_corrgan_reference.npz \
  --report /tmp/original_corrgan_parity.json \
  --checkpoint /tmp/original_corrgan_native.pt
```

The native checkpoint works with `corrgan-sample-unconditional`. TensorFlow is
not required for subsequent training or sampling.

## Explicit reconstruction choices

The reference discriminator has 64/128-channel 5 x 5 stride-two convolutions,
LeakyReLU(.3), dropout(.3), and a linear logit. We use Glorot initialization,
Adam with learning rate 1e-4 and betas (.9, .999), summed real/fake binary
cross-entropy for the discriminator and non-saturating generator loss. Both
gradients are computed from one shared fake batch before either optimizer steps.
This follows the mathematical pattern of the official
[TensorFlow DCGAN tutorial](https://www.tensorflow.org/tutorials/generative/dcgan),
with dimensions adapted to correlation matrices. It is a documented reference
choice, not evidence that the original CorrGAN used that recipe. PyTorch and
historical Keras batch-normalization running-variance conventions can also differ
during training even though imported evaluation states match.

The trainer loads only `matrices` from train/validation files; labels and scores
are never loaded. Its default configuration uses 50 epochs and batch size 64.
A fixed uniform validation subset preserves natural regime frequencies. All
feature scales are fitted on training data, and structural SWD selects the
checkpoint. Sampling preserves an independent latent RNG, copies the upper
triangle, sets the unit diagonal, and requires converged float64 nearest-
correlation repair (tolerance 1e-6, cap 1,000). Record validity before and after
repair. The sampler's stored labels describe selected reference cases only;
they do not condition generated matrices, so per-label scores do not measure
conditional generation quality.

## Execution evidence and remaining work

Two functional tests check updates to both networks, a single batch-normalization
update per training batch, independent sampling RNG, positive-definite repaired
outputs, and complete train/sample execution without labels, scores or test data.
A real Taiwan 80 x 80 CPU smoke run completed two optimization steps and generated
300 validation matrices. It stopped at `--max-steps 2`; it is not a full epoch or
an empirical reproduction. Its structural SWD of 7.8846 is only an execution
diagnostic. The local run is `artifacts/unconditional_smoke_20261005/`.

The full reference budget and replay comparison are now complete, as reported
below. The package also passes the shared isolated installation/source-archive
audit (103 tests across the three repositories). Remaining scientific evidence
includes training-seed variability, recovery of the original training recipe,
and successful fidelity reproduction. Preserve the weak result and distinguish
this reconstruction from categorical cCorrGAN. Matching the released generator
closes the architecture/import gap; it does not close the original training gap.


## Full reference budget completed (later October 5 milestone)

The declared 50-epoch, seed-1701 legacy-archive reconstruction completed 9,800
steps. Validation selected epoch 45 and its native score reproduces exactly.
The [complete validation diagnosis](unconditional_validation_results.md) preserves
selected/final/replay outcomes: selected structural SWD 2.487 ± .015 against
.114 ± .022 for replay, with zero measured learned precision/recall. All raw
draws need repair; convergence restores validity without solving fidelity.
This is one training seed and no test evaluation. Original recipe reproduction,
seed variability and a validated distributional match remain open.

The [post-run feature diagnosis](unconditional_feature_diagnosis.md) reproduces
all saved full scores before separating entries and spectra. Selected samples
average 9.27 eigenvalues at or below 2e-6, versus zero in validation; final
samples average 16.53. Entry-only distances also remain worse than replay.
This characterizes the failure without identifying its training cause or
replacing the selection metric.
