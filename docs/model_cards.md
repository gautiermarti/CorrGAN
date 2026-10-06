# CorrGAN family: development model cards

Updated 2026-10-05. These describe implemented research models and local evidence,
not published checkpoints. The package is `0.1.0rc1`; use the full Git revision
and its pinned CorrBench dependency when recording a run. Code is MIT licensed.
Data and recovered checkpoint redistribution require their own review.

## Unconditional CorrGAN

**Identity and purpose.** Native PyTorch implementation of the generator in the
author-released 80-dimensional CorrGAN SavedModel. Its intended research task is
sampling a distribution of anonymous financial correlation matrices. It takes
100 Gaussian latent coordinates and has no regime or continuous condition.
The 20 × 20 × 256 dense representation feeds three transposed convolutions.
The final tanh image is not necessarily symmetric or positive semidefinite.

**Output contract.** The supported sampler copies the upper triangle, sets the
diagonal to one and applies converged float64 nearest-correlation repair. Repair
uses a 1e-6 tolerance, a 1,000-iteration cap and a 1e-6 eigenvalue floor. Failure
to converge is an error. Reports retain validity before repair and its cost.
`--raw` skips repair but still prepares symmetry and the diagonal; it does not
mean the untouched convolutional image. Stored reference-case labels do not
condition this generator and do not support conditional-fidelity claims.

**Training and provenance.** The generator architecture is recovered; the
discriminator and optimizer are documented reference choices. Training uses
only matrix arrays from train and validation. The completed legacy S&P run uses
12,600 training and 2,700 validation matrices from a random archive split, 50
epochs and seed 1701. The archive has no dates or constituent identities. It is
not the return-derived Taiwan benchmark or a temporal holdout. Configuration:
`configs/corrgan_unconditional_reference.yaml`.

**Evidence.** The imported released generator matches TensorFlow on 192,000
entries to maximum absolute error 7.153e-7. That establishes inference parity.
The new 50-epoch reference fit does not reproduce realistic archive distributions:
validation selects epoch 45, with diagnostic structural SWD 2.487 ± .015 versus
.114 ± .022 for training replay. Both learned endpoints have zero measured
precision/recall and two-sample AUC 1.000. All 3,000 generated raw prepared
matrices need repair. The five draws measure generation variability for one fit;
the test split was never opened. See the [complete result](unconditional_validation_results.md)
and [architecture/recipe evidence](unconditional_reconstruction.md).

**Supported scope.** The released 80D generator has parity evidence. Other
multiples-of-four sizes are configurable architectures requiring new training
and validation. Passing geometry checks does not establish distributional
fidelity. This reconstruction is not ready to supply validated risk scenarios.
Sampling correlations alone does not generate returns, volatility, tail
dependence, temporal paths or previously unseen stress events.

**Availability.** Training and sampling commands are in the repository README.
Local selected/final checkpoints and parity artifacts are preserved. Original
weights are obtained separately; no checkpoint is distributed with this package.
Original discriminator/optimizer recovery and empirical training reproduction
remain open. A failure of this reference recipe does not establish failure of
the published CorrGAN method.

## Categorical cCorrGAN and continuous-condition extension

**Identity and purpose.** The categorical family samples 80D financial
correlation images from Gaussian latent noise and one of three regime labels.
The maintained code includes training, validation selection, sampling and
explicit repair-aware evaluation. The continuous-score option is a separate
research extension; its inputs and scores must not be mixed with categorical-only
comparisons. A constant categorical label does not turn this architecture into
the original unconditional CorrGAN generator.

**Data and model variants.** The legacy archive supplies categorical labels but
no dates, constituents or continuous performance information. Dated Taiwan data
start at curated adjusted returns, with frozen window construction, Ward ordering
and training-derived thresholds in CorrBench. The matched categorical protocol
uses `configs/ccorrgan_taiwan_categorical.yaml`: latent dimension 80, 128 channels,
three classes, 200 epochs and five declared training seeds. It uses identical
reference cases and validation structural selection to the flow comparison.
The original paper's continuous-conditioned 20-epoch configurations are separate.

**Output contract.** Network images require symmetry/unit-diagonal preparation
and converged nearest-correlation repair. Use the README evaluator command and
retain raw validity, repair magnitude, spectral effects and time. Neither a
successful optimization run nor repaired positive definiteness certifies realism.

**Evidence.** A dedicated importer reproduces the author-released categorical
HDF5 generator against TensorFlow on 192,000 entries, maximum error 2.623e-6.
Original checkpoint training membership is unknown. All ten earlier 200-epoch
validation selections reproduce exactly. In those Taiwan runs, later weights
improve entry summaries while worsening full structural SWD and near-floor
spectra. The [fidelity diagnosis](fidelity_diagnosis.md) preserves both outcomes.
The complete matched five-seed categorical reconstruction now passes its full
200-epoch audit and all five native validation selections reproduce exactly.
Structural SWD is 4.0390 ± .1537 after repair, precision is zero, and portfolio
SWD is 818.7222 ± 824.7008. This poor fidelity remains a reconstruction result,
not successful reproduction of the original published training.

**Limitations and availability.** Original published training/fidelity remains
unverified. An imported generator with unknown training membership cannot certify
held-out generalization. Overlapping historical return windows are dependent and
the paper periods were already inspected. Native checkpoints and generated
artifacts remain local; the package distributes code, configurations and audits.
The optional importer reads a separately obtained HDF5 model. No utility,
privacy guarantee, novel stress discovery or public checkpoint release is claimed.

For both families, use [release criteria](release_contract.md) before promoting a
development checkpoint to a coordinated public release.
