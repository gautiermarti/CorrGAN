# Unconditional reconstruction: entry and spectrum diagnosis

Post-run, validation only. All 15 saved artifacts and source arrays pass hash checks; every frozen full structural score reproduces exactly. No model is fitted, reselected or evaluated on test data.

| Weights | Full SWD | Entry-only SWD | Spectrum-only SWD |
| --- | ---: | ---: | ---: |
| selected | 2.4875 ± 0.0151 | 0.6648 ± 0.0271 | 3.2971 ± 0.0133 |
| final | 3.4176 ± 0.0828 | 1.1391 ± 0.0601 | 4.1973 ± 0.0821 |
| replay | 0.1143 ± 0.0222 | 0.1266 ± 0.0241 | 0.1220 ± 0.0305 |

Group scores use different projection spaces and are not additive decompositions of the full score. Marginal W2 is a diagnostic, not a replacement selection metric. Repaired eigenvalue counts describe a symptom and do not establish the training failure's cause. One fit, five draws; no original released-checkpoint performance claim.

## Eigenvalues near the repair floor

The real validation median minimum eigenvalue is 0.062996; none of its eigenvalues are at or below 2e-6.

| Weights | Mean count of eigenvalues ≤ 2e-6 per matrix | Mean over draws of median minimum eigenvalue |
| --- | ---: | ---: |
| selected | 9.270 | 0.00000100 |
| final | 16.528 | 0.00000100 |
| replay | 0.000 | 0.05808283 |

These near-floor spectra help explain why entrywise appearance is insufficient for this reconstruction. They do not identify which training choice caused the mismatch, and do not justify dropping spectral metrics or changing the declared selection rule.
