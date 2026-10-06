import numpy as np
import pytest
import torch
from corrbench.metrics import structural_features

from corrgan.validation import load_scores, structural_validation


def test_categorical_archive_requires_no_continuous_scores():
    source = {"labels": np.array([0, 1, 2])}
    np.testing.assert_array_equal(load_scores(source, continuous=False), [0, 0, 0])
    with pytest.raises(ValueError, match="requires observed sharpes"):
        load_scores(source, continuous=True)


def test_structural_validation_preserves_training_rng_and_repairs():
    class ToyGenerator(torch.nn.Module):
        latent_dim = 4

        def forward(self, noise, labels, scores):
            matrices = torch.zeros(len(labels), 4, 4)
            # Deliberately invalid before projection, with noise-dependent output.
            matrices[:, 0, 1] = 1.5 + 0.1 * noise[:, 0]
            return matrices

    generator = ToyGenerator()
    real = np.repeat(np.eye(4)[None], 6, axis=0)
    features, _ = structural_features(real)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    scores = torch.zeros(6)
    before = torch.random.get_rng_state().clone()
    first = structural_validation(
        generator,
        labels,
        scores,
        features,
        np.ones(features.shape[1]),
    )
    assert torch.equal(before, torch.random.get_rng_state())
    second = structural_validation(
        generator,
        labels,
        scores,
        features,
        np.ones(features.shape[1]),
    )
    assert first == second
    assert np.isfinite(first["validation_feature_swd"])
    assert first["validation_raw_valid_fraction"] == 0.0
