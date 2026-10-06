import pytest
import torch

from corrgan.models import (
    ConditionalCorrGANDiscriminator,
    ConditionalCorrGANGenerator,
)


def test_conditional_corrgan_shapes_and_gradients() -> None:
    generator = ConditionalCorrGANGenerator(channels=8)
    discriminator = ConditionalCorrGANDiscriminator(channels=8)
    noise = torch.randn(3, 80)
    labels = torch.tensor([0, 1, 2])
    continuous = torch.tensor([-0.5, 0.0, 0.5])
    matrices = generator(noise, labels, continuous)
    logits = discriminator(matrices, labels, continuous)
    assert matrices.shape == (3, 80, 80)
    assert logits.shape == (3,)
    assert torch.all(matrices.abs() <= 1)
    logits.mean().backward()
    assert generator.output.weight.grad is not None
    assert discriminator.output.weight.grad is not None


def test_continuous_condition_is_required_when_enabled() -> None:
    generator = ConditionalCorrGANGenerator(channels=8)
    with pytest.raises(ValueError, match="continuous condition"):
        generator(torch.randn(2, 80), torch.tensor([0, 1]))
