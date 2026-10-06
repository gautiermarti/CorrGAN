"""Conditional CorrGAN-style image generator and discriminator."""

from __future__ import annotations

import torch
from torch import nn


class ConditionalCorrGANGenerator(nn.Module):
    """PyTorch implementation of the published 80 by 80 cCorrGAN generator.

    The original architecture projects an 80-dimensional latent vector to a
    20 by 20 image, embeds the market regime alongside it, upsamples twice, and
    finishes with a 7 by 7 convolution.  We additionally admit one continuous
    condition so every Taiwan neural model receives the same information.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 80,
        classes: int = 3,
        channels: int = 128,
        continuous_condition: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.continuous_condition = continuous_condition
        self.label_embedding = nn.Embedding(classes, latent_dim)
        condition_dim = latent_dim + int(continuous_condition)
        self.noise_projection = nn.Linear(latent_dim, 20 * 20 * channels)
        self.condition_projection = nn.Linear(condition_dim, 20 * 20)
        self.up1 = nn.ConvTranspose2d(channels + 1, channels, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
        self.output = nn.Conv2d(channels, 1, 7, padding=3)
        self.activation = nn.LeakyReLU(0.2)

    def forward(
        self,
        noise: torch.Tensor,
        labels: torch.Tensor,
        continuous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = noise.shape[0]
        hidden = self.activation(self.noise_projection(noise))
        hidden = hidden.view(batch, -1, 20, 20)
        condition = self.label_embedding(labels)
        if self.continuous_condition:
            if continuous is None:
                raise ValueError("continuous condition is required")
            condition = torch.cat([condition, continuous.reshape(batch, 1)], dim=1)
        condition = self.condition_projection(condition).view(batch, 1, 20, 20)
        hidden = torch.cat([hidden, condition], dim=1)
        hidden = self.activation(self.up1(hidden))
        hidden = self.activation(self.up2(hidden))
        return torch.tanh(self.output(hidden)).squeeze(1)


class ConditionalCorrGANDiscriminator(nn.Module):
    """Conditional convolutional discriminator paired with cCorrGAN."""

    def __init__(
        self,
        *,
        matrix_dim: int = 80,
        classes: int = 3,
        channels: int = 128,
        continuous_condition: bool = True,
    ) -> None:
        super().__init__()
        self.matrix_dim = matrix_dim
        self.continuous_condition = continuous_condition
        self.label_embedding = nn.Embedding(classes, matrix_dim)
        condition_dim = matrix_dim + int(continuous_condition)
        self.condition_projection = nn.Linear(condition_dim, matrix_dim * matrix_dim)
        self.conv1 = nn.Conv2d(2, channels, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 4, stride=2, padding=1)
        final_dim = matrix_dim // 4
        self.output = nn.Linear(channels * final_dim * final_dim, 1)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.4)

    def forward(
        self,
        matrices: torch.Tensor,
        labels: torch.Tensor,
        continuous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = matrices.shape[0]
        condition = self.label_embedding(labels)
        if self.continuous_condition:
            if continuous is None:
                raise ValueError("continuous condition is required")
            condition = torch.cat([condition, continuous.reshape(batch, 1)], dim=1)
        condition = self.condition_projection(condition).view(
            batch, 1, self.matrix_dim, self.matrix_dim
        )
        hidden = torch.cat([matrices.unsqueeze(1), condition], dim=1)
        hidden = self.dropout(self.activation(self.conv1(hidden)))
        hidden = self.dropout(self.activation(self.conv2(hidden)))
        return self.output(hidden.flatten(1)).squeeze(1)
