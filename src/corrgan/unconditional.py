"""Unconditional generator recovered from the author's 80D SavedModel.

Generator inference is checked against the released checkpoint. The discriminator
below is a documented DCGAN reference, not a recovered original discriminator.
"""

from __future__ import annotations

import torch
from torch import nn


class SameConvTranspose2d(nn.ConvTranspose2d):
    """TensorFlow SAME convention for the released odd-kernel deconvolutions."""

    def __init__(self, input_channels, output_channels, *, stride):
        super().__init__(
            input_channels,
            output_channels,
            kernel_size=5,
            stride=stride,
            padding=2 if stride == 1 else 1,
            bias=False,
        )

    def forward(self, x, output_size=None):
        y = super().forward(x, output_size)
        # With k=5, s=2, TF crops one pixel on top/left and two on bottom/right.
        return y if self.stride == (1, 1) else y[:, :, :-1, :-1]


class OriginalCorrGANGenerator(nn.Module):
    """Published 80D architecture; configurable size variants require retraining."""

    def __init__(self, *, matrix_dim=80, latent_dim=100, channels=256):
        super().__init__()
        if matrix_dim < 8 or matrix_dim % 4 or channels < 4 or channels % 4:
            raise ValueError(
                "Matrix size and channels must be positive multiples of four"
            )
        self.matrix_dim, self.latent_dim, self.channels = (
            matrix_dim,
            latent_dim,
            channels,
        )
        side = matrix_dim // 4
        self.dense = nn.Linear(latent_dim, side * side * channels, bias=False)
        self.bn0 = nn.BatchNorm1d(side * side * channels, eps=0.001, momentum=0.01)
        self.deconv0 = SameConvTranspose2d(channels, channels // 2, stride=1)
        self.bn1 = nn.BatchNorm2d(channels // 2, eps=0.001, momentum=0.01)
        self.deconv1 = SameConvTranspose2d(channels // 2, channels // 4, stride=2)
        self.bn2 = nn.BatchNorm2d(channels // 4, eps=0.001, momentum=0.01)
        self.deconv2 = SameConvTranspose2d(channels // 4, 1, stride=2)
        self.activation = nn.LeakyReLU(0.3)

    def forward(self, noise):
        x = self.activation(self.bn0(self.dense(noise)))
        side = self.matrix_dim // 4
        # Keep the released dense/BN flattened NHWC order explicitly.
        x = x.reshape(len(noise), side, side, self.channels).permute(0, 3, 1, 2)
        x = self.activation(self.bn1(self.deconv0(x)))
        x = self.activation(self.bn2(self.deconv1(x)))
        return torch.tanh(self.deconv2(x))[:, 0]


class DCGANReferenceDiscriminator(nn.Module):
    """Two strided convolutions and dropout, explicitly a reconstruction choice."""

    def __init__(self, *, matrix_dim=80, channels=64, dropout=0.3):
        super().__init__()
        if matrix_dim % 4:
            raise ValueError("Matrix dimension must be divisible by four")
        self.network = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(1, channels, kernel_size=5, stride=2),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(channels, channels * 2, kernel_size=5, stride=2),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear((matrix_dim // 4) ** 2 * channels * 2, 1),
        )

    def forward(self, matrices):
        return self.network(matrices[:, None]).squeeze(-1)


def import_original_arrays(arrays):
    """Import named TensorFlow variables exported by the conversion utility."""
    model = OriginalCorrGANGenerator()
    state = model.state_dict()
    state["dense.weight"] = torch.from_numpy(arrays["dense/kernel:0"].T.copy())
    for target, source in [
        ("bn0", "batch_normalization"),
        ("bn1", "batch_normalization_1"),
        ("bn2", "batch_normalization_2"),
    ]:
        for to, old in [
            ("weight", "gamma"),
            ("bias", "beta"),
            ("running_mean", "moving_mean"),
            ("running_var", "moving_variance"),
        ]:
            state[f"{target}.{to}"] = torch.from_numpy(
                arrays[f"{source}/{old}:0"].copy()
            )
    for target, source in [
        ("deconv0", "conv2d_transpose"),
        ("deconv1", "conv2d_transpose_1"),
        ("deconv2", "conv2d_transpose_2"),
    ]:
        state[f"{target}.weight"] = torch.from_numpy(
            arrays[f"{source}/kernel:0"].transpose(3, 2, 0, 1).copy()
        )
    model.load_state_dict(state)
    return model.eval()
