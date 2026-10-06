"""Import the publicly released cCorrGAN generator without TensorFlow.

Only the hash-pinned 2021 artifact is supported. The conversion preserves the
NHWC dense-output order and TensorFlow convolution layouts in the NCHW model.
The weights are not bundled or downloaded by this module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from corrgan.models import ConditionalCorrGANGenerator

CCORRGAN_URL = (
    "https://sp500-histo.s3-ap-southeast-1.amazonaws.com/cCorrGAN_generator.h5"
)
CCORRGAN_SHA256 = "5a7f0fa2ad35e31e1b71621f7ee4a93f4c01100de74e1608b2054d0decb97984"


def nhwc_projection_to_nchw(
    kernel: np.ndarray, bias: np.ndarray, *, height: int, width: int, channels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder a dense projection whose output is reshaped to NHWC."""
    order = np.arange(height * width * channels).reshape(height, width, channels)
    order = order.transpose(2, 0, 1).reshape(-1)
    return kernel[:, order].T.copy(), bias[order].copy()


def load_published_ccorrgan(path: str | Path) -> ConditionalCorrGANGenerator:
    """Load the exact public checkpoint, rejecting changed/unrecognized files."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CCORRGAN_SHA256:
        raise ValueError("Unrecognized cCorrGAN HDF5: SHA-256 does not match")
    import h5py

    # Model construction must not perturb an ongoing training RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = ConditionalCorrGANGenerator(continuous_condition=False)
    with h5py.File(path, "r") as source:

        def weight(layer: str, name: str) -> np.ndarray:
            return source[f"model_weights/{layer}/{layer}/{name}:0"][:]

        dense_weight, dense_bias = nhwc_projection_to_nchw(
            weight("dense_7", "kernel"),
            weight("dense_7", "bias"),
            height=20,
            width=20,
            channels=128,
        )
        state = {
            "label_embedding.weight": weight("embedding_3", "embeddings"),
            "noise_projection.weight": dense_weight,
            "noise_projection.bias": dense_bias,
            "condition_projection.weight": weight("dense_6", "kernel").T,
            "condition_projection.bias": weight("dense_6", "bias"),
            # TF transpose conv: H,W,out,in; PyTorch: in,out,H,W.
            "up1.weight": weight("conv2d_transpose_2", "kernel").transpose(3, 2, 0, 1),
            "up1.bias": weight("conv2d_transpose_2", "bias"),
            "up2.weight": weight("conv2d_transpose_3", "kernel").transpose(3, 2, 0, 1),
            "up2.bias": weight("conv2d_transpose_3", "bias"),
            # TF regular conv: H,W,in,out; PyTorch: out,in,H,W.
            "output.weight": weight("conv2d_5", "kernel").transpose(3, 2, 0, 1),
            "output.bias": weight("conv2d_5", "bias"),
        }
    model.load_state_dict(
        {key: torch.from_numpy(value.copy()) for key, value in state.items()}
    )
    return model.eval()
