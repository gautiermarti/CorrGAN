"""Validation-only structural checkpoint selection for CorrGAN retraining."""

from __future__ import annotations

import numpy as np
import torch
from corrbench.geometry import nearest_correlation_torch
from corrbench.metrics import sliced_wasserstein, structural_features, validity_summary


def load_scores(source, *, continuous: bool) -> np.ndarray:
    """Require real scores for continuous models; allow categorical archives."""
    if continuous:
        if "sharpes" not in source:
            raise ValueError("Continuous conditioning requires observed sharpes.")
        return source["sharpes"].astype(np.float32)
    return np.zeros(len(source["labels"]), dtype=np.float32)


@torch.no_grad()
def structural_validation(
    generator: torch.nn.Module,
    labels: torch.Tensor,
    scores: torch.Tensor,
    real_features: np.ndarray,
    feature_scale: np.ndarray,
    *,
    seed: int = 44301,
    batch_size: int = 32,
) -> dict[str, float]:
    """Evaluate fixed noise on fixed validation cases, including converged repair.

    The local RNG does not consume the training noise stream. No test data is
    accepted by this helper; callers supply the validation subset explicitly.
    """
    rng = torch.Generator(labels.device).manual_seed(seed)
    raw, repaired = [], []
    generator.eval()
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        noise = torch.randn(
            stop - start,
            generator.latent_dim,
            generator=rng,
            device=labels.device,
        )
        matrices = generator(noise, labels[start:stop], scores[start:stop]).float()
        upper = torch.triu(matrices, diagonal=1)
        matrices = upper + upper.transpose(-1, -2)
        matrices += torch.eye(matrices.shape[-1], device=matrices.device)
        raw.append(matrices.cpu().numpy())
        valid, info = nearest_correlation_torch(
            matrices.double(),
            max_iterations=1000,
            tolerance=1e-6,
            return_info=True,
        )
        if not bool(torch.all(info["converged"])):
            raise RuntimeError("Validation repair did not converge in 1000 iterations")
        repaired.append(valid.cpu().numpy())
    features, _ = structural_features(np.concatenate(repaired))
    return {
        "validation_feature_swd": sliced_wasserstein(
            real_features,
            features,
            scale=feature_scale,
            seed=seed,
            device="cpu",
        ),
        "validation_raw_valid_fraction": validity_summary(np.concatenate(raw))[
            "valid_fraction"
        ],
    }
