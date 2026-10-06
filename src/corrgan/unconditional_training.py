"""Reference unconditional adversarial updates and audited CorrGAN sampling."""

from __future__ import annotations

import time

import numpy as np
import torch
from corrbench.geometry import nearest_correlation_torch
from corrbench.metrics import validity_summary


def adversarial_step(generator, discriminator, real, g_optimizer, d_optimizer):
    """Simultaneous non-saturating GAN update from one shared fake batch.

    Both gradients are computed at the same parameter state before either optimizer
    steps. The generator's batch-normalization state is updated once per batch.
    """
    noise = torch.randn(len(real), generator.latent_dim, device=real.device)
    fake = generator(noise)
    real_logits, fake_logits = discriminator(real), discriminator(fake)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    discriminator_loss = bce(real_logits, torch.ones_like(real_logits)) + bce(
        fake_logits, torch.zeros_like(fake_logits)
    )
    generator_loss = bce(fake_logits, torch.ones_like(fake_logits))
    if not bool(torch.isfinite(discriminator_loss) & torch.isfinite(generator_loss)):
        raise FloatingPointError("Nonfinite adversarial loss")
    g_optimizer.zero_grad(set_to_none=True)
    d_optimizer.zero_grad(set_to_none=True)
    d_parameters, g_parameters = (
        list(discriminator.parameters()),
        list(generator.parameters()),
    )
    d_gradients = torch.autograd.grad(
        discriminator_loss, d_parameters, retain_graph=True
    )
    g_gradients = torch.autograd.grad(generator_loss, g_parameters)
    for parameters, gradients in [
        (d_parameters, d_gradients),
        (g_parameters, g_gradients),
    ]:
        for parameter, gradient in zip(parameters, gradients, strict=True):
            if not bool(torch.all(torch.isfinite(gradient))):
                raise FloatingPointError("Nonfinite adversarial gradient")
            parameter.grad = gradient
    d_optimizer.step()
    g_optimizer.step()
    return float(generator_loss.detach()), float(discriminator_loss.detach())


@torch.no_grad()
def sample_unconditional(generator, *, samples, seed, batch_size=64, repair=True):
    if min(samples, batch_size) < 1:
        raise ValueError("Positive counts required")
    device = next(generator.parameters()).device
    rng = torch.Generator(device).manual_seed(seed)
    generator.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    prepared, outputs = [], []
    repair_seconds, iterations = 0.0, []
    for begin in range(0, samples, batch_size):
        noise = torch.randn(
            min(batch_size, samples - begin),
            generator.latent_dim,
            device=device,
            generator=rng,
        )
        raw = generator(noise)
        upper = torch.triu(raw, diagonal=1)
        matrix = (
            upper + upper.transpose(-1, -2) + torch.eye(raw.shape[-1], device=device)
        )
        prepared.append(matrix.cpu().numpy())
        if repair:
            repair_started = time.perf_counter()
            matrix, info = nearest_correlation_torch(
                matrix.double(),
                max_iterations=1000,
                tolerance=1e-6,
                return_info=True,
            )
            if not bool(torch.all(info["converged"])):
                raise RuntimeError("Nearest-correlation repair did not converge")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            repair_seconds += time.perf_counter() - repair_started
            iterations.append(int(info["iterations"]))
        outputs.append(matrix.cpu().numpy())
    elapsed = time.perf_counter() - started
    return np.concatenate(outputs), {
        "sampling_seconds": elapsed,
        "repair_seconds": repair_seconds,
        "repair": repair,
        "repair_dtype": "float64" if repair else None,
        "repair_converged": True if repair else None,
        "repair_max_iterations": max(iterations, default=0),
        "raw_validity_after_symmetry_and_diagonal": validity_summary(
            np.concatenate(prepared)
        ),
    }
