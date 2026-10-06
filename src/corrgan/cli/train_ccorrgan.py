#!/usr/bin/env python3
"""Train a conditional CorrGAN-style image baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from corrbench.metrics import structural_features

from corrgan.models import (
    ConditionalCorrGANDiscriminator,
    ConditionalCorrGANGenerator,
)
from corrgan.validation import load_scores, structural_validation


def initialize(module: torch.nn.Module) -> None:
    if isinstance(
        module,
        (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d),
    ):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.uniform_(module.weight, -0.05, 0.05)


def standardize(
    values: np.ndarray, scaler: dict[str, float] | None = None
) -> tuple[np.ndarray, dict[str, float]]:
    if scaler is None:
        scaler = {"mean": float(values.mean()), "std": float(values.std())}
    result = ((values - scaler["mean"]) / max(scaler["std"], 1e-8)).astype(np.float32)
    return result, scaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, help="Override the configured seed.")
    parser.add_argument("--epochs", type=int, help="Override configured epochs.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)

    config = yaml.safe_load(args.config.read_text())
    training = config["training"]
    seed = int(training["seed"] if args.seed is None else args.seed)
    epochs = int(training["epochs"] if args.epochs is None else args.epochs)
    training["seed"] = seed
    training["epochs"] = epochs
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    arrays = {}
    for split in ("train", "validation"):
        with np.load(args.data / f"{split}.npz") as source:
            arrays[split] = {
                "matrices": source["matrices"].astype(np.float32),
                "labels": source["labels"].astype(np.int64),
                "sharpes": load_scores(
                    source, continuous=bool(config["model"]["continuous_condition"])
                ),
            }
    train_score, scaler = standardize(arrays["train"]["sharpes"])
    validation_score, _ = standardize(arrays["validation"]["sharpes"], scaler)
    # Categorical-only archives have no continuous score to normalize.
    if not config["model"]["continuous_condition"]:
        scaler = {"mean": 0.0, "std": 1.0}
    train = torch.from_numpy(arrays["train"]["matrices"]).to(device)
    train_labels = torch.from_numpy(arrays["train"]["labels"]).to(device)
    train_score = torch.from_numpy(train_score).to(device)
    validation = torch.from_numpy(arrays["validation"]["matrices"]).to(device)
    validation_labels = torch.from_numpy(arrays["validation"]["labels"]).to(device)
    validation_score = torch.from_numpy(validation_score).to(device)

    generator = ConditionalCorrGANGenerator(**config["model"]).to(device)
    discriminator = ConditionalCorrGANDiscriminator(
        matrix_dim=train.shape[-1],
        classes=int(config["model"]["classes"]),
        channels=int(config["model"]["channels"]),
        continuous_condition=bool(config["model"]["continuous_condition"]),
    ).to(device)
    generator.apply(initialize)
    discriminator.apply(initialize)
    betas = (float(training["beta1"]), float(training["beta2"]))
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=float(training["learning_rate"]), betas=betas
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(training["learning_rate"]),
        betas=betas,
    )
    precision = training.get("precision", "float32")
    amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    batch_size = int(training["batch_size"])
    latent_dim = int(config["model"]["latent_dim"])
    smooth_real = 1.0 - float(training["label_smoothing"])
    gradient_clip = float(training["gradient_clip"])

    validation_count = min(256, validation.shape[0])
    validation_generator = torch.Generator(device).manual_seed(seed + 10_000)
    validation_indices = torch.randperm(
        validation.shape[0], generator=validation_generator, device=device
    )[:validation_count]
    validation_noise = torch.randn(
        validation_count,
        latent_dim,
        generator=validation_generator,
        device=device,
    )

    structural_every = int(training.get("structural_validation_every", 0))
    structural_indices = None
    if structural_every:
        per_class = int(training.get("validation_per_class", 100))
        selection_rng = np.random.default_rng(44301)
        structural_indices = np.concatenate(
            [
                selection_rng.choice(
                    np.flatnonzero(arrays["validation"]["labels"] == label),
                    per_class,
                    replace=False,
                )
                for label in range(int(config["model"]["classes"]))
            ]
        )
        train_features, _ = structural_features(arrays["train"]["matrices"])
        feature_scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
        real_features, _ = structural_features(
            arrays["validation"]["matrices"][structural_indices]
        )

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "history.json").exists():
        raise FileExistsError("Refusing to overwrite an existing training run")
    history = []
    global_step = 0
    best_validation = math.inf
    best_structural = math.inf
    (args.output / "config.yaml").write_text(yaml.safe_dump(config))
    import corrgan.models as model_module
    import corrgan.validation as validation_module

    provenance = {
        "protocol": config.get("protocol", "legacy_ccorrgan"),
        "source_hashes": {
            split: hashlib.sha256((args.data / f"{split}.npz").read_bytes()).hexdigest()
            for split in ["train", "validation"]
        },
        "code_hashes": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in [
                ("trainer", __file__),
                ("models", model_module.__file__),
                ("validation", validation_module.__file__),
            ]
        },
        "conditioning": "categorical_plus_score"
        if config["model"]["continuous_condition"]
        else "categorical_only",
        "torch": torch.__version__,
        "device": str(device),
        "selection": "minimum fixed-noise structural validation SWD"
        if structural_every
        else "validation generator loss",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    if structural_indices is not None:
        np.save(args.output / "selected_validation_indices.npy", structural_indices)
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        order = torch.randperm(train.shape[0], device=device)
        discriminator_losses = []
        generator_losses = []
        for start in range(0, train.shape[0] - batch_size + 1, batch_size):
            indices = order[start : start + batch_size]
            real = train[indices]
            labels = train_labels[indices]
            scores = train_score[indices]

            discriminator_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, latent_dim, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=precision != "float32",
            ):
                fake = generator(noise, labels, scores)
                real_logits = discriminator(real, labels, scores)
                fake_logits = discriminator(fake.detach(), labels, scores)
                discriminator_loss = 0.5 * (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        real_logits, torch.full_like(real_logits, smooth_real)
                    )
                    + torch.nn.functional.binary_cross_entropy_with_logits(
                        fake_logits, torch.zeros_like(fake_logits)
                    )
                )
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), gradient_clip)
            discriminator_optimizer.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            generator_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, latent_dim, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=precision != "float32",
            ):
                fake = generator(noise, labels, scores)
                fake_logits = discriminator(fake, labels, scores)
                generator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    fake_logits, torch.ones_like(fake_logits)
                )
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), gradient_clip)
            generator_optimizer.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            discriminator_losses.append(float(discriminator_loss.detach()))
            generator_losses.append(float(generator_loss.detach()))
            global_step += 1

        generator.eval()
        discriminator.eval()
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=precision != "float32",
            ),
        ):
            labels = validation_labels[validation_indices]
            scores = validation_score[validation_indices]
            fake = generator(validation_noise, labels, scores)
            fake_logits = discriminator(fake, labels, scores)
            real_logits = discriminator(validation[validation_indices], labels, scores)
            validation_generator_loss = float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    fake_logits, torch.ones_like(fake_logits)
                )
            )
            validation_discriminator_loss = float(
                0.5
                * (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        real_logits, torch.ones_like(real_logits)
                    )
                    + torch.nn.functional.binary_cross_entropy_with_logits(
                        fake_logits, torch.zeros_like(fake_logits)
                    )
                )
            )
        row = {
            "epoch": epoch + 1,
            "step": global_step,
            "discriminator_loss": float(np.mean(discriminator_losses)),
            "generator_loss": float(np.mean(generator_losses)),
            "validation_discriminator_loss": validation_discriminator_loss,
            "validation_generator_loss": validation_generator_loss,
            "validation_generated_mean": float(fake.float().mean()),
            "validation_generated_std": float(fake.float().std()),
            "elapsed_seconds": time.monotonic() - started,
        }
        if structural_every and (
            (epoch + 1) % structural_every == 0 or epoch == 0 or epoch + 1 == epochs
        ):
            row.update(
                structural_validation(
                    generator,
                    validation_labels[structural_indices],
                    validation_score[structural_indices],
                    real_features,
                    feature_scale,
                )
            )
            row["elapsed_seconds"] = time.monotonic() - started
        if device.type == "cuda":
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "model_config": config["model"],
            "matrix_dim": int(train.shape[-1]),
            "condition_scaler": scaler,
            "epoch": epoch + 1,
            "training_seed": seed,
            "history_row": row,
        }
        torch.save(checkpoint, args.output / "last.pt")
        checkpoint_every = int(training.get("checkpoint_every", 0))
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            torch.save(checkpoint, args.output / f"epoch_{epoch + 1:04d}.pt")
        if validation_generator_loss < best_validation:
            best_validation = validation_generator_loss
            torch.save(checkpoint, args.output / "best.pt")
        if row.get("validation_feature_swd", math.inf) < best_structural:
            best_structural = row["validation_feature_swd"]
            torch.save(checkpoint, args.output / "best_structural.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "complete.json").write_text(
        json.dumps(
            {
                "epochs_completed": epochs,
                "training_seconds": time.monotonic() - started,
                "best_structural_swd": best_structural if structural_every else None,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
