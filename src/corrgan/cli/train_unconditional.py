"""Train unconditional CorrGAN with a documented reference discriminator/recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from corrbench.metrics import sliced_wasserstein, structural_features, validity_summary

from corrgan.unconditional import DCGANReferenceDiscriminator, OriginalCorrGANGenerator
from corrgan.unconditional_training import adversarial_step, sample_unconditional


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    cfg = config["training"]
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    arrays, hashes = {}, {}
    for split in ["train", "validation"]:
        path = args.data / f"{split}.npz"
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as a:
            arrays[split] = a["matrices"].astype(np.float32)
        if arrays[split].ndim != 3 or not np.isfinite(arrays[split]).all():
            raise ValueError("Finite batched matrices required")
    n = arrays["train"].shape[-1]
    if arrays["train"].shape[-2] != n or arrays["validation"].shape[-2:] != (n, n):
        raise ValueError("Matrix dimensions differ")
    batch = int(cfg["batch_size"])
    if not 2 <= batch <= len(arrays["train"]) or cfg["epochs"] < 1:
        raise ValueError("Invalid batch/epoch count")
    model_config = {"matrix_dim": n, **config["model"]}
    generator = OriginalCorrGANGenerator(**model_config).to(args.device)
    discriminator_config = {"matrix_dim": n, **config["discriminator"]}
    discriminator = DCGANReferenceDiscriminator(**discriminator_config).to(args.device)
    for model in [generator, discriminator]:
        for module in model.modules():
            if isinstance(
                module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
            ):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    options = {"lr": cfg["learning_rate"], "betas": (cfg["beta1"], cfg["beta2"])}
    g_optimizer = torch.optim.Adam(generator.parameters(), **options)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), **options)
    train = torch.from_numpy(arrays["train"]).to(args.device)
    features, _ = structural_features(arrays["train"])
    scale = np.maximum(features.std(axis=0, ddof=1), 1e-6)
    selected = np.random.default_rng(cfg["validation_seed"]).choice(
        len(arrays["validation"]), int(cfg["validation_samples"]), replace=False
    )
    real_features, _ = structural_features(arrays["validation"][selected])
    args.output.mkdir(parents=True)
    np.save(args.output / "selected_validation_indices.npy", selected)
    (args.output / "config.yaml").write_text(yaml.safe_dump(config))
    import corrgan.unconditional as architecture_module
    import corrgan.unconditional_training as training_module

    provenance = {
        "scope": "unconditional retraining; original training recipe unverified",
        "conditioning": "none; labels and scores never loaded",
        "selection": "fixed uniform validation rows, minimum full structural SWD",
        "source_hashes": hashes,
        "code_hashes": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in [
                ("trainer", __file__),
                ("architecture", architecture_module.__file__),
                ("training", training_module.__file__),
            ]
        },
        "torch": torch.__version__,
        "device": args.device,
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    history, best, step = [], float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        generator.train()
        discriminator.train()
        order = torch.randperm(len(train), device=args.device)
        losses = []
        for start in range(0, len(order) - batch + 1, batch):
            losses.append(
                adversarial_step(
                    generator,
                    discriminator,
                    train[order[start : start + batch]],
                    g_optimizer,
                    d_optimizer,
                )
            )
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        row = {
            "epoch": epoch,
            "steps": step,
            "generator_loss": float(np.mean(losses, axis=0)[0]),
            "discriminator_loss": float(np.mean(losses, axis=0)[1]),
        }
        if epoch == 1 or epoch % cfg["validation_every"] == 0 or epoch == cfg["epochs"]:
            matrices, timing = sample_unconditional(
                generator,
                samples=len(selected),
                seed=cfg["validation_seed"],
                batch_size=cfg["sampling_batch_size"],
            )
            generated_features, _ = structural_features(matrices)
            row["validation_feature_swd"] = sliced_wasserstein(
                real_features,
                generated_features,
                scale=scale,
                seed=cfg["validation_seed"],
                device="cpu",
            )
            row["validity"] = validity_summary(matrices)
            row["sampling"] = timing
        row["elapsed_seconds"] = time.perf_counter() - started
        history.append(row)
        checkpoint = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "model_config": model_config,
            "discriminator_config": discriminator_config,
            "epoch": epoch,
            "history_row": row,
            "training_seed": cfg["seed"],
            "provenance": "reference reconstruction; original recipe unverified",
        }
        torch.save(checkpoint, args.output / "last.pt")
        score = row.get("validation_feature_swd", float("inf"))
        if score < best:
            best = score
            torch.save(checkpoint, args.output / "best_structural.pt")
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(row), flush=True)
        if args.max_steps is not None and step >= args.max_steps:
            break
    (args.output / "complete.json").write_text(
        json.dumps(
            {
                "epochs_completed": len(history),
                "steps": step,
                "stopped_at_max_steps": args.max_steps is not None
                and step >= args.max_steps,
                "seconds": time.perf_counter() - started,
                "best_structural_swd": best,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
