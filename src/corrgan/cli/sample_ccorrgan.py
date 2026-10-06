#!/usr/bin/env python3
"""Sample a trained conditional CorrGAN image generator."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from corrgan.models import ConditionalCorrGANGenerator
from corrgan.validation import load_scores


def stratified_indices(
    labels: np.ndarray, per_class: int, rng: np.random.Generator
) -> np.ndarray:
    return np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == label), per_class, replace=False)
            for label in range(3)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    torch.set_num_threads(4)

    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=True
    )
    model = ConditionalCorrGANGenerator(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    with np.load(args.data / f"{args.split}.npz") as source:
        all_labels = source["labels"]
        all_scores = load_scores(
            source,
            continuous=bool(checkpoint["model_config"]["continuous_condition"]),
        )
    rng = np.random.default_rng(
        args.seed if args.selection_seed is None else args.selection_seed
    )
    selected = stratified_indices(all_labels, args.samples_per_class, rng)
    labels = all_labels[selected].astype(np.int64)
    scores = all_scores[selected].astype(np.float32)
    scaler = checkpoint["condition_scaler"]
    standardized = ((scores - scaler["mean"]) / scaler["std"]).astype(np.float32)

    generated = []
    torch_generator = torch.Generator(args.device).manual_seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, labels.size, args.batch_size):
        stop = min(start + args.batch_size, labels.size)
        count = stop - start
        noise = torch.randn(
            count,
            int(checkpoint["model_config"]["latent_dim"]),
            generator=torch_generator,
            device=args.device,
        )
        batch_labels = torch.from_numpy(labels[start:stop]).to(args.device)
        batch_scores = torch.from_numpy(standardized[start:stop]).to(args.device)
        with torch.no_grad():
            matrices = model(noise, batch_labels, batch_scores)
        generated.append(matrices.cpu().numpy().astype(np.float32))
    matrices = np.concatenate(generated)
    sampling_seconds = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        matrices=matrices,
        labels=labels.astype(np.uint8),
        sharpes=scores,
        selected_test_indices=selected,
        checkpoint_epoch=checkpoint["epoch"],
        split=np.asarray(args.split),
        sampling_seconds=np.asarray(sampling_seconds),
    )
    print(f"wrote {matrices.shape} to {args.output}")


if __name__ == "__main__":
    main()
