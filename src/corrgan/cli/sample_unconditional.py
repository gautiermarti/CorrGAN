"""Generate unconditional CorrGAN samples on a uniform reference subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from corrgan.unconditional import OriginalCorrGANGenerator
from corrgan.unconditional_training import sample_unconditional


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=44301)
    p.add_argument("--selection-seed", type=int)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--raw",
        action="store_true",
        help="Skip PD repair, retain symmetry/unit diagonal",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    if args.output.suffix != ".npz":
        p.error("--output must end in .npz")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    generator = OriginalCorrGANGenerator(**checkpoint["model_config"]).to(args.device)
    generator.load_state_dict(checkpoint["generator"])
    selection_seed = args.seed if args.selection_seed is None else args.selection_seed
    with np.load(args.data / f"{args.split}.npz") as a:
        if a["matrices"].shape[-1] != generator.matrix_dim:
            raise ValueError("Data and checkpoint matrix sizes differ")
        indices = np.random.default_rng(selection_seed).choice(
            len(a["matrices"]), args.samples, replace=False
        )
        labels = (
            a["labels"][indices]
            if "labels" in a
            else np.zeros(args.samples, dtype=np.int64)
        )
    matrices, timing = sample_unconditional(
        generator,
        samples=args.samples,
        seed=args.seed,
        batch_size=args.batch_size,
        repair=not args.raw,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        matrices=matrices,
        labels=labels,
        selected_test_indices=indices,
        split=args.split,
        conditioning="unconditional",
        checkpoint_epoch=checkpoint["epoch"] if checkpoint["epoch"] is not None else -1,
        sampling_seconds=timing["sampling_seconds"],
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "conditioning": "none; stored labels describe reference cases only",
                "selection": "uniform rows, natural regime frequencies",
                "seed": args.seed,
                "selection_seed": selection_seed,
                "samples": args.samples,
                "checkpoint_epoch": checkpoint["epoch"],
                "timing": timing,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {args.samples} unconditional samples to {args.output}")


if __name__ == "__main__":
    main()
