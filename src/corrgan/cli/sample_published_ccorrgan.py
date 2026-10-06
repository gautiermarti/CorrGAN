"""Sample the hash-pinned public categorical cCorrGAN model in PyTorch."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from corrgan.legacy import CCORRGAN_SHA256, CCORRGAN_URL, load_published_ccorrgan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.samples_per_class < 1 or args.batch_size < 1:
        parser.error("sample counts and batch size must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    model = load_published_ccorrgan(args.model).to(args.device)
    labels = np.repeat(np.arange(3, dtype=np.int64), args.samples_per_class)
    noise = (
        np.random.default_rng(args.seed)
        .normal(size=(len(labels), 80))
        .astype(np.float32)
    )
    outputs = []
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(labels), args.batch_size):
            stop = start + args.batch_size
            matrices = model(
                torch.from_numpy(noise[start:stop]).to(args.device),
                torch.from_numpy(labels[start:stop]).to(args.device),
            )
            outputs.append(matrices.cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        matrices=np.concatenate(outputs),
        labels=labels,
        seed=args.seed,
        source_sha256=CCORRGAN_SHA256,
        source_url=CCORRGAN_URL,
        sampling_seconds=time.perf_counter() - started,
        scope="original checkpoint; raw images; original training membership unknown",
    )
    print(f"Saved {len(labels)} raw images to {args.output}")


if __name__ == "__main__":
    main()
