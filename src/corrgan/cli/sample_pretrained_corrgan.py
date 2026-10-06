#!/usr/bin/env python3
"""Sample the official 2020 TensorFlow CorrGAN SavedModel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    import tensorflow as tf

    model = tf.saved_model.load(str(args.model))
    generator = model.signatures["serving_default"]
    rng = np.random.default_rng(args.seed)
    generated = []
    for start in range(0, args.samples, args.batch_size):
        count = min(args.batch_size, args.samples - start)
        noise = rng.normal(size=(count, 100)).astype(np.float32)
        output = generator(dense_input=tf.convert_to_tensor(noise))
        generated.append(next(iter(output.values())).numpy()[..., 0])
    matrices = np.concatenate(generated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, matrices=matrices, seed=args.seed)
    print(
        f"wrote {matrices.shape} to {args.output}; "
        f"range=[{matrices.min():.4f}, {matrices.max():.4f}]"
    )


if __name__ == "__main__":
    main()
