"""Create or verify fixed TensorFlow outputs in separate framework environments.

Run with the TensorFlow environment and --framework tensorflow first; then use
the CorrGAN environment and --framework torch on the same --reference NPZ.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", choices=["tensorflow", "torch"], required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if args.framework == "tensorflow":
        import tensorflow as tf

        class CompatibleConv2DTranspose(tf.keras.layers.Conv2DTranspose):
            def __init__(self, *a, groups=1, **kw):
                if groups != 1:
                    raise ValueError(
                        "Only the historical groups=1 compatibility case is supported"
                    )
                super().__init__(*a, **kw)

        if args.reference.exists():
            raise FileExistsError(args.reference)
        model = tf.keras.models.load_model(
            args.model,
            compile=False,
            custom_objects={"Conv2DTranspose": CompatibleConv2DTranspose},
        )
        noise = np.random.default_rng(20261005).normal(size=(30, 80)).astype(np.float32)
        labels = np.tile(np.arange(3), 10).astype(np.int64)
        matrices = model([noise, labels[:, None]], training=False).numpy()[..., 0]
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.reference,
            noise=noise,
            labels=labels,
            matrices=matrices,
            tensorflow_version=tf.__version__,
        )
    else:
        import torch

        from corrgan.legacy import load_published_ccorrgan

        torch.set_num_threads(4)
        model = load_published_ccorrgan(args.model)
        with np.load(args.reference) as a, torch.no_grad():
            actual = model(
                torch.from_numpy(a["noise"]), torch.from_numpy(a["labels"])
            ).numpy()
            error = np.abs(actual - a["matrices"])
            np.testing.assert_allclose(actual, a["matrices"], atol=2e-5, rtol=2e-5)
            report = {
                "passed": True,
                "max_error": float(error.max()),
                "mean_error": float(error.mean()),
                "matrices": len(actual),
                "entries": int(actual.size),
                "atol": 2e-5,
                "rtol": 2e-5,
                "tensorflow": str(a["tensorflow_version"]),
                "torch": torch.__version__,
                "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(
                    args.reference.read_bytes()
                ).hexdigest(),
            }
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
