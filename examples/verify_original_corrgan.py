"""Export an original SavedModel reference on CPU, then verify native PyTorch.

Run separately in TensorFlow and PyTorch environments. Converted weights are
local artifacts, not automatically redistributed with the package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", choices=["tensorflow", "torch"], required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--report", type=Path)
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args()
    if args.framework == "tensorflow":
        import tensorflow as tf
        from tensorflow.core.protobuf import saved_model_pb2

        if args.reference.exists() or args.reference.with_suffix(".json").exists():
            raise FileExistsError(args.reference)
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        model = tf.saved_model.load(str(args.source))
        proto = saved_model_pb2.SavedModel()
        proto.ParseFromString((args.source / "saved_model.pb").read_bytes())
        metadata = json.loads(
            proto.meta_graphs[0].object_graph_def.nodes[0].user_object.metadata
        )
        noise = (
            np.random.default_rng(20261005).normal(size=(30, 100)).astype(np.float32)
        )
        samples = model.signatures["serving_default"](dense_input=tf.constant(noise))[
            "conv2d_transpose_2"
        ].numpy()[..., 0]
        arrays = {v.name: v.numpy() for v in model.variables}
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.reference, **arrays, noise=noise, samples=samples)
        args.reference.with_suffix(".json").write_text(
            json.dumps(
                {
                    "tensorflow": tf.__version__,
                    "model_config": metadata["model_config"],
                    "source_hashes": {
                        str(p.relative_to(args.source)): hashlib.sha256(
                            p.read_bytes()
                        ).hexdigest()
                        for p in sorted(args.source.rglob("*"))
                        if p.is_file()
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print("Exported 30 reference samples and original variable arrays")
    else:
        import torch

        from corrgan.unconditional import import_original_arrays

        if args.report is None or args.report.exists():
            raise ValueError("Provide a new --report path")
        if args.checkpoint is not None and args.checkpoint.exists():
            raise FileExistsError(args.checkpoint)
        torch.set_num_threads(4)
        with np.load(args.reference) as a:
            model = import_original_arrays(a)
            with torch.no_grad():
                generated = model(torch.from_numpy(a["noise"])).numpy()
            expected = a["samples"]
            np.testing.assert_allclose(generated, expected, atol=2e-5, rtol=2e-5)
            difference = float(np.max(np.abs(generated - expected)))
        report = {
            "scope": "published 80D generator inference; excludes training parity",
            "samples": 30,
            "entries": int(generated.size),
            "maximum_absolute_difference": difference,
            "torch": torch.__version__,
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "source": json.loads(args.reference.with_suffix(".json").read_text()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        if args.checkpoint is not None:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "generator": model.state_dict(),
                    "model_config": {
                        "matrix_dim": 80,
                        "latent_dim": 100,
                        "channels": 256,
                    },
                    "provenance": "original author-released checkpoint",
                    "epoch": None,
                },
                args.checkpoint,
            )
        print(
            json.dumps({"max_difference": difference, "entries": int(generated.size)})
        )


if __name__ == "__main__":
    main()
