"""Audit and evaluate the full reference CorrGAN run on validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from corrbench.metrics import (
    classifier_two_sample_auc,
    manifold_precision_recall,
    portfolio_functionals,
    sliced_wasserstein,
    structural_features,
    validity_summary,
)

from corrgan.unconditional import OriginalCorrGANGenerator
from corrgan.unconditional_training import sample_unconditional


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    frozen = {
        str(path): sha(path)
        for path in [
            Path(__file__),
            root / "docs/unconditional_validation_protocol.md",
            root / "src/corrgan/unconditional.py",
            root / "src/corrgan/unconditional_training.py",
        ]
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    while True:
        state = read(args.queue)
        if state["status"] == "complete":
            break
        if state["status"] == "failed" or not args.wait:
            raise RuntimeError("A full successful training budget is required")
        time.sleep(15)
    for filename, expected in {**state["source_sha256"], **frozen}.items():
        assert sha(Path(filename)) == expected, filename
    torch.set_num_threads(4)
    complete = read(args.run / "complete.json")
    assert complete["epochs_completed"] == 50
    assert complete["stopped_at_max_steps"] is False
    cfg = yaml.safe_load((args.run / "config.yaml").read_text())
    assert cfg == yaml.safe_load(
        (root / "configs/corrgan_unconditional_reference.yaml").read_text()
    )
    provenance = read(args.run / "provenance.json")
    arrays = {}
    for split in ["train", "validation"]:
        assert sha(args.data / f"{split}.npz") == provenance["source_hashes"][split]
        with np.load(args.data / f"{split}.npz") as data:
            arrays[split] = data["matrices"].astype(np.float32)
    assert complete["steps"] == 50 * (len(arrays["train"]) // 64)
    history = read(args.run / "history.json")
    assert [h["epoch"] for h in history] == list(range(1, 51))
    scored = [h for h in history if "validation_feature_swd" in h]
    selected = min(scored, key=lambda h: h["validation_feature_swd"])
    indices = np.load(args.run / "selected_validation_indices.npy")
    np.testing.assert_array_equal(
        indices,
        np.random.default_rng(44301).choice(
            len(arrays["validation"]), 300, replace=False
        ),
    )
    real = arrays["validation"][indices]
    train_features, feature_names = structural_features(arrays["train"])
    features, _ = structural_features(real)
    feature_scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
    train_portfolio, portfolio_names = portfolio_functionals(arrays["train"])
    portfolio_scale = np.maximum(train_portfolio.std(axis=0, ddof=1), 1e-6)
    real_portfolio, _ = portfolio_functionals(real)
    args.output.mkdir(parents=True)
    audit = {
        "scope": "Validation only; one training seed; original recipe unverified",
        "epochs": 50,
        "steps": complete["steps"],
        "selected_epoch": selected["epoch"],
        "source_sha256": provenance["source_hashes"],
        "code_sha256": frozen,
        "run_sha256": {
            name: sha(args.run / name)
            for name in [
                "history.json",
                "complete.json",
                "config.yaml",
                "provenance.json",
                "best_structural.pt",
                "last.pt",
                "selected_validation_indices.npy",
            ]
        },
        "native_validation_recorded": selected["validation_feature_swd"],
    }
    rows = []
    for weights, filename in [
        ("selected", "best_structural.pt"),
        ("final", "last.pt"),
        ("replay", None),
    ]:
        if filename:
            checkpoint = torch.load(
                args.run / filename, map_location="cpu", weights_only=True
            )
            expected_epoch = selected["epoch"] if weights == "selected" else 50
            assert checkpoint["epoch"] == expected_epoch
            assert checkpoint["training_seed"] == 1701
            generator = OriginalCorrGANGenerator(**checkpoint["model_config"]).cuda()
            generator.load_state_dict(checkpoint["generator"])
            if weights == "selected":
                native, _ = sample_unconditional(
                    generator, samples=300, seed=44301, batch_size=64, repair=True
                )
                native_features, _ = structural_features(native)
                value = sliced_wasserstein(
                    features,
                    native_features,
                    scale=feature_scale,
                    seed=44301,
                    device="cpu",
                )
                audit["native_validation_recomputed"] = value
                assert abs(value - selected["validation_feature_swd"]) < 2e-5
        for seed in range(2027, 2032):
            if filename:
                matrices, sampling = sample_unconditional(
                    generator, samples=300, seed=seed, batch_size=64, repair=True
                )
            else:
                replay_indices = np.random.default_rng(seed).integers(
                    len(arrays["train"]), size=300
                )
                matrices = arrays["train"][replay_indices]
                sampling = {"method": "uniform training replay with replacement"}
            path = args.output / f"{weights}_{seed}.npz"
            np.savez_compressed(
                path,
                matrices=matrices,
                selected_validation_indices=indices,
                split="validation",
                conditioning="unconditional",
                generation_seed=seed,
            )
            generated_features, _ = structural_features(matrices)
            generated_portfolio, _ = portfolio_functionals(matrices)
            manifold = manifold_precision_recall(
                features / feature_scale,
                generated_features / feature_scale,
                neighbors=5,
                seed=2027,
            )
            classifier = classifier_two_sample_auc(
                features / feature_scale, generated_features / feature_scale, seed=2027
            )
            row = {
                "weights": weights,
                "generation_seed": seed,
                "feature_swd": sliced_wasserstein(
                    features,
                    generated_features,
                    scale=feature_scale,
                    seed=2027,
                    device="cpu",
                ),
                "portfolio_swd": sliced_wasserstein(
                    real_portfolio,
                    generated_portfolio,
                    scale=portfolio_scale,
                    seed=2027,
                    device="cpu",
                ),
                "precision": manifold["precision"],
                "recall": manifold["recall"],
                "classifier_auc": classifier["auc"],
                "validity": validity_summary(matrices),
                "sampling": sampling,
                "sha256": sha(path),
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in [
                            "weights",
                            "generation_seed",
                            "feature_swd",
                            "portfolio_swd",
                        ]
                    }
                ),
                flush=True,
            )
    summary = []
    for method in ["selected", "final", "replay"]:
        subset = [row for row in rows if row["weights"] == method]
        metrics = {}
        for key in [
            "feature_swd",
            "portfolio_swd",
            "precision",
            "recall",
            "classifier_auc",
        ]:
            values = np.array([row[key] for row in subset])
            metrics[key] = {
                "mean": float(values.mean()),
                "generation_sd": float(values.std(ddof=1)),
            }
        summary.append({"weights": method, "metrics": metrics})
    report = {
        **audit,
        "status": "passed",
        "rows": rows,
        "summary": summary,
        "feature_names": feature_names,
        "portfolio_names": portfolio_names,
        "test_opened": False,
        "conditioning": "none",
        "training_seeds": 1,
        "generation_seeds": list(range(2027, 2032)),
        "limits": "Validation selected the model; these are diagnostic scores. "
        "Generation SD does not replicate training. No conditional or faithful "
        "original-training result is claimed.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
