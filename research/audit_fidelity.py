"""Diagnose frozen cGAN runs using train/validation data only.

Outputs are diagnostic, not a new holdout study. Published weights may have
seen the legacy archive; their membership and original training recipe are unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from corrbench.geometry import nearest_correlation_torch
from corrbench.metrics import sliced_wasserstein, structural_features
from scipy.stats import wasserstein_distance

from corrgan.legacy import CCORRGAN_SHA256, load_published_ccorrgan
from corrgan.models import ConditionalCorrGANDiscriminator, ConditionalCorrGANGenerator


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def sample(model, labels, scores, device, *, seed=44301):
    rng = torch.Generator(device).manual_seed(seed)
    outputs = []
    for start in range(0, len(labels), 32):
        y = torch.as_tensor(labels[start : start + 32], device=device)
        z = torch.randn(len(y), 80, generator=rng, device=device)
        s = torch.as_tensor(scores[start : start + 32], device=device)
        outputs.append(model(z, y, s).cpu().numpy())
    return np.concatenate(outputs)


def summarize(raw, real_features, train_scale, device):
    upper = np.triu(raw.astype(np.float64), 1)
    symmetric = upper + upper.swapaxes(-1, -2) + np.eye(raw.shape[-1])
    eigenvalues = np.linalg.eigvalsh(symmetric)
    repaired = []
    for start in range(0, len(raw), 32):
        x, info = nearest_correlation_torch(
            torch.as_tensor(symmetric[start : start + 32], device=device),
            max_iterations=1000,
            tolerance=1e-6,
            return_info=True,
        )
        if not bool(torch.all(info["converged"])):
            raise RuntimeError("Repair failed to converge")
        repaired.append(x.cpu().numpy())
    repaired = np.concatenate(repaired)
    f, names = structural_features(repaired)
    marginal = np.array(
        [
            wasserstein_distance(real_features[:, j], f[:, j]) / train_scale[j]
            for j in range(len(names))
        ]
    )
    eig_repaired = np.linalg.eigvalsh(repaired)
    result = {
        "feature_swd": sliced_wasserstein(
            real_features, f, scale=train_scale, seed=44301, device="cpu"
        ),
        # Diagnostic feature subsets; these do not replace the frozen metric.
        "entry_features_swd": sliced_wasserstein(
            real_features[:, :9],
            f[:, :9],
            scale=train_scale[:9],
            seed=44301,
            device="cpu",
        ),
        "features_without_logs_swd": sliced_wasserstein(
            real_features[:, :-2],
            f[:, :-2],
            scale=train_scale[:-2],
            seed=44301,
            device="cpu",
        ),
        "raw_pd_fraction": float(np.mean(eigenvalues[:, 0] > 0)),
        "raw_minimum_eigenvalue_median": float(np.median(eigenvalues[:, 0])),
        "repair_entry_rmse": float(np.sqrt(np.mean((repaired - symmetric) ** 2))),
        "near_floor_eigenvalue_fraction": float(np.mean(eig_repaired < 1e-5)),
        "repaired_smallest_eigenvalue_median": float(np.median(eig_repaired[:, 0])),
        "raw_diagonal_mean": float(np.diagonal(raw, axis1=-2, axis2=-1).mean()),
        "raw_asymmetry_rmse": float(
            np.sqrt(np.mean((raw - raw.swapaxes(-1, -2)) ** 2))
        ),
        "feature_marginal_w1_train_std": dict(
            zip(names, marginal.tolist(), strict=True)
        ),
        "real_feature_means": dict(
            zip(names, real_features.mean(0).tolist(), strict=True)
        ),
        "generated_feature_means": dict(zip(names, f.mean(0).tolist(), strict=True)),
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-root", type=Path, required=True)
    p.add_argument("--published-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    report = {
        "scope": "train/validation diagnosis; no test NPZ opened",
        "published_membership": "unknown; not an independent held-out baseline",
        "source_sha256": {},
        "markets": {},
        "published_checkpoint_sha256": CCORRGAN_SHA256,
        "script_sha256": digest(Path(__file__)),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    published = load_published_ccorrgan(args.published_model).to(args.device)
    base = args.research_root
    runs = base / "artifacts/neurreps_revision_20261003/retraining"
    for market, directory in [("taiwan", "taiwan_final_ward"), ("sp500", "corrgan")]:
        data = base / "data/processed" / directory
        arrays = {}
        for split in ["train", "validation"]:
            path = data / f"{split}.npz"
            report["source_sha256"][f"{directory}/{split}"] = digest(path)
            with np.load(path) as a:
                arrays[split] = {k: a[k] for k in ["matrices", "labels"]}
                if "sharpes" in a:
                    arrays[split]["sharpes"] = a["sharpes"]
        train, val = arrays["train"], arrays["validation"]
        train_features, _ = structural_features(train["matrices"])
        scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
        rng = np.random.default_rng(44301)
        indices = np.concatenate(
            [
                rng.choice(np.flatnonzero(val["labels"] == c), 100, replace=False)
                for c in range(3)
            ]
        )
        real = val["matrices"][indices]
        labels = val["labels"][indices].astype(np.int64)
        real_features, _ = structural_features(real)
        rows = []
        other_indices = np.concatenate(
            [
                rng.choice(
                    np.setdiff1d(np.flatnonzero(val["labels"] == c), indices),
                    100,
                    replace=False,
                )
                for c in range(3)
            ]
        )
        replay_indices = np.concatenate(
            [
                rng.choice(np.flatnonzero(train["labels"] == c), 100, replace=False)
                for c in range(3)
            ]
        )
        for name, matrices in [
            ("validation_vs_validation", val["matrices"][other_indices]),
            ("train_replay", train["matrices"][replay_indices]),
        ]:
            row = {
                "name": name,
                **summarize(matrices, real_features, scale, args.device),
            }
            rows.append(row)
            print(market, name, row["feature_swd"], flush=True)
        sources = [("published", None, published)]
        for seed in range(1701, 1706):
            for name in ["best_structural", "last"]:
                sources.append(
                    (
                        f"{seed}_{name}",
                        runs / f"{market}_ccorrgan_seed{seed}" / f"{name}.pt",
                        None,
                    )
                )
        for name, path, model in sources:
            row = {"name": name}
            if path:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                model = ConditionalCorrGANGenerator(**checkpoint["model_config"])
                model.load_state_dict(checkpoint["generator"])
                model = model.to(args.device).eval()
                row.update(checkpoint_sha256=digest(path), epoch=checkpoint["epoch"])
                continuous = checkpoint["model_config"]["continuous_condition"]
                if continuous:
                    scaler = checkpoint["condition_scaler"]
                    scores = (
                        (val["sharpes"][indices] - scaler["mean"])
                        / max(scaler["std"], 1e-8)
                    ).astype(np.float32)
                else:
                    scores = np.zeros(len(labels), np.float32)
            else:
                scores = np.zeros(len(labels), np.float32)
            raw = sample(model, labels, scores, args.device)
            row.update(summarize(raw, real_features, scale, args.device))
            probe = sample(
                model,
                np.repeat(np.arange(3), 32),
                np.zeros(96, np.float32),
                args.device,
            )
            row["fixed_condition_pixel_std_by_class"] = [
                float(probe[c * 32 : (c + 1) * 32].std(axis=0).mean()) for c in range(3)
            ]
            if path:
                d = ConditionalCorrGANDiscriminator(
                    continuous_condition=continuous,
                    channels=checkpoint["model_config"]["channels"],
                )
                d.load_state_dict(checkpoint["discriminator"])
                d = d.to(args.device).eval()
                with torch.no_grad():
                    y = torch.as_tensor(labels, device=args.device)
                    s = torch.as_tensor(scores, device=args.device)
                    for label, image in [("real", real), ("fake", raw)]:
                        logits = d(torch.as_tensor(image, device=args.device), y, s)
                        row[f"discriminator_{label}_probability_mean"] = float(
                            logits.sigmoid().mean()
                        )
                        row[f"discriminator_{label}_logit_std"] = float(logits.std())
                del d, checkpoint
            rows.append(row)
            print(
                market,
                name,
                "SWD",
                row["feature_swd"],
                "floor",
                row["near_floor_eigenvalue_fraction"],
                "noise std",
                row["fixed_condition_pixel_std_by_class"],
                flush=True,
            )
            if path:
                del model
        report["markets"][market] = {
            "validation_indices": indices.tolist(),
            "results": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
