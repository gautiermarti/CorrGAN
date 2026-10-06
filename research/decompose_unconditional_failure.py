"""Post-run, validation-only decomposition of the frozen reconstruction failure.

No model fitting, selection or test access. Reuses every saved selected/final/
replay draw, verifies hashes, and preserves the original full structural score.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from corrbench.metrics import sliced_wasserstein, structural_features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    audit = json.loads(args.report.read_text())
    assert audit["status"] == "passed" and audit["test_opened"] is False
    assert len(audit["rows"]) == 15
    arrays = {}
    for split in ["train", "validation"]:
        path = args.data / f"{split}.npz"
        assert sha(path) == audit["source_sha256"][split]
        with np.load(path) as data:
            arrays[split] = data["matrices"].astype(np.float32)
    index_path = args.run / "selected_validation_indices.npy"
    assert sha(index_path) == audit["run_sha256"][index_path.name]
    indices = np.load(index_path)
    np.testing.assert_array_equal(
        indices,
        np.random.default_rng(44301).choice(len(arrays["validation"]), 300, False),
    )
    train_features, names = structural_features(arrays["train"])
    real = arrays["validation"][indices]
    features, _ = structural_features(real)
    scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
    groups = {"entries": np.arange(9), "spectrum": np.arange(9, 21)}

    def spectrum_summary(matrices):
        eigenvalues = np.linalg.eigvalsh(matrices.astype(np.float64))
        return {
            "median_minimum_eigenvalue": float(np.median(eigenvalues[:, 0])),
            "median_eigenvalues_at_or_below_2e_minus6": float(
                np.median(np.sum(eigenvalues <= 2e-6, axis=1))
            ),
            "mean_eigenvalues_at_or_below_2e_minus6": float(
                np.mean(np.sum(eigenvalues <= 2e-6, axis=1))
            ),
            "median_log_condition_number": float(
                np.median(np.log(eigenvalues[:, -1] / eigenvalues[:, 0]))
            ),
        }

    rows = []
    for row in audit["rows"]:
        path = args.report.parent / f"{row['weights']}_{row['generation_seed']}.npz"
        assert sha(path) == row["sha256"]
        with np.load(path) as data:
            matrices = data["matrices"]
        generated, _ = structural_features(matrices)
        full = sliced_wasserstein(
            features, generated, scale=scale, seed=2027, device="cpu"
        )
        np.testing.assert_allclose(full, row["feature_swd"], rtol=0, atol=1e-12)
        # Equal-size empirical marginals have exact 1D W2 via sorted pairing.
        marginal = np.sqrt(
            np.mean(
                ((np.sort(features, axis=0) - np.sort(generated, axis=0)) / scale) ** 2,
                axis=0,
            )
        )
        rows.append(
            {
                "weights": row["weights"],
                "generation_seed": row["generation_seed"],
                "artifact_sha256": row["sha256"],
                "full_structural_swd": full,
                "group_swd": {
                    name: sliced_wasserstein(
                        features[:, columns],
                        generated[:, columns],
                        scale=scale[columns],
                        seed=2027,
                        device="cpu",
                    )
                    for name, columns in groups.items()
                },
                "feature_w2": dict(zip(names, marginal.tolist(), strict=True)),
                "spectrum": spectrum_summary(matrices),
            }
        )
    summaries = []
    for method in ["selected", "final", "replay"]:
        subset = [r for r in rows if r["weights"] == method]
        assert [r["generation_seed"] for r in subset] == list(range(2027, 2032))
        fields = {
            "full_structural_swd": [r["full_structural_swd"] for r in subset],
            **{
                name + "_swd": [r["group_swd"][name] for r in subset] for name in groups
            },
        }
        summaries.append(
            {
                "weights": method,
                "scores": {
                    name: {
                        "mean": float(np.mean(values)),
                        "generation_sd": float(np.std(values, ddof=1)),
                    }
                    for name, values in fields.items()
                },
                "feature_w2_mean": {
                    name: float(np.mean([r["feature_w2"][name] for r in subset]))
                    for name in names
                },
                "spectrum_draw_means": {
                    name: float(np.mean([r["spectrum"][name] for r in subset]))
                    for name in subset[0]["spectrum"]
                },
            }
        )
    result = {
        "scope": "Post-run decomposition of validation-only reference reconstruction",
        "selection_unchanged": True,
        "test_opened": False,
        "report_sha256": sha(args.report),
        "code_sha256": sha(Path(__file__)),
        "source_sha256": audit["source_sha256"],
        "groups": {
            name: [names[i] for i in columns] for name, columns in groups.items()
        },
        "real_validation_spectrum": spectrum_summary(real),
        "rows": rows,
        "summary": summaries,
        "interpretation": "Group scores use different projection spaces and are not "
        "additive decompositions of the full score. Marginal W2 is a diagnostic, "
        "not a replacement selection metric. Repaired eigenvalue counts describe "
        "a symptom and do not establish the training failure's cause. One fit, "
        "five draws; no original released-checkpoint performance claim.",
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Unconditional reconstruction: entry and spectrum diagnosis",
        "",
        "Post-run, validation only. All 15 saved artifacts and source arrays pass "
        "hash checks; every frozen full structural score reproduces exactly. "
        "No model is fitted, reselected or evaluated on test data.",
        "",
        "| Weights | Full SWD | Entry-only SWD | Spectrum-only SWD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        values = [
            row["scores"][key]
            for key in ["full_structural_swd", "entries_swd", "spectrum_swd"]
        ]
        lines.append(
            f"| {row['weights']} | "
            + " | ".join(f"{v['mean']:.4f} ± {v['generation_sd']:.4f}" for v in values)
            + " |"
        )
    lines += ["", result["interpretation"], ""]
    lines += [
        "## Eigenvalues near the repair floor",
        "",
        "The real validation median minimum eigenvalue is "
        f"{result['real_validation_spectrum']['median_minimum_eigenvalue']:.6f}; "
        "none of its eigenvalues are at or below 2e-6.",
        "",
        "| Weights | Mean count of eigenvalues ≤ 2e-6 per matrix | "
        "Mean over draws of median minimum eigenvalue |",
        "| --- | ---: | ---: |",
    ]
    for row in summaries:
        spectrum = row["spectrum_draw_means"]
        lines.append(
            f"| {row['weights']} | "
            f"{spectrum['mean_eigenvalues_at_or_below_2e_minus6']:.3f} | "
            f"{spectrum['median_minimum_eigenvalue']:.8f} |"
        )
    lines += [
        "",
        "These near-floor spectra help explain why entrywise appearance is "
        "insufficient for this reconstruction. They do not identify which "
        "training choice caused the mismatch, and do not justify dropping "
        "spectral metrics or changing the declared selection rule.",
        "",
    ]
    (args.output / "results.md").write_text("\n".join(lines))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
