"""Render the completed validation-only reconstruction, including its failure."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "passed" and not report["test_opened"]
    assert report["epochs"] == 50 and report["training_seeds"] == 1
    for filename, expected in report["code_sha256"].items():
        assert sha(Path(filename)) == expected
    for filename, expected in report["run_sha256"].items():
        assert sha(args.run / filename) == expected
    for row in report["rows"]:
        path = args.report.parent / f"{row['weights']}_{row['generation_seed']}.npz"
        assert sha(path) == row["sha256"]
    labels = {
        "selected": "CorrGAN selected (epoch 45)",
        "final": "CorrGAN final (epoch 50)",
        "replay": "Training replay",
    }
    for summary in report["summary"]:
        rows = [r for r in report["rows"] if r["weights"] == summary["weights"]]
        assert len(rows) == 5
        for name, values in summary["metrics"].items():
            array = np.array([r[name] for r in rows])
            np.testing.assert_allclose(
                [array.mean(), array.std(ddof=1)],
                [values["mean"], values["generation_sd"]],
                rtol=0,
                atol=1e-14,
            )
    result = {
        key: report[key]
        for key in [
            "scope",
            "epochs",
            "steps",
            "selected_epoch",
            "source_sha256",
            "run_sha256",
            "rows",
            "summary",
            "native_validation_recorded",
            "native_validation_recomputed",
            "conditioning",
            "training_seeds",
            "generation_seeds",
            "test_opened",
            "limits",
        ]
    }
    result["report_sha256"] = sha(args.report)
    result["render_code_sha256"] = sha(Path(__file__))
    result["finding"] = "The full reference training budget completed but does "
    result["finding"] += "not reproduce realistic archive distributions."
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "unconditional_validation_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    md = [
        "# Full unconditional reference-run validation — 2026-10-05",
        "",
        result["finding"],
        "",
        "The 50-epoch legacy-archive run completed 9,800 optimizer steps. "
        "Validation selected epoch 45; its native structural score 3.21389055 "
        "reproduces exactly. The trainer and evaluator never open test data or "
        "use regime labels to condition the generator.",
        "",
        "| Method | Structural SWD ↓ | Portfolio SWD ↓ | Precision | Recall "
        "| C2ST AUC (.5 ideal) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        m = row["metrics"]
        f, v = m["feature_swd"], m["portfolio_swd"]
        md.append(
            f"| {labels[row['weights']]} | {f['mean']:.4f} ± "
            f"{f['generation_sd']:.4f} | {v['mean']:.4f} ± "
            f"{v['generation_sd']:.4f} | {m['precision']['mean']:.4f} | "
            f"{m['recall']['mean']:.4f} | {m['classifier_auc']['mean']:.4f} |"
        )
    md += [
        "",
        "All 3,000 generated matrices fail raw validity after symmetry "
        "and unit-diagonal preparation. Converged float64 nearest-correlation "
        "repair makes all valid, with minimum eigenvalues near the configured "
        "1e-6 floor. Selected draws need at most 60 iterations per batch; final "
        "draws need at most 237. Repair resolves validity, not the distribution "
        "mismatch measured above.",
        "",
        "Scores use the same 300 uniform validation rows. Feature and portfolio "
        "scales come only from training matrices. Five generation seeds describe "
        "draw variability for one fitted model. The summary uses projection seed "
        "2027; native checkpoint selection used 44301 for both draws and "
        "projections, so its absolute score is not the summary score.",
        "",
        "The recovered native generator separately matches the author-released "
        "TensorFlow checkpoint to maximum error 7.153e-7. That inference result "
        "is unchanged. The current discriminator, optimizer and training recipe "
        "are explicit reference choices, not verified original training code. "
        "This result therefore diagnoses this reconstruction; it does not "
        "invalidate the published CorrGAN method or certify its reproduction.",
        "",
        "Validation has selected the checkpoint and the legacy archive is a "
        "random matrix split without dates/constituent identities. No untouched "
        "temporal generalization, conditional generation or five-training-seed "
        "claim follows. Preserve the failure while investigating the source "
        "training recipe and matched data/preprocessing.",
        "",
    ]
    (args.output / "unconditional_validation_results.md").write_text("\n".join(md))
    history = json.loads((args.run / "history.json").read_text())
    scored = [r for r in history if "validation_feature_swd" in r]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].plot(
        [r["epoch"] for r in scored],
        [r["validation_feature_swd"] for r in scored],
        marker="o",
        color="#587487",
    )
    axes[0].scatter(
        [45],
        [report["native_validation_recorded"]],
        s=90,
        color="#bd803d",
        zorder=5,
        label="Selected epoch",
    )
    axes[0].set(
        xlabel="Training epoch",
        ylabel="Native validation structural SWD ↓",
        ylim=(0, 9.2),
    )
    axes[0].legend(frameon=False, fontsize=9)
    values = [r["metrics"]["feature_swd"]["mean"] for r in report["summary"]]
    errors = [r["metrics"]["feature_swd"]["generation_sd"] for r in report["summary"]]
    axes[1].bar(
        np.arange(3),
        values,
        yerr=errors,
        capsize=4,
        color=["#177c75", "#587487", "#bd803d"],
    )
    axes[1].set_xticks(np.arange(3), ["Selected", "Final", "Replay"])
    axes[1].set_ylabel("Diagnostic structural SWD ↓")
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "A recovered generator is not a recovered training recipe",
        x=0.03,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.025,
        "Legacy S&P matrix archive · validation only · 1 training "
        "seed · five diagnostic generation seeds\n"
        "Left: selection draws/projections use seed 44301. Right: fixed projection "
        "seed 2027; whiskers show generation SD.",
        fontsize=8.5,
        color="#52616b",
    )
    fig.tight_layout(rect=[0, 0.14, 1, 0.91], w_pad=3)
    for extension in ["png", "pdf", "svg"]:
        path = args.output / f"unconditional_validation.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        if extension == "svg":
            path.write_text(
                "\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)
    print(result["finding"])


if __name__ == "__main__":
    main()
