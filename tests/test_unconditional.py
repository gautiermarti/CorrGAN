import json
import sys

import numpy as np
import pytest
import torch
import yaml

from corrgan.unconditional import DCGANReferenceDiscriminator, OriginalCorrGANGenerator
from corrgan.unconditional_training import adversarial_step, sample_unconditional


def test_reference_update_learns_both_networks_and_updates_bn_once():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    g = OriginalCorrGANGenerator(matrix_dim=8, channels=16, latent_dim=10)
    d = DCGANReferenceDiscriminator(matrix_dim=8, channels=4)
    g_initial, d_initial = g.dense.weight.clone(), next(d.parameters()).clone()
    optg, optd = torch.optim.Adam(g.parameters()), torch.optim.Adam(d.parameters())
    real = torch.eye(8).expand(4, -1, -1)
    losses = adversarial_step(g, d, real, optg, optd)
    assert np.isfinite(losses).all()
    assert not torch.equal(g_initial, g.dense.weight)
    assert not torch.equal(d_initial, next(d.parameters()))
    assert g.bn0.num_batches_tracked == 1
    assert g.bn1.num_batches_tracked == 1
    assert g.bn2.num_batches_tracked == 1
    state = {k: v.clone() for k, v in g.state_dict().items()}
    before = torch.random.get_rng_state().clone()
    matrices, info = sample_unconditional(g, samples=5, seed=7, batch_size=3)
    assert torch.equal(before, torch.random.get_rng_state())
    assert np.linalg.eigvalsh(matrices).min() > 0
    assert info["repair_converged"]
    for k, v in g.state_dict().items():
        assert torch.equal(v, state[k])


def test_unconditional_train_and_sample_need_no_labels_or_scores(tmp_path, monkeypatch):
    from corrgan.cli.sample_unconditional import main as sample_main
    from corrgan.cli.train_unconditional import main as train_main

    root = tmp_path / "data"
    root.mkdir()
    x = np.random.default_rng(17).normal(size=(24, 30, 8))
    matrices = np.stack([np.corrcoef(a.T) for a in x]).astype(np.float32)
    np.savez(root / "train.npz", matrices=matrices[:16])
    np.savez(root / "validation.npz", matrices=matrices[16:])
    config = {
        "model": {"latent_dim": 10, "channels": 16},
        "discriminator": {"channels": 4, "dropout": 0.3},
        "training": {
            "seed": 17,
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 0.0001,
            "beta1": 0.9,
            "beta2": 0.999,
            "validation_every": 1,
            "validation_samples": 6,
            "validation_seed": 44301,
            "sampling_batch_size": 3,
        },
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    run = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data",
            str(root),
            "--config",
            str(cfg),
            "--output",
            str(run),
            "--device",
            "cpu",
        ],
    )
    train_main()
    assert json.loads((run / "complete.json").read_text())["epochs_completed"] == 2
    with pytest.raises(FileExistsError):
        train_main()
    output = tmp_path / "samples.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample",
            "--data",
            str(root),
            "--checkpoint",
            str(run / "best_structural.pt"),
            "--output",
            str(output),
            "--samples",
            "6",
            "--device",
            "cpu",
        ],
    )
    sample_main()
    with np.load(output) as a:
        assert a["conditioning"].item() == "unconditional"
        assert a["matrices"].shape == (6, 8, 8)
        assert np.linalg.eigvalsh(a["matrices"]).min() > 0
