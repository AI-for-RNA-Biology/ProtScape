"""Independent sequence controls must never inherit ProtScape training."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from downstream_tasks import residue_probe as probe
from downstream_tasks.training.cv_utils import SplitPlan
from scripts.cscs.run_residue_ablation import REPO, sequence_tasks


@pytest.fixture
def experiment(tmp_path):
    cache = tmp_path / "residues"
    cache.mkdir()
    rng = np.random.default_rng(42)
    values = rng.normal(size=(26, 4)).astype(np.float16)
    genes = [f"G{i}" for i in range(12)] + ["g0"]
    np.save(cache / "residues.npy", values)
    np.save(cache / "offsets.npy", np.arange(0, 27, 2))
    (cache / "manifest.json").write_text(json.dumps(dict(genes=genes, embedding_dim=4)))
    (cache / "COMPLETE.json").write_text("{}")
    # Deliberately no global normalization file or pretrained model exists.
    means = values.reshape(13, 2, 4).astype(np.float32).mean(1)
    pd.DataFrame({"gene_name": genes, "ESM2-Embeddings": list(means)}).to_pickle(cache / "mean.plk")
    return tmp_path, cache, means


@pytest.mark.parametrize("mode", ["mlp", "attention"])
def test_pooler_is_fresh_and_normalization_is_train_only(experiment, mode):
    root, cache, means = experiment
    ids = probe.cache_indices(cache, ["G0", "G2", "g1"])
    assert ids.tolist() == [12, 2, 1]  # downstream last-record convention
    train_means = means[[12, 1, 2]]
    model = probe.ResidueClassifier(cache, mode, 3, train_means, 2)
    torch.testing.assert_close(model.pooler.feature_mean, torch.from_numpy(train_means.mean(0)))
    torch.testing.assert_close(model.pooler.feature_std, torch.from_numpy(train_means.std(0)))
    assert torch.count_nonzero(model.pooler.output.weight) == 0
    expected = (means[ids] - train_means.mean(0)) / train_means.std(0)
    torch.testing.assert_close(model.pooler(torch.tensor(ids)), torch.from_numpy(expected))
    model(torch.tensor(ids)).square().mean().backward()
    assert model.pooler.output.weight.grad.abs().sum() > 0
    assert model.head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["mlp", "attention", "swe"])
def test_independent_full_cv_and_skip_completed(experiment, monkeypatch, mode):
    root, cache, _ = experiment
    config = root / "source/configs"
    config.mkdir(parents=True)
    (config / "residue_ablation.yaml").write_text(yaml.safe_dump(
        dict(downstream=dict(epochs=2, patience=2, seed=42))))
    (root / "inputs.json").write_text(json.dumps(dict(tasks=[dict(task="tiny", csv="unused.csv")])))
    genes = [f"G{i}" for i in range(12)]
    labels = np.tile(np.eye(2, dtype=np.float32), (6, 1))
    folds = [np.array([2 * i, 2 * i + 1]) for i in range(6)]
    plan = SplitPlan(folds, 0, folds[1:], None, 5)
    split = root / "original_split.npz"
    np.savez(split, genes=genes, labels=labels, class_names=["a", "b"],
             **{f"fold_{i}": fold for i, fold in enumerate(folds)})
    monkeypatch.setattr(probe, "load_task_partition", lambda *a: (genes, labels, ["a", "b"], plan))
    monkeypatch.setattr(probe, "partition_path", lambda *a: split)
    probe.run(root, mode, "tiny", 3, .001)
    size = "ref3" if mode == "swe" else "h3"
    output = root / "downstream/tiny" / f"esm_{mode}_linear/{size}_lr0.001"
    result = pd.read_csv(output / "results.csv").iloc[0]
    assert result.n_cv_folds == 5 and not result.protscape_checkpoint_used
    assert np.isfinite(result.test_auprc_macro_mean)
    assert json.loads((output / "provenance.json").read_text())["supervision"] == "downstream_train_fold"
    with np.load(output / "split_indices.npz") as actual:
        np.testing.assert_array_equal(actual["fold_0"], folds[0])
    states = list(output.glob("fold_*/best.pt"))
    assert len(states) == 5
    times = [path.stat().st_mtime_ns for path in states]
    probe.run(root, mode, "tiny", 3, .001)
    assert times == [path.stat().st_mtime_ns for path in states]


@pytest.mark.parametrize("mode", ["mlp", "attention", "swe"])
def test_epoch_resume_matches_uninterrupted_fit(experiment, mode):
    root, cache, means = experiment
    ids = torch.arange(12)
    labels = np.tile(np.eye(2, dtype=np.float32), (6, 1))
    train, val = np.arange(2, 10), np.array([10, 11])
    models = []
    for name, stages in [("continuous", [3]), ("resumed", [1, 3])]:
        output = root / name
        output.mkdir()
        for epochs in stages:
            torch.manual_seed(42)
            model = probe.ResidueClassifier(cache, mode, 3, means[train], 2)
            probe.fit_fold(model, ids, labels, train, val, output,
                           lr=.001, epochs=epochs, patience=5, seed=42, batch_size=3)
        models.append(model)
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)


def test_queue_never_passes_pretrained_pooler_to_sequence_baseline(tmp_path, monkeypatch):
    from downstream_tasks.run import parse_args
    config = yaml.safe_load((REPO / "configs/residue_ablation.yaml").read_text())
    (tmp_path / "source/configs").mkdir(parents=True)
    (tmp_path / "source/configs/residue_ablation.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "inputs.json").write_text(json.dumps(dict(release="/unused_release")))
    tasks = list(sequence_tasks(tmp_path, [dict(task="corum", csv="unused.csv")], config["downstream"]))
    assert len(tasks) == 14
    for task in tasks:
        if task["key"].startswith("esm_"):
            assert "downstream_tasks.residue_probe" in task["command"]
            assert "--inference-root" not in task["command"]
        else:
            assert task["key"].startswith(("lr_esm_bos_", "lr_esm_mean_"))
            assert "/sequence/" not in " ".join(task["command"])
            monkeypatch.setattr("sys.argv", ["run", *task["command"][3:]])
            args = parse_args()
            assert args.epochs == 300 and args.patience == 50
