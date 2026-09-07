from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pretraining.rank_global_s2gae_validation as ranking


def test_shards_are_disjoint_and_cover_every_configuration():
    configurations = [{"name": str(index)} for index in range(54)]
    shards = [ranking.shard_runs(configurations, index, 4) for index in range(4)]
    names = [run["name"] for shard in shards for run in shard]
    assert len(names) == len(set(names)) == 54
    assert set(names) == {str(index) for index in range(54)}
    assert [len(shard) for shard in shards] == [14, 14, 13, 13]
    for index, count in ((0, 0), (4, 4), (-1, 4), (0, 55)):
        with pytest.raises(ValueError, match="shard"):
            ranking.shard_runs(configurations, index, count)


@pytest.fixture
def fake_sweep(tmp_path, monkeypatch):
    configurations = [
        {"name": f"run_{index}", "experiment_role": "sweep", "hidden_dim": 4,
         "num_layers": 2, "dropout": 0.4}
        for index in range(54)
    ]
    args = Namespace(
        runs_root=tmp_path / "runs", sweep_config=tmp_path / "sweep.yaml",
        networks_dir=tmp_path / "networks", esm2_embeddings=tmp_path / "esm.pkl",
        output_dir=tmp_path / "output", shard_index=0, shard_count=27, device="cpu",
    )
    args.sweep_config.write_text("synthetic sweep fixture")
    for index in (0, 27):
        path = args.runs_root / f"run_{index}" / "best_model_state_dict.pt"
        path.parent.mkdir(parents=True)
        path.write_text(f"synthetic checkpoint {index}")
    data = SimpleNamespace(
        features=torch.randn(4, 3),
        train_edge_index=torch.tensor([[0, 1], [1, 0]]),
        val_edge_index=torch.tensor([[0], [2]]),
        protein_names=["a", "b", "c", "d"],
        feature_mean=torch.zeros(3), feature_std=torch.ones(3),
        graph_fingerprint="graph", feature_fingerprint="features", split_fingerprint="split",
    )
    checkpoint = {
        **{key: value for key, value in vars(data).items()
           if key not in {"features", "train_edge_index", "val_edge_index"}},
        "best_global_val_ap": 0.9, "training_config": {"split_seed": 0},
        "git_commit": "training-commit", "model_type": "global_s2gae", "best_epoch": 4,
        "epoch": 4, "format_version": 3, "protocol": ranking.protocol_metadata(),
        "val_metrics": {"ap": 0.9}, "model_config": {"hidden_dim": 4},
    }
    calls = []
    models = []
    bank = torch.tensor([[[0] * 500], [[3] * 500]])
    monkeypatch.setattr(ranking, "load_sweep", lambda path: ({}, configurations))

    def select(selection_args):
        assert selection_args.checkpoint is None
        calls.append("select_all")
        return args.runs_root / "run_0" / "best_model_state_dict.pt", checkpoint, [
            {"run_name": config["name"], "status": "complete", "best_global_val_ap": 0.9,
             "validation_rank": index + 1}
            for index, config in enumerate(configurations)
        ]

    def load_data(networks, embeddings, **kwargs):
        assert calls == ["select_all"]
        assert kwargs == {"seed": 0, "verbose": False}
        calls.append("load_data")
        return data

    def build_bank(observed_data, **kwargs):
        assert observed_data is data
        assert kwargs == {"split": "val", "max_k": 500, "seed": 0}
        calls.append("build_bank")
        return bank

    class FrozenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def encode(self, features, edges):
            assert not torch.is_grad_enabled() and not self.training
            assert torch.equal(features, data.features)
            assert torch.equal(edges, data.train_edge_index)
            calls.append("encode_train_only")
            return features, [features]

    def build_model(observed_checkpoint, device):
        assert observed_checkpoint["model_type"] == "global_s2gae" and device.type == "cpu"
        model = FrozenModel()
        model.expected_ap = observed_checkpoint["val_metrics"]["ap"]
        models.append(model)
        calls.append("build_model")
        return model

    def score(model, observed_data, **kwargs):
        assert observed_data is data and model in models
        assert kwargs["negative_bank"] is bank
        assert kwargs["split"] == "val" and tuple(kwargs["k_values"]) == (1, 10, 50, 100, 500)
        assert not torch.is_grad_enabled() and not model.training
        calls.append("score_val_only")
        return {k: {"ap": model.expected_ap if k == 1 else 0.6, "f1": 0.8, "n_pos": 1, "n_neg": k}
                for k in kwargs["k_values"]}

    monkeypatch.setattr(ranking, "select_checkpoint", select)
    monkeypatch.setattr(ranking, "load_global_ppi_data", load_data)
    monkeypatch.setattr(ranking, "build_global_negative_bank", build_bank)
    monkeypatch.setattr(ranking, "load_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(ranking, "_validate_checkpoint", lambda path, ckpt: None)
    monkeypatch.setattr(ranking, "build_model", build_model)
    monkeypatch.setattr(ranking, "evaluate_global_edges", score)
    return args, data, checkpoint, calls, models


def test_loads_data_and_bank_once_and_scores_frozen_validation_only(fake_sweep):
    args, data, checkpoint, calls, models = fake_sweep
    summary = ranking.run(args)
    assert calls == ["select_all", "load_data", "build_bank"] + [
        "build_model", "encode_train_only", "score_val_only"] * 2
    assert len(models) == 2
    assert summary["n_scored_runs"] == 2 and summary["n_expected_runs"] == 54
    assert [row["run_name"] for row in summary["runs"]] == ["run_0", "run_27"]
    assert all(row["model_state_unchanged"] and row["baseline_ap_abs_error"] == 0
               and row["exploratory_ap500_rank"] is None for row in summary["runs"])
    assert summary["original_selection"]["unchanged"]
    assert summary["protocol"]["split"] == "validation_only"
    assert "not a search for high-k-optimal" in summary["interpretation"]
    assert len((args.output_dir / "metrics.csv").read_text().splitlines()) == 11
    assert (args.output_dir / "summary.json").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        ranking.run(args)


def test_baseline_reproduction_failure_does_not_publish_results(fake_sweep, monkeypatch):
    args, _, _, _, _ = fake_sweep
    monkeypatch.setattr(ranking, "evaluate_global_edges", lambda *a, **kw: {1: {"ap": 0.8}})
    with pytest.raises(ValueError, match="does not reproduce checkpoint"):
        ranking.run(args)
    assert not args.output_dir.exists()


def test_incomplete_sweep_is_rejected_before_loading_data(fake_sweep, monkeypatch):
    args, _, checkpoint, calls, _ = fake_sweep
    monkeypatch.setattr(ranking, "select_checkpoint", lambda _: (
        Path("run_0/best_model_state_dict.pt"), checkpoint, []))
    with pytest.raises(ValueError, match="All 54"):
        ranking.run(args)
    assert calls == [] and not args.output_dir.exists()


@pytest.mark.parametrize("snapshot_kind", ["best", "latest"])
def test_reference_best_or_latest_reuses_data_and_is_not_ranked(fake_sweep, monkeypatch, snapshot_kind):
    args, _, checkpoint, calls, models = fake_sweep
    reference = dict(checkpoint)
    if snapshot_kind == "latest":
        reference.update(
            epoch=9, val_metrics={"ap": 0.85}, best_global_val_ap=0.95,
            optimizer_state_dict={"state": {}, "param_groups": []},
            rng_state={"numpy": np.random.RandomState(0).get_state()},
        )
    args.reference_checkpoint = args.output_dir.parent / "reference.pt"
    torch.save(reference, args.reference_checkpoint)
    real_load, loads = torch.load, []

    def load_trusted(path, **kwargs):
        loads.append((path, kwargs))
        return real_load(path, **kwargs)

    monkeypatch.setattr(ranking.torch, "load", load_trusted)
    summary = ranking.run(args)
    result = summary["reference"]
    assert calls == ["select_all", "load_data", "build_bank"] + [
        "build_model", "encode_train_only", "score_val_only"] * 3
    assert len(loads) == 1 and loads[0][1] == {"map_location": "cpu", "weights_only": False}
    assert len(models) == 3 and summary["n_scored_runs"] == 2
    assert result["snapshot_kind"] == snapshot_kind
    assert result["actual_update"] == (10 if snapshot_kind == "latest" else 5)
    assert result["best_update"] == 5
    assert result["validation_ap_1"] == reference["val_metrics"]["ap"]
    assert result["historical_best_validation_ap"] == reference["best_global_val_ap"]
    assert result["baseline_ap_abs_error"] == 0
    assert not result["included_in_primary_ranking"]
    assert result["checkpoint_sha256"] == ranking.file_hash(args.reference_checkpoint)
    assert result["model_state_unchanged"]
    assert len((args.output_dir / "reference_metrics.csv").read_text().splitlines()) == 6
    assert len((args.output_dir / "metrics.csv").read_text().splitlines()) == 11
    assert [row["run_name"] for row in summary["runs"]] == ["run_0", "run_27"]
    assert summary["original_selection"]["unchanged"]


@pytest.mark.parametrize("corruption", ["protocol", "graph", "split_seed", "missing_rng", "invalid_update"])
def test_invalid_reference_checkpoint_is_rejected(fake_sweep, corruption):
    args, data, checkpoint, _, _ = fake_sweep
    reference = dict(checkpoint)
    if corruption == "protocol":
        reference["protocol"] = {"split": "test"}
    elif corruption == "graph":
        reference["graph_fingerprint"] = "other-graph"
    elif corruption == "split_seed":
        reference["training_config"] = {"split_seed": 10}
    elif corruption == "missing_rng":
        reference["optimizer_state_dict"] = {"state": {}}
    else:
        reference["epoch"] = 0
    path = args.output_dir.parent / "bad_reference.pt"
    torch.save(reference, path)
    with pytest.raises(ValueError):
        ranking.load_reference_checkpoint(path, data, 0)
