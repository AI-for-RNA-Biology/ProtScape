import copy

import numpy as np
import pytest
import torch

import pretraining.global_s2gae as global_s2gae
import pretraining.refine_global_s2gae as refinement
from pretraining.evaluate_global_s2gae import _validate_checkpoint, load_checkpoint
from pretraining.global_s2gae import GlobalPPIData, GlobalS2GAE, evaluate_global_edges, protocol_metadata
from pretraining.train_global_s2gae import capture_rng_state, cpu_state_dict, set_seed


def make_parent(tmp_path):
    set_seed(0)
    unique = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 3],
                           [1, 2, 3, 4, 5, 6, 7, 0, 3, 5, 4, 6]])

    def symmetric(edges):
        return torch.cat([edges, edges.flip(0)], dim=1)

    data = GlobalPPIData(
        features=torch.randn(8, 3), feature_mean=torch.zeros(3), feature_std=torch.ones(3),
        protein_names=[str(i) for i in range(8)], all_edge_index=symmetric(unique),
        train_edge_index=symmetric(unique[:, :8]), train_val_edge_index=symmetric(unique[:, :10]),
        val_edge_index=unique[:, 8:10], test_edge_index=unique[:, 10:], source_context_count=1,
        graph_fingerprint="graph", feature_fingerprint="features", split_fingerprint="split",
        split_counts={"shared_train": 8, "shared_val": 2, "shared_test": 2},
    )
    model = GlobalS2GAE(3, hidden_dim=4, num_layers=2, dropout=0.2, decode_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(2):
        model.train()
        optimizer.zero_grad()
        loss, _, _ = global_s2gae.masked_reconstruction_step(model, data, device=torch.device("cpu"))
        loss.backward()
        optimizer.step()
    metrics = evaluate_global_edges(model, data, split="val", k_values=[1], device=torch.device("cpu"))[1]
    checkpoint = {
        "format_version": 3, "model_type": "global_s2gae", "experiment_role": "sweep",
        "model_config": model.model_config, "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_state_dict": cpu_state_dict(model), "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(), "epoch": 1, "best_epoch": 0,
        "best_global_val_ap": metrics["ap"] + 0.01, "val_metrics": metrics,
        "training_config": {"epochs": 5000, "lr": 0.01, "mask_ratio": 0.5, "mask_type": "dm",
                            "k_negatives": 1, "seed": 0, "split_seed": 0,
                            "early_stopping_patience": 200, "early_stopping_min_delta": 0.0005},
        "protocol": protocol_metadata(), "git_commit": "parent_commit",
    }
    for key in ("protein_names", "feature_mean", "feature_std", "graph_fingerprint",
                "feature_fingerprint", "split_fingerprint", "split_counts"):
        checkpoint[key] = getattr(data, key)
    path = tmp_path / "parent_latest.pt"
    torch.save(checkpoint, path)
    return path, data


def test_adam_and_rng_are_restored_before_only_lr_changes(tmp_path):
    path, _ = make_parent(tmp_path)
    parent = torch.load(path, weights_only=False)
    model, optimizer = refinement.restore_training_state(parent, torch.device("cpu"), lr=0.001)
    assert refinement.state_hash(model.state_dict()) == refinement.state_hash(parent["model_state_dict"])
    assert refinement.state_hash(capture_rng_state()) == refinement.state_hash(parent["rng_state"])
    expected = copy.deepcopy(parent["optimizer_state_dict"])
    for group in expected["param_groups"]:
        group["lr"] = 0.001
    assert refinement.state_hash(optimizer.state_dict()) == refinement.state_hash(expected)
    assert all(state["step"].item() == 2 for state in optimizer.state.values())


def test_fixed_budget_uses_only_validation_and_keeps_initial_latest_candidate(tmp_path, monkeypatch):
    path, data = make_parent(tmp_path)
    parent = torch.load(path, weights_only=False)
    parent_hash = refinement.file_hash(path)
    baseline = parent["val_metrics"]["ap"]
    calls, banks = [], []

    def score(model, observed_data, **kwargs):
        assert observed_data is data and kwargs["split"] == "val"
        assert not model.training and not torch.is_grad_enabled()
        calls.append(kwargs["k_values"])
        banks.append(kwargs["negative_bank"])
        ap = baseline if len(calls) == 1 or len(kwargs["k_values"]) > 1 else baseline - 0.01
        return {k: {"ap": ap, "n_pos": 2, "n_neg": 2 * k} for k in kwargs["k_values"]}

    monkeypatch.setattr(refinement, "evaluate_global_edges", score)
    output = tmp_path / "phase"
    summary = refinement.run_refinement(path, data, output, lr=0.001, additional_updates=2,
                                        device=torch.device("cpu"))
    assert calls == [[1], [1], [1], [1, 10, 50, 100, 500]]
    assert all(bank is banks[0] for bank in banks)
    assert summary["completed_additional_updates"] == 2 and summary["best_phase_step"] == 0
    assert summary["best_global_val_ap"] == baseline
    assert summary["parent_best_global_val_ap"] == parent["best_global_val_ap"]
    assert refinement.file_hash(path) == parent_hash
    latest = torch.load(output / "latest_checkpoint.pt", weights_only=False)
    best_path = output / "best_model_state_dict.pt"
    best = load_checkpoint(best_path)
    _validate_checkpoint(best_path, best)
    assert "optimizer_state_dict" not in best and "rng_state" not in best
    assert "optimizer_state_dict" in latest and "rng_state" in latest
    assert latest["epoch"] == parent["epoch"] + 2
    assert all(state["step"].item() == 4 for state in latest["optimizer_state_dict"]["state"].values())
    assert best["epoch"] == parent["epoch"] and best["experiment_role"] == "sensitivity"
    assert refinement.state_hash(best["model_state_dict"]) == refinement.state_hash(parent["model_state_dict"])
    with pytest.raises(FileExistsError):
        refinement.run_refinement(path, data, output, lr=0.001, additional_updates=2, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="changed refinement configuration"):
        refinement.run_refinement(path, data, output, lr=0.003, additional_updates=2,
                                  device=torch.device("cpu"), resume=True)


def test_best_checkpoint_export_is_weights_only_safe_without_mutating_latest_payload(tmp_path):
    path, _ = make_parent(tmp_path)
    payload = torch.load(path, weights_only=False)
    payload["best_epoch"] = payload["epoch"]
    exported_path = tmp_path / "portable_best.pt"
    refinement.save_best_checkpoint(exported_path, payload)
    best = torch.load(exported_path, weights_only=True)
    _validate_checkpoint(exported_path, best)
    assert "optimizer_state_dict" not in best and "rng_state" not in best
    assert "optimizer_state_dict" in payload and "rng_state" in payload
    for key in ("model_state_dict", "model_config", "training_config", "protocol"):
        assert refinement.state_hash(best[key]) == refinement.state_hash(payload[key])


def test_lr_branches_use_same_first_mask_and_negative_samples(tmp_path, monkeypatch):
    path, data = make_parent(tmp_path)
    original_mask = global_s2gae.edge_mask_per_graph
    original_negatives = global_s2gae.structured_negative_sampling_k
    masks, negatives = [], []

    def record_mask(*args, **kwargs):
        result = original_mask(*args, **kwargs)
        masks.append(tuple(tensor.clone() for tensor in result))
        return result

    def record_negatives(*args, **kwargs):
        result = original_negatives(*args, **kwargs)
        negatives.append(result.clone())
        return result

    monkeypatch.setattr(global_s2gae, "edge_mask_per_graph", record_mask)
    monkeypatch.setattr(global_s2gae, "structured_negative_sampling_k", record_negatives)
    for lr in (0.01, 0.003, 0.001):
        refinement.run_refinement(path, data, tmp_path / str(lr), lr=lr, additional_updates=1,
                                  device=torch.device("cpu"))
    assert len(masks) == len(negatives) == 3
    for masked in masks[1:]:
        assert all(torch.equal(first, current) for first, current in zip(masks[0], masked))
    assert all(torch.equal(negatives[0], current) for current in negatives[1:])


def test_interrupted_phase_resumes_exact_optimizer_rng_and_budget(tmp_path, monkeypatch):
    path, data = make_parent(tmp_path)
    reference_dir, resumed_dir = tmp_path / "reference", tmp_path / "resumed"
    refinement.run_refinement(path, data, reference_dir, lr=0.003, additional_updates=3,
                              device=torch.device("cpu"))
    original_step = refinement.masked_reconstruction_step
    count = 0

    def interrupt(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("simulated interruption")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(refinement, "masked_reconstruction_step", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        refinement.run_refinement(path, data, resumed_dir, lr=0.003, additional_updates=3,
                                  device=torch.device("cpu"))
    monkeypatch.setattr(refinement, "masked_reconstruction_step", original_step)
    refinement.run_refinement(path, data, resumed_dir, lr=0.003, additional_updates=3,
                              device=torch.device("cpu"), resume=True)
    uninterrupted = torch.load(reference_dir / "latest_checkpoint.pt", weights_only=False)
    resumed = torch.load(resumed_dir / "latest_checkpoint.pt", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "refinement_history"):
        assert refinement.state_hash(resumed[key]) == refinement.state_hash(uninterrupted[key])
    assert resumed["refinement_history"][-1]["phase_step"] == 3
