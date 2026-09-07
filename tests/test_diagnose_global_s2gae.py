import copy
from types import SimpleNamespace

import pytest
import torch

import pretraining.diagnose_global_s2gae as diagnostic
from pretraining.global_s2gae import GlobalPPIData, GlobalS2GAE, evaluate_global_edges


def test_inference_controls_change_only_the_intended_input():
    features = torch.arange(12).reshape(4, 3).float()
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    original_features, original_edges = features.clone(), edges.clone()
    permutation = torch.tensor([3, 2, 0, 1])
    x, graph = diagnostic.diagnostic_inputs(features, edges, permutation, "train_topology")
    assert torch.equal(x, features) and torch.equal(graph, edges)
    x, graph = diagnostic.diagnostic_inputs(features, edges, permutation, "train_plus_self_loops")
    assert torch.equal(x, features)
    assert torch.equal(graph[:, :edges.size(1)], edges)
    assert torch.equal(graph[:, edges.size(1):], torch.arange(4).repeat(2, 1))
    x, graph = diagnostic.diagnostic_inputs(features, edges, permutation, "self_loops_only")
    assert torch.equal(x, features)
    assert torch.equal(graph, torch.arange(4).repeat(2, 1))
    x, graph = diagnostic.diagnostic_inputs(features, edges, permutation, "permuted_features")
    assert torch.equal(x, features[permutation]) and torch.equal(graph, edges)
    assert torch.equal(features, original_features)
    assert torch.equal(edges, original_edges)
    with pytest.raises(ValueError, match="Unknown diagnostic mode"):
        diagnostic.diagnostic_inputs(features, edges, permutation, "test")


def test_fingerprint_check_rejects_mismatched_data():
    data = SimpleNamespace(
        protein_names=["p1", "p2"], graph_fingerprint="graph", feature_fingerprint="feature",
        split_fingerprint="split", feature_mean=torch.zeros(3), feature_std=torch.ones(3),
    )
    checkpoint = vars(data).copy()
    diagnostic.validate_checkpoint_data(checkpoint, data)
    checkpoint["split_fingerprint"] = "other"
    with pytest.raises(ValueError, match="split"):
        diagnostic.validate_checkpoint_data(checkpoint, data)


def test_evaluation_is_frozen_validation_only_and_reuses_one_bank(monkeypatch):
    torch.manual_seed(0)
    model = GlobalS2GAE(3, hidden_dim=4, num_layers=2, dropout=0.4, decode_channels=4)
    data = SimpleNamespace(
        features=torch.randn(4, 3),
        train_edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        val_edge_index=torch.tensor([[0], [2]]),
    )
    initial_state = copy.deepcopy(model.state_dict())
    bank = torch.tensor([[[0] * 10], [[3] * 10]])
    calls = []

    def build_bank(observed_data, *, split, max_k, seed):
        assert observed_data is data and split == "val" and max_k == 10 and seed == 0
        calls.append("bank")
        return bank

    def evaluate(observed_model, observed_data, **kwargs):
        assert observed_model is model and observed_data is data
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in model.modules())
        assert kwargs["split"] == "val"
        assert kwargs["negative_bank"] is bank
        assert all(not value.requires_grad for value in kwargs["layer_embeddings"])
        calls.append("score")
        return {k: {"ap": 0.9, "f1": 0.8, "n_pos": 1, "n_neg": k} for k in kwargs["k_values"]}

    monkeypatch.setattr(diagnostic, "build_global_negative_bank", build_bank)
    monkeypatch.setattr(diagnostic, "evaluate_global_edges", evaluate)
    rows, audit = diagnostic.evaluate_modes(
        model, data, device=torch.device("cpu"), k_values=[10], split_seed=0,
        permutation_seed=7, expected_baseline_ap=0.9,
    )
    assert calls == ["bank"] + ["score"] * 4
    assert len(rows) == 8 and audit["k_values"] == [1, 10]
    assert audit["model_state_unchanged"]
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in initial_state.items())
    with pytest.raises(ValueError, match="does not reproduce checkpoint"):
        diagnostic.evaluate_modes(
            model, data, device=torch.device("cpu"), k_values=[10], split_seed=0,
            permutation_seed=7, expected_baseline_ap=0.8,
        )


def test_real_evaluator_reproduces_baseline_and_nested_negative_counts():
    torch.manual_seed(0)
    model = GlobalS2GAE(3, hidden_dim=4, num_layers=2, dropout=0.4, decode_channels=4).eval()
    unique_edges = torch.tensor([[0, 1, 0, 2, 3], [1, 2, 2, 3, 4]])

    def symmetric(edges):
        return torch.cat([edges, edges.flip(0)], dim=1)

    data = GlobalPPIData(
        features=torch.randn(6, 3), feature_mean=torch.zeros(3), feature_std=torch.ones(3),
        protein_names=[str(i) for i in range(6)], all_edge_index=symmetric(unique_edges),
        train_edge_index=symmetric(unique_edges[:, :2]),
        train_val_edge_index=symmetric(unique_edges[:, :4]),
        val_edge_index=unique_edges[:, 2:4], test_edge_index=unique_edges[:, 4:],
        source_context_count=1, graph_fingerprint="graph", feature_fingerprint="features",
        split_fingerprint="split", split_counts={"shared_val": 2},
    )
    baseline = evaluate_global_edges(
        model, data, split="val", k_values=[1], device=torch.device("cpu"), seed=0
    )[1]["ap"]
    rows, audit = diagnostic.evaluate_modes(
        model, data, device=torch.device("cpu"), k_values=[1, 10], split_seed=0,
        permutation_seed=0, expected_baseline_ap=baseline,
    )
    assert audit["baseline_ap"] == baseline
    assert len(rows) == 8
    assert all(row["n_pos"] == 2 and row["n_neg"] == 2 * row["k"] for row in rows)
