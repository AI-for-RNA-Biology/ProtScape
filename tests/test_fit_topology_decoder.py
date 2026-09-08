import json
from types import SimpleNamespace

import numpy as np
import torch

from pretraining.fit_topology_decoder import (
    CorrectionHead, SETTINGS, VARIANTS, fold_roles, symmetric_structure, fit_fold,
)


def test_initial_prediction_is_exactly_cf_score():
    values = torch.randn(17, 11)
    for variant in VARIANTS:
        model = CorrectionHead(variant, np.zeros(11), np.ones(11))
        assert torch.equal(model(values), values[:, 0])


def test_global_head_cannot_use_local_topology():
    model = CorrectionHead("mlp_global", np.zeros(11), np.ones(11))
    torch.nn.init.normal_(model.net[-1].weight)
    values = torch.randn(17, 11)
    changed = values.clone()
    changed[:, 6:] += 1000
    assert torch.equal(model(values), model(changed))
    assert model.columns.tolist() == list(range(6))


def test_ppr_symmetrization_is_endpoint_invariant():
    forward = np.array([[[[1., 3., 1., .5, .1]]]], dtype=np.float32)
    reverse = forward.copy()
    reverse[..., 4] = .3
    a = symmetric_structure(forward, np.array([3.]))
    b = symmetric_structure(reverse, np.array([1.]))
    np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_allclose(a[..., 4], np.log1p(.2 * 10000), atol=1e-6)


def test_nested_roles_are_disjoint_and_cover_every_pair():
    folds = torch.tensor([0, 1, 2, 0, 1, 2])
    for outer in range(3):
        train, monitor, evaluation = fold_roles(folds, outer)
        assert not torch.any(train & monitor)
        assert not torch.any((train | monitor) & evaluation)
        assert torch.all(train | monitor | evaluation)
        assert torch.all(folds[evaluation] == outer)


def test_outer_values_cannot_change_fitted_state_or_stopping(tmp_path, monkeypatch):
    monkeypatch.setitem(SETTINGS, "max_updates", 4)
    monkeypatch.setitem(SETTINGS, "eval_every", 2)
    monkeypatch.setitem(SETTINGS, "patience_updates", 4)
    monkeypatch.setitem(SETTINGS, "batch_size", 32)
    torch.manual_seed(0)
    n_positive = 18
    dataset = {"features": torch.randn(3, n_positive, 501, 11),
               "context_scores": torch.randn(3, n_positive, 501),
               "folds": torch.arange(n_positive) % 3,
               "cells": torch.arange(n_positive) // 6,
               "keys": torch.zeros(3, n_positive, 501, dtype=torch.long)}
    for label in ("original", "altered"):
        current = {key: tensor.clone() for key, tensor in dataset.items()}
        if label == "altered":
            current["features"][:, current["folds"] == 0] += 100
            current["context_scores"][:, current["folds"] == 0] -= 100
        args = SimpleNamespace(output_dir=tmp_path / label, variant="mlp_context_distilled", device="cpu")
        fit_fold(args, current, 0)
    left = tmp_path / "original/mlp_context_distilled/fold_0"
    right = tmp_path / "altered/mlp_context_distilled/fold_0"
    a = torch.load(left / "head.pt", weights_only=True)
    b = torch.load(right / "head.pt", weights_only=True)
    assert a["selected_updates"] == b["selected_updates"]
    assert all(torch.equal(value, b["state_dict"][key]) for key, value in a["state_dict"].items())
    assert json.loads((left / "completed.json").read_text())["outer_used_for_selection"] is False
