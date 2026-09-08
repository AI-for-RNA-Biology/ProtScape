import numpy as np
import scipy.sparse as sp
import torch
import json
from types import SimpleNamespace

from pretraining.investigate_global_topology import (
    adjacency, pair_keys, pair_fold, rewire_boundary, personalized_pagerank,
    topology_tables, negative_targets, probe_features, PROBES, analyze, complementarity,
)


def test_undirected_keys_and_grouping():
    edges = np.array([[1, 4, 6], [4, 1, 8]])
    assert np.array_equal(pair_keys(edges, 10), pair_keys(edges[::-1], 10))
    assert pair_fold(pair_keys(edges, 10))[0] == pair_fold(pair_keys(edges, 10))[1]
    assert set(pair_fold(np.arange(10000))) == {0, 1, 2}


def test_rewiring_changes_only_boundary_and_preserves_degrees():
    inside = np.arange(20) < 10
    edges = np.array([(i, (i + j) % 10 + 10) for i in range(10) for j in (0, 1, 2)] + [(0, 1), (10, 11)]).T
    edges = np.concatenate([edges, edges[::-1]], 1)
    before = adjacency(edges, 20)
    forbidden = {2 * 20 + 19}
    result, info = rewire_boundary(edges, inside, 20, 0, forbidden)
    after = adjacency(result, 20)
    assert np.array_equal(before.sum(0), after.sum(0))
    assert (before[:10, :10] != after[:10, :10]).nnz == 0
    assert (before[10:, 10:] != after[10:, 10:]).nnz == 0
    assert not (set(pair_keys(result, 20)) & forbidden)
    assert info["boundary_fraction_changed"] > .4
    assert result.shape[1] == after.nnz
    assert np.array_equal(result, rewire_boundary(edges, inside, 20, 0, forbidden)[0])


def test_ppr_matches_exact_solution_and_handles_isolated_nodes():
    matrix = adjacency(np.array([[0, 1, 1], [1, 2, 3]]), 5)
    sources = np.array([0, 2, 4])
    result, info = personalized_pagerank(matrix, sources, torch.device("cpu"))
    degree = np.asarray(matrix.sum(1)).ravel()
    transition = np.diag(np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)) @ matrix.toarray()
    for row, source in enumerate(sources):
        current = transition.copy()
        current[degree == 0, source] = 1
        exact = np.linalg.solve(np.eye(5) - .85 * current.T, .15 * np.eye(5)[:, source])
        np.testing.assert_allclose(result[row], exact, atol=2e-6)
    assert info["max_l1_iteration_residual"] <= 2e-6


def test_rewiring_preserves_duplicate_self_ppi_occurrences():
    edges = np.array([[0, 2, 0, 0, 1, 3], [2, 0, 0, 0, 3, 1]])
    inside = np.array([True, True, False, False])
    changed, _ = rewire_boundary(edges, inside, 4, 0)
    assert np.array_equal(np.bincount(edges[0], minlength=4), np.bincount(changed[0], minlength=4))
    assert (changed[0] == changed[1]).sum() == 2


def test_common_neighbor_and_resource_allocation_values():
    graph = adjacency(np.array([[0, 1, 1, 2], [1, 2, 3, 3]]), 4)
    tables = topology_tables(graph, np.array([0]), np.zeros((1, 4)))
    assert tables[1][0, 2] == 1
    np.testing.assert_allclose(tables[2][0, 2], 1 / 3)


def test_negative_banks_exclude_known_edges_and_preserve_pair_fold():
    size = 80
    mapping = np.arange(size)
    known = adjacency(np.array([[0, 0, 4], [1, 2, 7]]), size)
    values = np.arange(size, dtype=np.float32)
    fold = pair_fold(np.array([3]))[0]
    banks = negative_targets(0, mapping, known, fold, values, -values, values, -values, np.random.default_rng(0), k=10)
    assert set(banks) == {"random", "local_hard", "global_hard"}
    for name, bank in banks.items():
        assert len(bank) == 10
        assert not (set(bank) & {0, 1, 2})
        assert np.all(pair_fold(pair_keys(np.stack([np.zeros(10, int), bank]), size)) == fold)
        if name != "random":
            assert len(set(bank)) == 10


def test_probe_columns_do_not_confuse_local_and_global_gnn():
    artifact = {"local": np.ones((3, 2, 4, 5)), "global_features": np.ones((3, 2, 4, 5)) * 2,
                "scores": np.stack([np.zeros((3, 2, 4)) + i for i in range(5)])}
    design = probe_features(artifact)
    assert design.shape == (3, 2, 4, 12)
    assert np.all(design[..., 0] == 0) and np.all(design[..., 1] == 1)
    assert 1 not in PROBES["Global GNN + both structures"]
    assert 0 not in PROBES["Local GNN + both structures"]


def test_complete_crossfit_analysis_outputs(tmp_path):
    rng = np.random.default_rng(1)
    folds = np.repeat(np.arange(3), 2).astype(np.int8)
    possible = np.arange(100000)
    keys = np.stack([possible[pair_fold(possible) == fold][:1002].reshape(2, 501) for fold in range(3)]).reshape(6, 501)
    (tmp_path / "protocol.json").write_text(json.dumps({"selected": [1, 2]}))
    for cell in (1, 2):
        local = rng.random((3, 6, 501, 5), dtype=np.float32)
        np.savez(tmp_path / f"cell_{cell}.npz", local=local, global_features=local + .1,
                 scores=rng.normal(size=(5, 3, 6, 501)).astype(np.float32), folds=folds,
                 keys=np.broadcast_to(keys, (3, 6, 501)), info=np.array(json.dumps({"cell_id": cell})))
    analyze(SimpleNamespace(output_dir=tmp_path))
    import pandas as pd
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert len(summary) == 3 * 17 * 5
    assert np.all(summary["count"] == 2)
    assert np.all((summary["mean"] > 0) & (summary["mean"] <= 1))
    assert (tmp_path / "topology_interventions.png").exists()
    complementarity(SimpleNamespace(output_dir=tmp_path))
    fusion = pd.read_csv(tmp_path / "complementarity_summary.csv")
    assert len(fusion) == 2 * 3 * 3 * 3
    matched = fusion[(fusion.mixture == "random") & (fusion.model == "Global GNN + global structure")]
    old = summary[summary.model == "Global GNN + global structure"]
    checked = matched.merge(old, on=["bank", "model", "k"])
    np.testing.assert_allclose(checked.auprc, checked["mean"], atol=1e-10)
