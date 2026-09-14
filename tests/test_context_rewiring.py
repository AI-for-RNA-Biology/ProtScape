import numpy as np
import pandas as pd
import pytest

from pretraining.summarize_context_rewiring import analyze


def panel():
    return pd.DataFrame({"pair_key": [1] * 5 + [2] * 5,
                         "cell_id": list(range(5)) * 2, "cell": list("abcde") * 2,
                         "source_protein": ["A"] * 10, "target_protein": ["B"] * 10,
                         "score_global_context_free": [0.7] * 10,
                         "score_cell_context_free": [0.1, 0.2, 0.3, 0.4, 0.5] * 2,
                         "score_protscape": [0.2, 0.3, 0.4, 0.5, 0.6] + [0.6, 0.5, 0.4, 0.3, 0.2]})


def test_within_pair_variation_and_context_ranking():
    summary, pairs, cells, agreement, counts = analyze(panel())
    assert counts["n_pairs"] == 2 and counts["n_contexts"] == 5
    assert summary.iloc[0].mean_pair_sd_pp == 0
    np.testing.assert_allclose(pairs.context_rank_spearman, [1, -1])
    assert agreement["median_within_pair_spearman"] == 0
    assert cells.rms_context_deviation_global_context_free_pp.max() < 1e-12


def test_constant_context_scores_are_not_counted_as_agreement():
    data = panel()
    data["score_cell_context_free"] = 0.5
    _, _, _, agreement, _ = analyze(data)
    assert agreement["n_defined_pairs"] == 0
    assert agreement["n_constant_score_pairs"] == 2


def test_duplicate_pair_contexts_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(pd.concat([panel(), panel().iloc[[0]]]))


def test_pairwise_intersection_counts_contexts_not_duplicate_edges():
    import ast
    from pathlib import Path
    # Test the pure sampler without loading GPU/checkpoint libraries.
    tree = ast.parse((Path(__file__).parents[1] / "pretraining/evaluate_context_rewiring.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "shared_edge_panel")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "sampler", "exec"), namespace)
    keys, counts = namespace["shared_edge_panel"]([[1, 1, 2], [2, 3], [3, 4]])
    np.testing.assert_array_equal(keys, [2, 3])
    np.testing.assert_array_equal(counts, [2, 2])


def test_unobserved_panel_is_fixed_unique_and_not_in_reference():
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).parents[1] / "pretraining/evaluate_context_rewiring.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample_unobserved_pairs")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "sampler", "exec"), namespace)
    presence = np.ones((8, 4), dtype=bool)
    presence[7, 1:] = False
    known = {1, 2, 3}
    sampler = namespace["sample_unobserved_pairs"]
    keys = sampler(8, known, presence, 10)
    np.testing.assert_array_equal(keys, sampler(8, known, presence, 10))
    assert len(set(keys)) == 10 and not known.intersection(keys)
    assert np.all(keys // 8 < keys % 8)
    assert np.all(keys % 8 != 7)


def load_pure_function(name):
    """Keep numerical tests independent of model/checkpoint imports."""
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).parents[1] / "pretraining/evaluate_context_rewiring.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), namespace)
    return namespace[name]


def test_exhaustive_unobserved_equals_brute_force():
    enumerate_pairs = load_pure_function("all_unobserved_pairs")
    presence = np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool)
    n, known = len(presence), np.array([1, 4])
    expected = [(i * n + j, int((presence[i] & presence[j]).sum()))
                for i in range(n) for j in range(i + 1, n)
                if i * n + j not in known and (presence[i] & presence[j]).sum() >= 2]
    for block_size in [1, 2, 100]:
        keys, counts = enumerate_pairs(presence, known, block_size)
        assert list(zip(keys, counts)) == expected


def test_exhaustive_statistics_equal_original_analyzer_with_missing_contexts():
    statistics = load_pure_function("context_statistics")
    data = panel()
    # A sixth context for only one pair exercises missing-context rank handling.
    extra = data.iloc[[0]].copy()
    extra["cell_id"], extra["cell"] = 5, "f"
    data = pd.concat([data, extra], ignore_index=True)
    values = np.full((2, 6, 2), np.nan)
    for row in data.itertuples():
        values[row.pair_key - 1, row.cell_id] = [row.score_cell_context_free, row.score_protscape]
    variance, rho, sums, counts = statistics(values)
    _, pairs, cells, _, _ = analyze(data)
    np.testing.assert_allclose(variance[:, 0], (pairs.sd_cell_context_free_pp / 100) ** 2)
    np.testing.assert_allclose(variance[:, 1], (pairs.sd_protscape_pp / 100) ** 2)
    np.testing.assert_allclose(rho, pairs.context_rank_spearman)
    np.testing.assert_array_equal(counts, pairs.n_contexts)
    np.testing.assert_allclose(100 * np.sqrt(sums[:, 0] / cells.n_pairs), cells.rms_context_deviation_cell_context_free_pp)


def test_exhaustive_correlation_excludes_constant_and_low_coverage_pairs():
    statistics = load_pure_function("context_statistics")
    values = np.full((3, 6, 2), np.nan)
    values[0, :, 0], values[0, :, 1] = .5, np.arange(6) / 6
    values[1, :2] = [[.1, .2], [.2, .1]]
    values[2, :, 0] = [.1, .1, .2, .3, .4, .4]
    values[2, :, 1] = [.4, .4, .3, .2, .1, .1]
    _, rho, _, _ = statistics(values)
    assert np.isnan(rho[:2]).all()
    assert rho[2] == pytest.approx(-1)


def test_exhaustive_disk_pipeline_matches_dataframe_analysis(tmp_path):
    import json
    from scipy.stats import spearmanr
    statistics = load_pure_function("context_statistics")
    summarize = load_pure_function("summarize_exhaustive")
    finish = load_pure_function("finish_exhaustive")
    summarize.__globals__.update(json=json, pd=pd, context_statistics=statistics)
    finish.__globals__.update(json=json, pd=pd)
    rng = np.random.default_rng(7)
    values = rng.random((8, 6, 2)).astype(np.float32)
    values[0, 4:] = np.nan
    manifest = {"bank_sha256": "test", "n_shared_pairs": 8, "n_contexts": 6}
    counts = np.isfinite(values[:, :, 0]).sum(axis=1)
    manifest["n_pair_context_observations"] = int(counts.sum())
    (tmp_path / "BANK_COMPLETE.json").write_text(json.dumps(manifest))
    np.save(tmp_path / "pair_keys.npy", np.arange(8))
    np.save(tmp_path / "context_counts.npy", counts)
    rows = []
    for cell in range(6):
        indices = np.flatnonzero(np.isfinite(values[:, cell, 0]))
        np.save(tmp_path / f"cell_{cell}_pair_indices.npy", indices.astype(np.uint32))
        np.save(tmp_path / f"cell_{cell}_scores.npy", values[indices, cell])
        metadata = {"bank_sha256": "test", "n_pairs": len(indices), "index": cell,
                    "cell_id": cell, "cell": str(cell)}
        (tmp_path / f"cell_{cell}.json").write_text(json.dumps(metadata))
        for pair in indices:
            rows.append({"pair_key": pair, "cell_id": cell, "cell": str(cell),
                         "source_protein": str(pair), "target_protein": "X",
                         "score_global_context_free": .5,
                         "score_cell_context_free": float(values[pair, cell, 0]),
                         "score_protscape": float(values[pair, cell, 1])})
    for shard in range(4):
        summarize(tmp_path, shard)
    finish(tmp_path)
    result = pd.read_csv(tmp_path / "exhaustive_summary.csv").iloc[0]
    _, pairs, cells, _, _ = analyze(pd.DataFrame(rows), min_contexts=2)
    assert result.cf_mean_variance == pytest.approx(((pairs.sd_cell_context_free_pp / 100) ** 2).mean())
    assert result.protscape_mean_variance == pytest.approx(((pairs.sd_protscape_pp / 100) ** 2).mean())
    assert result.median_within_pair_spearman == pytest.approx(pairs.loc[pairs.n_contexts >= 5, "context_rank_spearman"].median())
    expected = [spearmanr(frame.score_cell_context_free, frame.score_protscape).statistic
                for _, frame in pd.DataFrame(rows).groupby("cell_id")]
    assert result.median_within_cell_spearman == pytest.approx(np.median(expected))
    actual_cells = pd.read_csv(tmp_path / "per_cell.csv")
    np.testing.assert_allclose(actual_cells.rms_context_deviation_protscape_pp, cells.rms_context_deviation_protscape_pp)
