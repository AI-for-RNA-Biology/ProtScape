import pandas as pd
import pytest

from exploration.analysis.pretraining_tables import (
    CONTEXT_FREE_MODEL_KEY,
    add_context_free_curve,
)


def test_add_context_free_curve_converts_context_macro_metrics_to_percent(tmp_path):
    contextual = pd.DataFrame(
        {
            "model_key": ["s2gae_att_k1_uni"],
            "k_negatives": [1],
            "pos_neg_ratio": ["1:1"],
            "score_percent": [89.9],
        }
    )
    metrics = pd.DataFrame(
        {
            "scope": ["cell_ppi_macro"] * 5,
            "k_negatives": [1, 10, 50, 100, 500],
            "ap": [0.9288, 0.6837, 0.4323, 0.3302, 0.1540],
            "f1": [0.8418, 0.6728, 0.5215, 0.4876, 0.4554],
        }
    )
    metrics_path = tmp_path / "global_test_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    combined = add_context_free_curve(contextual, metrics_path, "ap")
    added = combined[combined["model_key"] == CONTEXT_FREE_MODEL_KEY]

    assert added["k_negatives"].tolist() == [1, 10, 50, 100, 500]
    assert added["pos_neg_ratio"].tolist() == [
        "1:1",
        "1:10",
        "1:50",
        "1:100",
        "1:500",
    ]
    assert added["score_percent"].tolist() == pytest.approx(
        [92.88, 68.37, 43.23, 33.02, 15.40]
    )


def test_add_context_free_curve_requires_complete_1_to_k_bank(tmp_path):
    metrics_path = tmp_path / "incomplete.csv"
    pd.DataFrame(
        {
            "scope": ["global_unique_pairs"] * 2,
            "k_negatives": [1, 10],
            "ap": [0.9, 0.7],
        }
    ).to_csv(metrics_path, index=False)

    with pytest.raises(ValueError, match="expected"):
        add_context_free_curve(pd.DataFrame(), metrics_path, "ap")


def test_add_context_free_curve_rejects_global_unique_pair_scope(tmp_path):
    metrics_path = tmp_path / "wrong_scope.csv"
    pd.DataFrame(
        {
            "scope": ["global_unique_pairs"] * 5,
            "k_negatives": [1, 10, 50, 100, 500],
            "ap": [0.9, 0.7, 0.5, 0.4, 0.2],
        }
    ).to_csv(metrics_path, index=False)

    with pytest.raises(ValueError, match="cell_ppi_macro"):
        add_context_free_curve(pd.DataFrame(), metrics_path, "ap")
