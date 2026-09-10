from types import SimpleNamespace

import numpy as np
import pytest

from downstream_tasks.data.context_controls import replace_protein_instances, use_released_partition


def test_mean_preserves_cells_bag_sizes_and_source():
    original = {"A": [np.array([1., 3., 10.]), np.array([5., 7., 20.])]}
    result = replace_protein_instances(original, 2, "mean")
    np.testing.assert_array_equal(result["A"], [[3., 5., 10.], [3., 5., 20.]])
    np.testing.assert_array_equal(original["A"], [[1., 3., 10.], [5., 7., 20.]])
    assert replace_protein_instances(original, 2, "contextual") is original


def test_global_replacement_preserves_cell_suffix_and_requires_coverage():
    bags = {"A": [np.array([1., 3., 10.]), np.array([5., 7., 20.])]}
    result = replace_protein_instances(bags, 2, "global", {"A": np.array([9.])})
    np.testing.assert_array_equal(result["A"], [[9., 10.], [9., 20.]])
    with pytest.raises(ValueError, match="cover cohort"):
        replace_protein_instances(bags, 2, "global", {})


def test_released_partition_keeps_exact_cohort_and_folds(tmp_path):
    path = tmp_path / "split.npz"
    labels = np.array([[1], [0], [1], [0]])
    np.savez(path, genes=["A", "B", "C", "D"], labels=labels, class_names=["x"],
             fold_0=[3, 1], fold_1=[0, 2])
    plan = SimpleNamespace(folds=[np.array([0, 1]), np.array([2, 3])])
    result = use_released_partition(path, ["A", "B", "C", "D"], labels, ["x"], plan)
    assert result.test_fold_idx == 0
    np.testing.assert_array_equal(result.folds[0], [3, 1])
    with pytest.raises(ValueError, match="cohort"):
        use_released_partition(path, ["B", "A", "C", "D"], labels, ["x"], plan)
    with pytest.raises(ValueError, match="Labels"):
        use_released_partition(path, ["A", "B", "C", "D"], 1 - labels, ["x"], plan)


def test_partition_rejects_overlapping_folds(tmp_path):
    path = tmp_path / "split.npz"
    np.savez(path, genes=["A", "B"], labels=[[1], [0]], class_names=["x"], fold_0=[0], fold_1=[0])
    with pytest.raises(ValueError, match="exactly once"):
        use_released_partition(path, ["A", "B"], np.array([[1], [0]]), ["x"], SimpleNamespace(folds=[[], []]))
