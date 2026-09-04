import numpy as np
import torch
from torch_geometric.data import Data

from pretraining.compare_inference_topologies import (
    INFERENCE_ORDER,
    SPLIT_ORDER,
    canonical_edges,
    split_masks,
    top_fraction_jaccard,
)


def test_canonical_edges_preserve_occurrences_and_remove_direction():
    edges = torch.tensor([[3, 1, 2], [1, 3, 4]])
    assert torch.equal(
        canonical_edges(edges),
        torch.tensor([[1, 1, 2], [3, 3, 4]]),
    )


def test_split_masks_keep_targets_out_of_validation_and_test_topologies():
    graph = Data(
        train_mask=torch.tensor([True, False, False]),
        val_mask=torch.tensor([False, True, False]),
        test_mask=torch.tensor([False, False, True]),
    )
    train_message, train_target = split_masks(graph, "train")
    val_message, val_target = split_masks(graph, "validation")
    test_message, test_target = split_masks(graph, "test")

    assert torch.equal(train_message, train_target)
    assert torch.equal(val_message, graph.train_mask)
    assert torch.equal(val_target, graph.val_mask)
    assert torch.equal(test_message, graph.train_mask | graph.val_mask)
    assert torch.equal(test_target, graph.test_mask)


def test_top_fraction_jaccard_compares_ranked_edges():
    left = np.arange(10, dtype=float)
    assert top_fraction_jaccard(left, left, fraction=0.2) == 1.0
    assert top_fraction_jaccard(left, -left, fraction=0.2) == 0.0


def test_declared_orders_cover_three_models_and_splits():
    assert INFERENCE_ORDER == (
        "global_context_free",
        "cell_context_free",
        "protscape",
    )
    assert SPLIT_ORDER == ("train", "validation", "test")
