"""Shared context-specific PPI evaluation protocol."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


CONTEXTWISE_K_VALUES = (1, 10, 50, 100, 500)
CONTEXTWISE_NEGATIVE_BANK_SIZE = max(CONTEXTWISE_K_VALUES)


def binary_metrics(scores, labels) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    roc = 0.5 if np.unique(labels).size < 2 else roc_auc_score(labels, scores)
    return {
        "roc": float(roc),
        "ap": float(average_precision_score(labels, scores)),
        "acc": float(accuracy_score(labels, scores >= 0.5)),
        "f1": float(
            f1_score(
                labels,
                scores >= 0.5,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        ),
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
    }


def metrics_from_pos_neg(pos_scores, neg_scores) -> dict[str, float | int]:
    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate(
        [np.ones(pos_scores.size, dtype=np.int8), np.zeros(neg_scores.size, dtype=np.int8)]
    )
    return binary_metrics(scores, labels)


def sample_structured_negatives(
    pos_edges: torch.Tensor,
    all_positive_edges: torch.Tensor,
    num_nodes: int,
    max_k: int,
) -> torch.Tensor:
    """Sample the target-corruption bank used by contextual ProtScape."""
    device = pos_edges.device
    sources = pos_edges[0].detach().cpu().numpy().astype(np.int64, copy=False)
    positives = (
        all_positive_edges.detach().cpu().numpy().astype(np.int64, copy=False)
    )

    known_targets: dict[int, list[int]] = {}
    for left, right in zip(positives[0], positives[1]):
        known_targets.setdefault(int(left), []).append(int(right))
        known_targets.setdefault(int(right), []).append(int(left))

    source_rows: dict[int, list[int]] = {}
    for row, source in enumerate(sources):
        source_rows.setdefault(int(source), []).append(row)

    negative_targets = np.empty((sources.size, max_k), dtype=np.int64)
    for source, rows in source_rows.items():
        row_indices = np.asarray(rows, dtype=np.int64)
        forbidden = np.asarray(known_targets.get(source, ()), dtype=np.int64)
        needed = row_indices.size * max_k
        draws = []
        collected = 0
        while collected < needed:
            candidates = np.random.randint(
                0,
                num_nodes,
                size=max(4096, 2 * (needed - collected)),
            )
            keep = candidates != source
            if forbidden.size:
                keep &= ~np.isin(candidates, forbidden)
            candidates = candidates[keep]
            take = min(candidates.size, needed - collected)
            draws.append(candidates[:take])
            collected += take
        negative_targets[row_indices] = np.concatenate(draws).reshape(
            row_indices.size, max_k
        )

    negative_edges = np.stack(
        [np.repeat(sources, max_k), negative_targets.reshape(-1)], axis=0
    )
    return torch.from_numpy(negative_edges).to(device=device, dtype=torch.long)


def aggregate_context_metrics(
    metrics_by_k: dict[int, list[dict[str, float | int]]],
) -> dict[int, dict[str, float | int]]:
    """Unweighted macro-average across Cell-PPIs, matching the paper evaluator."""
    output: dict[int, dict[str, float | int]] = {}
    for k, rows in metrics_by_k.items():
        if not rows:
            raise ValueError(f"No context metrics were collected for k={k}.")
        output[k] = {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in ["roc", "ap", "acc", "f1"]
        }
        output[k]["n_pos"] = int(sum(int(row["n_pos"]) for row in rows))
        output[k]["n_neg"] = int(sum(int(row["n_neg"]) for row in rows))
        output[k]["n_cells"] = len(rows)
    return output
