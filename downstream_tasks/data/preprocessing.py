"""Normalize features and select genes with the required embeddings."""

from typing import Dict, List, Optional, Tuple
import numpy as np


def zscore_normalize_bags(
    bags: List[np.ndarray],
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Normalize every context using the supplied or newly fitted feature statistics."""
    if mean is None or std is None:
        all_vecs = np.concatenate(bags, axis=0)
        if mean is None:
            mean = np.mean(all_vecs, axis=0)
        if std is None:
            std = np.maximum(np.std(all_vecs, axis=0), eps)

    normalized = [(bag - mean) / std for bag in bags]
    return normalized, mean, std


def compute_class_weights(Y: np.ndarray) -> np.ndarray:
    """Weight positive examples by the negative-to-positive label ratio."""
    pos_counts = Y.sum(axis=0)
    neg_counts = Y.shape[0] - pos_counts
    # A label can have no positives in a training fold.
    pos_counts = np.maximum(pos_counts, 1)
    weights = neg_counts / pos_counts
    return weights.astype(np.float32)


def filter_genes_with_embeddings(
    genes: List[str],
    Y: np.ndarray,
    *embedding_dicts: Dict[str, np.ndarray],
) -> Tuple[List[str], np.ndarray, List[int]]:
    """Filter genes to only those present in all embedding dictionaries."""
    keep_idx = []
    for i, gene in enumerate(genes):
        gene_upper = gene.upper()
        if all(gene_upper in d for d in embedding_dicts):
            keep_idx.append(i)

    filtered_genes = [genes[i].upper() for i in keep_idx]
    filtered_Y = Y[keep_idx]
    return filtered_genes, filtered_Y, keep_idx
