# preprocessing.py
# Preprocessing utilities for embedding pooling and normalization

from typing import Dict, List, Optional, Tuple
import numpy as np


def _safe_std(arr: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    """Compute standard deviation with minimum value to avoid division by zero."""
    std = np.std(arr, axis=axis)
    return np.maximum(std, eps)


def mean_pool_contexts(
    gene_to_vecs: Dict[str, List[np.ndarray]],
    genes: List[str],
) -> np.ndarray:
    """
    Pool context vectors by computing mean for each gene.

    Args:
        gene_to_vecs: Dictionary mapping gene names to list of context vectors
        genes: List of gene names to process (defines order)

    Returns:
        Array of shape (n_genes, embed_dim) with mean-pooled vectors
    """
    pooled = []
    for gene in genes:
        vecs = gene_to_vecs.get(gene.upper(), [])
        if len(vecs) == 0:
            raise ValueError(f"Gene {gene} has no context vectors")
        stacked = np.stack(vecs, axis=0)
        pooled.append(np.mean(stacked, axis=0))
    return np.stack(pooled, axis=0).astype(np.float32)


def mean_std_pool_contexts(
    gene_to_vecs: Dict[str, List[np.ndarray]],
    genes: List[str],
) -> np.ndarray:
    """
    Pool context vectors by concatenating mean and standard deviation.

    Args:
        gene_to_vecs: Dictionary mapping gene names to list of context vectors
        genes: List of gene names to process (defines order)

    Returns:
        Array of shape (n_genes, embed_dim * 2) with mean+std pooled vectors
    """
    pooled = []
    for gene in genes:
        vecs = gene_to_vecs.get(gene.upper(), [])
        if len(vecs) == 0:
            raise ValueError(f"Gene {gene} has no context vectors")
        stacked = np.stack(vecs, axis=0)
        mean = np.mean(stacked, axis=0)
        std = _safe_std(stacked, axis=0)
        pooled.append(np.concatenate([mean, std], axis=0))
    return np.stack(pooled, axis=0).astype(np.float32)


def build_context_bags(
    gene_to_vecs: Dict[str, List[np.ndarray]],
    genes: List[str],
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Build context bags for ABMIL models.

    Args:
        gene_to_vecs: Dictionary mapping gene names to list of context vectors
        genes: List of gene names to process

    Returns:
        Tuple of (bags, keep_indices) where bags is list of numpy arrays
        and keep_indices are indices of genes that have context vectors
    """
    bags = []
    keep_idx = []
    for i, gene in enumerate(genes):
        vecs = gene_to_vecs.get(gene, [])
        if len(vecs) > 0:
            bags.append(np.stack(vecs, axis=0).astype(np.float32))
            keep_idx.append(i)
    return bags, keep_idx


def build_context_bags_with_cells(
    gene_to_vecs: Dict[str, List[np.ndarray]],
    gene_to_cells: Dict[str, List[str]],
    genes: List[str],
) -> Tuple[List[np.ndarray], List[int], List[List[str]]]:
    """
    Build context bags with cell ID tracking for stratification.

    Args:
        gene_to_vecs: Dictionary mapping gene names to list of context vectors
        gene_to_cells: Dictionary mapping gene names to list of cell IDs
        genes: List of gene names to process

    Returns:
        Tuple of (bags, keep_indices, cell_ids_per_bag)
    """
    bags = []
    keep_idx = []
    cell_ids = []
    for i, gene in enumerate(genes):
        vecs = gene_to_vecs.get(gene, [])
        cells = gene_to_cells.get(gene, [])
        if len(vecs) > 0:
            bags.append(np.stack(vecs, axis=0).astype(np.float32))
            keep_idx.append(i)
            cell_ids.append(cells if cells else ["unknown"])
    return bags, keep_idx, cell_ids


def zscore_normalize(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize features.

    Args:
        X: Array of shape (n_samples, n_features)
        mean: Optional precomputed mean (if None, computed from X)
        std: Optional precomputed std (if None, computed from X)
        eps: Minimum std value

    Returns:
        Tuple of (normalized_X, mean, std)
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.maximum(np.std(X, axis=0), eps)
    X_norm = (X - mean) / std
    return X_norm.astype(np.float32), mean, std


def zscore_normalize_bags(
    bags: List[np.ndarray],
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Z-score normalize context bags.

    Computes global statistics across all instances in all bags.

    Args:
        bags: List of numpy arrays, each of shape (n_instances, dim)
        mean: Optional precomputed mean
        std: Optional precomputed std
        eps: Minimum std value

    Returns:
        Tuple of (normalized_bags, mean, std)
    """
    if mean is None or std is None:
        all_vecs = np.concatenate(bags, axis=0)
        if mean is None:
            mean = np.mean(all_vecs, axis=0)
        if std is None:
            std = np.maximum(np.std(all_vecs, axis=0), eps)

    normalized = [(bag - mean) / std for bag in bags]
    return normalized, mean, std


def compute_class_weights(Y: np.ndarray) -> np.ndarray:
    """
    Compute class weights for imbalanced multi-label classification.

    Uses the inverse of class frequency as weight.

    Args:
        Y: Binary label matrix of shape (n_samples, n_classes)

    Returns:
        Weight array of shape (n_classes,)
    """
    pos_counts = Y.sum(axis=0)
    neg_counts = Y.shape[0] - pos_counts
    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1)
    weights = neg_counts / pos_counts
    return weights.astype(np.float32)


def filter_genes_with_embeddings(
    genes: List[str],
    Y: np.ndarray,
    *embedding_dicts: Dict[str, np.ndarray],
) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Filter genes to only those present in all embedding dictionaries.

    Args:
        genes: List of gene names
        Y: Label matrix
        *embedding_dicts: Variable number of embedding dictionaries to check

    Returns:
        Tuple of (filtered_genes, filtered_Y, keep_indices)
    """
    keep_idx = []
    for i, gene in enumerate(genes):
        gene_upper = gene.upper()
        if all(gene_upper in d for d in embedding_dicts):
            keep_idx.append(i)

    filtered_genes = [genes[i].upper() for i in keep_idx]
    filtered_Y = Y[keep_idx]
    return filtered_genes, filtered_Y, keep_idx
