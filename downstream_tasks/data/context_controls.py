"""Downstream controls that preserve context membership and cell vectors."""

import numpy as np


def replace_protein_instances(bags, protein_dim, mode, global_vectors=None):
    """Replace only each instance's protein prefix, never its cell-vector suffix."""
    if mode == "contextual":
        return bags
    if mode not in {"mean", "global"}:
        raise ValueError(f"Unknown protein context control: {mode}")
    result = {}
    for gene, rows in bags.items():
        values = np.stack(rows).astype(np.float32, copy=False)
        if mode == "mean":
            protein = values[:, :protein_dim].mean(axis=0, dtype=np.float64).astype(np.float32)
        else:
            if global_vectors is None or gene not in global_vectors:
                raise ValueError(f"CF embeddings do not cover cohort protein {gene}")
            protein = global_vectors[gene]
        repeated = np.broadcast_to(protein, (len(rows), len(protein)))
        result[gene] = list(np.concatenate([repeated, values[:, protein_dim:]], axis=1))
    return result


def use_released_partition(path, genes, labels, classes, split_plan):
    """Require the released cohort/labels exactly; replace generated fold indices."""
    with np.load(path, allow_pickle=False) as saved:
        if saved["genes"].astype(str).tolist() != list(genes):
            raise ValueError("Embedding/label cohort differs from the released partition")
        if saved["class_names"].astype(str).tolist() != list(classes):
            raise ValueError("Class order differs from the released partition")
        if not np.array_equal(saved["labels"], labels):
            raise ValueError("Labels differ from the released partition")
        folds = [saved[f"fold_{i}"].astype(np.int64) for i in range(len(split_plan.folds))]
    if not np.array_equal(np.sort(np.concatenate(folds)), np.arange(len(genes))):
        raise ValueError("Released folds must partition the cohort exactly once")
    split_plan.folds = folds
    split_plan.test_fold_idx = 0
    split_plan.cv_folds = folds[1:]
    split_plan.n_cv_folds = len(folds) - 1
    split_plan.stratification_method = "released_fixed_partition"
    return split_plan
