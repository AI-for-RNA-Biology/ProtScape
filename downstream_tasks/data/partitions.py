"""Shared protein cohorts and partitions for downstream benchmarks."""

from pathlib import Path

import numpy as np

from .task_loaders import get_task_loader
from ..training.cv_utils import SplitPlan, build_cv_splits


def partition_path(task, label_csv, dataset_mode="bulk"):
    root = Path(label_csv).parent
    suffix = "" if dataset_mode == "bulk" else f"_{dataset_mode}"
    if task == "corum":
        return root / f"split_indices{suffix}.npz"
    return root / "splits" / f"{task}{suffix}.npz"


def load_task_partition(task, label_csv, n_splits=6, dataset_mode="bulk"):
    """Read the dataset cohort in its saved order and validate labels and folds."""
    path = partition_path(task, label_csv, dataset_mode)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset partitions: {path}. "
            "Run python -m downstream_tasks.prepare_splits before training."
        )
    genes, labels, classes = get_task_loader(task, label_csv).load()
    genes = [gene.upper() for gene in genes]
    with np.load(path, allow_pickle=False) as saved:
        cohort = saved["genes"].astype(str).tolist()
        if len(set(genes)) != len(genes) or len(set(cohort)) != len(cohort):
            raise ValueError(f"Duplicate protein identifiers: {path}")
        cohort_set = set(cohort)
        keep = [i for i, gene in enumerate(genes) if gene in cohort_set]
        if [genes[i] for i in keep] != cohort:
            raise ValueError(f"Cohort/order differs from the dataset partition: {path}")
        if saved["class_names"].astype(str).tolist() != list(classes):
            raise ValueError(f"Class order differs from the dataset partition: {path}")
        labels = labels[keep]
        if not np.array_equal(saved["labels"], labels):
            raise ValueError(f"Labels differ from the dataset partition: {path}")
        folds = [saved[f"fold_{i}"].astype(np.int64) for i in range(n_splits)]
        if not np.array_equal(np.sort(np.concatenate(folds)), np.arange(len(cohort))):
            raise ValueError(f"Folds must partition the complete cohort: {path}")
    plan = SplitPlan(folds, 0, folds[1:], None, len(folds) - 1)
    return cohort, labels, classes, plan


def build_shared_split(genes, labels, embedding_loader, seed, n_splits,
                       gene_universe=None):
    """Intersect labels and embeddings, then stratify by labels and contexts."""
    genes = [gene.upper() for gene in genes]
    means, cells = embedding_loader.load_hc_mean(set(genes))
    shared = set(genes) & set(embedding_loader.load_esm()) & set(means)
    if gene_universe is not None:
        shared &= {gene.upper() for gene in gene_universe}
    keep = [i for i, gene in enumerate(genes) if gene in shared]
    cohort = [genes[i] for i in keep]
    labels = labels[keep]
    plan = build_cv_splits(
        labels, cell_ids_per_bag=[cells[gene] for gene in cohort],
        use_context_split=True, k_label=None, n_splits=n_splits, seed=seed,
    )
    return cohort, labels, plan


def load_training_partition(task, label_csv, loader, seed=42, n_splits=6,
                            dataset_mode="bulk"):
    genes, labels, classes, plan = load_task_partition(
        task, label_csv, n_splits, dataset_mode
    )
    # Compute label clusters for minibatch sampling; keep the dataset folds.
    cohort, _, computed = build_shared_split(genes, labels, loader, seed, n_splits)
    if cohort != genes:
        raise ValueError("Embeddings do not cover the complete dataset cohort.")
    plan.label_clusters = computed.label_clusters
    return genes, labels, classes, plan
