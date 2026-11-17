"""Keep fold 0 for testing and rotate validation over folds 1–5."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from ..utils.stratification import (
    build_context_stratified_folds,
    MultilabelStratifiedKFold,
    cluster_label_patterns,
)


@dataclass
class SplitPlan:
    """Container for CV split information."""
    folds: List[np.ndarray]  # List of fold index arrays (6 folds)
    test_fold_idx: int  # Which fold is the permanent test set (default: 0)
    cv_folds: List[np.ndarray]  # The 5 folds used for CV (folds 1-5)
    label_clusters: Optional[np.ndarray]  # Cluster assignments for stratified sampling
    n_cv_folds: int  # Number of CV folds (5)


def build_cv_splits(
    Y_ref: np.ndarray,
    *,
    cell_ids_per_bag: Optional[List[List[str]]] = None,
    use_context_split: bool = True,
    k_label: Optional[int] = None,
    n_splits: int = 6,
    seed: int = 42,
    test_fold_idx: int = 0,
) -> SplitPlan:
    """Build cross-validation splits with optional context stratification.
    
    Creates 6 stratified folds. Fold 0 is held out as permanent test set.
    Remaining 5 folds are used for 5-fold CV."""
    n_splits = int(n_splits)
    if len(Y_ref) < n_splits:
        raise ValueError(f"Need at least {n_splits} samples for CV.")

    if use_context_split:
        print("computing context-stratified folds...")
        if cell_ids_per_bag is None or len(cell_ids_per_bag) != len(Y_ref):
            raise ValueError("Context split requires cell_ids_per_bag with length matching Y_ref.")
        folds, label_clusters = build_context_stratified_folds(
            Y_ref,
            cell_ids_per_bag,
            seed=seed,
            k_label=k_label,
            n_splits=n_splits,
        )

        # Keep the context-stratified partition even if a rare label is absent
        # from a fold; changing stratifiers would change the held-out cohort.
    else:
        # Use standard multilabel stratified split
        splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for _, test_idx in splitter.split(np.zeros((len(Y_ref), 1)), Y_ref):
            folds.append(np.array(sorted(test_idx), dtype=int))
        k_label = int(max(1, min(k_label if k_label is not None else Y_ref.shape[1], len(Y_ref))))
        label_clusters = cluster_label_patterns(Y_ref, k_label, seed)

    # Extract CV folds (all except test fold)
    cv_folds = [folds[i] for i in range(n_splits) if i != test_fold_idx]

    return SplitPlan(
        folds=folds,
        test_fold_idx=test_fold_idx,
        cv_folds=cv_folds,
        label_clusters=label_clusters,
        n_cv_folds=len(cv_folds),
    )


def get_test_indices(split_plan: SplitPlan) -> np.ndarray:
    """Get the permanent held-out test set indices."""
    return np.asarray(split_plan.folds[split_plan.test_fold_idx], dtype=int)


def get_cv_train_val_indices(
    split_plan: SplitPlan,
    cv_fold: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get train/val indices for a specific CV fold."""
    n_cv = split_plan.n_cv_folds
    cv_fold = int(cv_fold) % n_cv
    
    val_idx = np.asarray(split_plan.cv_folds[cv_fold], dtype=int)
    
    train_parts = [
        np.asarray(split_plan.cv_folds[i], dtype=int)
        for i in range(n_cv)
        if i != cv_fold
    ]
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    print(f"computing train/val split: train {len(train_idx)} / val {len(val_idx)}")
    # All non-test indices, in fold order.
    trainval_idx = np.concatenate([np.asarray(f, dtype=int) for f in split_plan.cv_folds])
    
    return train_idx, val_idx, trainval_idx
