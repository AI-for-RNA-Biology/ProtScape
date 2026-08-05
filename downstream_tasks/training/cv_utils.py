# cv_utils.py
# Cross-validation utilities
#
# Strategy: 6-fold stratified split where:
#   - Fold 0 is permanently held out as the final test set
#   - Folds 1-5 are used for 5-fold CV (rotating val, rest are train)
#   - Model is trained 5 times, metrics averaged over the held out test set 
#   - Final test evaluation uses fold 0

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from ..utils.stratification import (
    build_context_stratified_folds,
    MultilabelStratifiedKFold,
    _compute_label_clusters,
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
    """
    Build cross-validation splits with optional context stratification.
    
    Creates 6 stratified folds. Fold 0 is held out as permanent test set.
    Remaining 5 folds are used for 5-fold CV.

    Args:
        Y_ref: Label matrix of shape (n_samples, n_labels)
        cell_ids_per_bag: Optional list of cell ID lists for context stratification
        use_context_split: Whether to use context-aware stratification
        k_label: Number of label clusters (None for auto)
        n_splits: Number of total folds (default 6)
        seed: Random seed
        test_fold_idx: Which fold to use as permanent test set (default 0)

    Returns:
        SplitPlan with folds and CV configuration
    """
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

        # Clustering multilabel patterns does not guarantee that every label is
        # represented in every fold. Prefer direct multilabel stratification
        # whenever that postcondition fails and the label has enough positives.
        label_totals = np.asarray(Y_ref).sum(axis=0)
        required_labels = label_totals >= n_splits
        fold_label_counts = np.stack(
            [np.asarray(Y_ref)[fold].sum(axis=0) for fold in folds]
        )
        missing_required = (fold_label_counts[:, required_labels] == 0).sum()
        if missing_required:
            print(
                "[WARN] Context-stratified folds omit "
                f"{int(missing_required)} required fold-label combinations; "
                "using direct multilabel stratification."
            )
            splitter = MultilabelStratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            folds = [
                np.asarray(test_idx, dtype=int)
                for _, test_idx in splitter.split(
                    np.zeros((len(Y_ref), 1)),
                    Y_ref,
                )
            ]
            fold_label_counts = np.stack(
                [np.asarray(Y_ref)[fold].sum(axis=0) for fold in folds]
            )
            if np.any(fold_label_counts[:, required_labels] == 0):
                raise RuntimeError(
                    "Multilabel stratification failed to represent every "
                    "eligible label in every fold."
                )
    else:
        # Use standard multilabel stratified split
        splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for _, test_idx in splitter.split(np.zeros((len(Y_ref), 1)), Y_ref):
            folds.append(np.array(sorted(test_idx), dtype=int))
        k_label = int(max(1, min(k_label if k_label is not None else Y_ref.shape[1], len(Y_ref))))
        label_clusters = _compute_label_clusters(Y_ref, k_label, seed)

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
    """
    Get train/val indices for a specific CV fold.
    
    Args:
        split_plan: The split plan with CV folds
        cv_fold: Which CV fold to use as validation (0 to n_cv_folds-1)
    
    Returns:
        Tuple of (train_idx, val_idx, trainval_idx)
        - train_idx: Indices for training (4 folds)
        - val_idx: Indices for validation (1 fold)
        - trainval_idx: All CV indices combined (for standardization stats)
    """
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
    # All CV folds combined (for computing standardization stats)
    trainval_idx = np.concatenate([np.asarray(f, dtype=int) for f in split_plan.cv_folds])
    
    return train_idx, val_idx, trainval_idx
