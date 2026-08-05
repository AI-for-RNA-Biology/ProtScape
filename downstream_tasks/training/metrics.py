# metrics.py
# Evaluation metrics for multi-label classification

from typing import Dict, List
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def auc_macro_ignore_empty(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Macro-averaged AUROC, ignoring classes with no positive examples.

    Args:
        y_true: True labels, shape (n_samples, n_classes)
        y_score: Predicted scores, shape (n_samples, n_classes)

    Returns:
        Macro-averaged AUROC score
    """
    aucs = []
    for j in range(y_true.shape[1]):
        if y_true[:, j].min() != y_true[:, j].max():
            try:
                aucs.append(roc_auc_score(y_true[:, j], y_score[:, j]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else float("nan")


def auprc_macro_ignore_empty(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Macro-averaged AUPRC (Average Precision), ignoring classes with no positive examples.

    Args:
        y_true: True labels, shape (n_samples, n_classes)
        y_score: Predicted scores, shape (n_samples, n_classes)

    Returns:
        Macro-averaged AUPRC score
    """
    pos = y_true.sum(axis=0)
    valid = pos > 0
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean([
        average_precision_score(y_true[:, j], y_score[:, j])
        for j in np.where(valid)[0]
    ]))


def f1_macro_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Macro-averaged F1 score with threshold.

    Args:
        y_true: True labels, shape (n_samples, n_classes)
        y_score: Predicted scores, shape (n_samples, n_classes)
        threshold: Classification threshold

    Returns:
        Macro-averaged F1 score
    """
    y_pred = (y_score >= threshold).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: True labels
        y_score: Predicted scores (probabilities)
        threshold: Classification threshold for F1

    Returns:
        Dictionary with auroc_macro, auprc_macro, f1_macro
    """
    return {
        "auroc_macro": auc_macro_ignore_empty(y_true, y_score),
        "auprc_macro": auprc_macro_ignore_empty(y_true, y_score),
        "f1_macro": f1_macro_threshold(y_true, y_score, threshold),
    }


def summarize_cv_metrics(
    fold_metrics: List[Dict[str, float]],
    split: str) -> Dict[str, float]:
    """
    Summarize metrics across CV folds.

    Args:
        fold_metrics: List of metric dictionaries, one per fold

    Returns:
        Dictionary with mean and std for each metric
    """
    if not fold_metrics:
        return {}

    skip_keys = {"fold"}
    keys = [k for k in fold_metrics[0].keys() if k not in skip_keys]
    summary = {}

    for key in keys:
        values = [m[key] for m in fold_metrics if not np.isnan(m[key])]
        if values:
            summary[f"{split}_{key}_mean"] = float(np.mean(values))
            summary[f"{split}_{key}_std"] = float(np.std(values))
        else:
            summary[f"{split}_{key}_mean"] = float("nan")
            summary[f"{split}_{key}_std"] = float("nan")

    return summary
