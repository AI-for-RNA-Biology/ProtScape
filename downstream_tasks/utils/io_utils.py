# io_utils.py
# I/O utilities for saving models, results, and checkpoints

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def setup_output_dirs(output_root: Path, task: str, inference_model: str, model_key: str) -> Path:
    """
    Setup output directory structure for a specific experiment.

    Creates:
    output_root/
        task/
            inference_model/
                model_key/
                    models/
                    training_curves/

    Returns:
        Path to the model_key directory
    """
    output_dir = output_root / task / inference_model / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "training_curves").mkdir(exist_ok=True)
    return output_dir



def save_model_weights(
    model: nn.Module,
    output_dir: Path,
    fold: int,
    hyperparam_subdir: Optional[str] = None,
) -> Path:
    """
    Save model weights for a specific fold.

    Args:
        model: PyTorch model
        output_dir: Output directory (should be the model_key directory)
        fold: Fold number
        hyperparam_subdir: Optional subfolder for hyperparameter combination

    Returns:
        Path to saved weights file
    """
    if hyperparam_subdir:
        models_dir = output_dir / "models" / hyperparam_subdir
    else:
        models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_path = models_dir / f"fold_{fold}.pt"

    # Save with atomic write
    tmp_path = weights_path.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), tmp_path)
    tmp_path.rename(weights_path)

    return weights_path


def load_model_weights(
    model: nn.Module,
    output_dir: Path,
    fold: int,
    hyperparam_subdir: Optional[str] = None,
) -> bool:
    """
    Load model weights for a specific fold.

    Args:
        model: PyTorch model
        output_dir: Output directory
        fold: Fold number
        hyperparam_subdir: Optional subfolder for hyperparameter combination

    Returns:
        True if weights were loaded, False if not found
    """
    if hyperparam_subdir:
        weights_path = output_dir / "models" / hyperparam_subdir / f"fold_{fold}.pt"
    else:
        weights_path = output_dir / "models" / f"fold_{fold}.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        return True
    return False


def model_weights_exist(
    output_dir: Path,
    n_folds: int,
    hyperparam_subdir: Optional[str] = None,
) -> bool:
    """
    Check if all fold weights exist.

    Args:
        output_dir: Output directory
        n_folds: Number of folds
        hyperparam_subdir: Optional subfolder for hyperparameter combination

    Returns:
        True if all fold weights exist
    """
    if hyperparam_subdir:
        models_dir = output_dir / "models" / hyperparam_subdir
    else:
        models_dir = output_dir / "models"
    if not models_dir.exists():
        return False
    for fold in range(n_folds):
        if not (models_dir / f"fold_{fold}.pt").exists():
            return False
    return True


def save_results_csv(
    results: Dict[str, Any],
    output_dir: Path,
    filename: str = "results.csv",
) -> Path:
    """
    Save results to a CSV file (atomic write for parallel safety).

    Args:
        results: Dictionary with result metrics
        output_dir: Output directory
        filename: CSV filename

    Returns:
        Path to saved CSV
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename

    # Convert to DataFrame
    df = pd.DataFrame([results])

    # Atomic write: write to temp file then rename
    tmp_path = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.rename(csv_path)

    return csv_path



def aggregate_results_from_subfolders(
    output_root: Path,
    task: str,
    inference_model: str,
) -> pd.DataFrame:
    """
    Aggregate results from all model subfolders for a given inference model.

    Args:
        output_root: Output root directory
        task: Task name
        inference_model: Inference model name

    Returns:
        DataFrame with aggregated results
    """
    task_dir = output_root / task / inference_model
    if not task_dir.exists():
        return pd.DataFrame()

    all_results = []
    for model_dir in task_dir.iterdir():
        if model_dir.is_dir():
            results_path = model_dir / "results.csv"
            if results_path.exists():
                df = pd.read_csv(results_path)
                all_results.append(df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def save_aggregated_results(
    output_root: Path,
    task: str,
    inference_model: str,
) -> Optional[Path]:
    """
    Save aggregated results for all models to a single CSV.

    Args:
        output_root: Output root directory
        task: Task name
        inference_model: Inference model name

    Returns:
        Path to aggregated CSV or None if no results
    """
    df = aggregate_results_from_subfolders(output_root, task, inference_model)
    if df.empty:
        return None

    csv_path = output_root / task / inference_model / "all_results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
