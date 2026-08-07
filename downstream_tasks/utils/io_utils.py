# io_utils.py
# I/O utilities for saving models, results, and checkpoints

from pathlib import Path
from typing import Any, Dict, Optional, Union
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
    fold: Union[int, str],
) -> Path:
    """Save one fold's model weights."""
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_path = models_dir / f"fold_{fold}.pt"

    torch.save(model.state_dict(), weights_path)

    return weights_path


def load_model_weights(
    model: nn.Module,
    output_dir: Path,
    fold: int,
) -> bool:
    """Load one fold's model weights if they exist."""
    weights_path = output_dir / "models" / f"fold_{fold}.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        return True
    return False


def model_weights_exist(
    output_dir: Path,
    n_folds: int,
) -> bool:
    """Return whether every expected fold weight exists."""
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
    Save results to a CSV file.

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

    df.to_csv(csv_path, index=False)

    return csv_path



def aggregate_task_results(output_root: Path, task: str) -> pd.DataFrame:
    """Collect every run-level ``results.csv`` for one downstream task."""
    task_dir = output_root / task
    if not task_dir.exists():
        return pd.DataFrame()

    all_results = []
    for results_path in sorted(task_dir.glob("*/*/results.csv")):
        df = pd.read_csv(results_path)
        if "inference_name" not in df:
            df["inference_name"] = results_path.parents[1].name
        if "output_model_key" not in df:
            df["output_model_key"] = results_path.parent.name
        all_results.append(df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def save_task_results(output_root: Path, task: str) -> Optional[Path]:
    """Write the task-level sweep summary used by the analysis plots."""
    df = aggregate_task_results(output_root, task)
    if df.empty:
        return None

    csv_path = output_root / task / "full_results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
