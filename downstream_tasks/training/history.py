# history.py
# Training history tracking and visualization

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class TrainingHistory:
    """Tracks training metrics per fold for logging and plotting."""

    model_name: str
    fold: int
    epochs: List[int] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_auprcs: List[float] = field(default_factory=list)
    val_aurocs: List[float] = field(default_factory=list)

    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_auprc: float,
        val_auroc: float = 0.0,
    ):
        """Record metrics for one epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_auprcs.append(val_auprc)
        self.val_aurocs.append(val_auroc)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame."""
        return pd.DataFrame({
            "model": self.model_name,
            "fold": self.fold,
            "epoch": self.epochs,
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "val_auprc": self.val_auprcs,
            "val_auroc": self.val_aurocs,
        })

    def get_best_epoch(self) -> int:
        """Get epoch with best validation AUPRC."""
        if not self.val_auprcs:
            return 0
        return self.epochs[int(np.argmax(self.val_auprcs))]

    def get_best_auprc(self) -> float:
        """Get best validation AUPRC."""
        if not self.val_auprcs:
            return 0.0
        return max(self.val_auprcs)


def plot_training_curves(
    histories: List[TrainingHistory],
    output_path: Path,
    model_name: str,
):
    """
    Plot training curves for all folds.

    Args:
        histories: List of TrainingHistory objects, one per fold
        output_path: Path to save the plot
        model_name: Model name for title
    """
    if plt is None or not histories:
        return

    n_folds = len(histories)
    fig, axes = plt.subplots(2, n_folds, figsize=(4 * n_folds, 6), squeeze=False)
    fig.suptitle(f"{model_name} - Training Curves", fontsize=14)

    for i, hist in enumerate(histories):
        # Loss plot
        ax_loss = axes[0, i]
        ax_loss.plot(hist.epochs, hist.train_losses, label="Train", alpha=0.8)
        ax_loss.plot(hist.epochs, hist.val_losses, label="Val", alpha=0.8)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"Fold {hist.fold}")
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

        # AUPRC plot
        ax_auprc = axes[1, i]
        ax_auprc.plot(hist.epochs, hist.val_auprcs, color="green", alpha=0.8)
        ax_auprc.set_xlabel("Epoch")
        ax_auprc.set_ylabel("Val AUPRC")
        ax_auprc.grid(True, alpha=0.3)

        if hist.val_auprcs:
            best_idx = int(np.argmax(hist.val_auprcs))
            ax_auprc.axvline(hist.epochs[best_idx], color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(
    history: TrainingHistory,
    output_path: Path,
    model_name: str,
):
    """Plot training curves for a single fold."""
    if plt is None or not history.epochs:
        return

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    fig.suptitle(f"{model_name} - Fold {history.fold}", fontsize=14)

    ax_loss, ax_auprc = axes
    ax_loss.plot(history.epochs, history.train_losses, label="Train", alpha=0.8)
    ax_loss.plot(history.epochs, history.val_losses, label="Val", alpha=0.8)
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    ax_auprc.plot(history.epochs, history.val_auprcs, color="green", alpha=0.8)
    ax_auprc.set_xlabel("Epoch")
    ax_auprc.set_ylabel("Val AUPRC")
    ax_auprc.grid(True, alpha=0.3)

    if history.val_auprcs:
        best_idx = int(np.argmax(history.val_auprcs))
        ax_auprc.axvline(history.epochs[best_idx], color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_histories_csv(
    histories: List[TrainingHistory],
    output_path: Path,
):
    """
    Save training histories to CSV (atomic write for parallel safety).

    Args:
        histories: List of TrainingHistory objects
        output_path: Path to save CSV
    """
    if not histories:
        return

    dfs = [h.to_dataframe() for h in histories]
    combined = pd.concat(dfs, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write
    tmp_path = output_path.with_suffix(".csv.tmp")
    combined.to_csv(tmp_path, index=False)
    tmp_path.rename(output_path)


def save_all_histories(
    all_histories: Dict[str, List[TrainingHistory]],
    output_dir: Path,
):
    """
    Save all training histories to CSV files and plot curves.

    Args:
        all_histories: Dict mapping model names to list of histories
        output_dir: Output directory
    """
    plots_dir = output_dir / "training_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for model_name, histories in all_histories.items():
        for hist in histories:
            all_dfs.append(hist.to_dataframe())
        if histories:
            safe_name = model_name.lower().replace(" ", "_").replace("+", "_")
            plot_training_curves(histories, plots_dir / f"{safe_name}.png", model_name)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_dir / "training_histories.csv", index=False)
