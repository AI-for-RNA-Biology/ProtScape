#!/usr/bin/env python3
"""Plot STRING-correlation diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from exploration.plotting.fonts import register_arial

register_arial()
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


CM = 1 / 2.54

SOURCE_STYLE = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "Arial:italic",
    "mathtext.sf": "Arial",
    "mathtext.tt": "Arial",
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "legend.fontsize": 6.0,
    "figure.titlesize": 7.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "savefig.transparent": True,
}

matplotlib.rcParams.update(SOURCE_STYLE)


def save_original(
    fig,
    output: Path,
    stem: str,
    *,
    pad_inches: float | None = None,
    pdf_dpi: int = 300,
):
    """Save with the transparent, tight-bbox settings of the source scripts."""
    kwargs = {"bbox_inches": "tight", "transparent": True}
    if pad_inches is not None:
        kwargs["pad_inches"] = pad_inches
    fig.savefig(output / f"{stem}.pdf", dpi=pdf_dpi, **kwargs)
    fig.savefig(output / f"{stem}.png", dpi=300, **kwargs)
    plt.close(fig)


def plot_string_correlations(ax, table):
    colors = {"Pearson": "#222222", "Spearman": "#B3B3B3"}
    combinations = ["BCE", "pHuber", "L1", "BCE+pHuber", "BCE+L1", "pHuber+L1", "BCE+pHuber+L1"]
    table = table.set_index("loss_combination").loc[combinations]
    labels = ["BCE", "pHuber", "L1", "BCE +\npHuber", "BCE +\nL1", "pHuber +\nL1", "BCE +\npHuber + L1"]
    x = np.arange(len(combinations))
    width = 0.18
    offset = 0.17
    for xoffset, column, name, legend_label in [
        (-offset, "pearson", "Pearson", "Pearson $r$"),
        (offset, "spearman", "Spearman", r"Spearman $\rho$"),
    ]:
        values = table[column].to_numpy()
        bars = ax.bar(
            x + xoffset,
            values,
            width=width,
            color=colors[name],
            edgecolor="#222222",
            linewidth=0.55,
            label=legend_label,
            zorder=2,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.004,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="#222222",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, table[["pearson", "spearman"]].to_numpy().max() + 0.045)
    ax.set_ylabel("Correlation with STRING score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_string(source, output):
    """Render STRING-validation correlations."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    string_correlations = pd.read_csv(source / "string_correlations.csv")

    panel_f_style = {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 7.701,
        "axes.labelsize": 7.701,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.labelsize": 6.701,
        "ytick.labelsize": 6.701,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.transparent": True,
    }
    with matplotlib.rc_context(panel_f_style):
        f_fig, f_ax = plt.subplots(
            figsize=(15 * CM, 4.5 * CM), constrained_layout=True
        )
        plot_string_correlations(f_ax, string_correlations)
        save_original(
            f_fig,
            output,
            "loss_descriptor_correlations_mean_std",
            pad_inches=0.03,
        )

