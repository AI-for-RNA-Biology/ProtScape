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
    "axes.labelsize": 8.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "legend.fontsize": 6.0,
    "figure.titlesize": 8.0,
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


def plot_loss_correlation_scatter(ax: plt.Axes, table: pd.DataFrame) -> None:
    combinations = ["BCE", "pHuber", "L1", "BCE+pHuber+L1"]
    labels = ["BCE", "pHuber", "L1", "Combined"]
    colors = {
        "BCE": "#a63603",
        "pHuber": "#d31529",
        "L1": "#9200bf",
        "Combined": "#222222",
    }
    rows = table.set_index("loss_combination").reindex(combinations).copy()
    if rows[["pearson", "spearman"]].isna().any().any():
        raise ValueError("Missing a loss combination required for the STRING scatter")
    rows["label"] = labels
    rows["pearson_percent"] = rows["pearson"] * 100.0
    rows["spearman_percent"] = rows["spearman"] * 100.0

    for row in rows.itertuples():
        ax.scatter(
            row.pearson_percent,
            row.spearman_percent,
            s=40,
            color=colors[row.label],
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            row.pearson_percent + 0.4,
            row.spearman_percent + 0.4,
            row.label,
            fontsize=7,
            ha="left",
            va="bottom",
            color="#222222",
        )

    ax.set_xlim(
        rows["pearson_percent"].min() - 2,
        rows["pearson_percent"].max() + 2,
    )
    ax.set_ylim(
        rows["spearman_percent"].min() - 2,
        rows["spearman_percent"].max() + 2,
    )
    ax.set_xlabel("P (loss, STRING) (%)")
    ax.set_ylabel(r"$\rho$ (loss, STRING) (%)")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")


def plot_string(source, output):
    """Render STRING-validation correlations."""
    matplotlib.rcParams.update(SOURCE_STYLE)
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

    scatter_fig, scatter_ax = plt.subplots(figsize=(4.0 * CM, 5.5 * CM))
    plot_loss_correlation_scatter(scatter_ax, string_correlations)
    save_original(scatter_fig, output, "loss_vs_string_correlations_scatter")
