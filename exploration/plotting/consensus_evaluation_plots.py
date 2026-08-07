#!/usr/bin/env python3
"""Plot loss-consensus evaluation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from exploration.plotting.fonts import register_arial

register_arial()
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


CM = 1 / 2.54

CLASS_NAMES = [
    "Consensus negative",
    "Weak disagreement negative",
    "Strong disagreement negative",
    "Strong disagreement positive",
    "Weak disagreement positive",
    "Consensus positive",
]
CLASS_LABELS = [
    "(---)",
    "(--)",
    "(-)",
    "(+)",
    "(++)",
    "(+++)",
]

SENSITIVITY_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 9,
    "mathtext.fontset": "dejavusans",
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "lines.linewidth": 1.3,
    "lines.markeredgewidth": 1.0,
    "patch.linewidth": 1.0,
    "hatch.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "axes.edgecolor": "#111111",
    "text.color": "#111111",
    "axes.labelcolor": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.transparent": False,
}


def read_table(source: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(source / filename)


def save_sensitivity_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        output / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def short_count(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 999_500:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def heatmap_text_color(value: float, norm: LogNorm) -> str:
    red, green, blue, _ = plt.get_cmap("viridis")(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "#111111" if luminance > 0.55 else "white"


def plot_split_count_heatmap(
    table: pd.DataFrame,
    output: Path,
    stem: str,
    title: str,
    log_floor: int,
    *,
    reverse_x: bool,
) -> None:
    matrix = (
        table.set_index("context_independent_consensus_class")
        .reindex(index=CLASS_NAMES, columns=CLASS_NAMES)
        .to_numpy(dtype=np.int64)
    )
    x_order = list(reversed(range(len(CLASS_NAMES)))) if reverse_x else list(range(len(CLASS_NAMES)))
    matrix = matrix[:, x_order]
    x_labels = [CLASS_LABELS[index] for index in x_order]
    values = matrix.astype(float) + 1
    norm = LogNorm(vmin=log_floor, vmax=max(log_floor + 1, float(values.max())))

    fig, ax = plt.subplots(figsize=(8.8 * CM, 8.0 * CM), constrained_layout=False)
    fig.subplots_adjust(left=0.24, right=0.84, bottom=0.22, top=0.86)
    image = ax.imshow(values, cmap="viridis", norm=norm)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Context-specific edge class", fontsize=9)
    ax.set_ylabel("Edge majority class across contexts", fontsize=9)
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8, fontweight="bold")
    ax.set_yticklabels(CLASS_LABELS, fontsize=8, fontweight="bold")
    ax.tick_params(axis="both", which="major", width=1.0, length=3.0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                short_count(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=8,
                color=heatmap_text_color(values[row, column], norm),
            )

    color_ax = inset_axes(
        ax,
        width="3.5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.03, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8, width=1.0, length=3.0)
    colorbar.set_label("Edge-context count, log scale", fontsize=9, labelpad=2)
    save_sensitivity_figure(fig, output, stem)


def style_sensitivity_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", labelsize=8, width=1.0, length=3.0)


def stability_count_label(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def draw_stability_panel(ax: plt.Axes, table: pd.DataFrame, label: str) -> None:
    class_names = CLASS_NAMES if label == "Labelled positive" else list(reversed(CLASS_NAMES))
    class_labels = CLASS_LABELS if label == "Labelled positive" else list(reversed(CLASS_LABELS))
    panel = table[table["label"] == label].set_index("modal_class").reindex(class_names)
    x = np.arange(len(CLASS_NAMES))
    stable = 100 * panel["stable_fraction"].fillna(0).to_numpy(dtype=float)
    unstable = 100 * panel["unstable_fraction"].fillna(0).to_numpy(dtype=float)
    counts = panel["n_unique_pairs"].fillna(0).to_numpy(dtype=float)
    ax.bar(x, stable, color="#9ecae1", width=0.55, label="Stable")
    ax.bar(x, unstable, bottom=stable, color="#de2d26", width=0.55, label="Unstable")
    for x_value, count in zip(x, counts):
        ax.text(
            x_value,
            102,
            stability_count_label(count),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111111",
        )
    title = "Labelled positives" if label == "Labelled positive" else "Labelled negatives"
    ax.set_title(title, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=8, fontweight="bold")
    ax.set_ylim(0, 110)
    style_sensitivity_axis(ax)


def plot_stability_outputs(table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18.0 * CM, 8.3 * CM), sharey=True)
    draw_stability_panel(axes[0], table, "Labelled positive")
    draw_stability_panel(axes[1], table, "Labelled negative")
    axes[0].set_ylabel("Unique edges (%)")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.34, top=0.88, wspace=0.12)
    save_sensitivity_figure(fig, output, "unique_edge_stable_unstable_by_majority_class")

    for label, slug in [
        ("Labelled positive", "positive"),
        ("Labelled negative", "negative"),
    ]:
        fig, ax = plt.subplots(figsize=(8.8 * CM, 7.9 * CM))
        draw_stability_panel(ax, table, label)
        ax.set_ylabel("Unique edges (%)")
        fig.subplots_adjust(left=0.20, right=0.98, bottom=0.34, top=0.88)
        stem = f"unique_edge_stable_unstable_labelled_{slug}_by_majority_class"
        save_sensitivity_figure(fig, output, stem)


def plot_consensus(source: str | Path, output: str | Path) -> None:
    """Render the consensus heatmap and stability plots."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    positive = read_table(source, "labelled_positive_class_counts.csv")
    negative = read_table(source, "labelled_negative_class_counts.csv")
    stability = read_table(source, "edge_stability.csv")

    with plt.rc_context(SENSITIVITY_RC):
        plot_split_count_heatmap(
            positive,
            output,
            "labelled_positive_majority_observed_counts_split",
            "Labelled positives",
            1_000,
            reverse_x=False,
        )
        plot_split_count_heatmap(
            negative,
            output,
            "labelled_negative_majority_observed_counts_split",
            "Labelled negatives",
            100_000,
            reverse_x=True,
        )
        plot_stability_outputs(stability, output)

