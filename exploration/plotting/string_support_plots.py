#!/usr/bin/env python3
"""Plot STRING-support results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from exploration.plotting.fonts import register_arial

register_arial()
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


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
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.titlesize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
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
    "savefig.transparent": True,
}

TITLE_FONT_SIZE = SENSITIVITY_RC["axes.titlesize"]
AXIS_LABEL_FONT_SIZE = SENSITIVITY_RC["axes.labelsize"]
TICK_FONT_SIZE = SENSITIVITY_RC["xtick.labelsize"]


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


def style_string_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONT_SIZE,
        width=1.0,
        length=3.0,
        pad=3,
    )


def string_count_label(value: float) -> str:
    if value >= 1_000_000:
        text = f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{text}M"
    if value >= 1_000:
        text = f"{value / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{text}k"
    return f"{value:.0f}"


POSITIVE_CLASS_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "positive_class", ["#fdd0a2", "#a63603"]
)
POSITIVE_CLASS_COLORS = [
    mcolors.to_hex(POSITIVE_CLASS_CMAP(position)) for position in (0.0, 0.5, 1.0)
]
STRING_BANDS = [
    ("low", "0 < score < 0.4", POSITIVE_CLASS_COLORS[0]),
    ("medium", "0.4 <= score < 0.8", POSITIVE_CLASS_COLORS[1]),
    ("high", "score >= 0.8", POSITIVE_CLASS_COLORS[2]),
]


def string_class_order(label: str) -> tuple[list[str], list[str]]:
    if label == "Labelled positive":
        return CLASS_NAMES, CLASS_LABELS
    return list(reversed(CLASS_NAMES)), list(reversed(CLASS_LABELS))


def draw_string_coverage_axis(
    ax: plt.Axes, score_bands: pd.DataFrame, label: str, *, show_ylabel: bool
) -> None:
    class_names, class_labels = string_class_order(label)
    table = score_bands[score_bands["label"] == label]
    counts = table.groupby("loss_class").first().reindex(class_names)
    total = counts["n_unique_pairs"].fillna(0).to_numpy(dtype=float)
    supported = counts["n_string_supported_pairs"].fillna(0).to_numpy(dtype=float)
    coverage = np.divide(100 * supported, total, out=np.zeros_like(total), where=total > 0)
    x = np.arange(len(class_names))
    ax.bar(x, 100, facecolor="none", edgecolor="#111111", linewidth=1.0, width=0.55)
    ax.bar(
        x,
        coverage,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        hatch="////",
        width=0.55,
    )
    for x_value, count in zip(x, total):
        ax.text(
            x_value,
            103,
            string_count_label(count),
            ha="center",
            va="bottom",
            fontsize=TICK_FONT_SIZE,
            color="#111111",
        )
    title = "Labelled positives" if label == "Labelled positive" else "Labelled negatives"
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=TICK_FONT_SIZE, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis="x", bottom=True, labelbottom=True)
    if show_ylabel:
        ax.set_ylabel("STRING edge coverage (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    else:
        ax.tick_params(axis="y", labelleft=False)
    style_string_axis(ax)


def draw_string_score_band_panel(
    ax: plt.Axes, score_bands: pd.DataFrame, label: str
) -> None:
    class_names, class_labels = string_class_order(label)
    table = score_bands[score_bands["label"] == label]
    x = np.arange(len(class_names))
    bottom = np.zeros(len(class_names), dtype=float)
    for band, band_label, color in STRING_BANDS:
        panel = (
            table[table["score_band"] == band]
            .set_index("loss_class")
            .reindex(class_names)
        )
        values = panel["fraction_string_supported_pairs"].fillna(0).to_numpy(dtype=float)
        ax.bar(
            x,
            100 * values,
            bottom=100 * bottom,
            width=0.55,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=band_label,
        )
        bottom += values
    for x_value in x:
        ax.add_patch(
            Rectangle(
                (x_value - 0.55 / 2, 0),
                0.55,
                100,
                fill=False,
                edgecolor="#006D2C",
                linewidth=1.0,
                zorder=4,
            )
        )
    counts = table.groupby("loss_class").first().reindex(class_names)[
        "n_string_supported_pairs"
    ]
    for x_value, count in zip(x, counts):
        ax.text(
            x_value,
            102,
            string_count_label(float(count)),
            ha="center",
            va="bottom",
            fontsize=TICK_FONT_SIZE,
            color="#111111",
        )
    title = "Labelled positives" if label == "Labelled positive" else "Labelled negatives"
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=TICK_FONT_SIZE, fontweight="bold")
    ax.set_ylim(0, 110)
    style_string_axis(ax)


def plot_string_coverage_by_label(
    score_bands: pd.DataFrame, output: Path, stem: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8 * CM, 4.5 * CM), sharey=True)
    for column, label in enumerate(["Labelled positive", "Labelled negative"]):
        draw_string_coverage_axis(
            axes[column], score_bands, label, show_ylabel=column == 0
        )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88, wspace=0.12)
    save_sensitivity_figure(fig, output, stem)


def plot_string_score_distribution_by_label(
    score_bands: pd.DataFrame, output: Path, stem: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8 * CM, 4.5 * CM), sharey=True)
    for column, label in enumerate(["Labelled positive", "Labelled negative"]):
        draw_string_score_band_panel(axes[column], score_bands, label)
    axes[0].set_ylabel("STRING edges (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88, wspace=0.12)
    save_sensitivity_figure(fig, output, stem)


def plot_string_support(score_bands: pd.DataFrame, output: Path) -> None:
    stem = "string_combined_string_supported_unique_edge_score_band_distribution_by_loss_class"
    plot_string_coverage_by_label(score_bands, output, f"{stem}_coverage")
    plot_string_score_distribution_by_label(
        score_bands, output, f"{stem}_distribution"
    )


def plot_string(source: str | Path, output: str | Path) -> None:
    """Render the STRING-support plot."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    score_bands = read_table(source, "string_score_bands.csv")
    with plt.rc_context(SENSITIVITY_RC):
        plot_string_support(score_bands, output)
