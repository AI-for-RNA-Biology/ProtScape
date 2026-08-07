#!/usr/bin/env python3
"""Plot the standalone Figure 2 panels."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatterMathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Input and output directories.
SOURCE_DATA_DIR = Path(
    "/storage/research/dbmr_luisierlab/temp/athomas/outputs_protscape_repo/"
    "figure_source_data/figure_2"
)
FIGURE_OUTPUT_DIR = Path(
    "/storage/research/dbmr_luisierlab/temp/athomas/outputs_protscape_repo/"
    "figures/figure_2"
)

CM = 1 / 2.54

CORE_MODEL_ORDER = [
    "pinnacle_random",
    "pinnacle_esm2_acm",
    "pinnacle_esm2",
    "gae_att",
    "s2gae_att_k1_uni",
]
MODEL_LABELS = {
    "pinnacle_random": "Pinnacle",
    "pinnacle_esm2_acm": "Pinnacle-ESM (ACM)",
    "pinnacle_esm2": "Pinnacle-ESM (GAT)",
    "gae_att": "ProtScape-GAE",
    "s2gae_att_k1_uni": "ProtScape",
}
MODEL_COLORS = {
    "pinnacle_random": "#2b2b2b",
    "pinnacle_esm2_acm": "#7a7a7a",
    "pinnacle_esm2": "#b0b0b0",
    "gae_att": "#1f77b4",
    "s2gae_att_k1_uni": "#e6550d",
}

# This is the order in the original loss-sensitivity output tables and legend.
LOSS_MODEL_ORDER = [
    "s2gae_att_k1_uni",
    "s2gae_att_k1_l1_do00",
    "s2gae_att_k1_phuber",
]
LOSS_LABELS = {
    "s2gae_att_k1_uni": "ProtScape-S2GAE BCE",
    "s2gae_att_k1_l1_do00": "ProtScape-S2GAE L1",
    "s2gae_att_k1_phuber": "ProtScape-S2GAE pHuber",
}
LOSS_COLORS = {
    "s2gae_att_k1_uni": "#e6550d",
    "s2gae_att_k1_l1_do00": "#54278f",
    "s2gae_att_k1_phuber": "#b85c00",
}

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

PRETRAINING_RC = {
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
    "savefig.format": "pdf",
    "savefig.transparent": True,
    "svg.fonttype": "none",
}

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


def save_pretraining_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_sensitivity_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        output / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def clean_pretraining_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.5, length=6, pad=5)
    ax.tick_params(axis="both", which="minor", width=1.2, length=3)


def percent_limits(values: pd.Series | np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    low = float(values.min())
    high = float(values.max())
    pad = max(1.0, 0.12 * (high - low))
    return max(0.0, low - pad), min(100.0, high + pad)


def model_handles(order: list[str], colors: dict[str, str], linewidth: float) -> list[Line2D]:
    return [Line2D([0], [0], color=colors[key], linewidth=linewidth) for key in order]


def draw_curve(
    ax: plt.Axes,
    table: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    ylabel: str,
    yticks: list[int] | None,
) -> None:
    rows = table[table["model_key"].isin(order)]
    for key in order:
        model = rows[rows["model_key"] == key].sort_values("k_negatives")
        ax.plot(
            model["k_negatives"],
            model["score_percent"],
            marker="o",
            markersize=3.5,
            linewidth=2,
            linestyle="-",
            color=colors[key],
            label=labels[key],
        )

    chance = table[table["model_key"] == "chance_auprc"].sort_values("k_negatives")
    if not chance.empty:
        ax.plot(
            chance["k_negatives"],
            chance["score_percent"],
            color="#222222",
            linestyle="--",
            linewidth=1,
            label="Chance AUPRC",
        )

    ax.set_xscale("log")
    ax.set_xticks([1, 10, 50, 100, 500])
    ax.set_xticklabels(["1", "10", "50", "100", "500"], fontsize=11)
    ax.set_xlabel("Negatives per positive edge", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(*percent_limits(table["score_percent"]))
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(value) for value in yticks], fontsize=11)
    clean_pretraining_axis(ax)


def plot_curve(
    table: pd.DataFrame,
    output: Path,
    stem: str,
    order: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    ylabel: str,
    yticks: list[int] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    draw_curve(
        ax,
        table,
        order,
        colors,
        labels,
        ylabel,
        yticks,
    )
    plt.tight_layout()
    save_pretraining_figure(fig, output, stem)


def plot_core_model_legend(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 0.45))
    ax.axis("off")
    ax.legend(
        model_handles(CORE_MODEL_ORDER, MODEL_COLORS, 2.5),
        [MODEL_LABELS[key] for key in CORE_MODEL_ORDER],
        frameon=False,
        loc="center",
        ncol=len(CORE_MODEL_ORDER),
        fontsize=11,
        handlelength=1.8,
    )
    save_pretraining_figure(fig, output, "core_model_legend")


def plot_metagraph_barplot(table: pd.DataFrame, output: Path, stem: str) -> None:
    rows = table.set_index("model_key").reindex(CORE_MODEL_ORDER)
    values = rows["metagraph_percent"].to_numpy(dtype=float)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(2.5, 3.0))
    ax.bar(
        x,
        values,
        color=[MODEL_COLORS[key] for key in CORE_MODEL_ORDER],
        edgecolor="#222222",
        linewidth=0.55,
        width=0.55,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(rows), fontsize=11)
    ax.set_xlabel("Models", fontsize=13)
    ax.set_ylabel("Metagraph AUPRC", fontsize=13)
    y_min = max(0, float(values.min()) - 5)
    y_max = min(100, float(values.max()) + 5)
    ax.set_ylim(y_min, y_max)
    for index, value in enumerate(values):
        ax.text(
            index,
            value + (y_max - y_min) * 0.02,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    clean_pretraining_axis(ax)
    plt.tight_layout()
    save_pretraining_figure(fig, output, stem)


def plot_parameter_counts(table: pd.DataFrame, output: Path, stem: str) -> None:
    rows = table.set_index("model_key").reindex(CORE_MODEL_ORDER)
    values = rows["parameter_count"].to_numpy(dtype=float)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(2.5, 3.0))
    ax.bar(
        x,
        values,
        color=[MODEL_COLORS[key] for key in CORE_MODEL_ORDER],
        edgecolor="#222222",
        linewidth=0.55,
        width=0.55,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(rows), fontsize=13)
    ax.set_xlabel("Models", fontsize=13)
    ax.set_yscale("log")
    for index, value in enumerate(values):
        ax.text(
            index,
            value * 1.12,
            f"{int(value / 1e6)}M",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Trainable parameters", fontsize=13)
    ax.set_ylim(max(1e7, float(values.min()) / 2), float(values.max()) * 1.85)
    ax.set_yticks([1e7, 1e8, 1e9])
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    clean_pretraining_axis(ax)
    save_pretraining_figure(fig, output, stem)


def plot_loss_legend(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 0.45))
    ax.axis("off")
    handles = model_handles(LOSS_MODEL_ORDER, LOSS_COLORS, 2.5)
    handles.append(Line2D([0], [0], color="#222222", linestyle="--", linewidth=1.25))
    labels = [LOSS_LABELS[key] for key in LOSS_MODEL_ORDER] + ["Chance AUPRC"]
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="center",
        ncol=len(labels),
        fontsize=11,
        handlelength=1.8,
    )
    save_pretraining_figure(fig, output, "loss_sensitivity_legend")


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


def style_string_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(
        axis="both", which="major", labelsize=8, width=1.0, length=3.0, pad=3
    )


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


def string_count_label(value: float) -> str:
    if value >= 1_000_000:
        text = f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{text}M"
    if value >= 1_000:
        text = f"{value / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{text}k"
    return f"{value:.0f}"


STRING_BANDS = [
    ("low", "0 < score < 0.4", "#D8B365"),
    ("medium", "0.4 <= score < 0.8", "#80CDC1"),
    ("high", "score >= 0.8", "#A1D99B"),
]


def string_class_order(label: str) -> tuple[list[str], list[str]]:
    if label == "Labelled positive":
        return CLASS_NAMES, CLASS_LABELS
    return list(reversed(CLASS_NAMES)), list(reversed(CLASS_LABELS))


def draw_string_coverage_axis(
    ax: plt.Axes, score_bands: pd.DataFrame, label: str, *, show_ylabel: bool
) -> None:
    class_names, _ = string_class_order(label)
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
        edgecolor="#006D2C",
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
            fontsize=8,
            color="#111111",
        )
    title = "Labelled positives" if label == "Labelled positive" else "Labelled negatives"
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("STRING edge coverage (%)", fontsize=9)
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
            fontsize=8,
            color="#111111",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=8, fontweight="bold")
    ax.set_ylim(0, 110)
    style_string_axis(ax)


def make_string_support_figure(score_bands: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18.0 * CM, 11.2 * CM),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [0.42, 1.0]},
    )
    for column, label in enumerate(["Labelled positive", "Labelled negative"]):
        draw_string_coverage_axis(axes[0, column], score_bands, label, show_ylabel=column == 0)
        draw_string_score_band_panel(axes[1, column], score_bands, label)
    axes[1, 0].set_ylabel("STRING edges (%)", fontsize=9)
    return fig, axes


def plot_string_support(score_bands: pd.DataFrame, output: Path) -> None:
    stem = "string_combined_string_supported_unique_edge_score_band_distribution_by_loss_class"
    fig, _ = make_string_support_figure(score_bands)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.90, wspace=0.12, hspace=0.32)
    save_sensitivity_figure(fig, output, stem)


def plot_pretraining(source: str | Path, output: str | Path) -> None:
    """Render the original pretraining plots used for Figure 2a-d."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panel_a_auprc = read_table(source, "robust_ppi_auprc.csv")
    panel_a_f1 = read_table(source, "robust_ppi_f1.csv")
    panel_b = read_table(source, "metagraph_auprc.csv")
    panel_c = read_table(source, "parameter_counts.csv")
    panel_d_auprc = read_table(source, "loss_sensitivity_auprc.csv")
    panel_d_f1 = read_table(source, "loss_sensitivity_f1.csv")

    with plt.rc_context(PRETRAINING_RC):
        plot_curve(
            panel_a_auprc,
            output,
            "core_robust_ppi_auprc",
            CORE_MODEL_ORDER,
            MODEL_COLORS,
            MODEL_LABELS,
            "AUPRC",
            [0, 20, 40, 60, 80, 100],
        )
        plot_curve(
            panel_a_f1,
            output,
            "core_robust_ppi_f1",
            CORE_MODEL_ORDER,
            MODEL_COLORS,
            MODEL_LABELS,
            "F1",
            [0, 20, 40, 60, 80],
        )
        plot_core_model_legend(output)

        plot_metagraph_barplot(panel_b, output, "metagraph_barplot_auprc")
        plot_parameter_counts(panel_c, output, "core_parameter_counts")

        plot_curve(
            panel_d_auprc,
            output,
            "loss_sensitivity_auprc",
            LOSS_MODEL_ORDER,
            LOSS_COLORS,
            LOSS_LABELS,
            "AUPRC",
            None,
        )
        plot_curve(
            panel_d_f1,
            output,
            "loss_sensitivity_f1",
            LOSS_MODEL_ORDER,
            LOSS_COLORS,
            LOSS_LABELS,
            "F1",
            None,
        )
        plot_loss_legend(output)


def plot_consensus(source: str | Path, output: str | Path) -> None:
    """Render the original heatmap and stability plots used for Figure 2e-f."""
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


def plot_string(source: str | Path, output: str | Path) -> None:
    """Render the original STRING-support plot used for Figure 2g."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    score_bands = read_table(source, "string_score_bands.csv")
    with plt.rc_context(SENSITIVITY_RC):
        plot_string_support(score_bands, output)


def plot_all(
    source: str | Path = SOURCE_DATA_DIR,
    output: str | Path = FIGURE_OUTPUT_DIR,
) -> None:
    """Render every standalone Figure 2 source plot from prepared CSVs."""
    source, output = Path(source), Path(output)
    if not source.exists():
        raise FileNotFoundError(f"Missing figure source data: {source}")
    plot_pretraining(source, output)
    plot_consensus(source, output)
    plot_string(source, output)
    print(f"Wrote individual Figure 2 plots to {output}")


def main() -> None:
    plot_all()


if __name__ == "__main__":
    main()
