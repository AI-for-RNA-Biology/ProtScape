#!/usr/bin/env python3
"""Plot therapeutic-target performance results."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


CM = 1 / 2.54
DOUBLE_COLUMN_WIDTH_CM = 18.0
SINGLE_COLUMN_WIDTH_CM = 8.8
MAX_FIGURE_HEIGHT_CM = 17.0
WIDE_SINGLE_SIZE = (18.0 * CM, 8.8 * CM)

AXIS_LINEWIDTH = 0.6
BAR_EDGE_LINEWIDTH = 0.55
DATA_LINEWIDTH = 0.8
TICK_LABEL_SIZE = 6.0
AXIS_LABEL_SIZE = 7.0
PANEL_TITLE_SIZE = 7.0
LEGEND_SIZE = 6.0

PLOT_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "font.size": 7.0,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "Arial:italic",
    "mathtext.sf": "Arial",
    "mathtext.tt": "Arial",
    "text.color": "#111111",
    "axes.labelcolor": "#111111",
    "axes.edgecolor": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "axes.linewidth": AXIS_LINEWIDTH,
    "lines.linewidth": DATA_LINEWIDTH,
    "lines.markeredgewidth": AXIS_LINEWIDTH,
    "patch.linewidth": AXIS_LINEWIDTH,
    "hatch.linewidth": AXIS_LINEWIDTH,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": PANEL_TITLE_SIZE,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
    "savefig.transparent": True,
}

READOUT_ORDER = [
    "lr_esm",
    "lr_prostt5",
    "lr_hc_cell",
    "lr_hc_cell_esm",
    "abmil8_hc_cell",
    "abmil8",
    "abmil8_pdl_hc_cell",
    "abmil8_pdl_id2_dropout",
]
READOUT_LABELS = {
    "lr_esm": "Linear probe\nESM2 sequence",
    "lr_prostt5": "Linear probe\nProstT5 sequence",
    "lr_hc_cell": "Mean-pool LR\nContextual instances",
    "lr_hc_cell_esm": "Mean-pool LR\nContextual + ESM2",
    "abmil8_hc_cell": "ABMIL\nContextual instances",
    "abmil8": "ABMIL\nContextual + ESM2",
    "abmil8_pdl_hc_cell": "ABMIL-PDL\nContextual instances",
    "abmil8_pdl_id2_dropout": "ABMIL-PDL\nContextual + ESM2",
}
BASELINES = {"lr_esm", "lr_prostt5"}
MODEL_ORDER = [
    "pinnacle_random_fixed",
    "pinnacle_esm_fixed",
    "pinnacle_esm2_acm",
    "gae_att_fixed_do06",
    "s2gae_att_k1_fixed_do04_uni",
]
MODEL_LABELS = {
    "lr_esm": "ESM2",
    "lr_prostt5": "ProstT5",
    "pinnacle_random_fixed": "Pinnacle",
    "pinnacle_esm_fixed": "Pinnacle-ESM2 (GAT)",
    "pinnacle_esm2_acm": "Pinnacle-ESM2 (ACM)",
    "gae_att_fixed_do06": "ProtScape-GAE",
    "s2gae_att_k1_fixed_do04_uni": "ProtScape",
}
MODEL_COLORS = {
    "lr_esm": "#222222",
    "lr_prostt5": "#666666",
    "pinnacle_random_fixed": "#2b2b2b",
    "pinnacle_esm_fixed": "#b0b0b0",
    "pinnacle_esm2_acm": "#7a7a7a",
    "gae_att_fixed_do06": "#1f77b4",
    "s2gae_att_k1_fixed_do04_uni": "#e6550d",
}


def figure_size(width_cm: float, height_cm: float) -> tuple[float, float]:
    return width_cm * CM, height_cm * CM


def clean_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#222222")
        ax.spines[side].set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=0.6,
        length=2.5,
        pad=5,
    )
    ax.tick_params(axis="both", which="minor", width=0.5, length=1.5)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    options = {"bbox_inches": "tight", "pad_inches": 0.03}
    fig.savefig(output / f"{stem}.pdf", **options)
    fig.savefig(output / f"{stem}.png", dpi=300, **options)
    plt.close(fig)


def plot_average_legend(rows: pd.DataFrame, output: Path) -> None:
    inference_order = [key for key in MODEL_ORDER if key in set(rows["inference_key"])]
    handles = [
        Patch(
            facecolor=MODEL_COLORS[key],
            edgecolor="white",
            label=MODEL_LABELS[key],
        )
        for key in inference_order
    ]
    handles.extend(
        [
            Patch(facecolor="white", edgecolor="#222222", label="ESM2"),
            Patch(
                facecolor="#f2f2f2",
                edgecolor="#666666",
                hatch="\\\\\\\\",
                label="ProstT5",
            ),
        ]
    )
    fig = plt.figure(figsize=figure_size(DOUBLE_COLUMN_WIDTH_CM, 2.2))
    fig.patch.set_facecolor("white")
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="center",
        ncol=min(5, len(handles)),
        handlelength=1.35,
        columnspacing=1.15,
        handletextpad=0.45,
    )
    save_figure(fig, output, "tt_average_core_legend")


def plot_average_metric(rows: pd.DataFrame, metric: str, output: Path) -> None:
    sub = rows.loc[rows["metric"].eq(metric)].copy()
    readouts = [key for key in READOUT_ORDER if key in set(sub["readout_key"])]
    inference_order = [key for key in MODEL_ORDER if key in set(sub["inference_key"])]
    x_positions = [0.0]
    for previous, current in zip(readouts[:-1], readouts[1:]):
        if previous in BASELINES and current in BASELINES:
            gap = 0.62
        elif previous in BASELINES:
            gap = 1.34
        else:
            gap = 0.98
        x_positions.append(x_positions[-1] + gap)
    x = np.asarray(x_positions)
    bar_width = min(0.15, 0.72 / max(len(inference_order), 1))
    fig, ax = plt.subplots(figsize=WIDE_SINGLE_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    baseline_styles = {
        "lr_esm": {
            "facecolor": "white",
            "edgecolor": "#222222",
            "linewidth": BAR_EDGE_LINEWIDTH,
        },
        "lr_prostt5": {
            "facecolor": "#f2f2f2",
            "edgecolor": "#666666",
            "hatch": "\\\\\\\\",
        },
    }
    for key, style in baseline_styles.items():
        row = sub.loc[sub["readout_key"].eq(key)]
        if row.empty:
            continue
        row = row.iloc[0]
        xpos = x[readouts.index(key)]
        ax.bar(xpos, row["score_percent"], width=bar_width, zorder=4, **style)
        ax.errorbar(
            xpos,
            row["score_percent"],
            yerr=row["sem_percent"],
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=DATA_LINEWIDTH,
            zorder=6,
        )

    for index, model in enumerate(inference_order):
        offset = (index - (len(inference_order) - 1) / 2.0) * bar_width
        heights = []
        errors = []
        for readout in readouts:
            row = sub.loc[
                sub["readout_key"].eq(readout) & sub["inference_key"].eq(model)
            ]
            heights.append(np.nan if row.empty else row.iloc[0]["score_percent"])
            errors.append(np.nan if row.empty else row.iloc[0]["sem_percent"])
        ax.bar(
            x + offset,
            heights,
            width=bar_width,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=BAR_EDGE_LINEWIDTH,
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            heights,
            yerr=errors,
            fmt="none",
            ecolor="#333333",
            capsize=2.0,
            linewidth=DATA_LINEWIDTH,
            zorder=5,
        )
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0.0, 101.0, 20.0))
    ax.set_ylabel(f"Mean test {metric.upper()}", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [READOUT_LABELS[key] for key in readouts],
        fontsize=TICK_LABEL_SIZE,
        rotation=25,
        ha="right",
    )
    ax.set_xlim(x[0] - 0.46, x[-1] + 0.48)
    clean_axes(ax)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.33, top=0.97)
    save_figure(fig, output, f"tt_average_{metric}_core")


def scatter_models(
    ax: plt.Axes,
    comparison: pd.DataFrame,
    keys: list[str],
    y: np.ndarray,
    size: float,
) -> None:
    for index, key in enumerate(keys):
        ax.scatter(
            100.0 * comparison[key],
            y,
            s=size * (1.16 if key == "s2gae_att_k1_fixed_do04_uni" else 1.0),
            facecolor="white" if key == "lr_esm" else MODEL_COLORS[key],
            edgecolor="#222222",
            linewidth=BAR_EDGE_LINEWIDTH,
            zorder=4 + index,
        )


def delta_color(delta: float, p_value: float) -> str:
    if delta < 0.0:
        return "#777777"
    if p_value < 0.05:
        return MODEL_COLORS["s2gae_att_k1_fixed_do04_uni"]
    return "#111111"


def comparison_bounds(comparison: pd.DataFrame, keys: list[str]) -> tuple[float, float]:
    values = 100.0 * comparison[keys].to_numpy().ravel()
    low = max(0.0, np.floor((values.min() - 4.0) / 5.0) * 5.0)
    high = min(100.0, np.ceil((values.max() + 4.0) / 5.0) * 5.0)
    if high - low < 20.0:
        midpoint = 0.5 * (low + high)
        low = max(0.0, np.floor((midpoint - 10.0) / 5.0) * 5.0)
        high = min(100.0, np.ceil((midpoint + 10.0) / 5.0) * 5.0)
    return low, high


def add_comparison_deltas(ax: plt.Axes, comparison: pd.DataFrame, y: np.ndarray) -> None:
    for y_pos, row in zip(y, comparison.itertuples(index=False)):
        pinnacle_value = "" if pd.isna(row.stars_vs_pinnacle) else str(row.stars_vs_pinnacle)
        esm2_value = "" if pd.isna(row.stars_vs_esm2) else str(row.stars_vs_esm2)
        pinnacle_stars = f" {pinnacle_value}" if pinnacle_value else ""
        esm_stars = f" {esm2_value}" if esm2_value else ""
        ax.text(
            1.035,
            y_pos,
            f"{row.delta_vs_pinnacle_percentage_points:+.1f}{pinnacle_stars}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=TICK_LABEL_SIZE,
            color=delta_color(row.delta_vs_pinnacle_percentage_points, row.p_vs_pinnacle),
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            1.24,
            y_pos,
            f"{row.delta_vs_esm2_percentage_points:+.1f}{esm_stars}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=TICK_LABEL_SIZE,
            color=delta_color(row.delta_vs_esm2_percentage_points, row.p_vs_esm2),
            fontweight="bold",
            clip_on=False,
        )
    for x, label in ((1.035, "Delta vs\nPinnacle"), (1.24, "Delta vs\nESM2")):
        ax.text(
            x,
            1.012,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=TICK_LABEL_SIZE,
            fontweight="bold",
            clip_on=False,
        )


def plot_disease_comparisons(comparison: pd.DataFrame, output: Path) -> None:
    y = np.arange(len(comparison), dtype=float)[::-1]
    original = ["lr_esm", "pinnacle_random_fixed", "s2gae_att_k1_fixed_do04_uni"]
    all_models = ["lr_esm", *MODEL_ORDER]
    x_low, x_high = comparison_bounds(comparison, original)

    fig, ax = plt.subplots(figsize=figure_size(DOUBLE_COLUMN_WIDTH_CM, MAX_FIGURE_HEIGHT_CM))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    scatter_models(ax, comparison, original, y, 54)
    add_comparison_deltas(ax, comparison, y)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(-0.8, len(comparison) + 0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(comparison["disease"], fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel("Test AUPRC (%)")
    ax.set_title("Per-disease performance improvement\n(ProtScape vs baselines)")
    clean_axes(ax)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white" if key == "lr_esm" else MODEL_COLORS[key],
            markeredgecolor="#222222",
            markeredgewidth=BAR_EDGE_LINEWIDTH,
            markersize=7.8 if key == "s2gae_att_k1_fixed_do04_uni" else 7.4,
            label=MODEL_LABELS[key],
        )
        for key in original
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        handletextpad=0.7,
    )
    ax.text(
        0.0,
        -0.14,
        "Mean delta vs Pinnacle: "
        f"{comparison['delta_vs_pinnacle_percentage_points'].mean():+.1f} pp "
        f"({(comparison['p_vs_pinnacle'] < 0.05).sum()}/{len(comparison)} p<0.05)\n"
        "Mean delta vs ESM2: "
        f"{comparison['delta_vs_esm2_percentage_points'].mean():+.1f} pp "
        f"({(comparison['p_vs_esm2'] < 0.05).sum()}/{len(comparison)} p<0.05)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TICK_LABEL_SIZE,
    )
    ax.text(
        1.035,
        -0.14,
        "* p < 0.05\n** p < 0.01\n*** p < 0.001",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TICK_LABEL_SIZE,
        clip_on=False,
    )
    fig.subplots_adjust(left=0.31, right=0.73, bottom=0.38, top=0.88)
    save_figure(
        fig,
        output,
        "tt_pdl_abmil_protscape_vs_pinnacle_and_esm2_auprc",
    )

    fig, ax = plt.subplots(figsize=figure_size(DOUBLE_COLUMN_WIDTH_CM, MAX_FIGURE_HEIGHT_CM))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    scatter_models(ax, comparison, original, y, 88)
    add_comparison_deltas(ax, comparison, y)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(-0.8, len(comparison) + 0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(comparison["disease"], fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel("Test AUPRC (%)", fontsize=AXIS_LABEL_SIZE)
    clean_axes(ax)
    fig.subplots_adjust(left=0.30, right=0.75, bottom=0.11, top=0.93)
    save_figure(
        fig,
        output,
        "tt_pdl_abmil_protscape_vs_pinnacle_and_esm2_auprc_clean",
    )

    all_low, all_high = comparison_bounds(comparison, all_models)
    fig, ax = plt.subplots(figsize=figure_size(DOUBLE_COLUMN_WIDTH_CM, MAX_FIGURE_HEIGHT_CM))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    scatter_models(ax, comparison, all_models, y, 88)
    add_comparison_deltas(ax, comparison, y)
    ax.set_xlim(all_low, all_high)
    ax.set_ylim(-0.8, len(comparison) + 0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(comparison["disease"], fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel("Test AUPRC (%)", fontsize=AXIS_LABEL_SIZE)
    clean_axes(ax)
    fig.subplots_adjust(left=0.30, right=0.75, bottom=0.11, top=0.93)
    save_figure(fig, output, "tt_pdl_abmil_all_models_per_disease_auprc_clean")


def plot_performance(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(PLOT_RC):
        mean = pd.read_csv(source / "mean_performance.csv")
        plot_average_metric(mean, "auprc", output)
        plot_average_legend(mean, output)
        plot_disease_comparisons(
            pd.read_csv(source / "disease_model_comparison.csv"), output
        )
