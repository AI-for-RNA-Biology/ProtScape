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
BAR_EDGE_COLOR = "#222222"
BAR_EDGE_LINEWIDTH = 0.55
DATA_LINEWIDTH = 0.8
TICK_LABEL_SIZE = 6.0
AXIS_LABEL_SIZE = 8.0
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
    "figure.titlesize": PANEL_TITLE_SIZE,
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
    "savefig.format": "pdf",
    "svg.fonttype": "none",
    "savefig.dpi": 300,
    "savefig.transparent": True,
}

READOUT_ORDER = [
    "lr_esm",
    "lr_prostt5",
    "lr_hc_cell",
    "lr_hc_cell_esm",
    "abmil8_pdl_hc_cell",
    "abmil8_pdl_id2_dropout",
]
READOUT_LABELS = {
    "lr_esm": "ESM2",
    "lr_prostt5": "ProstT5",
    "lr_hc_cell": "Mean-MIL\nContextual",
    "lr_hc_cell_esm": "Mean-MIL\nContextual + ESM2",
    "abmil8_pdl_hc_cell": "ABMIL\nContextual",
    "abmil8_pdl_id2_dropout": "ABMIL\nContextual + ESM2",
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
    "s2gae_phuber_uni": "ProtScape (pHuber)",
    "s2gae_l1_uni": "ProtScape (L1)",
}
MODEL_COLORS = {
    "lr_esm": "#222222",
    "lr_prostt5": "#666666",
    "pinnacle_random_fixed": "#2b2b2b",
    "pinnacle_esm_fixed": "#b0b0b0",
    "pinnacle_esm2_acm": "#7a7a7a",
    "gae_att_fixed_do06": "#1f77b4",
    "s2gae_att_k1_fixed_do04_uni": "#e6550e",
    "s2gae_phuber_uni": "#ff1529",
    "s2gae_l1_uni": "#9200bf",
}

LOSS_MODEL_ORDER = [
    "s2gae_att_k1_fixed_do04_uni",
    "s2gae_phuber_uni",
    "s2gae_l1_uni",
]
LOSS_MODEL_LABELS = {
    "s2gae_att_k1_fixed_do04_uni": "BCE",
    "s2gae_phuber_uni": "pHuber",
    "s2gae_l1_uni": "L1",
}
LOSS_READOUT_ORDER = [
    "lr_hc_cell",
    "lr_hc_cell_esm",
    "abmil8_pdl_id2_dropout",
]
LOSS_READOUT_LABELS = {
    "lr_hc_cell": "Mean-MIL\n(1)",
    "lr_hc_cell_esm": "Mean-MIL\n(2)",
    "abmil8_pdl_id2_dropout": "ABMIL\n(2)",
}
LOSS_CONTEXT_NOTE = "(1) Contextual     (2) Contextual + ESM2"
N_DISEASES = 15


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
    sub = rows.loc[
        rows["metric"].eq(metric)
        & rows["inference_key"].isin([*BASELINES, *MODEL_ORDER])
    ].copy()
    readouts = [key for key in READOUT_ORDER if key in set(sub["readout_key"])]
    inference_order = [key for key in MODEL_ORDER if key in set(sub["inference_key"])]
    bar_width = 0.25
    group_widths = {
        key: bar_width if key in BASELINES else bar_width * len(inference_order)
        for key in readouts
    }
    x_positions = [0.0]
    for previous, current in zip(readouts[:-1], readouts[1:]):
        gap = (
            group_widths[previous] + group_widths[current]
        ) / 2.0 + bar_width * 1.3
        x_positions.append(x_positions[-1] + gap)
    x = np.asarray(x_positions)
    figure_width_cm = DOUBLE_COLUMN_WIDTH_CM * (0.40 + bar_width * 0.7)
    fig, ax = plt.subplots(figsize=figure_size(figure_width_cm, 5.5))
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
            edgecolor=BAR_EDGE_COLOR,
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
    y_max = float(np.nanmax(sub["score_percent"].to_numpy(dtype=float)))
    y_top = min(100.0, np.ceil((y_max + 2.0) / 5.0) * 5.0)
    y_min = 10.0
    ax.set_ylim(y_min, max(y_top, y_min))
    ax.set_yticks(np.asarray([10.0, 30.0, 50.0, 70.0]))
    metric_label = "macro F1" if metric == "f1" else metric.upper()
    ax.set_ylabel(
        f"Mean {metric_label} across diseases (%)",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [READOUT_LABELS[key] for key in readouts],
        fontsize=TICK_LABEL_SIZE,
        rotation=0,
        ha="center",
    )
    left_margin = group_widths[readouts[0]] / 2.0 + bar_width * 0.3
    right_margin = group_widths[readouts[-1]] / 2.0 + bar_width * 0.3
    ax.set_xlim(x[0] - left_margin, x[-1] + right_margin)
    clean_axes(ax)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.34, top=0.97)
    save_figure(fig, output, f"tt_average_{metric}_core")


def loss_rows(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = rows.loc[
        rows["metric"].eq(metric)
        & rows["inference_key"].isin(LOSS_MODEL_ORDER)
        & rows["readout_key"].isin(LOSS_READOUT_ORDER)
    ].copy()
    expected = pd.MultiIndex.from_product(
        [LOSS_READOUT_ORDER, LOSS_MODEL_ORDER],
        names=["readout_key", "inference_key"],
    )
    coverage = sub.set_index(["readout_key", "inference_key"])["n_tasks"].reindex(
        expected
    )
    incomplete = coverage[coverage.ne(N_DISEASES)]
    if not incomplete.empty:
        raise ValueError(
            f"The {metric.upper()} loss benchmark requires all {N_DISEASES} diseases:\n"
            + incomplete.to_string()
        )
    return sub


def add_loss_legend(ax: plt.Axes) -> None:
    handles = [
        Patch(
            facecolor=MODEL_COLORS[key],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LINEWIDTH,
            label=LOSS_MODEL_LABELS[key],
        )
        for key in LOSS_MODEL_ORDER
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        title="(1) Contextual\n(2) Contextual + ESM2",
        title_fontsize=LEGEND_SIZE,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )


def plot_loss_metric(
    rows: pd.DataFrame,
    metric: str,
    output: Path,
    *,
    include_legend: bool = False,
) -> None:
    sub = loss_rows(rows, metric)
    bar_width = 0.25
    group_width = bar_width * len(LOSS_MODEL_ORDER)
    x_positions = [0.0]
    for _ in LOSS_READOUT_ORDER[1:]:
        x_positions.append(x_positions[-1] + group_width + bar_width * 1.3)
    x = np.asarray(x_positions)

    margin = group_width / 2.0 + bar_width * 0.3
    loss_axis_span = (x[-1] - x[0]) + 2.0 * margin
    main_axis_span = 7.275
    main_width_cm = DOUBLE_COLUMN_WIDTH_CM * (0.40 + bar_width * 0.7)
    figure_width_cm = main_width_cm * loss_axis_span / main_axis_span
    if include_legend:
        figure_width_cm *= (0.98 - 0.08) / (0.72 - 0.08)

    fig, ax = plt.subplots(figsize=figure_size(figure_width_cm, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for index, model in enumerate(LOSS_MODEL_ORDER):
        offset = (index - (len(LOSS_MODEL_ORDER) - 1) / 2.0) * bar_width
        values = sub.loc[sub["inference_key"].eq(model)].set_index("readout_key")
        values = values.reindex(LOSS_READOUT_ORDER)
        heights = values["score_percent"].to_numpy(dtype=float)
        errors = values["sem_percent"].to_numpy(dtype=float)
        ax.bar(
            x + offset,
            heights,
            width=bar_width,
            color=MODEL_COLORS[model],
            edgecolor=BAR_EDGE_COLOR,
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

    y_upper = 82.0 if metric == "f1" else 80.0
    ax.set_ylim(40.0, y_upper)
    ax.set_yticks(np.arange(40.0, 80.1, 10.0))
    metric_label = "macro F1" if metric == "f1" else metric.upper()
    ax.set_ylabel(
        f"Mean {metric_label} across diseases (%)",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [LOSS_READOUT_LABELS[key] for key in LOSS_READOUT_ORDER],
        fontsize=TICK_LABEL_SIZE,
        rotation=0,
        ha="center",
    )
    ax.set_xlim(x[0] - margin, x[-1] + margin)
    clean_axes(ax)
    if include_legend:
        add_loss_legend(ax)
        fig.subplots_adjust(left=0.08, right=0.72, bottom=0.34, top=0.97)
        suffix = "_with_legend"
    else:
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.34, top=0.97)
        suffix = ""
    save_figure(
        fig,
        output,
        f"tt_average_{metric}_loss_model_supplement{suffix}",
    )


def plot_loss_legend(output: Path) -> None:
    handles = [
        Patch(
            facecolor=MODEL_COLORS[key],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LINEWIDTH,
            label=LOSS_MODEL_LABELS[key],
        )
        for key in LOSS_MODEL_ORDER
    ]
    fig = plt.figure(figsize=figure_size(SINGLE_COLUMN_WIDTH_CM, 1.6))
    fig.patch.set_facecolor("white")
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=len(handles),
        handlelength=1.35,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.text(
        0.5,
        0.22,
        LOSS_CONTEXT_NOTE,
        ha="center",
        va="center",
        fontsize=LEGEND_SIZE,
    )
    save_figure(fig, output, "loss_model_supplement_legend")


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
        plot_average_metric(mean, "f1", output)
        plot_average_legend(mean, output)
        for metric in ("auprc", "f1"):
            plot_loss_metric(mean, metric, output)
            plot_loss_metric(mean, metric, output, include_legend=True)
        plot_loss_legend(output)
        plot_disease_comparisons(
            pd.read_csv(source / "disease_model_comparison.csv"), output
        )
