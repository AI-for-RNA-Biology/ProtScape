#!/usr/bin/env python3
"""Plot CORUM context-analysis results."""

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

CM = 1 / 2.54
AXIS_LINEWIDTH = 0.6
BAR_EDGE_LINEWIDTH = 0.55
DATA_LINEWIDTH = 0.8
TICK_LABEL_SIZE = 6.0
AXIS_LABEL_SIZE = 7.0
PANEL_TITLE_SIZE = 7.0
LEGEND_SIZE = 6.0
SINGLE_COLUMN_WIDTH_CM = 8.8
DOUBLE_COLUMN_WIDTH_CM = 18.0
COMPACT_SINGLE_SIZE = (SINGLE_COLUMN_WIDTH_CM * CM, 6.6 * CM)

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

MAIN_CONTEXT_MODEL_ORDER = [
    "pinnacle_random",
    "pinnacle_esm",
    "pinnacle_acm",
    "gae_bce",
    "s2gae_bce_uni",
]
MAIN_MODEL_ORDER = ["lr_esm", "lr_prostt5"] + MAIN_CONTEXT_MODEL_ORDER
BASELINE_MODELS = {"lr_esm", "lr_prostt5"}
MODEL_LABELS = {
    "lr_esm": "ESM2",
    "lr_prostt5": "ProstT5",
    "pinnacle_random": "Pinnacle",
    "pinnacle_esm": "Pinnacle-ESM2 (GAT)",
    "pinnacle_acm": "Pinnacle-ESM2 (ACM)",
    "gae_bce": "ProtScape-GAE",
    "s2gae_bce_uni": "ProtScape",
    "s2gae_phuber_uni": "ProtScape (pHuber)",
    "s2gae_l1_uni": "ProtScape (L1)",
}
MODEL_COLORS = {
    "lr_esm": "#222222",
    "lr_prostt5": "#666666",
    "pinnacle_random": "#2b2b2b",
    "pinnacle_esm": "#b0b0b0",
    "pinnacle_acm": "#7a7a7a",
    "gae_bce": "#1f77b4",
    "s2gae_bce_uni": "#e6550d",
    "s2gae_phuber_uni": "#b85c00",
    "s2gae_l1_uni": "#54278f",
}

DRIVER_SPECS = [
    {"key": "complex_size", "label": "Complex size"},
    {"key": "cell_ppi_member_coverage", "label": "Cell-PPI node coverage"},
    {"key": "cell_ppi_edge_coverage", "label": "Cell-PPI edge coverage"},
    {"key": "global_ppi_edge_density", "label": "Global PPI edge density"},
]
XMIL_METRICS = [
    {"key": "context_member_coverage", "label": "Cell-PPI node coverage"},
    {"key": "context_edge_coverage", "label": "Cell-PPI edge coverage"},
]



def figure_size(width_cm: float, height_cm: float) -> tuple[float, float]:
    return width_cm * CM, height_cm * CM


def read_table(source: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(source / filename, float_precision="round_trip")


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        output / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


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


def place_bar_y_axis_at_zero(ax: plt.Axes) -> None:
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, length=0)
    ax.axvline(0.0, color="#222222", linewidth=AXIS_LINEWIDTH, zorder=0)


def significance_symbol(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def model_line_handle(model_key: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=MODEL_COLORS[model_key],
        linestyle="--" if model_key in BASELINE_MODELS else "-",
        marker="o",
        markersize=4.0,
        linewidth=DATA_LINEWIDTH,
    )


def plot_driver_legend(output: Path) -> None:
    contextual = [key for key in MAIN_MODEL_ORDER if key not in BASELINE_MODELS]
    baselines = [key for key in MAIN_MODEL_ORDER if key in BASELINE_MODELS]
    fig = plt.figure(figsize=figure_size(DOUBLE_COLUMN_WIDTH_CM, 2.9))
    fig.text(
        0.02,
        0.84,
        "Contextual models",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.76,
        0.84,
        "Context-free baselines",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        ha="left",
    )
    contextual_legend = fig.legend(
        [model_line_handle(key) for key in contextual],
        [MODEL_LABELS[key] for key in contextual],
        loc="center left",
        bbox_to_anchor=(0.02, 0.39),
        ncol=3,
        handlelength=2.0,
        columnspacing=1.1,
        labelspacing=0.65,
    )
    fig.add_artist(contextual_legend)
    fig.legend(
        [model_line_handle(key) for key in baselines],
        [MODEL_LABELS[key] for key in baselines],
        loc="center left",
        bbox_to_anchor=(0.76, 0.39),
        ncol=1,
        handlelength=2.0,
        labelspacing=0.65,
    )
    save_figure(fig, output, "corum_main_competitors_per_complex_legend")


def plot_per_complex_drivers(
    summary: pd.DataFrame,
    bins: pd.DataFrame,
    correlations: pd.DataFrame,
    output: Path,
) -> None:
    scores = 100.0 * summary["mean_score"].dropna().to_numpy(dtype=float)
    ylim = (
        max(0.0, 10.0 * np.floor((scores.min() - 5.0) / 10.0)),
        min(100.0, 10.0 * np.ceil((scores.max() + 5.0) / 10.0)),
    )
    fig = plt.figure(figsize=figure_size(DOUBLE_COLUMN_WIDTH_CM, 10.3))
    grid = fig.add_gridspec(
        2,
        len(DRIVER_SPECS),
        height_ratios=[0.80, 1.12],
        hspace=0.25,
        wspace=0.32,
    )
    trend_axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    rho_axes = [fig.add_subplot(grid[1, index]) for index in range(4)]
    rho_min, rho_max = correlations["spearman"].min(), correlations["spearman"].max()
    rho_xlim = (
        min(-0.15, 0.05 * np.floor((rho_min - 0.03) / 0.05)),
        max(0.30, 0.05 * np.ceil((rho_max + 0.05) / 0.05)),
    )

    for index, driver in enumerate(DRIVER_SPECS):
        key = driver["key"]
        driver_bins = bins[(bins["driver"] == key) & bins["n_complexes"].gt(0)]
        x_lookup = dict(
            zip(driver_bins["driver_bin"], driver_bins["percentile_midpoint"])
        )
        ax = trend_axes[index]
        for model_key in MAIN_MODEL_ORDER:
            model_rows = summary[
                (summary["driver"] == key)
                & (summary["inference_key"] == model_key)
            ].copy()
            model_rows["x"] = model_rows["driver_bin"].map(x_lookup)
            model_rows = model_rows.dropna(subset=["x"]).sort_values("x")
            ax.plot(
                model_rows["x"],
                100.0 * model_rows["mean_score"],
                color=MODEL_COLORS[model_key],
                linestyle="--" if model_key in BASELINE_MODELS else "-",
                marker="o",
                markersize=3.5,
                linewidth=DATA_LINEWIDTH,
                alpha=1.0 if model_key in {"s2gae_bce_uni", "gae_bce"} else 0.78,
                zorder=4 if model_key == "s2gae_bce_uni" else 3,
            )
        ax.set_title(driver["label"], loc="left", fontweight="bold", pad=3.0)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(*ylim)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)
        else:
            ax.set_aspect(1.0 / ax.get_data_ratio())
        ax.set_xlabel("Percentile")
        clean_axes(ax)
        ax.tick_params(labelleft=index == 0)
        if index == 0:
            ax.set_ylabel("Test AUPRC")

        rho_ax = rho_axes[index]
        stat_rows = (
            correlations[correlations["driver"] == key]
            .set_index("inference_key")
            .loc[MAIN_MODEL_ORDER]
            .reset_index()
        )
        positions = np.arange(len(stat_rows), dtype=float)
        for position, stat in zip(positions, stat_rows.itertuples(index=False)):
            rho_ax.barh(
                position,
                stat.spearman,
                height=0.58,
                color=MODEL_COLORS[stat.inference_key],
                edgecolor=(
                    "#333333" if stat.inference_key in BASELINE_MODELS else "none"
                ),
                linewidth=BAR_EDGE_LINEWIDTH,
                hatch="////" if stat.inference_key in BASELINE_MODELS else None,
            )
            symbol = significance_symbol(stat.spearman_p)
            suffix = "" if symbol == "ns" else symbol
            displayed = 0.0 if abs(stat.spearman) < 0.005 else stat.spearman
            text_x = stat.spearman + 0.006 if stat.spearman >= 0 else 0.008
            rho_ax.text(
                text_x,
                position,
                f"{displayed:.2f}{suffix}",
                ha="left",
                va="center",
                fontsize=TICK_LABEL_SIZE,
            )
        rho_ax.set_xlim(*rho_xlim)
        rho_ax.set_ylim(-0.7, len(stat_rows) - 0.3)
        rho_ax.set_yticks(positions)
        rho_ax.invert_yaxis()
        rho_ax.set_yticklabels(
            [MODEL_LABELS[key] for key in MAIN_MODEL_ORDER] if index == 0 else []
        )
        clean_axes(rho_ax)
        place_bar_y_axis_at_zero(rho_ax)
        rho_ax.tick_params(axis="y", labelleft=index == 0)
        rho_ax.set_xlabel("Spearman $\\rho$")

    fig.subplots_adjust(left=0.15, right=0.93, bottom=0.11, top=0.96)
    save_figure(fig, output, "corum_main_competitors_auprc_driver_summary")


def plot_xmil_summary(
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    output: Path,
) -> None:
    summary = summary[summary["evidence_score"] == "complete_positive_lrp"].copy()
    correlations = correlations[
        correlations["evidence_score"] == "complete_positive_lrp"
    ].copy()
    lower = 100.0 * summary["mean_lrp"].min()
    upper = 100.0 * summary["mean_lrp"].max()
    padding = max(0.08, 0.10 * (upper - lower))
    ylim = min(0.0, lower - padding), max(0.0, upper + padding)

    fig = plt.figure(figsize=figure_size(DOUBLE_COLUMN_WIDTH_CM, 13.0))
    grid = fig.add_gridspec(
        2,
        len(XMIL_METRICS),
        height_ratios=[1.0, 1.12],
        hspace=0.25,
        wspace=0.34,
    )
    trend_axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    rho_axes = [fig.add_subplot(grid[1, index]) for index in range(2)]
    rho_min = float(correlations["median_spearman"].min())
    rho_max = float(correlations["median_spearman"].max())
    rho_xlim = (
        min(-0.05, 0.05 * np.floor((rho_min - 0.02) / 0.05)),
        max(0.46, 0.05 * np.ceil((rho_max + 0.06) / 0.05)),
    )

    for index, metric in enumerate(XMIL_METRICS):
        trend_ax, rho_ax = trend_axes[index], rho_axes[index]
        for model_key in MAIN_CONTEXT_MODEL_ORDER:
            model_rows = summary[
                (summary["inference_key"] == model_key)
                & (summary["coverage_metric"] == metric["key"])
            ].sort_values("percentile_midpoint")
            trend_ax.plot(
                model_rows["percentile_midpoint"],
                100.0 * model_rows["mean_lrp"],
                color=MODEL_COLORS[model_key],
                marker="o",
                markersize=3.2,
                linewidth=DATA_LINEWIDTH,
                alpha=1.0 if model_key in {"s2gae_bce_uni", "gae_bce"} else 0.82,
            )
        trend_ax.set_title(metric["label"], loc="left", fontweight="bold", pad=3.0)
        trend_ax.set_xlim(0, 100)
        trend_ax.set_xticks([0, 50, 100])
        trend_ax.set_ylim(*ylim)
        trend_ax.set_xlabel("Coverage percentile across complex-context pairs")
        clean_axes(trend_ax)
        trend_ax.axhline(0.0, color="#999999", linewidth=AXIS_LINEWIDTH, zorder=0)
        trend_ax.tick_params(labelleft=index == 0)
        if index == 0:
            trend_ax.set_ylabel("Mean context attribution\nin bin (%)")

        stat_rows = (
            correlations[correlations["coverage_metric"] == metric["key"]]
            .set_index("inference_key")
            .loc[MAIN_CONTEXT_MODEL_ORDER]
            .reset_index()
        )
        positions = np.arange(len(stat_rows), dtype=float)
        for position, stat in zip(positions, stat_rows.itertuples(index=False)):
            rho_ax.barh(
                position,
                stat.median_spearman,
                height=0.58,
                color=MODEL_COLORS[stat.inference_key],
                edgecolor="none",
            )
            rho_ax.text(
                stat.median_spearman + 0.008,
                position,
                f"{stat.median_spearman:.2f}",
                ha="left",
                va="center",
                fontsize=TICK_LABEL_SIZE,
            )
        rho_ax.set_xlim(*rho_xlim)
        rho_ax.set_ylim(-0.7, len(stat_rows) - 0.3)
        rho_ax.set_yticks(positions)
        rho_ax.set_yticklabels(
            [MODEL_LABELS[key] for key in MAIN_CONTEXT_MODEL_ORDER]
            if index == 0
            else []
        )
        rho_ax.invert_yaxis()
        clean_axes(rho_ax)
        place_bar_y_axis_at_zero(rho_ax)
        rho_ax.tick_params(axis="y", labelleft=index == 0)
        rho_ax.set_xlabel(
            "Median per-complex $\\rho$:\n"
            "evidence vs coverage across contexts"
        )

    fig.subplots_adjust(left=0.25, right=0.975, bottom=0.12, top=0.96)
    save_figure(fig, output, "corum_xmil_main_models_context_summary")


def plot_xmil_legend(output: Path) -> None:
    fig = plt.figure(figsize=figure_size(DOUBLE_COLUMN_WIDTH_CM, 1.9))
    fig.legend(
        [model_line_handle(key) for key in MAIN_CONTEXT_MODEL_ORDER],
        [MODEL_LABELS[key] for key in MAIN_CONTEXT_MODEL_ORDER],
        loc="center",
        ncol=3,
        handlelength=2.0,
        columnspacing=1.2,
    )
    save_figure(fig, output, "corum_xmil_main_models_complete_positive_legend")


def plot_context_analysis(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(PLOT_RC):
        plot_per_complex_drivers(
            read_table(source, "corum_per_complex_driver_summary.csv"),
            read_table(source, "corum_per_complex_driver_bins.csv"),
            read_table(source, "corum_per_complex_driver_correlations.csv"),
            output,
        )
        plot_driver_legend(output)

        plot_xmil_summary(
            read_table(source, "corum_xmil_binned_summary.csv"),
            read_table(source, "corum_xmil_correlation_summary.csv"),
            output,
        )
        plot_xmil_legend(output)
