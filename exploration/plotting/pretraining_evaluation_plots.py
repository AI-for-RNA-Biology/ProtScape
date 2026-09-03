#!/usr/bin/env python3
"""Plot pretraining evaluation results."""

from __future__ import annotations

import argparse
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
from matplotlib.ticker import LogFormatterMathtext

from exploration.analysis.pretraining_tables import (
    CONTEXT_FREE_MODEL_KEY,
    add_context_free_curve,
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
    CONTEXT_FREE_MODEL_KEY: "Context-free ProtScape",
}
MODEL_COLORS = {
    "pinnacle_random": "#2b2b2b",
    "pinnacle_esm2_acm": "#7a7a7a",
    "pinnacle_esm2": "#b0b0b0",
    "gae_att": "#1f77b4",
    "s2gae_att_k1_uni": "#e6550d",
    CONTEXT_FREE_MODEL_KEY: "#d31529",
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
    "s2gae_att_k1_l1_do00": "#9200bf",
    "s2gae_att_k1_phuber": "#d31529",
}


PRETRAINING_RC = {
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
    "savefig.format": "pdf",
    "savefig.transparent": True,
}


def read_table(source: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(source / filename)


def save_pretraining_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    options = {"dpi": 300, "bbox_inches": "tight", "transparent": True}
    fig.savefig(output / f"{stem}.svg", **options)
    fig.savefig(output / f"{stem}.pdf", **options)
    fig.savefig(output / f"{stem}.png", **options)
    plt.close(fig)
def clean_pretraining_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#222222")
        ax.spines[side].set_linewidth(matplotlib.rcParams["axes.linewidth"])
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=matplotlib.rcParams["xtick.labelsize"],
        width=matplotlib.rcParams["xtick.major.width"],
        length=matplotlib.rcParams["xtick.major.size"],
        pad=matplotlib.rcParams["xtick.major.pad"],
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=matplotlib.rcParams["xtick.minor.width"],
        length=matplotlib.rcParams["xtick.minor.size"],
    )


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
    ax.set_xticklabels(["1", "10", "50", "100", "500"])
    ax.set_xlabel("Negatives per positive edge")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*percent_limits(table["score_percent"]))
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(value) for value in yticks])
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
    fig, ax = plt.subplots(figsize=(5.0 * CM, 5.5 * CM))
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


def plot_core_model_legend(output: Path, order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.8 * CM, 4.5 * CM))
    ax.axis("off")
    ax.legend(
        model_handles(order, MODEL_COLORS, 2.5),
        [MODEL_LABELS[key] for key in order],
        frameon=False,
        loc="center",
        ncol=len(order),
        handlelength=1.8,
    )
    save_pretraining_figure(fig, output, "core_model_legend")


def plot_metagraph_barplot(table: pd.DataFrame, output: Path, stem: str) -> None:
    rows = table.set_index("model_key").reindex(CORE_MODEL_ORDER)
    values = rows["metagraph_percent"].to_numpy(dtype=float)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.75 * CM, 5.5 * CM))
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
    ax.set_xticklabels([""] * len(rows))
    ax.set_xlabel("Models")
    ax.set_ylabel("Metagraph AUPRC")
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
            fontsize=matplotlib.rcParams["xtick.labelsize"],
        )
    clean_pretraining_axis(ax)
    plt.tight_layout()
    save_pretraining_figure(fig, output, stem)


def plot_metagraph_metric_scatter(
    table: pd.DataFrame,
    output: Path,
    stem: str,
) -> None:
    """Plot metagraph F1 against metagraph AUPRC for the released models."""
    pivot = (
        table[table["metric"].isin(["ap", "f1"])]
        .pivot(index="model_key", columns="metric", values="metagraph_score")
        .reindex(CORE_MODEL_ORDER)
    )
    values = 100.0 * pivot[["ap", "f1"]]

    fig, ax = plt.subplots(figsize=(4.5 * CM, 5 * CM))
    for key in CORE_MODEL_ORDER:
        x = values.loc[key, "f1"]
        y = values.loc[key, "ap"]
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter(x, y, s=70, color=MODEL_COLORS[key], linewidths=0)

    ax.set_xlabel("1:1 F1 (%)")
    ax.set_ylabel("1:1 AUPRC (%)")
    ax.set_xlim(*percent_limits(values["f1"]))
    ax.set_ylim(*percent_limits(values["ap"]))
    ax.set_xticks([84, 88, 92, 96])
    clean_pretraining_axis(ax)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.16, top=0.96)
    save_pretraining_figure(fig, output, stem)


def plot_parameter_counts(table: pd.DataFrame, output: Path, stem: str) -> None:
    rows = table.set_index("model_key").reindex(CORE_MODEL_ORDER)
    values = rows["parameter_count"].to_numpy(dtype=float)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.75 * CM, 5.5 * CM))
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
    ax.set_xticklabels([""] * len(rows))
    ax.set_xlabel("Models")
    ax.set_yscale("log")
    ax.set_ylabel("Trainable parameters")
    ax.set_ylim(max(1e7, float(values.min()) / 2), float(values.max()) * 1.85)
    ax.set_yticks([1e7, 1e8, 1e9])
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
    clean_pretraining_axis(ax)
    plt.tight_layout()
    save_pretraining_figure(fig, output, stem)


def plot_loss_legend(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4 * CM, 4.5 * CM))
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
        handlelength=1.8,
    )
    save_pretraining_figure(fig, output, "loss_sensitivity_legend")


def plot_pretraining(
    source: str | Path,
    output: str | Path,
    context_free_metrics: str | Path | None = None,
) -> None:
    """Render the main pretraining evaluation plots."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panel_a_auprc = read_table(source, "robust_ppi_auprc.csv")
    panel_a_f1 = read_table(source, "robust_ppi_f1.csv")
    core_ppi_model_order = list(CORE_MODEL_ORDER)
    if context_free_metrics is not None:
        panel_a_auprc = add_context_free_curve(
            panel_a_auprc, context_free_metrics, "ap"
        )
        panel_a_f1 = add_context_free_curve(panel_a_f1, context_free_metrics, "f1")
        core_ppi_model_order.append(CONTEXT_FREE_MODEL_KEY)
    panel_b = read_table(source, "metagraph_auprc.csv")
    metagraph_metrics = read_table(source, "metagraph_metrics.csv")
    panel_c = read_table(source, "parameter_counts.csv")
    panel_d_auprc = read_table(source, "loss_sensitivity_auprc.csv")
    panel_d_f1 = read_table(source, "loss_sensitivity_f1.csv")

    with plt.rc_context(PRETRAINING_RC):
        plot_curve(
            panel_a_auprc,
            output,
            "core_robust_ppi_auprc",
            core_ppi_model_order,
            MODEL_COLORS,
            MODEL_LABELS,
            "AUPRC",
            [0, 20, 40, 60, 80, 100],
        )
        plot_curve(
            panel_a_f1,
            output,
            "core_robust_ppi_f1",
            core_ppi_model_order,
            MODEL_COLORS,
            MODEL_LABELS,
            "F1",
            [0, 20, 40, 60, 80],
        )
        plot_core_model_legend(output, core_ppi_model_order)

        plot_metagraph_barplot(panel_b, output, "metagraph_barplot_auprc")
        plot_metagraph_metric_scatter(
            metagraph_metrics,
            output,
            "core_metagraph_f1_vs_ap",
        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-free-metrics", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_pretraining(
        args.source,
        args.output,
        context_free_metrics=args.context_free_metrics,
    )


if __name__ == "__main__":
    main()
