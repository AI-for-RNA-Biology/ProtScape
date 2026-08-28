#!/usr/bin/env python3
"""Plot pretraining diagnostics from analysis tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from exploration.plotting.fonts import register_arial

register_arial()
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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

matplotlib.rcParams.update(SOURCE_STYLE)


MODEL_ORDER = [
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

POOLING_ORDER = ["gae_att", "gae_vn", "gae_learnedvn"]
POOLING_LABELS = {
    "gae_att": "Attention",
    "gae_vn": "Virtual node",
    "gae_learnedvn": "Learned virtual node",
}
POOLING_COLORS = {
    "gae_att": "#1f77b4",
    "gae_vn": "#74c476",
    "gae_learnedvn": "#238b45",
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
    fig.savefig(output / f"{stem}.svg", dpi=300, **kwargs)
    fig.savefig(output / f"{stem}.pdf", dpi=pdf_dpi, **kwargs)
    fig.savefig(output / f"{stem}.png", dpi=300, **kwargs)
    plt.close(fig)


def clean_2d_axis(ax):
    """Axis styling shared by the original sensitivity plots."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.spines["left"].set_linewidth(matplotlib.rcParams["axes.linewidth"])
    ax.spines["bottom"].set_linewidth(matplotlib.rcParams["axes.linewidth"])
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=matplotlib.rcParams["xtick.labelsize"],
        width=matplotlib.rcParams["xtick.major.width"],
        length=matplotlib.rcParams["xtick.major.size"],
        pad=matplotlib.rcParams["xtick.major.pad"],
    )


def plot_contextwise_boxplots(legend_ax, axes, auprc, f1):
    legend_order = [
        "pinnacle_random",
        "pinnacle_esm2_acm",
        "pinnacle_esm2",
        "gae_att",
        "s2gae_att_k1_uni",
    ]
    handles = [Line2D([0], [0], color=MODEL_COLORS[key], linewidth=2.5) for key in legend_order]
    legend_ax.axis("off")
    legend_ax.legend(
        handles,
        [MODEL_LABELS[key] for key in legend_order],
        ncol=len(legend_order),
        loc="center",
        frameon=False,
        handlelength=1.8,
    )

    for ax, table, ylabel, yticks in zip(
        axes,
        [auprc, f1],
        ["[1:1]  AUPRC (%)", "[1:1]  F1 (%)"],
        [[50, 60, 70, 80, 90], [40, 50, 60, 70, 80]],
    ):
        values = [
            table.loc[table["model_key"] == key, "score"].dropna().to_numpy()
            for key in MODEL_ORDER
        ]
        boxes = ax.boxplot(
            values,
            positions=np.arange(len(MODEL_ORDER)),
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.0},
            whiskerprops={"color": "#777777", "linewidth": 0.9},
            capprops={"color": "#777777", "linewidth": 0.9},
        )
        for patch, key in zip(boxes["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[key])
            patch.set_edgecolor(MODEL_COLORS[key])
            patch.set_linewidth(1.0)

        scores = table["score"].dropna().to_numpy()
        ax.set_ylim(max(0, scores.min() - 2), min(100, scores.max() + 2))
        ax.set_yticks(yticks)
        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels([""] * len(MODEL_ORDER))
        ax.set_xlabel("Models")
        ax.set_ylabel(ylabel)
        clean_2d_axis(ax)


def plot_pooling(legend_ax, ax, table):
    legend_ax.axis("off")
    handles = [Line2D([0], [0], color=POOLING_COLORS[key], linewidth=2.2) for key in POOLING_ORDER]
    legend_ax.legend(
        handles,
        [POOLING_LABELS[key] for key in POOLING_ORDER],
        ncol=3,
        loc="center",
        frameon=False,
    )

    metrics = ["PPI - AUPRC", "PPI - F1", "Metagraph - AUPRC", "Metagraph - F1"]
    offsets = {"gae_att": -0.18, "gae_vn": 0.0, "gae_learnedvn": 0.18}
    for boundary in [0.5, 1.5, 2.5]:
        ax.axvline(boundary, color="#9f9f9f", linestyle=":", linewidth=1.2, zorder=1)
    for key in POOLING_ORDER:
        scores = table.loc[table["model_key"] == key].set_index("metric").loc[metrics, "score"]
        xpos = np.arange(4) + offsets[key]
        ax.scatter(
            xpos,
            scores,
            s=40,
            color=POOLING_COLORS[key],
            linewidths=0,
            label=POOLING_LABELS[key],
            zorder=3,
        )
        ax.hlines(
            scores,
            xpos - 0.055,
            xpos + 0.055,
            color=POOLING_COLORS[key],
            linestyle="--" if key in {"gae_vn", "gae_learnedvn"} else "-",
            linewidth=2.2,
            zorder=3,
        )

    ax.set_ylim(55, 100)
    ax.set_yticks([55, 70, 85, 100])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["AUPRC", "F1", "AUPRC", "F1"])
    ax.set_ylabel("[1:1] Score (%)")
    ax.text(0.25, 1.03, "PPI", transform=ax.transAxes, ha="center", va="bottom")
    ax.text(0.75, 1.03, "Metagraph", transform=ax.transAxes, ha="center", va="bottom")
    clean_2d_axis(ax)


def plot_pretraining(source, output):
    """Render model-performance and pooling diagnostics."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    auprc = pd.read_csv(source / "contextwise_ppi_auprc.csv")
    f1 = pd.read_csv(source / "contextwise_ppi_f1.csv")
    pooling = pd.read_csv(source / "pooling_sensitivity.csv")

    a_legend_fig, a_legend_ax = plt.subplots(figsize=(6.8 * CM, 4.5 * CM))
    a_auprc_fig, a_auprc_ax = plt.subplots(figsize=(3.5 * CM, 4.5 * CM))
    a_f1_fig, a_f1_ax = plt.subplots(figsize=(3.5 * CM, 4.5 * CM))
    plot_contextwise_boxplots(
        a_legend_ax, [a_auprc_ax, a_f1_ax], auprc, f1
    )
    save_original(a_legend_fig, output, "contextwise_core_model_legend")
    save_original(a_auprc_fig, output, "contextwise_ppi_auprc")
    save_original(a_f1_fig, output, "contextwise_ppi_f1")

    b_legend_fig, b_legend_ax = plt.subplots(figsize=(7 * CM, 2.2 * CM))
    b_fig, b_ax = plt.subplots(figsize=(5 * CM, 4.5 * CM))
    plot_pooling(b_legend_ax, b_ax, pooling)
    b_fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.86)
    save_original(b_legend_fig, output, "pooling_sensitivity_legend")
    save_original(b_fig, output, "pooling_sensitivity")
