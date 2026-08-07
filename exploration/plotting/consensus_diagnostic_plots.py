#!/usr/bin/env python3
"""Plot loss-consensus diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from exploration.plotting.fonts import register_arial

register_arial()
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


CLASS_ORDER = [
    "Consensus negative",
    "Weak disagreement negative",
    "Strong disagreement negative",
    "Strong disagreement positive",
    "Weak disagreement positive",
    "Consensus positive",
]
CLASS_LABELS = {
    "Consensus negative": "(---)",
    "Weak disagreement negative": "(--)",
    "Strong disagreement negative": "(-)",
    "Strong disagreement positive": "(+)",
    "Weak disagreement positive": "(++)",
    "Consensus positive": "(+++)",
}
CLASS_COLORS = {
    name: mcolors.to_hex(matplotlib.cm.get_cmap("seismic")(position))
    for name, position in zip(CLASS_ORDER, [0.02, 0.18, 0.40, 0.60, 0.82, 0.98])
}
EDGE_LABELS = ["Labelled positive", "Labelled negative"]


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


def class_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CLASS_COLORS[name],
            markeredgecolor="none",
            markersize=6,
            label=CLASS_LABELS[name],
        )
        for name in CLASS_ORDER
    ]


def clean_3d_axis(ax):
    pad = 0.0
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["linewidth"] = 0
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#d9d9d9")
        if "ticklabel" in axis._axinfo:
            axis._axinfo["ticklabel"]["space"] = 0
            axis._axinfo["ticklabel"]["va"] = "center"
        if "label" in axis._axinfo:
            axis._axinfo["label"]["space"] = 0
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=matplotlib.rcParams["xtick.labelsize"],
        width=matplotlib.rcParams["xtick.major.width"],
        pad=pad,
    )
    ax.xaxis.set_tick_params(pad=pad)
    ax.yaxis.set_tick_params(pad=pad)
    ax.zaxis.set_tick_params(pad=pad)


def add_threshold_planes(ax):
    grid = np.array([0.0, 1.0])
    yy, zz = np.meshgrid(grid, grid)
    xx, zz_y = np.meshgrid(grid, grid)
    xx_z, yy_z = np.meshgrid(grid, grid)
    style = {"color": "#999999", "alpha": 0.06, "linewidth": 0, "shade": False}
    ax.plot_surface(np.full_like(yy, 0.5), yy, zz, **style)
    ax.plot_surface(xx, np.full_like(xx, 0.5), zz_y, **style)
    ax.plot_surface(xx_z, yy_z, np.full_like(xx_z, 0.5), **style)


def plot_3d_legend(legend_ax):
    legend_ax.axis("off")
    legend = legend_ax.legend(
        handles=class_legend_handles(),
        title="Loss class",
        ncol=1,
        loc="center",
        frameon=False,
        fontsize=matplotlib.rcParams["legend.fontsize"],
        title_fontsize=matplotlib.rcParams["font.size"],
        labelspacing=2.2,
    )
    for label in legend.get_texts():
        label.set_fontweight("bold")


def plot_3d_score_space(table, *, show_values=True):
    if show_values:
        fig = plt.figure(figsize=(16 * CM, 8 * CM))
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.92, wspace=0.08)
    else:
        fig = plt.figure(figsize=(10 * CM, 5 * CM))

    for index, edge_label in enumerate(EDGE_LABELS, start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        left = 0.06 + (index - 1) * 0.48
        ax.set_position([left, 0.14, 0.47, 0.78])
        labelled = table[table["edge_label"] == edge_label]
        for class_name in CLASS_ORDER:
            rows = labelled[labelled["consensus_class"] == class_name]
            ax.scatter(
                rows["BCE"],
                rows["pHuber"],
                rows["L1"],
                s=0.65,
                alpha=0.10,
                color=CLASS_COLORS[class_name],
                linewidths=0,
                rasterized=False,
            )
        add_threshold_planes(ax)
        ax.set_title(edge_label)
        if show_values:
            ax.set_xlabel("BCE", labelpad=-1.0)
            ax.set_ylabel("pHuber", labelpad=-1.0)
            ax.set_zlabel("")
            ax.text2D(
                0.86,
                0.54,
                "L1",
                transform=ax.transAxes,
                fontsize=matplotlib.rcParams["axes.labelsize"],
            )
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_zticks([0, 0.5, 1])
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
        try:
            ax.set_proj_type("ortho")
        except AttributeError:
            pass
        ax.view_init(elev=22, azim=-42)
        if show_values:
            clean_3d_axis(ax)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.tick_params(
                labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False,
            )

    return fig


def plot_mean_std(axes, table):
    for ax, edge_label in zip(axes, EDGE_LABELS):
        labelled = table[table["edge_label"] == edge_label]
        for class_name in CLASS_ORDER:
            rows = labelled[labelled["consensus_class"] == class_name]
            ax.scatter(
                rows["prediction_mean"],
                rows["prediction_std"],
                s=0.65,
                alpha=0.10,
                color=CLASS_COLORS[class_name],
                linewidths=0,
                rasterized=False,
            )
        ax.axvline(0.5, color="#222222", linewidth=1.0, linestyle="--", alpha=0.75)
        ax.set_title(edge_label)
        ax.set_xlabel("Prediction mean")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        clean_2d_axis(ax)
    axes[0].set_ylabel("Prediction std")


def plot_loss_similarity(axes, table):
    colors = {"Labelled positives": CLASS_COLORS["Consensus positive"],
              "Labelled negatives": CLASS_COLORS["Consensus negative"]}

    panels = [
        ("Binned Spearman score correlation", "Before thresholding"),
        ("Thresholded Spearman call correlation", "After thresholding"),
    ]
    comparisons = ["BCE vs pHuber", "BCE vs L1"]
    groups = ["Labelled positives", "Labelled negatives"]
    x = np.arange(2)
    width = 0.55 / 2
    legend_handles = None
    legend_labels = None
    for ax, (metric, title) in zip(axes, panels):
        rows = table[table["metric"] == metric]
        offsets = np.linspace(-width * (len(groups) - 1) / 2, width * (len(groups) - 1) / 2, len(groups))
        for offset, group in zip(offsets, groups):
            values = rows[rows["group"] == group].set_index("comparison").loc[comparisons, "value"] * 100
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=colors[group],
                edgecolor="#222222",
                linewidth=0.55,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 2.5,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["pHuber", "L1"])
        ax.set_ylim(0, 108)
        clean_2d_axis(ax)
        if legend_handles is None:
            legend_handles = [
                Patch(
                    facecolor=colors[group],
                    edgecolor="#222222",
                    linewidth=0.55,
                    label=group,
                )
                for group in groups
            ]
            legend_labels = groups
    axes[0].set_ylabel(r"$\rho$ (BCE, loss) (%)")
    return legend_handles, legend_labels


def plot_consensus(source, output):
    """Render loss-score geometry and agreement diagnostics."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    score_sample = pd.read_csv(source / "score_sample.csv.gz")
    sample_summary = pd.read_csv(source / "score_sample_summary.csv")
    loss_similarity = pd.read_csv(source / "loss_similarity.csv")

    observed = score_sample.groupby(["edge_label", "consensus_class"]).size()
    expected = sample_summary.set_index(["label", "loss_class"])["edge_contexts_sampled"]
    if not observed.sort_index().equals(expected.sort_index()):
        raise ValueError("Score sample does not match its sample summary")

    c_legend_fig, c_legend_ax = plt.subplots(figsize=(2 * CM, 4.5 * CM), dpi=300)
    c_legend_ax.set_position([0.0, 0.0, 1.0, 1.0])
    c_legend_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    plot_3d_legend(c_legend_ax)
    save_original(c_legend_fig, output, "loss_score_space_3d_legend", pad_inches=0.02, pdf_dpi=100)

    c_fig = plot_3d_score_space(score_sample, show_values=True)
    save_original(c_fig, output, "loss_score_space_3d", pad_inches=0.02, pdf_dpi=100)

    c_clean_fig = plot_3d_score_space(score_sample, show_values=False)
    save_original(
        c_clean_fig,
        output,
        "loss_score_space_3d_without_labels",
        pad_inches=0.02,
        pdf_dpi=100,
    )

    d_fig, d_axes = plt.subplots(
        1, 2, figsize=(7 * CM, 4.5 * CM), sharex=True, sharey=True
    )
    plot_mean_std(d_axes, score_sample)
    d_fig.subplots_adjust(left=0.10, right=0.99, bottom=0.38, top=0.93, wspace=0.22)
    save_original(d_fig, output, "loss_score_mean_variance_consensus_classes")

    e_legend_fig, e_legend_ax = plt.subplots(figsize=(5 * CM, 2 * CM))
    e_fig, e_axes = plt.subplots(
        1, 2, figsize=(5 * CM, 4.5 * CM), sharey=True
    )
    legend_handles, legend_labels = plot_loss_similarity(e_axes, loss_similarity)
    e_fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.86, wspace=0.18)
    e_legend_ax.axis("off")
    e_legend_fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=len(legend_labels),
        loc="center",
    )
    save_original(e_legend_fig, output, "loss_similarity_to_bce_legend")
    save_original(e_fig, output, "loss_similarity_to_bce_barplots")

