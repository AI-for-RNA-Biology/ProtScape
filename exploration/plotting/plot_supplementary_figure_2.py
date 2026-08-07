#!/usr/bin/env python3
"""Plot the individual components of Supplementary Figure 2."""

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


# Input and output directories.
SOURCE_DATA_DIR = Path(
    "/storage/research/dbmr_luisierlab/temp/athomas/outputs_protscape_repo/"
    "figure_source_data/supplementary_figure_2"
)
FIGURE_OUTPUT_DIR = Path(
    "/storage/research/dbmr_luisierlab/temp/athomas/outputs_protscape_repo/"
    "figures/supplementary_figure_2"
)

import matplotlib.pyplot as plt


CM = 1 / 2.54

# Preserve the source analysis style, with Arial forced so every regenerated
# Supplementary Figure 2 component uses the same typeface.
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
        fontsize=11,
        handlelength=1.8,
    )

    for ax, table, ylabel, yticks in zip(
        axes,
        [auprc, f1],
        ["AUPRC 1:1 (%)", "F1 1:1 (%)"],
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
    ax.set_ylabel("Score 1:1 (%)")
    ax.text(0.25, 1.03, "PPI", transform=ax.transAxes, ha="center", va="bottom")
    ax.text(0.75, 1.03, "Metagraph", transform=ax.transAxes, ha="center", va="bottom")
    clean_2d_axis(ax)


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


def plot_pretraining(source, output):
    """Render panels a-b from pretraining-evaluation tables."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    auprc = pd.read_csv(source / "contextwise_ppi_auprc.csv")
    f1 = pd.read_csv(source / "contextwise_ppi_f1.csv")
    pooling = pd.read_csv(source / "pooling_sensitivity.csv")

    a_legend_fig, a_legend_ax = plt.subplots(figsize=(6.8, 0.45))
    a_auprc_fig, a_auprc_ax = plt.subplots(figsize=(5.5 * CM, 4.5 * CM))
    a_f1_fig, a_f1_ax = plt.subplots(figsize=(5.5 * CM, 4.5 * CM))
    plot_contextwise_boxplots(
        a_legend_ax, [a_auprc_ax, a_f1_ax], auprc, f1
    )
    save_original(a_legend_fig, output, "core_model_legend")
    save_original(a_auprc_fig, output, "supp_contextwise_ppi_auprc_curated")
    save_original(a_f1_fig, output, "supp_contextwise_ppi_f1_curated")

    b_legend_fig, b_legend_ax = plt.subplots(figsize=(7 * CM, 2.2 * CM))
    b_fig, b_ax = plt.subplots(figsize=(7 * CM, 4.5 * CM))
    plot_pooling(b_legend_ax, b_ax, pooling)
    b_fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.86)
    save_original(b_legend_fig, output, "pooling_sensitivity_legend")
    save_original(b_fig, output, "pooling_sensitivity")


def plot_consensus(source, output):
    """Render panels c-e from loss-consensus analysis tables."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    score_sample = pd.read_csv(source / "score_sample.csv.gz")
    sample_summary = pd.read_csv(source / "score_sample_summary.csv")
    loss_similarity = pd.read_csv(source / "loss_similarity.csv")

    observed = score_sample.groupby(["edge_label", "consensus_class"]).size()
    expected = sample_summary.set_index(["label", "loss_class"])["edge_contexts_sampled"]
    if not observed.sort_index().equals(expected.sort_index()):
        raise ValueError("Panel c/d score sample does not match its sample summary")

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


def plot_string(source, output):
    """Render panel f from the STRING-validation table."""
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



def plot_all(source=SOURCE_DATA_DIR, output=FIGURE_OUTPUT_DIR):
    """Render every standalone Supplementary Figure 2 plot from prepared CSVs."""
    source, output = Path(source), Path(output)
    if not source.exists():
        raise FileNotFoundError(f"Missing figure source data: {source}")
    plot_pretraining(source, output)
    plot_consensus(source, output)
    plot_string(source, output)
    print(f"Wrote individual Supplementary Figure 2 plots to {output}")


def main():
    plot_all()


if __name__ == "__main__":
    main()
