#!/usr/bin/env python3
"""Plot therapeutic-target attribution results."""

from __future__ import annotations

import re
import textwrap
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


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

MODEL_ORDER = [
    "pinnacle_random_fixed",
    "pinnacle_esm_fixed",
    "pinnacle_esm2_acm",
    "gae_att_fixed_do06",
    "s2gae_att_k1_fixed_do04_uni",
]
MODEL_LABELS = {
    "pinnacle_random_fixed": "Pinnacle",
    "pinnacle_esm_fixed": "Pinnacle-ESM2 (GAT)",
    "pinnacle_esm2_acm": "Pinnacle-ESM2 (ACM)",
    "gae_att_fixed_do06": "ProtScape-GAE",
    "s2gae_att_k1_fixed_do04_uni": "ProtScape",
}
MODEL_COLORS = {
    "pinnacle_random_fixed": "#2b2b2b",
    "pinnacle_esm_fixed": "#b0b0b0",
    "pinnacle_esm2_acm": "#7a7a7a",
    "gae_att_fixed_do06": "#1f77b4",
    "s2gae_att_k1_fixed_do04_uni": "#e6550d",
}

CELL_CLASSES = [
    "Immune",
    "Stromal",
    "Muscle",
    "Epithelial",
    "Endocrine",
    "Blood",
    "Secretory",
    "Pigment",
    "Glial",
    "Neuronal",
    "Stromal-Epithelial",
    "Endothelial",
]
CELL_CLASS_LABELS = {"Stromal-Epithelial": "Stromal–\nepithelial"}
CELL_CLASS_COLORS = {
    "Neuronal": "#4E79A7",
    "Glial": "#9C755F",
    "Epithelial": "#E15759",
    "Immune": "#59A14F",
    "Stromal": "#F28E2B",
    "Endothelial": "#76B7B2",
    "Muscle": "#79706E",
    "Blood": "#B07AA1",
    "Secretory": "#86BCB6",
    "Endocrine": "#EDC948",
    "Stromal-Epithelial": "#AF7AA1",
    "Pigment": "#555555",
}

TASK_ORDER = [
    ("therapeutic_target_mondo_0005180", "Parkinson disease"),
    ("therapeutic_target_efo_0000305", "Breast carcinoma"),
    ("therapeutic_target_efo_1001207", "Systolic heart failure"),
    ("therapeutic_target_efo_0000571", "Lung adenocarcinoma"),
    ("therapeutic_target_efo_0003767", "Inflammatory bowel disease"),
    ("therapeutic_target_efo_0000676", "Psoriasis"),
    ("therapeutic_target_mondo_0007915", "Systemic lupus erythematosus"),
    ("therapeutic_target_efo_1001249", "NASH"),
    ("therapeutic_target_mondo_0004979", "Asthma"),
    ("therapeutic_target_mondo_0005148", "Type 2 diabetes"),
    ("therapeutic_target_efo_0003884", "Chronic kidney disease"),
    ("therapeutic_target_efo_0000685", "Rheumatoid arthritis"),
    ("therapeutic_target_efo_0000341", "COPD"),
    ("therapeutic_target_efo_0001361", "Pulmonary arterial hypertension"),
    ("therapeutic_target_efo_0000274", "Atopic eczema"),
]


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


def plot_signed_heatmap(matrix: pd.DataFrame, output: Path) -> None:
    matrix = matrix.set_index("disease").reindex(
        index=[disease for _, disease in TASK_ORDER], columns=CELL_CLASSES
    )
    fig = plt.figure(figsize=figure_size(DOUBLE_COLUMN_WIDTH_CM, 15.5), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(18.0, 0.68),
        left=0.285,
        right=0.925,
        bottom=0.19,
        top=0.95,
        wspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [CELL_CLASS_LABELS.get(value, value) for value in matrix.columns],
        rotation=42,
        ha="right",
        rotation_mode="anchor",
        fontsize=TICK_LABEL_SIZE,
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=TICK_LABEL_SIZE)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=AXIS_LINEWIDTH)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar_ax = fig.add_subplot(grid[0, 1])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(
        "Mean signed contribution per cellular context (% of total |LRP|)",
        fontsize=AXIS_LABEL_SIZE,
    )
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, length=2)
    colorbar.outline.set_linewidth(AXIS_LINEWIDTH)
    save_figure(fig, output, "test_disease_cell_class_signed_lrp")


def wrap_label(value: object) -> str:
    return textwrap.fill(str(value), width=34, break_long_words=False)


def plot_focal_targets(top: pd.DataFrame, output: Path) -> None:
    stems = {
        (
            "therapeutic_target_mondo_0005180",
            "HTR1A",
        ): "parkinson_htr1a_cell_type_attribution",
        (
            "therapeutic_target_efo_0000571",
            "ERBB3",
        ): "lung_adenocarcinoma_erbb3_cell_type_attribution",
    }
    shared_max = float(top["shared_x_axis_max_pct"].iloc[0])
    shown_classes = []
    for keys, stem in stems.items():
        values = top.loc[top["task"].eq(keys[0]) & top["gene"].eq(keys[1])].sort_values("rank")
        shown_classes.extend(values["cell_class"].tolist())
        y = np.arange(len(values))
        fig, ax = plt.subplots(figsize=WIDE_SINGLE_SIZE, facecolor="white")
        fig.subplots_adjust(left=0.48, right=0.97, bottom=0.20, top=0.96)
        ax.scatter(
            values["mean_overall_positive_contribution_pct"],
            y,
            s=105,
            c=[CELL_CLASS_COLORS[value] for value in values["cell_class"]],
            edgecolors="none",
            zorder=3,
        )
        clean_axes(ax)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [wrap_label(value) for value in values["display_cell_label"]],
            fontsize=TICK_LABEL_SIZE,
        )
        ax.invert_yaxis()
        ax.set_xlim(0.0, shared_max)
        ax.set_xlabel(
            "Mean contribution to complete positive attribution (%)",
            fontsize=AXIS_LABEL_SIZE,
        )
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
        ax.spines["left"].set_visible(False)
        save_figure(fig, output, stem)

    shown_classes = list(dict.fromkeys(shown_classes))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=CELL_CLASS_COLORS[cell_class],
            markeredgecolor="none",
            markersize=9.5,
            label=cell_class,
        )
        for cell_class in shown_classes
    ]
    fig = plt.figure(figsize=figure_size(4.4, 3.2), facecolor="white")
    fig.legend(
        handles=handles,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handletextpad=0.45,
        labelspacing=0.65,
    )
    save_figure(fig, output, "focal_target_cell_class_legend")


def plot_absolute_relevance(values: pd.DataFrame, output: Path) -> None:
    model_order = [MODEL_LABELS[key] for key in MODEL_ORDER]
    model_colors = {MODEL_LABELS[key]: MODEL_COLORS[key] for key in MODEL_ORDER}
    rng = np.random.default_rng(18)
    fig, ax = plt.subplots(figsize=figure_size(5.5, 4.5))
    grouped = [
        100.0 * values.loc[values["model"].eq(model), "q_absolute_context"].to_numpy()
        for model in model_order
    ]
    positions = np.arange(len(model_order))
    boxes = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#777777", "linewidth": 0.9},
        capprops={"color": "#777777", "linewidth": 0.9},
    )
    for patch, model in zip(boxes["boxes"], model_order):
        patch.set(
            facecolor=model_colors[model],
            edgecolor=model_colors[model],
            alpha=1.0,
            linewidth=1.0,
        )
    for median in boxes["medians"]:
        median.set_zorder(4)
    for position, (model, model_values) in enumerate(zip(model_order, grouped)):
        jitter = rng.uniform(-0.14, 0.14, size=len(model_values))
        ax.scatter(
            np.full(len(model_values), position) + jitter,
            model_values,
            s=9,
            color=model_colors[model],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
    ax.set_xlim(-0.55, len(model_order) - 0.45)
    ax.set_xticks(positions)
    ax.set_xticklabels(["" for _ in model_order])
    all_values = np.concatenate(grouped)
    ax.set_ylim(max(0.0, all_values.min() - 2.0), min(100.0, all_values.max() + 2.0))
    ax.set_yticks([20, 40, 60, 80])
    ax.set_ylabel("Absolute contextual\nrelevance (%)", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("Models", fontsize=AXIS_LABEL_SIZE)
    clean_axes(ax)
    save_figure(fig, output, "absolute_contextual_relevance_fraction")


def file_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def plot_disease_top_contexts(top: pd.DataFrame, output: Path) -> None:
    shared_max = float(top["shared_x_axis_max_pct"].iloc[0])
    for task, disease in TASK_ORDER:
        values = top.loc[top["task"].eq(task)].sort_values("rank")
        y = np.arange(len(values))
        fig, ax = plt.subplots(figsize=WIDE_SINGLE_SIZE, facecolor="white")
        fig.subplots_adjust(left=0.48, right=0.97, bottom=0.18, top=0.86)
        ax.scatter(
            values["mean_positive_contribution_pct"],
            y,
            s=105,
            c=[CELL_CLASS_COLORS[value] for value in values["cell_class"]],
            edgecolors="none",
            zorder=3,
        )
        clean_axes(ax)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [wrap_label(value) for value in values["display_cell_label"]],
            fontsize=TICK_LABEL_SIZE,
        )
        ax.invert_yaxis()
        ax.set_xlim(0.0, shared_max)
        ax.set_title(disease, fontsize=PANEL_TITLE_SIZE, pad=10)
        ax.set_xlabel(
            "Mean contribution to complete positive attribution (%)",
            fontsize=AXIS_LABEL_SIZE,
        )
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
        ax.spines["left"].set_visible(False)
        save_figure(
            fig,
            output,
            f"{file_slug(disease)}_top5_cell_type_attribution",
        )

    shown = set(top["cell_class"])
    classes = [cell_class for cell_class in CELL_CLASSES if cell_class in shown]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=CELL_CLASS_COLORS[cell_class],
            markeredgecolor="none",
            markersize=9.5,
            label=cell_class,
        )
        for cell_class in classes
    ]
    height = min(MAX_FIGURE_HEIGHT_CM, 0.68 * len(classes) + 0.4)
    fig = plt.figure(
        figsize=figure_size(SINGLE_COLUMN_WIDTH_CM / 2.0, height),
        facecolor="white",
    )
    fig.legend(
        handles=handles,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handletextpad=0.45,
        labelspacing=0.65,
    )
    save_figure(fig, output, "disease_top_context_cell_class_legend")


def plot_attributions(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(PLOT_RC):
        plot_signed_heatmap(
            pd.read_csv(source / "cell_class_signed_contribution_percent.csv"),
            output,
        )
        plot_focal_targets(
            pd.read_csv(source / "focal_target_top_contexts.csv"), output
        )
        plot_absolute_relevance(
            pd.read_csv(source / "absolute_contextual_relevance_diseases.csv"),
            output,
        )
        plot_disease_top_contexts(
            pd.read_csv(source / "disease_top_contexts.csv"), output
        )
