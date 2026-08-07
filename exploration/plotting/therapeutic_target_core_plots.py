#!/usr/bin/env python3
"""Plot the therapeutic-target panels for Figures 5 and S5."""

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

PAPER_RC = {
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
        ("therapeutic_target_mondo_0005180", "HTR1A"): "D1_parkinson_HTR1A_cell_type_dots",
        ("therapeutic_target_efo_0000571", "ERBB3"): "E1_lung_ERBB3_cell_type_dots",
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
    save_figure(fig, output, "common_cell_class_legend")


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
    save_figure(fig, output, "S17_absolute_contextual_relevance_fraction")


def file_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def plot_disease_top_contexts(top: pd.DataFrame, output: Path) -> None:
    shared_max = float(top["shared_x_axis_max_pct"].iloc[0])
    for index, (task, disease) in enumerate(TASK_ORDER, start=1):
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
        save_figure(fig, output, f"S{index:02d}_{file_slug(disease)}_top5_cell_type_dots")

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
    save_figure(fig, output, "S16_common_cell_class_legend")


def plot_all(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(PAPER_RC):
        mean = pd.read_csv(source / "mean_performance.csv")
        plot_average_metric(mean, "auprc", output)
        plot_average_legend(mean, output)
        plot_disease_comparisons(
            pd.read_csv(source / "disease_model_comparison.csv"), output
        )
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
