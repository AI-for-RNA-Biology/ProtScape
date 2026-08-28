#!/usr/bin/env python3
"""Plot CORUM dataset and benchmark results."""

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
from matplotlib.patches import Patch


CM = 1 / 2.54
AXIS_LINEWIDTH = 0.6
BAR_EDGE_COLOR = "#222222"
BAR_EDGE_LINEWIDTH = 0.55
DATA_LINEWIDTH = 0.8
TICK_LABEL_SIZE = 6.0
AXIS_LABEL_SIZE = 8.0
PANEL_TITLE_SIZE = 7.0
LEGEND_SIZE = 6.0
SINGLE_COLUMN_WIDTH_CM = 8.8
DOUBLE_COLUMN_WIDTH_CM = 18.0

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
MAIN_CONTEXT_MODEL_ORDER = [
    "pinnacle_random",
    "pinnacle_esm",
    "pinnacle_acm",
    "gae_bce",
    "s2gae_bce_uni",
]
MAIN_MODEL_ORDER = ["lr_esm", "lr_prostt5"] + MAIN_CONTEXT_MODEL_ORDER
LOSS_MODEL_ORDER = ["s2gae_bce_uni", "s2gae_phuber_uni", "s2gae_l1_uni"]
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
LOSS_MODEL_LABELS = {
    **MODEL_LABELS,
    "s2gae_bce_uni": "ProtScape (BCE)",
}
MODEL_COLORS = {
    "lr_esm": "#222222",
    "lr_prostt5": "#666666",
    "pinnacle_random": "#2b2b2b",
    "pinnacle_esm": "#b0b0b0",
    "pinnacle_acm": "#7a7a7a",
    "gae_bce": "#1f77b4",
    "s2gae_bce_uni": "#e6550e",
    "s2gae_phuber_uni": "#ff1529",
    "s2gae_l1_uni": "#9200bf",
}


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


def plot_complex_size_distribution(
    stats: pd.DataFrame,
    distribution: pd.DataFrame,
    output: Path,
) -> None:
    row = stats.iloc[0]
    x = distribution["n_members"].to_numpy(dtype=float)
    y = distribution["n_complexes"].to_numpy(dtype=float)
    median_members = float(np.median(np.repeat(x, y.astype(int))))

    fig, ax = plt.subplots(figsize=figure_size(6.0, 5.0))
    ax.bar(
        x,
        y,
        width=0.85,
        color="#595959",
        edgecolor="#222222",
        linewidth=BAR_EDGE_LINEWIDTH,
        zorder=3,
    )
    ax.axvline(
        median_members,
        color="#e6550d",
        linestyle="--",
        linewidth=DATA_LINEWIDTH,
        zorder=4,
    )
    ax.text(
        0.98,
        0.94,
        f"n={int(row['n_complexes']):,}\nmedian={median_members:.0f}\n"
        f"range {int(x.min())}-{int(x.max())}",
        ha="right",
        va="top",
        fontsize=TICK_LABEL_SIZE,
        linespacing=1.05,
        transform=ax.transAxes,
    )
    ax.set_xlabel("Members per complex", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("CORUM complexes", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(max(0, x.min() - 1.0), x.max() + 5.0)
    ax.set_ylim(0, max(y) * 1.20)
    ax.set_xticks([10, 20, 30, 50, 75, 100])
    clean_axes(ax)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.20, top=0.96)
    save_figure(fig, output, "corum_complex_size_distribution")


def plot_complexes_per_protein_distribution(
    stats: pd.DataFrame,
    distribution: pd.DataFrame,
    output: Path,
) -> None:
    row = stats.iloc[0]
    maximum = int(distribution["complexes_per_protein"].max())
    full_x = pd.DataFrame({"complexes_per_protein": np.arange(1, maximum + 1)})
    distribution = full_x.merge(
        distribution, on="complexes_per_protein", how="left"
    ).fillna(0)
    label_x = distribution["complexes_per_protein"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=figure_size(6.0, 5.0))
    ax.bar(
        label_x,
        distribution["n_proteins"],
        color="#4a4a4a",
        edgecolor="#222222",
        linewidth=BAR_EDGE_LINEWIDTH,
        zorder=3,
    )
    ax.text(
        0.98,
        0.94,
        f"n={int(row['n_proteins']):,}\n"
        f"median={row['median_complexes_per_protein']:.0f}\n"
        f"max={int(row['max_complexes_per_protein'])}",
        ha="right",
        va="top",
        fontsize=TICK_LABEL_SIZE,
        linespacing=1.05,
        transform=ax.transAxes,
    )
    ax.set_xlabel("Complexes per protein", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Proteins", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(0.35, maximum + 0.65)
    ax.set_ylim(0, max(distribution["n_proteins"]) * 1.18)
    ax.set_xticks(label_x)
    clean_axes(ax)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.20, top=0.96)
    save_figure(fig, output, "corum_complexes_per_protein_distribution")


def present_models(rows: pd.DataFrame, order: list[str]) -> list[str]:
    available = set(rows["inference_key"])
    return [key for key in order if key in available]


def plot_performance_legend(
    rows: pd.DataFrame,
    output: Path,
    stem: str,
    model_order: list[str],
    labels: dict[str, str],
    include_baselines: bool,
) -> None:
    model_order = present_models(rows, model_order)
    handles = [
        Patch(facecolor=MODEL_COLORS[key], edgecolor="white", label=labels[key])
        for key in model_order
    ]
    if include_baselines and "lr_esm" in set(rows["readout_key"]):
        handles.append(
            Patch(facecolor="white", edgecolor="#222222", label="ESM2")
        )
    if include_baselines and "lr_prostt5" in set(rows["readout_key"]):
        handles.append(
            Patch(
                facecolor="#f2f2f2",
                edgecolor="#666666",
                hatch="\\\\\\\\",
                label="LR ProstT5",
            )
        )

    fig = plt.figure(
        figsize=figure_size(SINGLE_COLUMN_WIDTH_CM, max(2.5, 0.86 * len(handles) + 0.8))
    )
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="center",
        ncol=1,
        handlelength=1.8,
    )
    save_figure(fig, output, stem)


def plot_performance_metric(
    rows: pd.DataFrame,
    metric: str,
    output: Path,
    stem: str,
    model_order: list[str],
) -> None:
    sub = rows[rows["metric"] == metric].copy()
    readouts = [key for key in READOUT_ORDER if key in set(sub["readout_key"])]
    model_order = present_models(sub, model_order)
    bar_width = 0.25
    group_widths = {
        key: bar_width if key in BASELINE_MODELS else bar_width * len(model_order)
        for key in readouts
    }
    x_positions = [0.0]
    for index in range(1, len(readouts)):
        previous, current = readouts[index - 1], readouts[index]
        gap = (
            (group_widths[previous] + group_widths[current]) / 2.0
            + bar_width * 1.3
        )
        x_positions.append(x_positions[-1] + gap)
    x = np.asarray(x_positions, dtype=float)

    plot_width = DOUBLE_COLUMN_WIDTH_CM * max(0.45, 0.40 + bar_width * 0.7)
    fig, ax = plt.subplots(figsize=figure_size(plot_width, 5.5))
    baseline_styles = {
        "lr_esm": {"facecolor": "white", "edgecolor": "#222222"},
        "lr_prostt5": {
            "facecolor": "#f2f2f2",
            "edgecolor": "#666666",
            "hatch": "\\\\\\\\",
        },
    }
    for key, style in baseline_styles.items():
        baseline = sub[sub["readout_key"] == key]
        if baseline.empty or key not in readouts:
            continue
        ax.bar(
            float(x[readouts.index(key)]),
            float(baseline.iloc[0]["score_percent"]),
            width=bar_width,
            zorder=4,
            **style,
        )

    for index, model_key in enumerate(model_order):
        offset = (index - (len(model_order) - 1) / 2.0) * bar_width
        heights = []
        for readout in readouts:
            if readout in BASELINE_MODELS:
                heights.append(np.nan)
                continue
            match = sub[
                (sub["readout_key"] == readout)
                & (sub["inference_key"] == model_key)
            ]
            heights.append(
                float(match.iloc[0]["score_percent"]) if not match.empty else np.nan
            )
        ax.bar(
            x + offset,
            heights,
            width=bar_width,
            color=MODEL_COLORS[model_key],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LINEWIDTH,
            zorder=3,
        )

    if metric == "auprc":
        ax.set_ylim(0.0, 80.0)
        ax.set_yticks(np.arange(0.0, 81.0, 20.0))
    elif metric == "f1":
        ax.set_ylim(0.0, 60.0)
        ax.set_yticks(np.arange(0.0, 61.0, 20.0))
    ax.set_ylabel("AUPRC (%)" if metric == "auprc" else f"{metric.upper()} (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [READOUT_LABELS[key] for key in readouts],
        fontsize=TICK_LABEL_SIZE,
        rotation=0,
        ha="center",
    )
    if len(x):
        left_margin = group_widths[readouts[0]] / 2.0 + bar_width * 0.3
        right_margin = group_widths[readouts[-1]] / 2.0 + bar_width * 0.3
        ax.set_xlim(x[0] - left_margin, x[-1] + right_margin)
    clean_axes(ax)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.34, top=0.97)
    save_figure(fig, output, stem)


def plot_benchmark(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(PLOT_RC):
        stats = read_table(source, "corum_dataset_statistics.csv")
        plot_complex_size_distribution(
            stats,
            read_table(source, "corum_complex_size_distribution.csv"),
            output,
        )
        plot_complexes_per_protein_distribution(
            stats,
            read_table(source, "corum_complexes_per_protein_distribution.csv"),
            output,
        )

        main_performance = read_table(source, "corum_main_performance.csv")
        for metric in ("auprc", "f1"):
            plot_performance_metric(
                main_performance,
                metric,
                output,
                f"corum_{metric}_performance_main_competitors",
                MAIN_CONTEXT_MODEL_ORDER,
            )
        plot_performance_legend(
            main_performance,
            output,
            "corum_performance_main_competitors_legend",
            MAIN_CONTEXT_MODEL_ORDER,
            MODEL_LABELS,
            include_baselines=True,
        )

        loss_performance = read_table(source, "corum_loss_performance.csv")
        for metric in ("auprc", "f1"):
            plot_performance_metric(
                loss_performance,
                metric,
                output,
                f"corum_{metric}_performance_protscape_losses",
                LOSS_MODEL_ORDER,
            )
        plot_performance_legend(
            loss_performance,
            output,
            "corum_performance_protscape_losses_legend",
            LOSS_MODEL_ORDER,
            LOSS_MODEL_LABELS,
            include_baselines=False,
        )
