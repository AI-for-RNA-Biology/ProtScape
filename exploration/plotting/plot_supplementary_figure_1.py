#!/usr/bin/env python
"""Plot the standalone network-statistics panels for Supplementary Figure 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from exploration.plotting.fonts import register_arial

register_arial()
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial:italic",
        "mathtext.sf": "Arial",
        "mathtext.tt": "Arial",
        "axes.linewidth": 0.8,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Input and output directories.
SOURCE_DATA_DIR = Path(
    "/storage/research/dbmr_luisierlab/temp/athomas/outputs_protscape_repo/"
    "figure_source_data/supplementary_figure_1"
)
FIGURE_OUTPUT_DIR = Path(
    "/storage/research/dbmr_luisierlab/temp/athomas/outputs_protscape_repo/"
    "figures/supplementary_figure_1"
)


CM = 1 / 2.54
BLACK = "#222222"
BAR_EDGE = "#222222"
AXIS_LINEWIDTH = 1.0
LEGEND_SIZE = 8.0

SOURCE_COLORS = {
    "Tabula Sapiens": "#4E79A7",
    "HBCA": "#59A14F",
    "Tabula Sapiens + HBCA": "#B07AA1",
    "ALS": "#E15759",
}
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


def style_axes(ax, grid_axis=None, labelsize=8.5):
    """Original ``paper_style.style_axes`` used by the source scripts."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.tick_params(
        labelsize=labelsize,
        color=BLACK,
        labelcolor=BLACK,
        width=0.8,
    )
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E9ECEF", linewidth=0.7)
        ax.set_axisbelow(True)


def save_figure(fig, output, stem):
    """Use the exact layout and export settings of the source scripts."""
    fig.tight_layout()
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gene_retention(data, output):
    """Source: ``filtered_gene_proportions.py`` compact plot."""
    specs = [
        ("Reliable / Measured", "reliable_gene_fraction"),
        ("Selected / Reliable", "selected_gene_fraction"),
        ("LCC / Selected", "lcc_among_selected_fraction"),
    ]
    sources = ["Tabula Sapiens", "HBCA", "ALS"]
    source_labels = {"Tabula Sapiens": "Tabula", "HBCA": "HBCA", "ALS": "ALS"}
    offsets = {"Tabula Sapiens": -0.24, "HBCA": 0.0, "ALS": 0.24}
    fontsize = 10

    fig, ax = plt.subplots(figsize=(10 * CM, 5 * CM))
    legend_handles = []

    for metric_pos, (_, metric) in enumerate(specs, start=1):
        for source in sources:
            values = (
                100.0
                * data.loc[data["source"].eq(source), metric]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=float)
            )
            if values.size == 0:
                continue
            color = SOURCE_COLORS[source]
            position = metric_pos + offsets[source]
            box = ax.boxplot(
                [values],
                positions=[position],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
            )
            box["boxes"][0].set_facecolor(color)
            box["boxes"][0].set_alpha(1.0)
            box["boxes"][0].set_edgecolor(BAR_EDGE)
            box["boxes"][0].set_linewidth(0.7)
            for key in ["whiskers", "caps", "medians"]:
                for artist in box[key]:
                    artist.set_color(BAR_EDGE)
                    artist.set_linewidth(0.8)

    for source in sources:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=SOURCE_COLORS[source],
                linewidth=2.8,
                linestyle="-",
                label=source_labels[source],
            )
        )

    ax.set_xlim(0.5, len(specs) + 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(1, len(specs) + 1))
    ax.set_xticklabels([label for label, _ in specs], fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=max(fontsize - 2, 6))
    ax.set_ylabel("Retained gene fraction (%)")
    style_axes(ax, grid_axis=None)
    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        prop={"size": fontsize},
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    save_figure(fig, output, "preprocessing_fraction_compact_by_source")


def plot_ppi_sizes(data, output):
    """Source: ``ppi_graph_statistics.py`` PPI node/edge scatter."""
    fig, ax = plt.subplots(figsize=(8.8 * CM, 8.0 * CM))
    fontsize = LEGEND_SIZE
    mean_proteins = data["n_proteins"].mean()
    mean_edges = data["n_edges"].mean()
    mean_proteins_rounded = int(round(mean_proteins))
    mean_edges_rounded = int(round(mean_edges))

    for source, subset in data.groupby("source", sort=False):
        ax.scatter(
            subset["n_proteins"],
            subset["n_edges"],
            s=34,
            alpha=0.88,
            color=SOURCE_COLORS.get(source, "#777777"),
            edgecolors="none",
            linewidths=0,
        )

    label_font = {"family": "sans-serif", "size": fontsize}
    ax.set_xlabel(r"Protein count ($10^3$)", fontdict=label_font)
    ax.set_ylabel(r"Edge count ($10^3$)", fontdict=label_font)
    ax.ticklabel_format(style="plain", axis="both")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x / 1000)}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x / 1000)}"))
    ax.tick_params(axis="both", labelsize=fontsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("sans-serif")
        label.set_fontsize(fontsize)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.axvline(
        mean_proteins,
        color="#444444",
        linestyle="--",
        linewidth=AXIS_LINEWIDTH,
        zorder=0,
        ymin=0,
        ymax=(mean_edges - ylim[0]) / (ylim[1] - ylim[0]),
    )
    ax.axhline(
        mean_edges,
        color="#444444",
        linestyle="--",
        linewidth=AXIS_LINEWIDTH,
        zorder=0,
        xmin=0,
        xmax=(mean_proteins - xlim[0]) / (xlim[1] - xlim[0]),
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="--",
            linewidth=AXIS_LINEWIDTH,
            label=f"Mean protein: {mean_proteins_rounded}",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="--",
            linewidth=AXIS_LINEWIDTH,
            label=f"Mean edge: {mean_edges_rounded}",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        prop={"family": "sans-serif", "size": fontsize},
    )
    style_axes(ax, grid_axis=None)
    save_figure(fig, output, "ppi_nodes_vs_edges")


def plot_jaccard_boxplots(data, output):
    """Source: ``ppi_graph_statistics.py`` node/edge Jaccard boxplot."""
    labels = ["node", "edge"]
    values = [data["node_jaccard"].dropna(), data["edge_jaccard"].dropna()]
    fig, ax = plt.subplots(figsize=(4 * CM, 4.8 * CM))
    box = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        widths=0.45,
        showfliers=False,
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#D8A7B1")
        patch.set_edgecolor(BAR_EDGE)
        patch.set_alpha(0.95)
    for whisker in box["whiskers"]:
        whisker.set_color(BAR_EDGE)
        whisker.set_linewidth(0.8)
    for cap in box["caps"]:
        cap.set_color(BAR_EDGE)
        cap.set_linewidth(0.8)
    for median in box["medians"]:
        median.set_color(BLACK)
        median.set_linewidth(1.1)

    ax.set_ylabel("Jaccard similarity")
    style_axes(ax, grid_axis=None)
    save_figure(fig, output, "ppi_node_edge_jaccard_boxplot")


def format_mds_axes(ax, fontsize=8):
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontfamily("sans-serif")
        label.set_fontsize(fontsize)
    tick_fontsize = max(fontsize - 2, 4)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("sans-serif")
        label.set_fontsize(tick_fontsize)
    ax.grid(False)


def plot_mds_by_group(
    data,
    group_col,
    color_map,
    group_order,
    title,
    stem,
    output,
):
    """Source: ``ppi_jaccard_mds_embedding.py`` plot and separate legend."""
    fig, ax = plt.subplots(figsize=(12 * CM, 12 * CM))
    handles = []
    labels = []
    for group in group_order:
        subset = data[data[group_col].eq(group)]
        if subset.empty:
            continue
        if group_col == "database":
            label = f"{group} \n(n={len(subset)})"
        else:
            label = f"{group} (n={len(subset)})"
        handle = ax.scatter(
            subset["mds_1"],
            subset["mds_2"],
            s=30,
            color=color_map.get(group, "#9A9A9A"),
            edgecolors="black",
            linewidths=0.1,
            alpha=0.94,
            label=label,
        )
        handles.append(handle)
        labels.append(label)
    format_mds_axes(ax, fontsize=8)
    save_figure(fig, output, stem)

    legend_fig, legend_ax = plt.subplots(figsize=(2.8 * CM, 8.0 * CM))
    legend_ax.axis("off")
    legend = legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        fontsize=8,
        ncol=1,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_fontfamily("sans-serif")
        text.set_fontsize(8)
    legend_fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save_figure(legend_fig, output, f"{stem}_legend")


def plot_metagraph_edges(data, output):
    """Source: ``dataset_statistics.py`` metagraph edge-type bars."""
    plot_data = data.copy()
    plot_data["edge_type"] = plot_data["edge_type"].replace(
        {
            "Cell-cell communication": "Cell-Cell",
            "Context-to-tissue": "Cell-Tissue",
        }
    )
    colors = [
        SOURCE_COLORS.get(label, CELL_CLASS_COLORS.get(label, "#D7B8C2"))
        for label in plot_data["edge_type"]
    ]

    fig, ax = plt.subplots(figsize=(4 * CM, 6 * CM))
    ax.bar(
        plot_data["edge_type"],
        plot_data["n_edges"],
        color=colors,
        edgecolor=BAR_EDGE,
        linewidth=0.6,
        width=0.35,
    )
    ax.set_ylabel(
        r"Edge counts ($10^3$)",
        fontdict={"family": "sans-serif", "size": 10},
    )
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f"{x / 1000:.0f}" if x != 0 else "0")
    )
    style_axes(ax, grid_axis=None)
    save_figure(fig, output, "metagraph_edge_type_counts")


def plot_tissue_counts(data, output):
    """Source: ``tissue_context_counts.py`` horizontal tissue bars."""
    plot_data = data.sort_values("n_contexts")
    fig, ax = plt.subplots(figsize=(11 * CM, 7 * CM))
    ax.barh(
        plot_data["tissue_name"],
        plot_data["n_contexts"],
        color="#D8A7B1",
        edgecolor=BAR_EDGE,
        linewidth=0.3,
        height=0.4,
    )
    ax.set_xlabel("Associated cellular contexts", fontsize=8)
    ax.set_ylabel("")
    ax.tick_params(
        axis="y",
        labelsize=5,
        color=BLACK,
        labelcolor=BLACK,
        width=0.8,
    )
    ax.tick_params(
        axis="x",
        labelsize=5,
        color=BLACK,
        labelcolor=BLACK,
        width=0.8,
    )
    ax.set_title("")
    average = plot_data["n_contexts"].mean()
    ax.axvline(
        average,
        color=BAR_EDGE,
        linestyle="--",
        linewidth=1.1,
        label=f"mean: {average:.1f}",
    )
    ax.set_ylim(-0.5, len(plot_data) - 0.5)
    plt.tight_layout()
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save_figure(fig, output, "tissue_context_counts")


def plot_context_tissue_histogram(data, output):
    """Source: ``tissue_context_counts.py`` tissue assignments histogram."""
    values = data["n_tissues"].dropna()
    fig, ax = plt.subplots(figsize=(3.5 * CM, 5 * CM))
    ax.hist(
        values,
        bins=range(1, int(values.max()) + 2),
        color="#D8A7B1",
        edgecolor=BAR_EDGE,
        linewidth=0.45,
        rwidth=0.9,
    )
    ax.set_xlabel("Tissues per context", fontsize=8, labelpad=3)
    ax.set_ylabel("Contexts", fontsize=8, labelpad=3)
    ax.set_xticks([0, 10, 20])
    ax.set_title("", fontsize=8)
    style_axes(ax, grid_axis=None, labelsize=5)
    mean_value = values.mean()
    ax.axhline(
        mean_value,
        color=BAR_EDGE,
        linestyle="--",
        linewidth=1.1,
        label=f"mean: {mean_value:.1f}",
    )
    ax.legend(frameon=False, loc="upper right", fontsize=5)
    save_figure(fig, output, "context_tissue_assignment_histogram")


def plot_all(source=SOURCE_DATA_DIR, output=FIGURE_OUTPUT_DIR):
    """Render the original standalone source plots from prepared CSV tables."""
    source = Path(source)
    output = Path(output)
    if not source.exists():
        raise FileNotFoundError(f"Missing figure source data: {source}")
    output.mkdir(parents=True, exist_ok=True)

    retention = pd.read_csv(source / "gene_retention.csv")
    ppi_metrics = pd.read_csv(source / "ppi_metrics.csv")
    jaccard = pd.read_csv(source / "pairwise_jaccard.csv")
    mds = pd.read_csv(source / "edge_jaccard_mds.csv")
    metagraph_edges = pd.read_csv(source / "metagraph_edges.csv")
    context_counts = pd.read_csv(source / "context_tissue_counts.csv")
    tissue_counts = pd.read_csv(source / "tissue_context_counts.csv")

    plot_gene_retention(retention, output)
    plot_ppi_sizes(ppi_metrics, output)
    plot_jaccard_boxplots(jaccard, output)
    plot_mds_by_group(
        mds,
        "database",
        SOURCE_COLORS,
        ["Tabula Sapiens", "HBCA", "Tabula Sapiens + HBCA", "ALS"],
        "PPI Jaccard MDS by database",
        "bulk_ppi_jaccard_mds_by_database",
        output,
    )
    plot_mds_by_group(
        mds,
        "cell_type_class",
        CELL_CLASS_COLORS,
        [
            "Neuronal",
            "Epithelial",
            "Immune",
            "Stromal",
            "Glial",
            "Endothelial",
            "Muscle",
            "Blood",
            "Secretory",
            "Endocrine",
            "Stromal-Epithelial",
            "Pigment",
        ],
        "PPI Jaccard MDS by cell class",
        "bulk_ppi_jaccard_mds_by_cell_class",
        output,
    )
    plot_metagraph_edges(metagraph_edges, output)
    plot_tissue_counts(tissue_counts, output)
    plot_context_tissue_histogram(context_counts, output)

    print(f"Wrote individual Supplementary Figure 1 plots to {output}")


def main():
    plot_all()


if __name__ == "__main__":
    main()
