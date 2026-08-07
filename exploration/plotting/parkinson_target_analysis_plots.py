#!/usr/bin/env python3
"""Plot Parkinson target-discovery results."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from exploration.plotting.fonts import register_arial

register_arial()
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


CM = 1 / 2.54
PROTSCAPE_COLOR = "#E85D24"
PINNACLE_COLOR = "#2F2F2F"
AXIS_COLOR = "#222222"
GRID_COLOR = "#E5E5E5"
AXIS_LINEWIDTH = 0.6
BAR_EDGE_LINEWIDTH = 0.55
DATA_LINEWIDTH = 0.8
TICK_LABEL_SIZE = 6.0
PANEL_TITLE_SIZE = 7.0
MODULE_PLOT_HEIGHT_CM = 10.5
MODULE_TERM_LABEL_SIZE = 8.5
MODULE_AXIS_LABEL_SIZE = 8.5
MODULE_TICK_LABEL_SIZE = 8.0
MODULE_TITLE_SIZE = 9.5

MODULE_COLORS = {
    1: "#4477AA",
    2: "#66CCEE",
    3: "#7E57C2",
    4: "#CCBB44",
    5: "#228833",
    6: "#CC79A7",
}
MODULE_LABELS = {
    1: "Class-A GPCR and monoaminergic receptors",
    2: "GABA/cholinergic ligand-gated receptors",
    3: "Ionotropic glutamate/NMDA receptor signalling",
    4: "Metabotropic glutamate and Class-C GPCR signalling",
    5: "Voltage-gated ion channels and excitability",
    6: "DNA replication/repair",
}
NETWORK_LABEL_OFFSETS = {
    "POLE": (-8, 8),
    "ESR1": (-8, 8),
    "GABRB3": (-2, 9),
    "GLRA1": (-8, 8),
    "GRIN1": (-8, -8),
    "HCN1": (8, -8),
    "MTNR1B": (0, 9),
}
STRING_CATEGORY_LABELS = {
    "Process": "GO biological process",
    "Function": "GO molecular function",
    "Pfam": "Pfam",
}

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
    "axes.labelsize": 7.0,
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
    "legend.fontsize": TICK_LABEL_SIZE,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.transparent": True,
}


def figure_size(width_cm: float, height_cm: float) -> tuple[float, float]:
    return width_cm * CM, height_cm * CM


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
        ax.spines[side].set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(
        axis="both", which="major", labelsize=TICK_LABEL_SIZE,
        width=0.6, length=2.5, pad=5,
    )
    ax.tick_params(axis="both", which="minor", width=0.5, length=1.5)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "pad_inches": 0.03}
    fig.savefig(output_dir / f"{stem}.pdf", **options)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, **options)
    plt.close(fig)


def lighten_color(color: str, white_fraction: float = 0.58) -> tuple[float, ...]:
    """Blend a module colour with white for ProtScape-candidate segments."""
    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    return tuple(rgb + white_fraction * (1.0 - rgb))


def plot_candidate_recovery(source: Path, output: Path) -> None:
    curve = pd.read_csv(source / "parkinson_candidate_recovery_curve.csv")
    summary = pd.read_csv(source / "parkinson_candidate_recovery_summary.csv").set_index("model")
    colors = {"protscape": PROTSCAPE_COLOR, "pinnacle": PINNACLE_COLOR}

    fig, ax = plt.subplots(figsize=figure_size(8.8, 8.0))
    for model in ["protscape", "pinnacle"]:
        rows = curve[curve["model"].eq(model)]
        ax.step(
            np.log10(rows["other_proteins_screened"] + 1),
            rows["test_positive_recall_percent"],
            where="post", color=colors[model], linewidth=DATA_LINEWIDTH,
        )
        point = summary.loc[model]
        count = int(point["other_proteins_screened_before_full_recovery"])
        x = np.log10(count + 1)
        ax.scatter(
            x, 100, s=45, color=colors[model], edgecolor="white",
            linewidth=BAR_EDGE_LINEWIDTH, zorder=4,
        )
        ax.axvline(
            x, color=colors[model], linestyle=(0, (2, 2)),
            linewidth=AXIS_LINEWIDTH,
        )
        ax.annotate(
            f"{count:,}", xy=(x, 100),
            xytext=(-2 if model == "pinnacle" else 0, -10),
            textcoords="offset points", color=colors[model],
            fontsize=TICK_LABEL_SIZE, weight="bold",
            ha="right" if model == "pinnacle" else "center", va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
            zorder=5,
        )

    ax.set_xlim(0, np.log10(12001))
    ax.set_ylim(0, 105)
    ax.set_xticks(np.log10(np.array([0, 1, 10, 100, 1000, 10000]) + 1))
    ax.set_xticklabels(["0", "1", "10", "100", "1,000", "10,000"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Other screened proteins encountered")
    ax.set_ylabel("Held-out Parkinson targets recovered (%)")
    clean_axes(ax, grid_axis="y")
    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.20, top=0.96)
    save_figure(fig, output, "parkinson_candidate_recovery_curve")


def plot_external_support(source: Path, output: Path) -> None:
    summary = pd.read_csv(source / "parkinson_candidate_external_support.csv")
    support_order = [
        "current_opentargets_parkinson_association_non_literature_only",
        "approved_human_drugbank_target_any_indication",
    ]
    support_labels = {
        support_order[0]: "Current Parkinson\nassociation",
        support_order[1]: "Approved human\nDrugBank target",
    }
    model_specs = [
        ("protscape", PROTSCAPE_COLOR),
        ("pinnacle", PINNACLE_COLOR),
    ]
    indexed = summary.set_index(["support_flag", "model"])
    y_positions = np.arange(len(support_order))[::-1]
    fig, ax = plt.subplots(figsize=figure_size(18.0, 9.1))
    for y, flag in zip(y_positions, support_order):
        values = [indexed.loc[(flag, model)] for model, _ in model_specs]
        percentages = [float(value["support_percent"]) for value in values]
        ax.plot(percentages, [y, y], color="#A5A5A5", linewidth=DATA_LINEWIDTH, zorder=1)
        for model_index, ((_, color), value) in enumerate(zip(model_specs, values)):
            percentage = float(value["support_percent"])
            ax.scatter(
                percentage, y, s=82, color=color, edgecolor="white",
                linewidth=BAR_EDGE_LINEWIDTH, zorder=3,
            )
            ax.annotate(
                f"{percentage:.1f}%\n"
                f"({int(value['supported_candidates'])}/{int(value['candidate_count'])})",
                xy=(percentage, y), xytext=(0, 8 if model_index == 0 else -8),
                textcoords="offset points", color=color,
                fontsize=TICK_LABEL_SIZE, weight="bold", ha="center",
                va="bottom" if model_index == 0 else "top",
                annotation_clip=False,
            )
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, len(support_order) - 0.45)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(y_positions)
    ax.set_yticklabels([support_labels[flag] for flag in support_order])
    ax.set_xlabel("Candidates with external support (%)")
    ax.set_title("Candidates with probability ≥0.5", pad=8)
    clean_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.18, top=0.90)
    save_figure(fig, output, "parkinson_candidate_external_support")


def plot_synaptic_completion(source: Path, output: Path) -> None:
    table = pd.read_csv(source / "parkinson_synaptic_group_completion.csv")
    table = table.sort_values("group_order")
    y_positions = np.arange(len(table))[::-1]
    fig, ax = plt.subplots(figsize=figure_size(18.0, 6.1))
    for y, row in zip(y_positions, table.itertuples()):
        ax.plot(
            [row.protscape_completion_rank, row.pinnacle_completion_rank], [y, y],
            color="#A5A5A5", linewidth=DATA_LINEWIDTH,
            solid_capstyle="round", zorder=1,
        )
        for rank, color, offset, valign in [
            (row.protscape_completion_rank, PROTSCAPE_COLOR, (0, 7), "bottom"),
            (row.pinnacle_completion_rank, PINNACLE_COLOR, (0, -8), "top"),
        ]:
            ax.scatter(
                rank, y, s=78, color=color, edgecolor="white",
                linewidth=BAR_EDGE_LINEWIDTH, zorder=3,
            )
            ax.annotate(
                f"{int(rank):,}", xy=(rank, y), xytext=offset,
                textcoords="offset points", ha="center", va=valign,
                color=color, fontsize=TICK_LABEL_SIZE, weight="bold",
            )
    ax.set_xscale("log")
    ax.set_xlim(0.7, 12_000)
    ax.set_xticks([1, 10, 100, 1_000, 10_000])
    ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.minorticks_off()
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row.group} ({row.member_label})" for row in table.itertuples()])
    ax.set_ylim(-0.55, len(table) - 0.45)
    ax.set_xlabel("Rank of last recovered member among 13,303 label-excluded proteins")
    clean_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.27, top=0.96)
    save_figure(fig, output, "parkinson_synaptic_group_completion_depth")


def network_graph(source: Path) -> tuple[nx.Graph, pd.DataFrame]:
    edges = pd.read_csv(source / "parkinson_string_network_edges.csv")
    nodes = pd.read_csv(source / "parkinson_leiden_nodes.csv")
    graph = nx.from_pandas_edgelist(
        edges, "protein_a", "protein_b",
        edge_attr=["experimental_score", "combined_score"],
    )
    if set(graph) != set(nodes["protein"]):
        raise ValueError("Released connected network nodes and edges disagree")
    return graph, nodes


def annotate_network(ax: plt.Axes, graph: nx.Graph, position: dict[str, np.ndarray]) -> None:
    for node, (horizontal, vertical) in NETWORK_LABEL_OFFSETS.items():
        if node not in graph:
            continue
        ax.annotate(
            node, xy=position[node], xytext=(horizontal, vertical),
            textcoords="offset points",
            ha="left" if horizontal >= 0 else "right",
            va="bottom" if vertical >= 0 else "top",
            fontsize=TICK_LABEL_SIZE, weight="semibold", color="#242424",
            zorder=5,
            bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    x_values = np.array([coordinates[0] for coordinates in position.values()])
    y_values = np.array([coordinates[1] for coordinates in position.values()])
    ax.set_xlim(x_values.min() - 0.05 * np.ptp(x_values), x_values.max() + 0.05 * np.ptp(x_values))
    ax.set_ylim(y_values.min() - 0.05 * np.ptp(y_values), y_values.max() + 0.05 * np.ptp(y_values))
    ax.set_axis_off()


def plot_role_network(source: Path, output: Path, stem: str, known_color: str) -> None:
    graph, nodes = network_graph(source)
    position = nx.spring_layout(graph, seed=42, k=1.2 / graph.number_of_nodes() ** 0.5)
    fig, ax = plt.subplots(figsize=figure_size(18.0, 15.5))
    nx.draw_networkx_edges(
        graph, position, ax=ax, edge_color="#BDBDBD", alpha=0.33,
        width=AXIS_LINEWIDTH,
    )
    for role, fill_color, border_color in [
        ("benchmark_positive", known_color, "#222222" if known_color == "white" else "#4A4A4A"),
        ("candidate", PROTSCAPE_COLOR, "#222222" if known_color == "white" else "#8A3517"),
    ]:
        role_nodes = nodes.loc[nodes["node_role"].eq(role), "protein"].tolist()
        nx.draw_networkx_nodes(
            graph, position, nodelist=role_nodes, node_color=fill_color,
            edgecolors=border_color, linewidths=BAR_EDGE_LINEWIDTH,
            node_size=36 if known_color == "white" else 38, ax=ax,
        )
    annotate_network(ax, graph, position)
    save_figure(fig, output, stem)


def plot_leiden_network(source: Path, output: Path) -> None:
    graph, nodes = network_graph(source)
    position = nx.spring_layout(graph, seed=42, k=1.2 / graph.number_of_nodes() ** 0.5)
    cluster = nodes.set_index("protein")["leiden_cluster"].astype(int).to_dict()
    fig, ax = plt.subplots(figsize=figure_size(18.0, 15.5))
    inter_edges = [(a, b) for a, b in graph.edges() if cluster[a] != cluster[b]]
    nx.draw_networkx_edges(
        graph, position, edgelist=inter_edges, edge_color="#BDBDBD",
        alpha=0.12, width=AXIS_LINEWIDTH, ax=ax,
    )
    for module_id in sorted(MODULE_COLORS):
        module_nodes = nodes.loc[nodes["leiden_cluster"].eq(module_id)]
        subgraph = graph.subgraph(module_nodes["protein"])
        nx.draw_networkx_edges(
            graph, position, edgelist=list(subgraph.edges()),
            edge_color=MODULE_COLORS[module_id], alpha=0.30,
            width=AXIS_LINEWIDTH, ax=ax,
        )
        for role, fill_color in [("benchmark_positive", "white"), ("candidate", PROTSCAPE_COLOR)]:
            role_nodes = module_nodes.loc[module_nodes["node_role"].eq(role), "protein"].tolist()
            nx.draw_networkx_nodes(
                graph, position, nodelist=role_nodes, node_color=fill_color,
                node_size=36, edgecolors=MODULE_COLORS[module_id],
                linewidths=BAR_EDGE_LINEWIDTH, ax=ax,
            )
    annotate_network(ax, graph, position)
    save_figure(fig, output, "parkinson_known_candidates_leiden_modules")


def plot_leiden_legend(output: Path) -> None:
    """Render the separate role/module legend emitted by the source script."""
    fig, ax = plt.subplots(figsize=figure_size(18.0, 10.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(MODULE_LABELS) + 2)

    for row, (label, fill_color) in enumerate(
        [
            ("ProtScape-predicted candidate", PROTSCAPE_COLOR),
            ("Known benchmark target", "white"),
        ]
    ):
        y = len(MODULE_LABELS) + 1.5 - row
        ax.scatter(
            0.043, y, s=45, facecolor=fill_color, edgecolor="#4A4A4A",
            linewidth=BAR_EDGE_LINEWIDTH,
        )
        ax.text(0.085, y, label, ha="left", va="center", fontsize=TICK_LABEL_SIZE)

    for row, module_id in enumerate(sorted(MODULE_LABELS)):
        y = len(MODULE_LABELS) - row - 0.5
        ax.scatter(
            0.043, y, s=45, facecolor="white",
            edgecolor=MODULE_COLORS[module_id], linewidth=BAR_EDGE_LINEWIDTH,
        )
        ax.text(
            0.085, y, f"M{module_id}  {MODULE_LABELS[module_id]}",
            ha="left", va="center", fontsize=TICK_LABEL_SIZE,
        )
    ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.97)
    save_figure(fig, output, "protscape_leiden_modules_legend")


def plot_string_enrichment(source: Path, output: Path) -> None:
    terms = pd.read_csv(source / "parkinson_string_selected_terms.csv")
    display = pd.read_csv(source / "parkinson_string_main_enrichment.csv")
    terms = terms.sort_values(["category_order", "term_order"]).reset_index(drop=True)
    y_by_index = {}
    header_y = {}
    cursor = float(len(terms) + len(STRING_CATEGORY_LABELS) * 0.9)
    for category in STRING_CATEGORY_LABELS:
        indices = terms.index[terms["category"].eq(category)].tolist()
        if not indices:
            continue
        header_y[category] = cursor
        cursor -= 0.8
        for index in indices:
            y_by_index[index] = cursor
            cursor -= 1.0
        cursor -= 0.45
    y = np.array([y_by_index[index] for index in terms.index])

    fig, axes = plt.subplots(
        1, 2, figsize=figure_size(18.0, 11.4), sharex=True, sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08},
    )
    set_specs = [
        ("known_targets", "Known targets", "#4A4A4A"),
        ("protscape_candidates", "ProtScape\nnovel targets", "#222222"),
    ]
    color_limit = max(1.0, display["minus_log10_fdr"].replace([np.inf], np.nan).max())
    for ax, (set_name, title, edge) in zip(axes, set_specs):
        values = terms[["category", "term", "description"]].merge(
            display[display["set"].eq(set_name)],
            on=["category", "term", "description"], how="left",
        )
        available = values["number_of_genes"].notna()
        plotted = values.loc[available]
        scatter = ax.scatter(
            plotted["log2_fold_enrichment"], y[available.to_numpy()],
            s=20 + 4.0 * plotted["number_of_genes"],
            c=plotted["minus_log10_fdr"], cmap="viridis", vmin=0,
            vmax=color_limit, edgecolor=edge, linewidth=BAR_EDGE_LINEWIDTH, zorder=3,
        )
        ax.axvline(0, color="#D0D0D0", linewidth=AXIS_LINEWIDTH)
        ax.set_title(title)
        ax.set_xlabel(r"$\log_2$ fold enrichment")
        for name in ("top", "right", "left"):
            ax.spines[name].set_visible(False)
        ax.tick_params(axis="y", length=0)

    x_values = pd.to_numeric(display["log2_fold_enrichment"], errors="coerce").dropna()
    x_min, x_max = min(0.0, float(x_values.min())), float(x_values.max())
    padding = 0.04 * max(x_max - x_min, 1.0)
    for ax in axes:
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_xticks(np.arange(0.0, 2.0 * np.floor(x_max / 2.0) + 0.1, 2.0))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(terms["description"].tolist())
    for category, label in STRING_CATEGORY_LABELS.items():
        indices = terms.index[terms["category"].eq(category)].tolist()
        if not indices:
            continue
        axes[0].text(
            -0.05, header_y[category], label,
            transform=axes[0].get_yaxis_transform(), ha="right", va="bottom",
            fontsize=PANEL_TITLE_SIZE, weight="bold",
        )
        if max(indices) < len(terms) - 1:
            for ax in axes:
                ax.axhline(y[max(indices)] - 0.7, color="#D8D8D8", linewidth=0.5)
    axes[0].set_ylim(cursor + 0.6, max(header_y.values()) + 0.5)
    fig.subplots_adjust(left=0.34, right=0.74, bottom=0.16, top=0.92)
    handles = [
        axes[1].scatter([], [], s=20 + 4.0 * value, facecolor="white",
                        edgecolor="#555555", linewidth=BAR_EDGE_LINEWIDTH)
        for value in [10, 30, 60]
    ]
    fig.legend(
        handles, ["10", "30", "60"], title="Genes", loc="lower left",
        bbox_to_anchor=(0.82, 0.14), bbox_transform=fig.transFigure,
        frameon=False, scatterpoints=1, handletextpad=0.7,
        labelspacing=0.8, borderaxespad=0.0,
    )
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.035, pad=0.04)
    colorbar.set_label(r"$-\log_{10}$(FDR)")
    save_figure(fig, output, "parkinson_string_functional_enrichment")


def plot_reactome_modules(source: Path, output: Path) -> None:
    display = pd.read_csv(source / "parkinson_reactome_module_enrichment.csv")
    for module_id in sorted(MODULE_COLORS):
        rows = display[display["leiden_cluster"].eq(module_id)].sort_values(
            ["fdr", "fold_enrichment", "module_hits", "term"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
        if rows.empty:
            continue
        y = np.arange(len(rows), dtype=float)
        labels = [textwrap.fill(label, width=37, break_long_words=False) for label in rows["description"]]
        significance = rows["minus_log10_fdr"].to_numpy(dtype=float)
        known_fraction = (
            rows["benchmark_positive_hits"] / rows["module_hits"]
        ).to_numpy(dtype=float)
        known_width = significance * known_fraction
        candidate_width = significance - known_width

        fig, ax = plt.subplots(
            figsize=figure_size(18.0, MODULE_PLOT_HEIGHT_CM)
        )
        ax.barh(
            y, known_width, height=0.55,
            color=MODULE_COLORS[module_id], edgecolor="none",
        )
        ax.barh(
            y, candidate_width, left=known_width, height=0.55,
            color=lighten_color(MODULE_COLORS[module_id]), edgecolor="none",
        )
        ax.barh(
            y, significance, height=0.55, color="none", edgecolor="#222222",
            linewidth=BAR_EDGE_LINEWIDTH,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=MODULE_TERM_LABEL_SIZE)
        ax.invert_yaxis()
        ax.set_xlim(0, float(rows["minus_log10_fdr"].max()) * 1.04)
        ax.set_xlabel(
            r"$-\log_{10}$(FDR)",
            fontsize=MODULE_AXIS_LABEL_SIZE,
        )
        ax.set_title(
            textwrap.fill(MODULE_LABELS[module_id], width=52, break_long_words=False),
            fontsize=MODULE_TITLE_SIZE, fontweight="bold", pad=12,
        )
        clean_axes(ax)
        ax.tick_params(axis="x", labelsize=MODULE_TICK_LABEL_SIZE)
        ax.tick_params(axis="y", labelsize=MODULE_TERM_LABEL_SIZE)
        fig.subplots_adjust(left=0.52, right=0.98, bottom=0.17, top=0.88)
        save_figure(fig, output, f"parkinson_M{module_id}_reactome_top5")


def plot_reactome_hit_composition_legend(output: Path) -> None:
    color = "#555555"
    handles = [
        Patch(
            facecolor=color,
            edgecolor="#222222",
            linewidth=BAR_EDGE_LINEWIDTH,
            label="Known benchmark targets",
        ),
        Patch(
            facecolor=lighten_color(color),
            edgecolor="#222222",
            linewidth=BAR_EDGE_LINEWIDTH,
            label="ProtScape-predicted candidates",
        ),
    ]
    fig, ax = plt.subplots(figsize=figure_size(8.8, 4.0))
    ax.legend(
        handles=handles,
        title="Proteins contributing to enriched term",
        loc="center",
        ncol=1,
        fontsize=MODULE_TERM_LABEL_SIZE,
        title_fontsize=MODULE_TERM_LABEL_SIZE,
        frameon=False,
    )
    ax.set_axis_off()
    save_figure(fig, output, "parkinson_reactome_hit_composition_legend")


def plot_all(source: Path, output: Path) -> None:
    source = Path(source)
    output = Path(output)
    with matplotlib.rc_context(PLOT_RC):
        plot_candidate_recovery(source, output)
        plot_external_support(source, output)
        plot_synaptic_completion(source, output)
        plot_leiden_network(source, output)
        plot_leiden_legend(output)
        plot_string_enrichment(source, output)

        plot_role_network(
            source, output,
            "parkinson_known_candidates_experimental_network", "#B8B8B8",
        )
        plot_reactome_modules(source, output)
        plot_reactome_hit_composition_legend(output)
