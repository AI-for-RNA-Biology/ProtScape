"""Section 6 (plots) + section 8: giant-network, module-density, and per-module
enrichment/pathway visualisations.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import shapiro, ttest_ind, wilcoxon

from .condition_enrichment import _fmt_p
from .gene_expression import CYTO_CONDITION_LABELS, CYTO_CONDITIONS
from .layout import module_centroid_positions, points_per_data_unit

EDGE_TYPE_COLORS = {"ct": "firebrick", "vcp": (4 / 255, 144 / 255, 158 / 255),
                     "known": (0, 0, 0, 0.08), "other": (1, 1, 1, 0)}
EDGE_TYPE_WIDTHS = {"ct": 1.3, "vcp": 1.3, "known": 0.4, "other": 0.4}

PATHWAY_CATEGORIES = ["RCTM", "Function", "SMART", "WikiPathways"]


# --- section 6 plots (padj-annotated per-module bar charts) -----------------

def plot_distribution_novel_edges(enrichment_new_known: pd.DataFrame, out_path: Path) -> None:
    """Section A: known vs new % of edges per module, padj annotated -
    `distribution_novel_edges_per_module.pdf` in the R script."""
    modules = enrichment_new_known.index.to_numpy()
    x = np.arange(len(modules))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(8, len(modules) * 0.35), 5))
    ax.bar(x - width / 2, enrichment_new_known["perc_known"], width,
           color="white", edgecolor="black", label="known")
    ax.bar(x + width / 2, enrichment_new_known["perc_new"], width, color="black", label="new")
    ax.set_xticks(x)
    ax.set_xticklabels(modules, fontsize=6, rotation=90)
    ax.set(xlabel="module ID", ylabel="% of edges per module")
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    ymax = enrichment_new_known[["perc_known", "perc_new"]].max(axis=1)
    for xi, (m, ym) in enumerate(zip(modules, ymax)):
        ax.text(xi, ym + 0.5, _fmt_p(enrichment_new_known.loc[m, "padj"]),
                fontsize=4, rotation=90, ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_enrichment_compared_known(enrichment_cond_known: pd.DataFrame, mod_palette: dict,
                                    condition_cols, out_path: Path, per_page: int = 12) -> None:
    """Section B: per-module grid, one panel per module - `known` bar plus one bar per
    condition, padj annotated - `enrichment_per_module_compared_known.pdf` in the R script.
    All bars (including `known`) use the same column-normalized percentage: this
    module's share of ALL edges of that type in the graph, matching R's
    `per_edges_modules`."""
    perc_cols = [f"perc_{c}" for c in condition_cols]
    padj_cols = [f"padj_{c}" for c in condition_cols]
    labels = ["known"] + list(condition_cols)
    modules = enrichment_cond_known.index.to_numpy()

    with PdfPages(out_path) as pdf:
        for start in range(0, len(modules), per_page):
            chunk = modules[start:start + per_page]
            fig, axes = plt.subplots(4, 3, figsize=(9, 11))
            for ax, m in zip(axes.flat, chunk):
                row = enrichment_cond_known.loc[m]
                heights = [row["perc_known"]] + [row[c] for c in perc_cols]
                bar_colors = (["lightgray", "black"] * (len(condition_cols) // 2 + 1))[:len(condition_cols)]
                ax.bar(range(len(heights)), heights, color=[mod_palette[int(m)]] + bar_colors)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=4, rotation=90)
                ax.set_ylabel(f"% edges mod {int(m)}", fontsize=6)
                for xi, c in enumerate(padj_cols, start=1):
                    ax.text(xi, heights[xi], _fmt_p(row[c]), fontsize=6, rotation=90, ha="center", va="bottom")
            for ax in axes.flat[len(chunk):]:
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {out_path}")


def plot_enrichment_vcp_ct(enrichment_vcp_ct: pd.DataFrame, mod_palette: dict,
                            out_path: Path, n_nodes_map: dict, per_page: int = 12) -> None:
    """Section C: per-module grid comparing VCP vs CTRL at each timepoint -
    `enrichment_per_module_vcp_ct.pdf` in the R script. The R version's bar
    labels for this plot swap the last two bars (`"d35_vcp"`/`"d35_ct"` are
    drawn over `ct_35`/`vcp_35` in that order) - the labels below
    (`ct_d22, vcp_22, ct_35, vcp_35`) match the actual bar/data order instead."""
    labels = ["ct_d22", "vcp_22", "ct_35", "vcp_35"]
    cols = ["perc_ct_d22", "perc_vcp_22", "perc_ct_35", "perc_vcp_35"]

    with PdfPages(out_path) as pdf:
        for start in range(0, len(enrichment_vcp_ct), per_page):
            chunk = enrichment_vcp_ct.iloc[start:start + per_page]
            fig, axes = plt.subplots(4, 3, figsize=(9, 11))
            for ax, (_, row) in zip(axes.flat, chunk.iterrows()):
                m = int(row["module"])
                heights = [row[c] for c in cols]
                ax.bar(range(4), heights, color=["lightgray", mod_palette[m], "lightgray", mod_palette[m]])
                ax.set_xticks(range(4))
                ax.set_xticklabels(labels, fontsize=4, rotation=90)
                ax.set_ylabel(f"% edges mod {m}", fontsize=6)
                ax.text(0.5, max(heights[0], heights[1]), _fmt_p(row["padj_d22"]),
                        fontsize=6, ha="center", va="bottom")
                ax.text(2.5, max(heights[2], heights[3]), _fmt_p(row["padj_d35"]),
                        fontsize=6, ha="center", va="bottom")
                ax.set_title(f"nedges={int(row['n_tot'])}, nprot={n_nodes_map[m]}", fontsize=5)
            for ax in axes.flat[len(chunk):]:
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {out_path}")


# --- section 8: giant network / module density -------------------------------

def plot_giant(giant: nx.Graph, module_of: dict, mod_ids, mod_palette: dict,
               layout: dict, day: str = "d22", ax=None) -> None:
    edge_type_attr = f"edge_type_{day}"
    ax = ax or plt.gca()

    # Draw low-information edges (known/other - almost all of them) first and
    # faint, so they read as background texture instead of visually merging
    # different-colored modules together; condition-specific edges are drawn
    # on top afterwards so they stay legible against that background.
    background_edges = [(u, v) for u, v, d in giant.edges(data=True) if d[edge_type_attr] in ("known", "other")]
    foreground_edges = [(u, v) for u, v, d in giant.edges(data=True) if d[edge_type_attr] in ("ct", "vcp")]

    nx.draw_networkx_edges(giant, layout, edgelist=background_edges,
                            edge_color=EDGE_TYPE_COLORS["known"], width=EDGE_TYPE_WIDTHS["known"], ax=ax)
    fg_colors = [EDGE_TYPE_COLORS[giant.edges[u, v][edge_type_attr]] for u, v in foreground_edges]
    fg_widths = [EDGE_TYPE_WIDTHS[giant.edges[u, v][edge_type_attr]] for u, v in foreground_edges]
    nx.draw_networkx_edges(giant, layout, edgelist=foreground_edges,
                            edge_color=fg_colors, width=fg_widths, ax=ax)

    node_colors = [mod_palette[module_of[n]] for n in giant.nodes()]
    nx.draw_networkx_nodes(giant, layout, node_size=10, node_color=node_colors,
                            edgecolors="grey", linewidths=0.3, ax=ax)

    ax.legend(handles=[
        plt.Line2D([0], [0], color="gray", lw=2, label="Known PPI edge"),
        plt.Line2D([0], [0], color="firebrick", lw=2, label=f"New edge (CT {day})"),
        plt.Line2D([0], [0], color=EDGE_TYPE_COLORS["vcp"], lw=2, label=f"New edge (VCP {day})"),
    ], loc="upper right", fontsize=7, frameon=False)
    ax.set_axis_off()


def plot_giant_simple(giant: nx.Graph, module_of: dict, mod_palette: dict, layout: dict,
                       edges: str = "known", ax=None) -> None:
    """Simplified companion to `plot_giant()`: `edges="known"` draws only
    known-PPI edges; `edges="all"` draws known edges as background plus every new
    edge (any day, any of CT/VCP) pooled into a single style - no discrimination
    between d22/d35 or CT/VCP, just "known" vs "new"."""
    ax = ax or plt.gca()
    known_edges = [(u, v) for u, v, d in giant.edges(data=True) if d["present_in_ppi"] == 1]

    if edges == "known":
        nx.draw_networkx_edges(giant, layout, edgelist=known_edges,
                                edge_color=(0, 0, 0, 0.3), width=0.6, ax=ax)
        legend_handles = [plt.Line2D([0], [0], color="grey", lw=2, label="Known PPI edge")]
    elif edges == "all":
        new_edges = [(u, v) for u, v, d in giant.edges(data=True) if d["present_in_ppi"] == 0]
        nx.draw_networkx_edges(giant, layout, edgelist=known_edges,
                                edge_color=EDGE_TYPE_COLORS["known"], width=EDGE_TYPE_WIDTHS["known"], ax=ax)
        nx.draw_networkx_edges(giant, layout, edgelist=new_edges,
                                edge_color="black", width=0.8, ax=ax)
        legend_handles = [
            plt.Line2D([0], [0], color="gray", lw=2, label="Known PPI edge"),
            plt.Line2D([0], [0], color="black", lw=2, label="New edge (any condition/timepoint)"),
        ]
    else:
        raise ValueError(f"edges must be 'known' or 'all', got {edges!r}")

    node_colors = [mod_palette[module_of[n]] for n in giant.nodes()]
    nx.draw_networkx_nodes(giant, layout, node_size=10, node_color=node_colors,
                            edgecolors="grey", linewidths=0.3, ax=ax)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, frameon=False)
    ax.set_axis_off()


def _canonical_module_subgraph(giant: nx.Graph, module_of: dict, module_id: int) -> nx.Graph:
    """Build the module subgraph in stored node/edge order for seeded layouts."""
    node_set = {n for n, m in module_of.items() if m == module_id}
    sub_g = nx.Graph()
    sub_g.add_nodes_from(n for n in giant.nodes() if n in node_set)
    for u, v, d in giant.edges(data=True):
        if u in node_set and v in node_set:
            sub_g.add_edge(u, v, **d)
    return sub_g


def plot_focus_module(giant: nx.Graph, module_of: dict, mod_palette: dict,
                       module_id: int, day: str = "d22", seed: int = 12,
                       show_labels: bool = True, ax=None) -> None:
    """`day="d22"`/`"d35"` (default): colors this module's edges by that
    day's `edge_type_{day}` attribute - known PPI grey, new-in-CT firebrick,
    new-in-VCP yellow (any edge new at the OTHER day, or "other", is drawn
    invisible). `day="all"` instead ignores day/genotype entirely and just
    uses `present_in_ppi` - every known edge grey, every new edge (found at
    ANY day/genotype) black - a single "everything this module has" summary
    page, complementing the d22-only/d35-only pages."""
    sub_g = _canonical_module_subgraph(giant, module_of, module_id)
    layout = nx.spring_layout(sub_g, seed=seed)
    ax = ax or plt.gca()

    if day == "all":
        known_edges = [(u, v) for u, v, d in sub_g.edges(data=True) if d["present_in_ppi"] == 1]
        new_edges = [(u, v) for u, v, d in sub_g.edges(data=True) if d["present_in_ppi"] == 0]
        nx.draw_networkx_edges(sub_g, layout, edgelist=known_edges, edge_color="grey", width=0.6, ax=ax)
        nx.draw_networkx_edges(sub_g, layout, edgelist=new_edges, edge_color="black", width=1.0, ax=ax)
        legend_handles = [
            plt.Line2D([0], [0], color="grey", lw=2, label="Known PPI edge"),
            plt.Line2D([0], [0], color="black", lw=2, label="New edge (any condition/timepoint)"),
        ]
    else:
        edge_type_attr = f"edge_type_{day}"
        focus_colors = {"ct": "firebrick", "vcp": "yellow", "known": "grey", "other": (1, 1, 1, 0)}
        edge_colors = [focus_colors[d[edge_type_attr]] for _, _, d in sub_g.edges(data=True)]
        edge_widths = [1.3 if d[edge_type_attr] in ("ct", "vcp") else 0.6 for _, _, d in sub_g.edges(data=True)]
        nx.draw_networkx_edges(sub_g, layout, edge_color=edge_colors, width=edge_widths, ax=ax)
        legend_handles = [
            plt.Line2D([0], [0], color="grey", lw=2, label="Known PPI edge"),
            plt.Line2D([0], [0], color="firebrick", lw=2, label=f"New ct {day}"),
            plt.Line2D([0], [0], color="yellow", lw=2, label=f"New vcp {day}"),
        ]

    nx.draw_networkx_nodes(sub_g, layout, node_size=10, node_color=[mod_palette[module_id]], ax=ax)
    if show_labels:
        nx.draw_networkx_labels(sub_g, layout, font_size=6, ax=ax)

    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, frameon=False)
    ax.set_title(f"module {module_id}" + (" (all edges)" if day == "all" else ""), fontsize=8)
    ax.set_axis_off()


def _paired_wilcoxon_p(log2fc_df: pd.DataFrame, members: list, ctrl_col: str, vcp_col: str) -> float:
    """Paired (not independent-samples) test: CTRL vs VCP log2FC for the
    SAME set of genes at one timepoint - each gene is its own pair (its
    CTRL-vs-D0 and VCP-vs-D0 log2FC), so Wilcoxon signed-rank is the right
    test, not e.g. Mann-Whitney (which assumes two independent samples).
    `NaN` if fewer than 1 gene has both values, or if every paired
    difference is exactly 0 (wilcoxon raises on that degenerate case)."""
    paired = log2fc_df.reindex(members)[[ctrl_col, vcp_col]].dropna()
    if len(paired) < 1:
        return np.nan
    try:
        return wilcoxon(paired[ctrl_col], paired[vcp_col]).pvalue
    except ValueError:
        return np.nan


def _normal_enough(values: np.ndarray, alpha: float = 0.05) -> bool:
    """Shapiro-Wilk normality check - `False` (i.e. "don't trust a
    normal-theory test here") whenever there's too little data for the test
    to mean anything (`< 3` values) or the test itself rejects normality."""
    if len(values) < 3:
        return False
    try:
        return shapiro(values).pvalue > alpha
    except ValueError:
        return False


def _welch_p_if_normal(log2fc_df: pd.DataFrame, members: list, ctrl_col: str, vcp_col: str):
    """Welch's t-test (unequal-variance, INDEPENDENT samples - CTRL and VCP
    values are NOT gene-matched here, unlike `_paired_wilcoxon_p`'s primary
    test) for CTRL vs VCP log2FC, only computed if both groups individually
    pass a Shapiro-Wilk normality check - `None` otherwise, since a
    normal-theory test on visibly non-normal data (common for log2FC, which
    is often skewed/heavy-tailed) can be pulled around by a handful of
    outlier genes in a way rank-based Wilcoxon isn't."""
    ctrl_vals = log2fc_df.reindex(members)[ctrl_col].dropna().to_numpy()
    vcp_vals = log2fc_df.reindex(members)[vcp_col].dropna().to_numpy()
    if not (_normal_enough(ctrl_vals) and _normal_enough(vcp_vals)):
        return None
    return ttest_ind(ctrl_vals, vcp_vals, equal_var=False).pvalue


def plot_module_log2fc_boxplot(ax, log2fc_df: pd.DataFrame, members: list, module_id: int,
                                mod_palette: dict, conditions: list = CYTO_CONDITIONS,
                                condition_labels: dict = CYTO_CONDITION_LABELS) -> None:
    """Boxplot of `log2fc_df` across every gene in `members`, one box per
    entry in `conditions` (default `CYTO_CONDITIONS` - CTRL/VCP x D3/D7/D14/
    D22/D35, matching `gene_expression.cyto_log2fc_all_timepoints()`'s
    output) - shows how this module's genes moved from D0 overall, not
    just at the PPI-edge
    level. `conditions` MUST alternate CTRL/VCP per timepoint (`ctrl_d3,
    vcp_d3, ctrl_d7, vcp_d7, ...`) - box coloring, pairing for the
    significance tests below, and the CTRL/VCP legend all assume that
    order. Genes in `members` absent from `log2fc_df` are silently dropped
    (not 0-filled, unlike the gene-focus expression boxes - here it's a
    real distribution, not a per-node annotation, so a missing gene should
    just not count rather than pull the distribution toward 0).

    A paired Wilcoxon signed-rank p-value (CTRL vs VCP, same genes - the
    primary test, since the two groups are gene-matched, not independent)
    is annotated above each timepoint's pair of boxes, together with
    Welch's t-test p-value IF (and only if) both groups pass a Shapiro-Wilk
    normality check - Welch's treats CTRL/VCP as independent samples
    (ignoring the gene-level pairing), so it's shown as a secondary,
    normal-theory cross-check, not a replacement for the Wilcoxon result."""
    labels = [condition_labels[c] for c in conditions]
    data = [log2fc_df.reindex(members)[c].dropna().to_numpy() for c in conditions]
    colors = ["lightgray" if i % 2 == 0 else mod_palette[module_id] for i in range(len(conditions))]

    bp = ax.boxplot(data, labels=labels, showfliers=False, widths=0.6, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")

    pad = 0.05 * max((max((d.max() for d in data if len(d)), default=1.0)
                       - min((d.min() for d in data if len(d)), default=0.0)), 1e-6)
    for k in range(len(conditions) // 2):
        ctrl_col, vcp_col = conditions[2 * k], conditions[2 * k + 1]
        p_wilcoxon = _paired_wilcoxon_p(log2fc_df, members, ctrl_col, vcp_col)
        p_welch = _welch_p_if_normal(log2fc_df, members, ctrl_col, vcp_col)
        label = f"Wilcoxon {_fmt_p(p_wilcoxon)}"
        if p_welch is not None:
            label += f"\nWelch {_fmt_p(p_welch)}"
        d_ctrl, d_vcp = data[2 * k], data[2 * k + 1]
        y = max(d_ctrl.max() if len(d_ctrl) else -np.inf, d_vcp.max() if len(d_vcp) else -np.inf) + pad
        ax.text(2 * k + 1.5, y, label, fontsize=4, ha="center", va="bottom")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_ylabel("log2FC vs D0", fontsize=6)
    ax.set_title(f"module {module_id} (n={len(members)} genes)", fontsize=6)


def save_module_log2fc_boxplots(module_ids, log2fc_df: pd.DataFrame, module_of: dict, mod_palette: dict,
                                 out_path: Path, conditions: list = CYTO_CONDITIONS,
                                 condition_labels: dict = CYTO_CONDITION_LABELS) -> None:
    """Saves a multi-page PDF - ONE page per module in `module_ids` (not a
    grid: with `len(conditions)` boxes per panel - 10, by default, D3-D35 x
    CTRL/VCP - a small grid panel doesn't leave room to read the per-day
    significance annotations), each `plot_module_log2fc_boxplot` - the
    log2FC-vs-D0 distribution across every gene in that module. Default
    `conditions`/`condition_labels` are the full D3/D7/D14/D22/D35 x
    CTRL/VCP time course from the Cytoplasmic-only expression matrix
    (`gene_expression.cyto_log2fc_all_timepoints`)."""
    if not module_ids:
        print(f"No modules of interest for {out_path.name}, skipping")
        return

    with PdfPages(out_path) as pdf:
        for m in module_ids:
            members = [n for n, mm in module_of.items() if mm == m]
            fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * len(conditions)), 5))
            plot_module_log2fc_boxplot(ax, log2fc_df, members, m, mod_palette,
                                        conditions=conditions, condition_labels=condition_labels)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {out_path}")


def _cluster_row_order(data: pd.DataFrame) -> pd.DataFrame:
    """Reorders `data`'s rows by hierarchical clustering (average-linkage,
    Euclidean, `scipy.cluster.hierarchy.leaves_list`) on their values across
    columns - the same "similar rows end up near each other" reordering
    seaborn's `clustermap` applies, so the heatmap doesn't just show genes
    in whatever incidental order they were listed in. Missing values are
    0-filled for the distance computation ONLY (doesn't touch what's
    actually displayed - real NaNs stay NaN, still rendered grey). No-op
    (order unchanged) for <=2 rows, or if linkage degenerates (e.g. every
    row identical)."""
    if len(data) <= 2:
        return data
    try:
        order = leaves_list(linkage(data.fillna(0.0).to_numpy(), method="average", metric="euclidean"))
    except Exception:
        return data
    return data.iloc[order]


def plot_module_log2fc_heatmap(ax, data: pd.DataFrame, module_id: int, condition_labels: list,
                                cmap_name: str = "PuOr", n_members: int = None) -> None:
    """Heatmap (rows=genes, columns=conditions) of `data` - log2FC-vs-D0
    values, one row per gene in the module, one column per entry in
    `condition_labels` - complementing `plot_module_log2fc_boxplot`'s
    distribution-only view with the per-gene detail. Rows are hierarchically
    clustered first (`_cluster_row_order`) so similar genes group together,
    like seaborn's `clustermap`; gene names are NOT drawn on the y-axis
    (there are typically far too many to read individually - the point of
    this view is the pattern, not looking up one specific gene by eye).

    `data` is expected to already have any gene with NO value across every
    `condition_labels` column dropped (see `save_module_log2fc_heatmaps`) -
    an all-missing row would just render as a blank grey stripe. That means
    `len(data)` (rows actually drawn) can be smaller than the module's true
    membership count; pass that as `n_members` so the title states both
    explicitly, rather than silently disagreeing with
    `plot_module_log2fc_boxplot`'s title (which always reports the full
    membership, since it drops NaNs per-condition/per-box instead of
    dropping whole genes).

    Drawn with `pcolormesh`, NOT `imshow`: `imshow` embeds a rasterized
    bitmap in the PDF, which reads as blurry/antialiased at most zoom levels
    in a PDF viewer; `pcolormesh` draws actual vector quadrilaterals, so
    every cell stays a crisp, sharp-edged square no matter how far you zoom
    in - "one square = one gene x one condition", not a smoothed image.
    Equal aspect forces those cells to be visually square, not just
    rectangular. Diverging color scale, symmetric around 0 (`TwoSlopeNorm`),
    sharpened to THIS module's own min/max magnitude rather than a fixed
    global range; missing values (gene not covered at that condition)
    render in light grey rather than silently taking some in-range color."""
    data = _cluster_row_order(data)
    values = data.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = np.abs(finite).max() if finite.size else 1.0
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgrey")

    masked = np.ma.masked_invalid(values)
    mesh = ax.pcolormesh(masked, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.08)
    ax.set_xlim(0, values.shape[1])
    ax.set_ylim(0, values.shape[0])
    ax.invert_yaxis()  # first (post-clustering) row at the top, like imshow's default
    ax.set_aspect("equal")

    ax.set_yticks([])
    ax.set_xticks(np.arange(len(condition_labels)) + 0.5)
    ax.set_xticklabels(condition_labels, fontsize=7, rotation=90)
    if n_members is not None and n_members != len(data):
        title_n = f"n={len(data)}/{n_members} genes with data"
    else:
        title_n = f"n={len(data)} genes"
    ax.set_title(f"module {module_id} ({title_n}) - log2FC vs D0", fontsize=9)

    fig = ax.figure
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02, label="log2FC vs D0")


def save_module_log2fc_heatmaps(module_ids, log2fc_df: pd.DataFrame, module_of: dict, out_path: Path,
                                 conditions: list = CYTO_CONDITIONS,
                                 condition_labels: dict = CYTO_CONDITION_LABELS) -> None:
    """Saves a multi-page PDF - ONE heatmap per page (not a grid, unlike
    `save_module_log2fc_boxplots` - a heatmap needs its own page to keep
    cells legibly sized) - one page per module in `module_ids`, each
    `plot_module_log2fc_heatmap`. Page size scales with the module's own
    gene count AND `len(conditions)`, at a fixed inches-per-cell, so the
    equal-aspect square cells `plot_module_log2fc_heatmap` draws fill the
    page instead of rendering tiny (if the figure were some fixed size not
    matching the data's row/column ratio) or getting clipped. Default
    `conditions`/`condition_labels` are the full D3/D7/D14/D22/D35 x
    CTRL/VCP time course from the Cytoplasmic-only expression matrix
    (`gene_expression.cyto_log2fc_all_timepoints`)."""
    if not module_ids:
        print(f"No modules of interest for {out_path.name}, skipping")
        return

    labels = [condition_labels[c] for c in conditions]
    cell_size = 0.15  # inches per cell, both directions
    with PdfPages(out_path) as pdf:
        for m in module_ids:
            members = sorted(n for n, mm in module_of.items() if mm == m)
            data = log2fc_df.reindex(members)[conditions].dropna(how="all")
            width = len(conditions) * cell_size + 2.5  # + colorbar/margins/xtick labels
            height = max(2.5, len(data) * cell_size + 1.2)  # + title margin
            fig, ax = plt.subplots(figsize=(width, height))
            plot_module_log2fc_heatmap(ax, data, m, labels, n_members=len(members))
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {out_path}")


def module_density_graph(giant: nx.Graph, module_of: dict, mod_ids, edge_filter=None) -> nx.Graph:
    """Collapses `giant` to one node per module - a simplified view of the giant
    network for when the node-level plot is too dense to read. Each module-node
    carries its true protein count (`size`); each edge (including self-loops) is
    an edge *density*: actual edges found between (or within) the corresponding
    modules, divided by the number of possible edges (`n_i * n_j` between two
    modules, `n*(n-1)/2` within one). `edge_filter(data) -> bool` optionally
    restricts which of `giant`'s edges count (e.g. known-PPI only); `None` counts
    every edge. Inter-module edges with zero actual edges are omitted; self-loops
    are always present, even at density 0."""
    module_nodes = {m: [n for n, mm in module_of.items() if mm == m] for m in mod_ids}
    sizes = {m: len(nodes) for m, nodes in module_nodes.items()}

    counts = {}
    for u, v, data in giant.edges(data=True):
        if edge_filter is not None and not edge_filter(data):
            continue
        mu, mv = module_of[u], module_of[v]
        key = (mu, mv) if mu <= mv else (mv, mu)
        counts[key] = counts.get(key, 0) + 1

    mod_g = nx.Graph()
    for m in mod_ids:
        mod_g.add_node(m, size=sizes[m])

    for m in mod_ids:
        n = sizes[m]
        possible = n * (n - 1) / 2
        actual = counts.get((m, m), 0)
        density = actual / possible if possible > 0 else 0.0
        mod_g.add_edge(m, m, weight=density, n_edges=actual, n_possible=possible)

    for i in range(len(mod_ids)):
        for j in range(i + 1, len(mod_ids)):
            a, b = mod_ids[i], mod_ids[j]
            possible = sizes[a] * sizes[b]
            actual = counts.get((a, b), 0)
            if actual > 0:
                mod_g.add_edge(a, b, weight=actual / possible, n_edges=actual, n_possible=possible)

    return mod_g


def module_density_matrix(mod_g: nx.Graph, mod_ids) -> pd.DataFrame:
    """Plain module x module connectivity matrix from `module_density_graph()`'s
    output - diagonal is within-module density (the self-loops), off-diagonal is
    between-module density (0 where `module_density_graph` omitted a zero-density
    inter-module edge)."""
    mat = pd.DataFrame(0.0, index=mod_ids, columns=mod_ids)
    for m in mod_ids:
        mat.loc[m, m] = mod_g.edges[m, m]["weight"]
    for u, v in mod_g.edges():
        if u != v:
            w = mod_g.edges[u, v]["weight"]
            mat.loc[u, v] = w
            mat.loc[v, u] = w
    return mat


def plot_module_density_graph(mod_g: nx.Graph, mod_palette: dict, pos: dict, ax=None,
                               size_scale: float = 15.0, min_node_size: float = 60.0,
                               max_node_size: float = 3000.0, min_edge_width: float = 0.15,
                               max_edge_width: float = 1.2, max_self: float = None,
                               max_inter: float = None, title: str = None) -> None:
    """Draws `module_density_graph()`'s output: node size ~ sqrt(true module size)
    (matplotlib's marker `s` is already an *area*, so sqrt-scaling the input
    keeps perceived size from exploding across a huge module-size range), and
    self-loop / inter-module edge thickness ~ density, normalized *separately*
    (within-module density's denominator ~n^2/2 is typically much larger than
    between-module density's ~n_i*n_j, so sharing one scale would collapse
    every inter-module edge to an indistinguishable minimum width).

    Self-loops are drawn manually as circles positioned just outside each
    node's actual rendered radius (via `points_per_data_unit`), rather than
    using networkx's built-in self-loop rendering - that built-in loop size
    doesn't scale enough with `node_size`, so on a wide size range it stays
    small enough to be completely hidden behind large nodes.

    By default the normalization ("strongest" = max observed weight) is
    computed from `mod_g` alone; pass `max_self`/`max_inter` explicitly (e.g.
    the max across both a known-only and a known+new graph) to put multiple
    calls on one shared, directly-comparable scale."""
    ax = ax or plt.gca()
    nodes = list(mod_g.nodes())
    node_size_map = {
        m: float(np.clip(size_scale * np.sqrt(mod_g.nodes[m]["size"]), min_node_size, max_node_size))
        for m in nodes
    }
    node_size_list = [node_size_map[m] for m in nodes]

    layout = {m: np.asarray(pos[m], dtype=float) for m in nodes}
    inter_edges = [(u, v) for u, v in mod_g.edges() if u != v]

    xs = [layout[m][0] for m in nodes]
    ys = [layout[m][1] for m in nodes]
    provisional_pad = 0.25 * max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    ax.set_xlim(min(xs) - provisional_pad, max(xs) + provisional_pad)
    ax.set_ylim(min(ys) - provisional_pad, max(ys) + provisional_pad)
    ax.set_aspect("equal")

    ppdu = points_per_data_unit(ax)
    max_marker_radius_data = max(np.sqrt(s / np.pi) / ppdu for s in node_size_list)
    final_pad = max(provisional_pad, max_marker_radius_data * 2.5)
    ax.set_xlim(min(xs) - final_pad, max(xs) + final_pad)
    ax.set_ylim(min(ys) - final_pad, max(ys) + final_pad)
    ppdu = points_per_data_unit(ax)  # xlim changed above, so this must be recomputed

    self_weights = np.array([mod_g.edges[m, m]["weight"] for m in nodes])
    if max_self is None:
        max_self = self_weights.max() if len(self_weights) and self_weights.max() > 0 else 1.0
    inter_weights = np.array([mod_g.edges[u, v]["weight"] for u, v in inter_edges]) if inter_edges else np.array([])
    if max_inter is None:
        max_inter = inter_weights.max() if len(inter_weights) and inter_weights.max() > 0 else 1.0

    if inter_edges:
        inter_widths = (min_edge_width + (max_edge_width - min_edge_width)
                         * (inter_weights / max_inter)).tolist()
        nx.draw_networkx_edges(mod_g, layout, edgelist=inter_edges, width=inter_widths,
                                edge_color="grey", node_size=node_size_list, ax=ax)

    nx.draw_networkx_nodes(mod_g, layout, node_size=node_size_list,
                            node_color=[mod_palette[m] for m in nodes],
                            edgecolors="black", linewidths=0.6, ax=ax)
    nx.draw_networkx_labels(mod_g, layout, font_size=8, ax=ax)

    for m in nodes:
        marker_radius_data = np.sqrt(node_size_map[m] / np.pi) / ppdu
        loop_r = max(marker_radius_data * 0.7, final_pad * 0.03)
        lw = min_edge_width + (max_edge_width - min_edge_width) * (mod_g.edges[m, m]["weight"] / max_self)
        cx, cy = layout[m]
        circle = plt.Circle((cx, cy + marker_radius_data + loop_r * 0.95), loop_r, fill=False,
                             edgecolor=mod_palette[m], linewidth=lw, zorder=5)
        ax.add_patch(circle)

    if title:
        ax.set_title(title)
    ax.set_axis_off()


def save_giant_network_pdf(giant: nx.Graph, module_of: dict, mod_ids, mod_palette: dict,
                            layout: dict, out_path: Path, focus_modules=()) -> None:
    """Multi-page network overview - `plotGiant_nextwork.pdf` (typo kept in the R
    filename, fixed here) in the R script: d22 edges, d35 edges (per-day,
    CT-vs-VCP-colored, via `plot_giant`), then two simplified pages via
    `plot_giant_simple` (known-only, then known+new pooled), one highlight page
    per module in `focus_modules`, then one page with the module-density view
    side by side for known-only vs. known+new, sharing both the layout
    (`module_centroid_positions`) and the edge-width normalization scale so the
    two are directly visually comparable. Also saves the two underlying module
    x module density matrices as CSVs (`module_density_known.csv`,
    `module_density_known_and_new.csv`)."""
    with PdfPages(out_path) as pdf:
        for day in ("d22", "d35"):
            fig, ax = plt.subplots(figsize=(8, 8))
            plot_giant(giant, module_of, mod_ids, mod_palette, layout, day=day, ax=ax)
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_giant_simple(giant, module_of, mod_palette, layout, edges="known", ax=ax)
        ax.set_title("Known PPI edges only")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_giant_simple(giant, module_of, mod_palette, layout, edges="all", ax=ax)
        ax.set_title("Known + new edges (all conditions/timepoints pooled)")
        pdf.savefig(fig)
        plt.close(fig)

        for m in focus_modules:
            fig, ax = plt.subplots(figsize=(8, 8))
            node_colors = [mod_palette[m] if module_of[n] == m else "white" for n in giant.nodes()]
            nx.draw_networkx_edges(giant, layout, edge_color="lightgrey", width=0.5, ax=ax)
            nx.draw_networkx_nodes(giant, layout, node_size=10, node_color=node_colors,
                                    edgecolors="grey", linewidths=0.3, ax=ax)
            ax.set_title(f"module {m}", fontsize=8)
            ax.set_axis_off()
            pdf.savefig(fig)
            plt.close(fig)

        mod_pos = module_centroid_positions(module_of, layout)

        mod_g_known = module_density_graph(giant, module_of, mod_ids,
                                            edge_filter=lambda d: d["present_in_ppi"] == 1)
        mod_g_all = module_density_graph(giant, module_of, mod_ids, edge_filter=None)

        shared_max_self = max(
            max((mod_g_known.edges[m, m]["weight"] for m in mod_ids), default=0.0),
            max((mod_g_all.edges[m, m]["weight"] for m in mod_ids), default=0.0),
        ) or 1.0
        shared_max_inter = max(
            max((d["weight"] for u, v, d in mod_g_known.edges(data=True) if u != v), default=0.0),
            max((d["weight"] for u, v, d in mod_g_all.edges(data=True) if u != v), default=0.0),
        ) or 1.0

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        plot_module_density_graph(mod_g_known, mod_palette, pos=mod_pos, ax=axes[0],
                                   max_self=shared_max_self, max_inter=shared_max_inter,
                                   title="Module density - known PPI only")
        plot_module_density_graph(mod_g_all, mod_palette, pos=mod_pos, ax=axes[1],
                                   max_self=shared_max_self, max_inter=shared_max_inter,
                                   title="Module density - known + new edges")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved {out_path}")

    known_matrix_path = out_path.parent / "module_density_known.csv"
    all_matrix_path = out_path.parent / "module_density_known_and_new.csv"
    module_density_matrix(mod_g_known, mod_ids).to_csv(known_matrix_path)
    module_density_matrix(mod_g_all, mod_ids).to_csv(all_matrix_path)
    print(f"Saved {known_matrix_path}")
    print(f"Saved {all_matrix_path}")


def plot_pathways(all_enrichment: pd.DataFrame, mod_palette: dict, category: str,
                   module_id: int, top_n: int = 5, ax=None) -> None:
    """Top-`top_n` STRING terms (by -log10 p) for one module/category -
    `PlotPathways()` in the R script."""
    ax = ax or plt.gca()
    sub = all_enrichment[(all_enrichment["category"] == category) & (all_enrichment["module"] == module_id)]
    if sub.empty:
        ax.axis("off")
        return

    vals = -np.log10(sub["p_value"].to_numpy())
    names = sub["description"].to_numpy()
    order = np.argsort(vals)[::-1][:top_n]
    vals, names = vals[order], names[order]
    if len(vals) < top_n:
        pad = top_n - len(vals)
        vals = np.concatenate([vals, np.zeros(pad)])
        names = np.concatenate([names, [""] * pad])

    ypos = np.arange(top_n)[::-1]
    ax.barh(ypos, vals, color=mod_palette[module_id])
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=4)
    ax.set_xlabel("-log10(P)", fontsize=5)
    ax.set_title(f"{category} module {module_id}", fontsize=5)


MOI_BUCKET_COLORS = {"vcp": "forestgreen", "ct": "firebrick", "ns": "lightgray"}


def plot_module_volcano(enrichment_vcp_ct: pd.DataFrame, day: str, out_path: Path,
                         fe_cutoff: float = 1.2, padj_cutoff: float = 0.01,
                         max_edges: "int | None" = None) -> None:
    """Volcano plot, one point per module: x = `log2(FE_cond_d{day})`, y =
    `-log10(padj_d{day})`, colored by the same three-way call
    `modules_of_interest` makes at that timepoint - green if
    `FE_cond_d{day} > fe_cutoff` (VCP-enriched), red if `< 1/fe_cutoff`
    (CTRL-enriched), gray otherwise - both gated on `padj_d{day} < padj_cutoff`,
    matching `condition_enrichment.modules_of_interest` exactly. Unlike a
    per-module bar chart, plot area doesn't grow with the number of modules -
    only the (typically few) significant ones get a module-ID label, so it
    stays readable at 88+ modules.

    `max_edges`, if given, restricts the plot to modules with fewer than that
    many total internal edges (`n_tot`, the same count `enrichment_vcp_ct`
    reports elsewhere) - useful for looking at small/sparse modules without
    the large ones dominating the point cloud."""
    fe_col, padj_col = f"FE_cond_d{day}", f"padj_d{day}"
    df = enrichment_vcp_ct
    if max_edges is not None:
        df = df[df["n_tot"] < max_edges]
    fe = df[fe_col].to_numpy()
    log2fe = np.log2(fe)

    # BH-adjustment can round a tiny p-value to exactly 0, which would make
    # -log10 infinite - floor at the smallest representable positive float.
    padj_raw = df[padj_col].to_numpy()
    padj = np.clip(padj_raw, np.finfo(float).tiny, 1.0)
    neglog10_padj = -np.log10(padj)

    significant = padj_raw < padj_cutoff
    is_vcp = significant & (fe > fe_cutoff)
    is_ct = significant & (fe < 1 / fe_cutoff)
    colors = np.where(is_vcp, MOI_BUCKET_COLORS["vcp"],
                       np.where(is_ct, MOI_BUCKET_COLORS["ct"], MOI_BUCKET_COLORS["ns"]))

    fig, ax = plt.subplots(figsize=(3.5, 6))
    ax.scatter(log2fe, neglog10_padj, c=colors, edgecolor="black", linewidth=0.4, s=80, zorder=3)

    ax.axhline(-np.log10(padj_cutoff), color="black", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(np.log2(fe_cutoff), color="black", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(np.log2(1 / fe_cutoff), color="black", linewidth=0.6, linestyle="--", zorder=1)

    to_label = is_vcp | is_ct
    for m, x, y in zip(df["module"].to_numpy()[to_label], log2fe[to_label], neglog10_padj[to_label]):
        ax.annotate(str(int(m)), (x, y), fontsize=6, ha="center", va="bottom",
                    xytext=(0, 2), textcoords="offset points")

    title = f"n modules={len(df)}" + (f", n_tot<{max_edges}" if max_edges is not None else "")
    ax.set(xlabel=f"log2(FE_cond_d{day})  (VCP vs CTRL fold enrichment)", ylabel=f"-log10(padj_d{day})", title=title)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MOI_BUCKET_COLORS["vcp"],
                   markeredgecolor="black", markersize=7,
                   label=f"VCP-enriched (FE>{fe_cutoff}, padj<{padj_cutoff})"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MOI_BUCKET_COLORS["ct"],
                   markeredgecolor="black", markersize=7,
                   label=f"CTRL-enriched (FE<{1 / fe_cutoff:.2f}, padj<{padj_cutoff})"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MOI_BUCKET_COLORS["ns"],
                   markeredgecolor="black", markersize=7, label="not significant"),
    ], loc="best", fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_fe_significance_csv(enrichment_vcp_ct: pd.DataFrame, day: str, out_path: Path,
                              max_edges: "int | None" = None) -> None:
    """The exact `(module, FE, Pval)` table `plot_module_volcano` plots for
    this `day`, as a CSV - one row per module, `FE` = `FE_cond_d{day}` (VCP
    vs. CTRL fold enrichment) and `Pval` = `padj_d{day}` (Benjamini-Hochberg
    FDR-adjusted, from `condition_enrichment.enrichment_vcp_vs_ct`'s Fisher's
    exact test - see that function's docstring). `max_edges`, if given,
    applies the same `n_tot < max_edges` restriction `plot_module_volcano`
    uses for its "small modules" variant, so this can mirror either PDF."""
    fe_col, padj_col = f"FE_cond_d{day}", f"padj_d{day}"
    df = enrichment_vcp_ct
    if max_edges is not None:
        df = df[df["n_tot"] < max_edges]
    out = df[["module", fe_col, padj_col]].rename(columns={fe_col: "FE", padj_col: "Pval"})
    out = out.sort_values("module").reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def plot_fraction_module_oi(enrichment_vcp_ct: pd.DataFrame, mod_palette: dict, module_id: int, ax=None) -> None:
    """Plot the percentage of edges per condition for one module."""
    ax = ax or plt.gca()
    row = enrichment_vcp_ct.set_index("module").loc[module_id]
    labels = ["ct_d22", "vcp_22", "ct_35", "vcp_35"]
    heights = [row["perc_ct_d22"], row["perc_vcp_22"], row["perc_ct_35"], row["perc_vcp_35"]]

    ax.bar(range(4), heights, color=["lightgray", mod_palette[module_id], "lightgray", mod_palette[module_id]])
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=4, rotation=90)
    ax.set_ylabel(f"% of edges per mod {module_id}", fontsize=6)
    ax.text(0.5, max(heights[0], heights[1]), _fmt_p(row["padj_d22"]), fontsize=4, ha="center", va="bottom")
    ax.text(2.5, max(heights[2], heights[3]), _fmt_p(row["padj_d35"]), fontsize=4, ha="center", va="bottom")
    ax.set_title(f"ntot edges={int(row['n_tot'])}", fontsize=5)


def save_modules_of_interest_pdf(module_ids, enrichment_vcp_ct: pd.DataFrame, all_enrichment: pd.DataFrame,
                                  mod_palette: dict, giant: nx.Graph, module_of: dict, out_path: Path,
                                  seed_focus: int = 12) -> None:
    """Per-bucket PDF (`modules_oi_<bucket>.pdf` in the R script): for each module of
    interest, the condition fraction bar chart + top-15 STRING terms per pathway
    category, followed by a zoomed network view at d22, d35, and finally the
    same module with every known edge (grey) + every new edge from any
    day/genotype pooled together (black) - see `plot_focus_module(day="all")`."""
    if not module_ids:
        print(f"No modules of interest for {out_path.name}, skipping")
        return

    with PdfPages(out_path) as pdf:
        for m in module_ids:
            fig, axes = plt.subplots(1, 5, figsize=(16, 6))
            plot_fraction_module_oi(enrichment_vcp_ct, mod_palette, m, ax=axes[0])
            for ax, cat in zip(axes[1:], PATHWAY_CATEGORIES):
                plot_pathways(all_enrichment, mod_palette, cat, m, top_n=15, ax=ax)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            for day in ("d22", "d35", "all"):
                fig, ax = plt.subplots(figsize=(6, 6))
                plot_focus_module(giant, module_of, mod_palette, module_id=m, day=day,
                                   seed=seed_focus, show_labels=True, ax=ax)
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved {out_path}")
