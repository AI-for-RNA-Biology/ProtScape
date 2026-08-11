"""Section 9: connected components of a single new-edge condition within each
module of interest, with STRING enrichment for the larger ones.

For each module of interest, restrict that module's induced subgraph to edges
of a single new-edge condition column (`vcp_35`, `ct_35`, `vcp_22`, or
`ct_22`), then split into connected components - only that condition column's
edges are used to find components; known-PPI and the complementary
condition's edges play no role in *which* genes get grouped together. Each
component is a fully-connected cluster of proteins linked *only* by that
condition's new edges within that module - isolated proteins with no such
edge are dropped.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .layout import module_layout


def extract_edge_components(giant: nx.Graph, module_of: dict, module_ids, edge_col: str) -> pd.DataFrame:
    """For each module in `module_ids`, restrict that module's induced subgraph to
    edges where `edge_col` (e.g. "vcp_35", "ct_35") is 1, and split into
    connected components - the fully-connected clusters of proteins linked only
    by that condition's new edges within that module. Nodes with no such edge in
    this module are dropped (a node only appears if it has at least one
    qualifying edge, so every component has >= 2 genes)."""
    rows = []
    for m in module_ids:
        mod_nodes = [n for n, mm in module_of.items() if mm == m]
        sub_g = giant.subgraph(mod_nodes)
        sel_edges = [(u, v) for u, v, d in sub_g.edges(data=True) if d[edge_col] == 1]
        sel_g = nx.Graph()
        sel_g.add_edges_from(sel_edges)

        for c, comp_nodes in enumerate(nx.connected_components(sel_g), start=1):
            comp_g = sel_g.subgraph(comp_nodes)
            rows.append({
                "module": m, "component": c,
                "n_nodes": comp_g.number_of_nodes(), "n_edges": comp_g.number_of_edges(),
                "genes": sorted(comp_nodes),
            })
    # Explicit `columns=` so the result still has a "genes" column (etc.) even
    # when `rows` is empty (e.g. no modules of interest for this condition) -
    # `pd.DataFrame([])` on its own has no columns at all, which breaks
    # `save_edge_components`'s `components_df["genes"]` access downstream.
    return pd.DataFrame(rows, columns=["module", "component", "n_nodes", "n_edges", "genes"])


def plot_edge_component_full(giant: nx.Graph, genes: list, mod_palette: dict,
                              module_id: int, day: str, layout: dict, ax=None) -> None:
    """Draws only this component: the induced subgraph on `genes`, restricted
    to known-PPI / new-ct / new-vcp edges among them (`edge_type_<day>` =
    "other" edges are dropped, not just made invisible). Nodes outside
    `genes` are omitted entirely - not drawn, not even as pale dots. Uses the
    same `layout` `module_layout` computed for the FULL module, so node
    positions match across components/pages from the same module."""
    comp_g = giant.subgraph(genes)
    edge_type_attr = f"edge_type_{day}"
    focus_colors = {"ct": "firebrick", "vcp": "yellow", "known": "grey"}
    edges_to_draw = [(u, v) for u, v, d in comp_g.edges(data=True) if d[edge_type_attr] in focus_colors]
    edge_colors = [focus_colors[comp_g.edges[u, v][edge_type_attr]] for u, v in edges_to_draw]
    edge_widths = [1.3 if comp_g.edges[u, v][edge_type_attr] in ("ct", "vcp") else 0.6 for u, v in edges_to_draw]

    ax = ax or plt.gca()
    nx.draw_networkx_edges(comp_g, layout, edgelist=edges_to_draw, edge_color=edge_colors, width=edge_widths, ax=ax)

    nx.draw_networkx_nodes(comp_g, layout, nodelist=genes, node_size=10, node_color=[mod_palette[module_id]],
                            edgecolors="black", linewidths=0.4, ax=ax)
    nx.draw_networkx_labels(comp_g, layout, labels={g: g for g in genes}, font_size=5, ax=ax)
    ax.legend(handles=[
        plt.Line2D([0], [0], color="grey", lw=2, label="Known PPI edge"),
        plt.Line2D([0], [0], color="firebrick", lw=2, label=f"New ct {day}"),
        plt.Line2D([0], [0], color="yellow", lw=2, label=f"New vcp {day}"),
    ], loc="upper right", fontsize=6, frameon=False)
    ax.set_title(f"module {module_id} - component ({len(genes)} genes)", fontsize=8)
    ax.set_axis_off()


def plot_component_enrichment(enrichment_df: pd.DataFrame, module_id: int, component_id: int,
                               top_n: int = 8, ax=None) -> None:
    ax = ax or plt.gca()
    sub = enrichment_df[(enrichment_df["module"] == module_id) & (enrichment_df["component"] == component_id)]
    if sub.empty:
        ax.text(0.5, 0.5, "no significant\nSTRING enrichment", ha="center", va="center", fontsize=7)
        ax.set_axis_off()
        return

    sub = sub.sort_values("p_value").head(top_n).iloc[::-1]
    vals = -np.log10(sub["p_value"].to_numpy())
    names = sub["description"].to_numpy()
    ax.barh(range(len(vals)), vals, color="steelblue")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("-log10(P)", fontsize=6)
    ax.set_title("STRING Function enrichment", fontsize=7)


def save_edge_components(components_df: pd.DataFrame, giant: nx.Graph, module_of: dict, mod_palette: dict,
                          out_dir: Path, basename: str, edge_col: str,
                          enrichment_df: "pd.DataFrame | None" = None, seed: int = 12) -> None:
    """Saves one multi-page PDF - for each module/component, one page with all
    three edge types among its genes (known-PPI, CT-new, VCP-new), plus a
    STRING enrichment panel alongside it for components above the 10-gene
    threshold if `enrichment_df` is given. Also saves one CSV with the gene
    list per component, and (if `enrichment_df` is non-empty) one CSV of the
    pooled STRING enrichment results."""
    day = f"d{edge_col.split('_')[-1]}"
    pdf_path = out_dir / f"{basename}.pdf"
    csv_path = out_dir / f"{basename}.csv"

    module_layouts = {}
    with PdfPages(pdf_path) as pdf:
        for _, row in components_df.iterrows():
            m = row["module"]
            if m not in module_layouts:
                module_layouts[m] = module_layout(giant, module_of, m, seed=seed)
            layout = module_layouts[m]

            has_enrichment = enrichment_df is not None and row["n_nodes"] > 10
            if has_enrichment:
                fig, (ax_net, ax_enrich) = plt.subplots(1, 2, figsize=(12, 6))
                plot_edge_component_full(giant, row["genes"], mod_palette, m, day, layout, ax=ax_net)
                plot_component_enrichment(enrichment_df, row["module"], row["component"], ax=ax_enrich)
                fig.tight_layout()
            else:
                fig, ax = plt.subplots(figsize=(7, 7))
                plot_edge_component_full(giant, row["genes"], mod_palette, m, day, layout, ax=ax)
            pdf.savefig(fig)
            plt.close(fig)

    out = components_df.copy()
    out["genes"] = out["genes"].apply(lambda g: ";".join(g))
    out.to_csv(csv_path, index=False)
    print(f"Saved {pdf_path}")
    print(f"Saved {csv_path}")

    if enrichment_df is not None and len(enrichment_df):
        enrich_path = out_dir / f"{basename}_string_enrichment.csv"
        enrichment_df.to_csv(enrich_path, index=False)
        print(f"Saved {enrich_path}")
