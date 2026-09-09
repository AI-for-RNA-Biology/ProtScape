"""2-hop ego-network plot for one gene, one panel per context, with node
positions kept consistent across panels so the same gene lands in the same
spot everywhere and panels can be compared visually.

Runs automatically as part of ALS_rewiring_analysis.py, for
config.NEIGHBORHOOD_GENE (default: ALS2CL) against
config.NEIGHBORHOOD_EDGES_CSV (`als_rewiring_neighborhood_edges_csv` in
configs/paths.yaml) - both overridable via --gene/--csv for ad hoc
exploration of a different gene or CSV export. This reads a standalone
observed-PPI neighborhood CSV (prepare_neighborhood.py exports with
source/target/*_hop columns), not INTERMEDIATE_DIR/giant.pickle - so it
doesn't share state with the rest of this package; any CSV with the same
column layout works.

Expected input columns: context, source, target, source_hop, target_hop.
Edges are kept when both endpoints are within --max-hop of --gene (default:
2, i.e. drop the 3-hop shell - it makes the plot too dense to read).

Layout: a spring layout is computed once on the UNION of edges across every
context in the CSV; each node keeps that layout's angle but has its radius
snapped to its own hop count (0 = center, 1 = inner ring, 2 = outer ring, ...
one ring per integer hop up to --max-hop). Because the union layout - not
each context's own subgraph - decides the angle, a gene's position is the
same in every panel regardless of which contexts it actually appears in.
Nodes within a ring are then re-spaced evenly by that angle (order preserved)
to reduce label overlap.

Node color is NOT hop (the radius already shows that) - it's cross-condition
presence, compared day by day using the CSV's own condition/day columns.
Within each day (e.g. d22), a node is black if it's in both the CTRL and VCP
graphs for that day, green if only in CTRL, pink if only in VCP - reused
as-is for every other day found in the CSV. Requires exactly one CTRL and one
VCP context per day; a day that doesn't fit that shape falls back to gray for
every node in it.

Writes:
    OUT_DIR/neighborhood_<gene>/<gene>_neighborhood_<max-hop>hop_by_context.pdf
    (one row of panels per 2 contexts, ALS2CL/--gene at the center of each)

Usage:
    python neighborhood_by_context.py                    # config.NEIGHBORHOOD_GENE, default CSV/out-dir
    python neighborhood_by_context.py \\
        --csv /path/to/some_other_gene_neighborhood_edges.csv \\
        --gene SOMEOTHERGENE \\
        --out-dir /path/to/figures/SOMEOTHERGENE_neighborhood

"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import config

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def compute_union_layout(df, gene, max_hop, seed=42, radius_step=0.55):
    """Spring-layout the union of all contexts' edges once, then snap each
    node's radius to its own hop count so every context panel can reuse the
    same (angle, ring) position for a given gene. `radius_step` sets the
    gap between consecutive hop rings."""
    G_all = nx.Graph()
    node_hop = {}
    for _, r in df.iterrows():
        s, t, sh, th = r["source"], r["target"], r["source_hop"], r["target_hop"]
        G_all.add_edge(s, t)
        node_hop[s] = min(node_hop.get(s, sh), sh)
        node_hop[t] = min(node_hop.get(t, th), th)
    node_hop[gene] = 0

    spring_pos = nx.spring_layout(G_all, seed=seed, k=0.6, iterations=200)
    cx, cy = spring_pos[gene]
    angles = {n: np.arctan2(y - cy, x - cx) for n, (x, y) in spring_pos.items()}

    pos = {gene: (0.0, 0.0)}
    radii = {hop: hop * radius_step for hop in range(max_hop + 1)}
    for hop in range(1, max_hop + 1):
        ring_nodes = sorted((n for n, h in node_hop.items() if h == hop), key=lambda n: angles.get(n, 0))
        n = len(ring_nodes)
        for i, node in enumerate(ring_nodes):
            theta = 2 * np.pi * i / n if n else 0
            pos[node] = (radii[hop] * np.cos(theta), radii[hop] * np.sin(theta))

    return pos, node_hop, radii


BOTH_COLOR = "black"
CTRL_ONLY_COLOR = "#2ca02c"   # green
VCP_ONLY_COLOR = "#e377c2"    # pink
UNPAIRED_COLOR = "gray"       # day doesn't have exactly one CTRL + one VCP context


def build_context_graphs(df, gene, contexts):
    graphs = {}
    for ctx in contexts:
        sub = df[df.context == ctx]
        Gc = nx.Graph()
        Gc.add_edges_from(zip(sub.source, sub.target))
        Gc.add_node(gene)
        graphs[ctx] = Gc
    return graphs


def build_presence_colors(df, contexts, graphs):
    """(context, node) -> color, comparing CTRL vs VCP within each day found
    in the CSV's own condition/day columns. See module docstring."""
    meta = df[df.context.isin(contexts)][["context", "condition", "day"]].drop_duplicates().set_index("context")
    colors = {}
    for day, group in meta.groupby("day"):
        ctrl_ctxs = group.index[group["condition"] == "CTRL"].tolist()
        vcp_ctxs = group.index[group["condition"] == "VCP"].tolist()
        if len(ctrl_ctxs) == 1 and len(vcp_ctxs) == 1:
            ctrl_ctx, vcp_ctx = ctrl_ctxs[0], vcp_ctxs[0]
            ctrl_nodes = set(graphs[ctrl_ctx].nodes())
            vcp_nodes = set(graphs[vcp_ctx].nodes())
            for ctx in (ctrl_ctx, vcp_ctx):
                for node in graphs[ctx].nodes():
                    if node in ctrl_nodes and node in vcp_nodes:
                        colors[(ctx, node)] = BOTH_COLOR
                    elif node in ctrl_nodes:
                        colors[(ctx, node)] = CTRL_ONLY_COLOR
                    else:
                        colors[(ctx, node)] = VCP_ONLY_COLOR
        else:
            for ctx in group.index:
                for node in graphs[ctx].nodes():
                    colors[(ctx, node)] = UNPAIRED_COLOR
    return colors


def plot_by_context(graphs, gene, contexts, pos, node_hop, radii, node_colors, out_path, ncols=2):
    nrows = -(-len(contexts) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, ctx in zip(axes, contexts):
        Gc = graphs[ctx]

        for hop, r in radii.items():
            if r > 0:
                ax.add_patch(plt.Circle((0, 0), r, fill=False, linestyle="--",
                                         linewidth=0.7, color="lightgray", zorder=0))

        for u, v in Gc.edges():
            (x1, y1), (x2, y2) = pos[u], pos[v]
            ax.plot([x1, x2], [y1, y2], color="gray", linewidth=0.6, alpha=0.6, zorder=1)

        for node in Gc.nodes():
            x, y = pos[node]
            hop = node_hop.get(node, max(radii))
            color = node_colors.get((ctx, node), UNPAIRED_COLOR)
            ax.scatter(x, y, s=200 if hop == 0 else 60, color=color,
                       edgecolors="black", linewidths=0.5, zorder=3)

            if hop == 0:
                offset, ha, va = (0, 14), "center", "bottom"
            else:
                # Push the label radially outward (away from the center gene) rather
                # than always straight up, so labels on the outer ring don't sit on
                # top of - or overlap - their (often black) node.
                norm = np.hypot(x, y) or 1.0
                ux, uy = x / norm, y / norm
                offset = (ux * 12, uy * 12)
                ha = "left" if ux > 0.15 else ("right" if ux < -0.15 else "center")
                va = "bottom" if uy > 0.15 else ("top" if uy < -0.15 else "center")
            ax.annotate(node, (x, y), fontsize=7 if hop >= 2 else 8, ha=ha, va=va,
                        xytext=offset, textcoords="offset points", zorder=4)

        ax.set_title(f"{ctx}  (n={Gc.number_of_nodes()} nodes, {Gc.number_of_edges()} edges)", fontsize=13)
        ax.set_aspect("equal")
        ax.axis("off")
        # Fixed additive padding (not proportional to radii) - a margin that scales
        # with radius_step would cancel out any change to the ring spacing itself,
        # since the axes auto-scale to fill the figure either way.
        lim = max(radii.values()) + 0.45 if max(radii.values()) > 0 else 1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    for ax in axes[len(contexts):]:
        ax.axis("off")

    legend_handles = [
        mpatches.Patch(color=BOTH_COLOR, label="present in both (CTRL & VCP, same day)"),
        mpatches.Patch(color=CTRL_ONLY_COLOR, label="CTRL only"),
        mpatches.Patch(color=VCP_ONLY_COLOR, label="VCP only"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), fontsize=11, frameon=False)
    fig.suptitle(f"{gene} {max(radii)}-hop neighborhood by context (node positions consistent across panels)",
                 fontsize=15, y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None, help="Neighborhood-edges CSV (context, source, target, "
                                                      "source_hop, target_hop columns). Default: "
                                                      "config.NEIGHBORHOOD_EDGES_CSV.")
    parser.add_argument("--gene", default=None, help="Center gene, hop 0. Default: config.NEIGHBORHOOD_GENE.")
    parser.add_argument("--max-hop", type=int, default=2, help="Keep edges where both endpoints are within "
                                                                 "this many hops of --gene. Default: 2.")
    parser.add_argument("--out-dir", default=None, help="Directory to save the plot in - must be somewhere "
                                                          "you own, not a shared/temp directory another "
                                                          "process might rewrite. Default: "
                                                          "OUT_DIR/neighborhood_<gene>.")
    parser.add_argument("--out-name", default=None, help="Output filename (default: "
                                                           "<gene>_neighborhood_<max-hop>hop_by_context.png).")
    parser.add_argument("--seed", type=int, default=42, help="Spring-layout seed. Default: 42.")
    parser.add_argument("--radius-step", type=float, default=0.07, help="Gap between consecutive hop "
                                                                          "rings. Default: 0.07.")
    args = parser.parse_args()

    csv_path = args.csv or config.NEIGHBORHOOD_EDGES_CSV
    gene = args.gene or config.NEIGHBORHOOD_GENE
    out_dir = Path(args.out_dir) if args.out_dir else config.OUT_DIR / f"neighborhood_{gene}"

    df = pd.read_csv(csv_path)
    df = df[(df.source_hop <= args.max_hop) & (df.target_hop <= args.max_hop)].copy()
    contexts = sorted(df.context.unique())

    pos, node_hop, radii = compute_union_layout(df, gene, args.max_hop, seed=args.seed,
                                                 radius_step=args.radius_step)
    graphs = build_context_graphs(df, gene, contexts)
    node_colors = build_presence_colors(df, contexts, graphs)

    out_name = args.out_name or f"{gene}_neighborhood_{args.max_hop}hop_by_context.pdf"
    out_path = out_dir / out_name
    plot_by_context(graphs, gene, contexts, pos, node_hop, radii, node_colors, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
