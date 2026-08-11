"""For each day in the neighborhood-edges CSV, find the genes that are
"new" in one condition's 2-hop ego network but not the other's (CTRL-only /
VCP-only, same day-pairing convention as neighborhood_by_context.py's
node coloring and neighborhood_by_context_edges.py's edge coloring), and
heatmap their Cytoplasmic gene expression - genes as rows, one column per
CTRL/VCP x timepoint "sample" - across every timepoint the source matrix
has, for both CTRL and VCP.

Source expression matrix (already filtered/normalized) - same one
gene_expression.py's CYTO_GENE_EXPRESSION_CSV points at
(`als_rewiring_cyto_gene_expression_csv` in configs/paths.yaml).
Loaded via gene_expression.load_cyto_gene_expression (keeps only
`_Cytoplasmic_` sample columns, drops `_Nuclear_`), then
gene_expression.condition_means (log2(x+1) per sample, THEN averaged across
that condition's replicates - the pipeline's standard mean-of-logs
convention) over every genotype x timepoint in the matrix (D0/D3/D7/D14/
D22/D35 x CTRL/VCP), not just the D22/D35 days the neighborhood CSV itself
covers - so a day's new-gene set gets its full time course, not just that
one day's value. Each column is therefore one CTRL/VCP x timepoint
"sample" - the replicate-mean, not an individual replicate.

Three heatmaps are produced per gene set: the raw log2(expr+1) values, a
row-standardized (z-scored per gene, across its own 12 columns) version, and
a row min-max-normalized (scaled to [0, 1] per gene, across its own 12
columns) version - the latter two both show each gene's own up/down SHAPE
across time/genotype independent of its absolute expression level, which
the raw heatmap's shared color scale can otherwise wash out for
lowly-expressed genes sitting next to highly-expressed ones; min-max keeps
every gene's range on the same fixed [0, 1] scale (easier to eyeball
"where in its own range is this point" at a glance) where z-score keeps it
in standard-deviation units (easier to compare how PEAKED vs FLAT a gene's
own trajectory is).

For gene sets with more than 1 gene, a boxplot (one box per CTRL/VCP x
timepoint sample, log2(expr+1) across all genes in that set, points
jittered on top) is also produced - the distribution-summary view
complementing the heatmap's per-gene detail. Sets with only 0 or 1 gene are
skipped (nothing to summarize a distribution over).

Rows are hierarchically clustered (average-linkage, Euclidean, on whatever
values that particular heatmap displays - raw/z-score/min-max each get
their OWN clustering, computed after normalization) via plotting.py's
`_cluster_row_order` - the same reordering plot_module_log2fc_heatmap uses,
so genes with similar patterns end up near each other instead of listed in
whatever order they happened to be in - unlike that function's own usage
though, gene names ARE still drawn here (these gene sets are small enough,
usually well under plot_module_log2fc_heatmap's typical module size, to
read individually).

Runs automatically as part of ALS_rewiring_analysis.py, right after
neighborhood_by_context.py, for the same config.NEIGHBORHOOD_GENE/
config.NEIGHBORHOOD_EDGES_CSV defaults (overridable via --gene/--csv).
Imports neighborhood_by_context's build_context_graphs, and plotting.py's
`_cluster_row_order`, directly (same directory / already-installed package,
no duplication); the day/CTRL-vs-VCP gene-set logic here is the same
comparison neighborhood_by_context.build_presence_colors makes, just
returning gene sets instead of a per-node color. Heatmap style (pcolormesh,
not imshow, for crisp vector cells in the PDF; one page per gene set, sized
to its own row count) matches plotting.plot_module_log2fc_heatmap.

Writes:
    <out-dir>/<gene>_new_gene_expression.csv (log2(x+1) mean expression,
        one row per gene, one column per genotype x timepoint, plus a
        `day`/`new_in` column recording which day/condition-comparison the
        gene came from)
    <out-dir>/<gene>_new_gene_expression_heatmap.pdf (raw log2(expr+1),
        one page per day x {CTRL-only, VCP-only} gene set)
    <out-dir>/<gene>_new_gene_expression_heatmap_zscore.pdf (same pages,
        row-standardized, z-score)
    <out-dir>/<gene>_new_gene_expression_heatmap_minmax.pdf (same pages,
        row-standardized, min-max to [0, 1])
    <out-dir>/<gene>_new_gene_expression_boxplot.pdf (one page per day x
        {CTRL-only, VCP-only} gene set with >1 gene)
    <out-dir>/<gene>_new_gene_expression_heatmap{,_zscore,_minmax}_d22d35.pdf
        (same three heatmaps, columns restricted to CTRL/VCP x {D22, D35} -
        z-score/min-max recomputed on just those 4 columns, not sliced out
        of the 12-column version)

Usage:
    python neighborhood_new_gene_expression.py                    # config.NEIGHBORHOOD_GENE, default CSV/out-dir
    python neighborhood_new_gene_expression.py \\
        --csv /path/to/some_other_gene_neighborhood_edges.csv \\
        --gene SOMEOTHERGENE \\
        --out-dir /path/to/figures/SOMEOTHERGENE_neighborhood
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, TwoSlopeNorm

import _bootstrap  # noqa: F401
from exploration.analysis.ALS_rewiring import config, gene_expression
from exploration.analysis.ALS_rewiring.plotting import _cluster_row_order

from neighborhood_by_context import build_context_graphs

ALL_DAYS = ["d0", "d3", "d7", "d14", "d22", "d35"]
DAY_LABELS = {"d0": "D0", "d3": "D3", "d7": "D7", "d14": "D14", "d22": "D22", "d35": "D35"}
ALL_CONDITIONS = [f"{geno}_{day}" for day in ALL_DAYS for geno in ("ctrl", "vcp")]
COND_LABELS = {f"{geno}_{day}": f"{geno.upper()} {DAY_LABELS[day]}" for day in ALL_DAYS for geno in ("ctrl", "vcp")}

# The two days the neighborhood CSV itself actually covers - a heatmap variant
# restricted to just these, dropping D0/D3/D7/D14.
D22_D35_CONDITIONS = [f"{geno}_{day}" for day in ("d22", "d35") for geno in ("ctrl", "vcp")]

NEW_IN_LABELS = {"ctrl_only": "new in CTRL only", "vcp_only": "new in VCP only"}

CTRL_COLOR = "#2ca02c"   # green - matches neighborhood_by_context*'s CTRL-only color
VCP_COLOR = "#e377c2"    # pink - matches neighborhood_by_context*'s VCP-only color


def day_gene_sets(df, contexts, graphs):
    """day -> {"ctrl_ctx", "vcp_ctx", "ctrl_only", "vcp_only", "both"} -
    same CTRL-vs-VCP day pairing as
    neighborhood_by_context.build_presence_colors, but returning the
    actual gene sets instead of a per-node color. Days without exactly one
    CTRL + one VCP context are skipped (nothing to compare)."""
    meta = df[df.context.isin(contexts)][["context", "condition", "day"]].drop_duplicates().set_index("context")
    out = {}
    for day, group in meta.groupby("day"):
        ctrl_ctxs = group.index[group["condition"] == "CTRL"].tolist()
        vcp_ctxs = group.index[group["condition"] == "VCP"].tolist()
        if len(ctrl_ctxs) != 1 or len(vcp_ctxs) != 1:
            print(f"day {day}: not exactly one CTRL + one VCP context, skipping")
            continue
        ctrl_ctx, vcp_ctx = ctrl_ctxs[0], vcp_ctxs[0]
        ctrl_nodes = set(graphs[ctrl_ctx].nodes())
        vcp_nodes = set(graphs[vcp_ctx].nodes())
        out[day] = {
            "ctrl_ctx": ctrl_ctx, "vcp_ctx": vcp_ctx,
            "ctrl_only": ctrl_nodes - vcp_nodes,
            "vcp_only": vcp_nodes - ctrl_nodes,
            "both": ctrl_nodes & vcp_nodes,
        }
    return out


def build_matrix(expr_means, genes, conditions=ALL_CONDITIONS):
    present = sorted(g for g in genes if g in expr_means.index)
    missing = sorted(set(genes) - set(present))
    mat = expr_means.loc[present, conditions] if present else pd.DataFrame(columns=conditions)
    return mat, missing


def zscore_rows(mat):
    """Per-gene z-score across its own 12 genotype x timepoint columns
    (ddof=0). A gene with zero variance across every column (std == 0)
    would divide by zero - its row is left as NaN (rendered light grey by
    plot_heatmap) rather than silently becoming 0 everywhere."""
    if mat.empty:
        return mat
    std = mat.std(axis=1, ddof=0).replace(0, np.nan)
    return mat.sub(mat.mean(axis=1), axis=0).div(std, axis=0)


def minmax_rows(mat):
    """Per-gene min-max scaling to [0, 1] across its own 12 genotype x
    timepoint columns. A gene with zero range (max == min) would divide by
    zero - its row is left as NaN (rendered light grey by plot_heatmap)
    rather than silently becoming 0 everywhere."""
    if mat.empty:
        return mat
    rmin, rmax = mat.min(axis=1), mat.max(axis=1)
    rng = (rmax - rmin).replace(0, np.nan)
    return mat.sub(rmin, axis=0).div(rng, axis=0)


def plot_heatmap(ax, mat, title, cmap_name, cbar_label, symmetric=False, vrange=None):
    if mat.empty:
        ax.axis("off")
        ax.set_title(f"{title} - no genes found in expression matrix", fontsize=9)
        return

    values = mat.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgrey")
    if vrange is not None:
        norm = Normalize(vmin=vrange[0], vmax=vrange[1])
    elif symmetric:
        vmax = np.abs(finite).max() if finite.size else 1.0
        norm = TwoSlopeNorm(vmin=-(vmax or 1.0), vcenter=0.0, vmax=(vmax or 1.0))
    else:
        vmin = finite.min() if finite.size else 0.0
        vmax = finite.max() if finite.size else 1.0
        norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)

    masked = np.ma.masked_invalid(values)
    mesh = ax.pcolormesh(masked, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.4)
    ax.set_xlim(0, values.shape[1])
    ax.set_ylim(0, values.shape[0])
    ax.invert_yaxis()

    ax.set_xticks(np.arange(len(mat.columns)) + 0.5)
    ax.set_xticklabels([COND_LABELS[c] for c in mat.columns], rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(mat.index)) + 0.5)
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_title(f"{title} (n={len(mat.index)} genes)", fontsize=10)

    fig = ax.figure
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)


HEATMAP_STYLES = {
    None: dict(cmap_name="Blues", cbar_label="log2(expr + 1), Cytoplasmic", symmetric=False, vrange=None),
    "zscore": dict(cmap_name="RdBu_r", cbar_label="z-score (per gene, across its 12 samples)",
                   symmetric=True, vrange=None),
    "minmax": dict(cmap_name="Blues", cbar_label="min-max normalized (per gene, across its 12 samples)",
                   symmetric=False, vrange=(0.0, 1.0)),
}


def _own_other_ctx(sets, new_in):
    other_ctx = sets["vcp_ctx"] if new_in == "ctrl_only" else sets["ctrl_ctx"]
    own_ctx = sets["ctrl_ctx"] if new_in == "ctrl_only" else sets["vcp_ctx"]
    return own_ctx, other_ctx


def save_heatmap_pdf(gene_sets, expr_means, out_path, normalize=None, conditions=ALL_CONDITIONS):
    style = HEATMAP_STYLES[normalize]
    cell_w, cell_h = 0.35, 0.22  # inches per column / per row

    with PdfPages(out_path) as pdf:
        for day in sorted(gene_sets):
            sets = gene_sets[day]
            for new_in in ("ctrl_only", "vcp_only"):
                genes = sets[new_in]
                mat, missing = build_matrix(expr_means, genes, conditions=conditions)
                if missing:
                    print(f"  D{day} {NEW_IN_LABELS[new_in]}: {len(missing)}/{len(genes)} gene(s) not in the "
                          f"expression matrix - {missing}")
                if normalize == "zscore":
                    mat = zscore_rows(mat)
                elif normalize == "minmax":
                    mat = minmax_rows(mat)
                mat = _cluster_row_order(mat)

                own_ctx, other_ctx = _own_other_ctx(sets, new_in)
                title = f"D{day} - {NEW_IN_LABELS[new_in]} ({own_ctx} not in {other_ctx})"

                n_rows = max(len(mat.index), 1)
                fig_w = max(len(conditions) * cell_w + 2.5, 6.0)  # floor so long titles don't get clipped
                fig_h = n_rows * cell_h + 1.5
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                plot_heatmap(ax, mat, title, **style)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    print(f"Saved {out_path}")


def plot_boxplot(ax, mat, title):
    xpos = np.arange(len(ALL_CONDITIONS))
    data = [mat[c].dropna().to_numpy() for c in ALL_CONDITIONS]
    box_colors = [CTRL_COLOR if c.startswith("ctrl_") else VCP_COLOR for c in ALL_CONDITIONS]

    bp = ax.boxplot(data, positions=xpos, widths=0.6, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    for median in bp["medians"]:
        median.set_color("black")

    rng = np.random.default_rng(0)
    for x, vals, color in zip(xpos, data, box_colors):
        if len(vals):
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(x + jitter, vals, color=color, edgecolors="black", linewidths=0.3, s=18, alpha=0.8, zorder=3)

    ax.set_xticks(xpos)
    ax.set_xticklabels([COND_LABELS[c] for c in ALL_CONDITIONS], rotation=90, fontsize=8)
    ax.set_ylabel("log2(expr + 1), Cytoplasmic")
    ax.set_title(f"{title} (n={len(mat.index)} genes)", fontsize=10)


def save_boxplot_pdf(gene_sets, expr_means, out_path):
    with PdfPages(out_path) as pdf:
        n_pages = 0
        for day in sorted(gene_sets):
            sets = gene_sets[day]
            for new_in in ("ctrl_only", "vcp_only"):
                genes = sets[new_in]
                mat, missing = build_matrix(expr_means, genes)
                if len(mat.index) <= 1:
                    print(f"  D{day} {NEW_IN_LABELS[new_in]}: only {len(mat.index)} gene(s), skipping boxplot "
                          "(needs more than 1 gene)")
                    continue

                own_ctx, other_ctx = _own_other_ctx(sets, new_in)
                title = f"D{day} - {NEW_IN_LABELS[new_in]} ({own_ctx} not in {other_ctx})"

                fig, ax = plt.subplots(figsize=(8, 5))
                plot_boxplot(ax, mat, title)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                n_pages += 1
    if n_pages:
        print(f"Saved {out_path}")
    else:
        print(f"No gene set had more than 1 gene - {out_path} has no pages")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None, help="Neighborhood-edges CSV (context, source, target, "
                                                      "source_hop, target_hop, condition, day columns). "
                                                      "Default: config.NEIGHBORHOOD_EDGES_CSV.")
    parser.add_argument("--gene", default=None, help="Center gene, hop 0. Default: config.NEIGHBORHOOD_GENE.")
    parser.add_argument("--max-hop", type=int, default=2, help="Keep edges where both endpoints are within "
                                                                 "this many hops of --gene. Default: 2.")
    parser.add_argument("--out-dir", default=None, help="Directory to save the CSV/PDFs in - must be "
                                                          "somewhere you own, not a shared/temp directory "
                                                          "another process might rewrite. Default: "
                                                          "OUT_DIR/neighborhood_<gene>.")
    args = parser.parse_args()

    csv_path_in = args.csv or config.NEIGHBORHOOD_EDGES_CSV
    gene = args.gene or config.NEIGHBORHOOD_GENE
    out_dir = Path(args.out_dir) if args.out_dir else config.OUT_DIR / f"neighborhood_{gene}"

    df = pd.read_csv(csv_path_in)
    df = df[(df.source_hop <= args.max_hop) & (df.target_hop <= args.max_hop)].copy()
    contexts = sorted(df.context.unique())

    graphs = build_context_graphs(df, gene, contexts)
    gene_sets = day_gene_sets(df, contexts, graphs)

    expr_cyto = gene_expression.load_cyto_gene_expression()
    expr_means = gene_expression.condition_means(expr_cyto, conditions=ALL_CONDITIONS)

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for day, sets in gene_sets.items():
        for new_in, genes in (("ctrl_only", sets["ctrl_only"]), ("vcp_only", sets["vcp_only"])):
            present = [g for g in sorted(genes) if g in expr_means.index]
            for g in present:
                row = {"day": day, "new_in": new_in, "gene": g}
                row.update(expr_means.loc[g].to_dict())
                rows.append(row)
    csv_path = out_dir / f"{gene}_new_gene_expression.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    save_heatmap_pdf(gene_sets, expr_means, out_dir / f"{gene}_new_gene_expression_heatmap.pdf",
                      normalize=None)
    save_heatmap_pdf(gene_sets, expr_means, out_dir / f"{gene}_new_gene_expression_heatmap_zscore.pdf",
                      normalize="zscore")
    save_heatmap_pdf(gene_sets, expr_means, out_dir / f"{gene}_new_gene_expression_heatmap_minmax.pdf",
                      normalize="minmax")
    save_boxplot_pdf(gene_sets, expr_means, out_dir / f"{gene}_new_gene_expression_boxplot.pdf")

    # Same three heatmaps, restricted to D22/D35 only (dropping D0/D3/D7/D14) -
    # z-score/min-max are recomputed on just these 4 columns, not sliced out of
    # the 12-column version, so each gene's own D22/D35-only range/spread drives
    # its normalization.
    save_heatmap_pdf(gene_sets, expr_means, out_dir / f"{gene}_new_gene_expression_heatmap_d22d35.pdf",
                      normalize=None, conditions=D22_D35_CONDITIONS)
    save_heatmap_pdf(gene_sets, expr_means,
                      out_dir / f"{gene}_new_gene_expression_heatmap_zscore_d22d35.pdf",
                      normalize="zscore", conditions=D22_D35_CONDITIONS)
    save_heatmap_pdf(gene_sets, expr_means,
                      out_dir / f"{gene}_new_gene_expression_heatmap_minmax_d22d35.pdf",
                      normalize="minmax", conditions=D22_D35_CONDITIONS)


if __name__ == "__main__":
    main()
