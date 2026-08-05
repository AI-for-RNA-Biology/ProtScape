#!/usr/bin/env python
"""Step 3: induce a global PPI on each reliable-gene set and retain its LCC."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import ks_2samp

from .config import CL_PATH, GLOBAL_PPI, HBCA_GENE_METADATA, HBCA_INTERMEDIATE, TABULA_INTERMEDIATE
from .utils import pairwise_jaccard


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _resolve_reliable_gene_file(base_dir: str, override: str | None) -> str:
    if override:
        return override
    for filename in ("reliable_genes_specific.json", "reliable_genes.json"):
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No reliable-gene file found in {base_dir}")


@lru_cache(None)
def _load_cl_names() -> dict[str, str]:
    if not os.path.exists(CL_PATH):
        logger.warning("Cell Ontology file not found: %s", CL_PATH)
        return {}
    import obonet

    graph = obonet.read_obo(CL_PATH)
    return {
        cl_id: attributes["name"]
        for cl_id, attributes in graph.nodes(data=True)
        if isinstance(attributes.get("name"), str)
    }


def normalize_gene_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().split(".", 1)[0]


def is_ensembl_identifier(value: str) -> bool:
    return value.strip().upper().startswith(("ENSG", "ENST", "ENSRN", "ENSMUSG"))


def extract_cl_id(cell_type: str) -> str:
    """Convert supported cell-type labels to the PINNACLE ``CL_...`` form."""
    label = str(cell_type).strip()
    if re.match(r"^CL:\d+_[A-Z]+_", label):
        return label.replace("CL:", "CL_", 1)
    match = re.search(r"\[CL:(\d+)\]", label)
    if match:
        return f"CL_{match.group(1)}"
    if re.match(r"^CL:\d+$", label):
        return label.replace(":", "_")
    return label


def load_global_ppi(path: str) -> nx.Graph:
    graph = nx.read_edgelist(path, nodetype=str)
    logger.info("Loaded global PPI: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _clean_symbol(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    symbol = str(value).strip()
    return None if not symbol or symbol.lower() in {"nan", "none", "null"} else symbol


def map_genes(adata: sc.AnnData, gene_metadata_path: str | None) -> dict[str, str]:
    """Map AnnData gene identifiers to uppercase symbols, resolving duplicates by mean expression."""
    genes = adata.var.copy()
    genes["gene_id"] = adata.var_names.astype(str)
    genes["gene_id_normalized"] = genes["gene_id"].map(normalize_gene_identifier)

    symbol_column = next(
        (name for name in ("gene_symbol", "symbol", "gene_name", "hgnc_symbol") if name in genes),
        None,
    )
    symbols = genes[symbol_column].map(_clean_symbol) if symbol_column else None

    if symbols is None and gene_metadata_path and os.path.exists(gene_metadata_path):
        metadata = pd.read_csv(gene_metadata_path)
        id_column = next(
            (
                name
                for name in metadata
                if name.lower()
                in {"gene_identifier", "ensembl_id", "ensembl_gene_id", "ensembl", "gene_id"}
            ),
            None,
        )
        metadata_symbol_column = next(
            (name for name in metadata if name.lower() in {"gene_symbol", "symbol", "gene_name", "hgnc_symbol"}),
            None,
        )
        if id_column and metadata_symbol_column:
            metadata = metadata[[id_column, metadata_symbol_column]].dropna()
            metadata[id_column] = metadata[id_column].map(normalize_gene_identifier)
            metadata[metadata_symbol_column] = metadata[metadata_symbol_column].map(_clean_symbol)
            lookup = (
                metadata.loc[metadata[id_column] != ""]
                .drop_duplicates(id_column)
                .set_index(id_column)[metadata_symbol_column]
            )
            symbols = genes["gene_id_normalized"].map(lookup)

    if symbols is None:
        symbols = genes["gene_id"].map(_clean_symbol)

    symbols = symbols.astype(object)
    genes["symbol"] = symbols.where(symbols.notna(), genes["gene_id"]).astype(str).str.strip()
    mean_column = "mean_counts" if "mean_counts" in genes else "mean_cpm" if "mean_cpm" in genes else None
    genes["mean_expression"] = genes[mean_column] if mean_column else 0.0
    selected = (
        genes.sort_values("mean_expression", ascending=False)
        .groupby("symbol", sort=False)
        .head(1)
    )

    mapping = selected["symbol"].str.upper().to_dict()
    mapping.update(
        {
            normalize_gene_identifier(gene_id): symbol
            for gene_id, symbol in mapping.items()
            if normalize_gene_identifier(gene_id)
        }
    )
    mapping.update({symbol: symbol for symbol in mapping.values()})
    logger.info("Mapped %d gene identifiers to symbols", len(mapping))
    return mapping


def map_to_symbols(gene_ids: list[str], mapping: dict[str, str]) -> list[str]:
    symbols = []
    seen = set()
    for gene_id in gene_ids:
        normalized = normalize_gene_identifier(gene_id)
        symbol = mapping.get(gene_id) or mapping.get(normalized)
        if symbol is None or pd.isna(symbol):
            candidate = normalized.upper()
            if not candidate or is_ensembl_identifier(candidate):
                continue
            symbol = candidate
        symbol = str(symbol).strip().upper()
        if symbol and symbol.lower() not in {"nan", "none", "null"} and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


def _summary_row(cell_type: str, genes: list[str], symbols: list[str], genes_in_ppi: list[str]) -> dict:
    return {
        "cell_type": cell_type,
        "input_genes": len(genes),
        "ranked_genes": len(symbols),
        "genes_considered": len(symbols),
        "genes_mapped": len(symbols),
        "nodes_in_ppi": len(genes_in_ppi),
        "subgraph_nodes": 0,
        "subgraph_edges": 0,
        "lcc_size": 0,
        "lcc_edges": 0,
        "lcc_median_degree": 0.0,
        "global_median_degree": 0.0,
        "pct_REG_in_PPI": 100.0 * len(genes_in_ppi) / max(1, len(genes)),
        "pct_REG_kept": 0.0,
        "ks_distance": float("inf"),
        "final_nodes": 0,
        "final_edges": 0,
        "status": "skipped_no_ppi",
    }


def build_ppi_networks(
    gene_sets: dict[str, list[str]],
    gene_mapping: dict[str, str],
    global_ppi: nx.Graph,
) -> tuple[dict[str, list[str]], dict[str, nx.Graph], pd.DataFrame, tuple[float, float]]:
    """Build one induced LCC per cell type, retaining the established 200-node cutoff."""
    node_sets = {}
    graphs = {}
    summary = []

    for cell_type, genes in sorted(gene_sets.items()):
        symbols = sorted(set(map_to_symbols(genes, gene_mapping)))
        genes_in_ppi = [symbol for symbol in symbols if symbol in global_ppi]
        row = _summary_row(cell_type, genes, symbols, genes_in_ppi)
        if not genes_in_ppi:
            logger.warning("%s: no mapped genes in the global PPI", cell_type)
            summary.append(row)
            continue

        subgraph = global_ppi.subgraph(genes_in_ppi).copy()
        lcc_nodes = max(nx.connected_components(subgraph), key=len)
        lcc = subgraph.subgraph(lcc_nodes).copy()
        row.update(
            {
                "subgraph_nodes": subgraph.number_of_nodes(),
                "subgraph_edges": subgraph.number_of_edges(),
                "lcc_size": lcc.number_of_nodes(),
                "lcc_edges": lcc.number_of_edges(),
                "pct_REG_kept": 100.0 * lcc.number_of_nodes() / max(1, len(genes)),
            }
        )
        if lcc.number_of_nodes() < 200:
            row.update(status="skipped_small_lcc", ks_distance=float("nan"))
            summary.append(row)
            logger.info("%s: discarded %d-node LCC (<200)", cell_type, lcc.number_of_nodes())
            continue

        local_degrees = np.fromiter((degree for _, degree in lcc.degree()), dtype=float)
        global_degrees = np.fromiter((global_ppi.degree(node) for node in lcc_nodes), dtype=float)
        row.update(
            {
                "lcc_median_degree": float(np.median(local_degrees)),
                "global_median_degree": float(np.median(global_degrees)),
                "ks_distance": float(ks_2samp(local_degrees, global_degrees).statistic),
                "final_nodes": lcc.number_of_nodes(),
                "final_edges": lcc.number_of_edges(),
                "status": "kept",
            }
        )
        node_sets[cell_type] = sorted(lcc_nodes)
        graphs[cell_type] = lcc
        summary.append(row)
        logger.info("%s: retained %d nodes and %d edges", cell_type, lcc.number_of_nodes(), lcc.number_of_edges())

    summary_df = pd.DataFrame(summary)
    kept = summary_df.loc[summary_df["status"] == "kept", "lcc_size"] if not summary_df.empty else pd.Series(dtype=float)
    thresholds = (
        (float(kept.quantile(0.10)), float(kept.quantile(0.90)))
        if not kept.empty
        else (float("nan"), float("nan"))
    )
    return node_sets, graphs, summary_df, thresholds


def _edgelist_filename(cell_type: str) -> str:
    standardized = extract_cl_id(cell_type)
    if re.match(r"^CL_\d+$", standardized):
        cl_name = _load_cl_names().get(standardized.replace("_", ":", 1), "unknown")
        filename = f"{cl_name}_[{standardized}]"
    else:
        filename = standardized
    return (
        filename.lower()
        .replace(" ", "_")
        .replace(":", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def write_outputs(
    output_dir: str,
    ppi_dict: dict[str, list[str]],
    graphs: dict[str, nx.Graph],
    summary: pd.DataFrame,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with (output / "ppi_celltype_list.csv").open("w") as handle:
        for index, (cell_type, genes) in enumerate(sorted(ppi_dict.items())):
            handle.write(f"{index}\t{extract_cl_id(cell_type)}\t{','.join(genes)}\n")

    edgelist_dir = output / "ppi_edgelists"
    if edgelist_dir.exists():
        shutil.rmtree(edgelist_dir)
    edgelist_dir.mkdir()
    with (output / "inventory.txt").open("w") as inventory:
        for cell_type, graph in graphs.items():
            path = edgelist_dir / f"{_edgelist_filename(cell_type)}.txt"
            nx.write_edgelist(graph, path, data=False)
            inventory.write(f"{path}\n")

    summary.to_csv(output / "ppi_summary.csv", index=False)


def _save_figure(fig: plt.Figure, directory: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(directory / filename, dpi=150)
    plt.close(fig)


def generate_qc(
    output_dir: str,
    summary: pd.DataFrame,
    global_ppi: nx.Graph,
    thresholds: tuple[float, float],
) -> None:
    """Write network-size, coverage, degree, and distribution diagnostics."""
    kept = summary[summary["status"] == "kept"].copy()
    if kept.empty:
        logger.warning("No retained networks; skipping QC")
        return

    qc_dir = Path(output_dir) / "ppi_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    q_low, q_high = thresholds
    global_nodes = global_ppi.number_of_nodes()
    global_edges = global_ppi.number_of_edges()
    statistics = {
        "cell_types_retained": int(len(kept)),
        "median_subgraph_nodes": float(kept["subgraph_nodes"].median()),
        "median_lcc_nodes": float(kept["lcc_size"].median()),
        "median_lcc_edges": float(kept["lcc_edges"].median()),
        "median_lcc_median_degree": float(kept["lcc_median_degree"].median()),
        "median_global_median_degree": float(kept["global_median_degree"].median()),
        "median_ks_distance": float(kept["ks_distance"].median()),
        "median_pct_reg_in_PPI": float(kept["pct_REG_in_PPI"].median()),
        "median_pct_reg_kept": float(kept["pct_REG_kept"].median()),
        "lcc_size_p10": None if np.isnan(q_low) else float(q_low),
        "lcc_size_p90": None if np.isnan(q_high) else float(q_high),
        "global_nodes": global_nodes,
        "global_edges": global_edges,
        "global_avg_degree": float(2 * global_edges / global_nodes) if global_nodes else 0.0,
    }
    with (qc_dir / "summary_statistics.json").open("w") as handle:
        json.dump(statistics, handle, indent=2)

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(kept["lcc_size"], bins=25, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvspan(q_low, q_high, color="red", alpha=0.1, label="Selection band (P10–P90)")
    ax.axvline(kept["lcc_size"].median(), color="black", linestyle="--", label="Median LCC")
    ax.set(xlabel="LCC size (nodes)", ylabel="Cell types", title="Distribution of LCC sizes")
    ax.legend()
    _save_figure(fig, qc_dir, "lcc_size_distribution.png")

    ordered = kept.sort_values("lcc_size", ascending=False)
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(positions, ordered["lcc_size"], color="steelblue", edgecolor="black", alpha=0.8, label="LCC size")
    ax.plot(positions, ordered["subgraph_nodes"], color="black", linewidth=2, label="Total nodes (pre-LCC)")
    ax.axhline(q_low, color="red", linestyle="--", linewidth=1.5, label="P10 threshold")
    ax.axhline(q_high, color="orange", linestyle="--", linewidth=1.5, label="P90 threshold")
    ax.set_xticks(positions)
    ax.set_xticklabels(ordered["cell_type"], rotation=90, ha="right", fontsize=8)
    ax.set(xlabel="Cell type (sorted by LCC size)", ylabel="Number of nodes", title="PPI size per cell type (LCC vs total subgraph)")
    ax.legend()
    _save_figure(fig, qc_dir, "lcc_vs_total_nodes.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(kept["lcc_median_degree"], bins=15, color="seagreen", edgecolor="black", alpha=0.7)
    ax.axvline(kept["lcc_median_degree"].median(), color="red", linestyle="--", label="LCC median")
    ax.axvline(kept["global_median_degree"].median(), color="orange", linestyle="--", label="Global median")
    ax.set(xlabel="Median degree (cell-type LCC)", ylabel="Cell types", title="Median degree across retained networks")
    ax.legend()
    _save_figure(fig, qc_dir, "median_degree_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(kept["subgraph_nodes"], kept["lcc_size"], c=kept["ks_distance"], cmap="viridis", alpha=0.7)
    ax.axhline(q_low, color="red", linestyle="--", linewidth=1.5, label="P10 threshold")
    ax.axhline(q_high, color="orange", linestyle="--", linewidth=1.5, label="P90 threshold")
    ax.set(xlabel="Total nodes before LCC", ylabel="LCC size", title="Relationship between induced subgraph size and LCC size")
    fig.colorbar(scatter, ax=ax, label="KS distance")
    ax.legend()
    _save_figure(fig, qc_dir, "subgraph_vs_lcc.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(kept["lcc_size"], kept["lcc_edges"], c=kept["lcc_size"], cmap="viridis", alpha=0.75, edgecolor="k", linewidth=0.2)
    x_values = np.linspace(0, kept["lcc_size"].max() * 1.05, 200)
    for degree, color in ((5, "steelblue"), (10, "orange"), (20, "green"), (50, "crimson")):
        ax.plot(x_values, degree / 2 * x_values, "--", color=color, label=f"avg_degree={degree}")
    ax.set(xlabel="Number of nodes (LCC size)", ylabel="Number of edges (LCC)", title="Network size: nodes vs. edges")
    ax.legend(loc="upper left")
    fig.colorbar(scatter, ax=ax, label="LCC size")
    _save_figure(fig, qc_dir, "nodes_vs_edges.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(kept["global_median_degree"], kept["lcc_median_degree"], c="purple", alpha=0.7)
    limit = max(kept["global_median_degree"].max(), kept["lcc_median_degree"].max(), 1)
    ax.plot([0, limit], [0, limit], color="grey", linestyle="--", label="y = x")
    ax.set(xlabel="Median degree (global PPI, same node set)", ylabel="Median degree (cell-type LCC)", title="Median degree alignment with global PPI")
    ax.legend()
    _save_figure(fig, qc_dir, "median_degree_comparison.png")

    valid_ks = kept["ks_distance"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(valid_ks, bins=20, color="cornflowerblue", edgecolor="black", alpha=0.7)
    ax.axvline(valid_ks.median(), color="red", linestyle="--", label="Median")
    ax.set(xlabel="KS distance (LCC vs global degree)", ylabel="Cell types", title="Distribution of KS distances")
    ax.legend()
    _save_figure(fig, qc_dir, "ks_distance_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(kept["subgraph_nodes"], kept["ks_distance"], c=kept["lcc_size"], cmap="plasma", alpha=0.7)
    ax.set(xlabel="Total nodes before LCC", ylabel="KS distance to global PPI", title="KS distance vs. induced subgraph size")
    fig.colorbar(scatter, ax=ax, label="LCC size")
    _save_figure(fig, qc_dir, "ks_distance_vs_subgraph_nodes.png")


def _describe(values) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _plot_similarity_histogram(values: list[float], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=30, color="slateblue", edgecolor="black", alpha=0.75)
    ax.axvline(np.median(values), color="red", linestyle="--", label="Median")
    ax.set(xlabel="Jaccard similarity", ylabel="Pairs", title=title)
    ax.legend()
    _save_figure(fig, path.parent, path.name)


def analyze_similarity(
    ppi_graphs: dict[str, nx.Graph],
    summary: pd.DataFrame,
    output_dir: str,
) -> None:
    """Summarize pairwise node and edge overlap among retained PPIs."""
    kept = summary[summary["status"] == "kept"]
    graphs = {
        cell_type: ppi_graphs[cell_type]
        for cell_type in kept["cell_type"]
        if cell_type in ppi_graphs and ppi_graphs[cell_type].number_of_nodes() > 0
    }
    if len(graphs) < 2:
        logger.warning("Need at least two PPIs for similarity analysis")
        return

    node_jaccard = pairwise_jaccard({name: set(graph.nodes()) for name, graph in graphs.items()})
    edge_jaccard = pairwise_jaccard(
        {name: {tuple(sorted(edge)) for edge in graph.edges()} for name, graph in graphs.items()}
    )
    qc_dir = Path(output_dir) / "ppi_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    statistics = {
        "node_counts": _describe(kept["lcc_size"]),
        "edge_counts": _describe(kept["lcc_edges"]),
        "node_jaccard": _describe(node_jaccard),
        "edge_jaccard": _describe(edge_jaccard),
    }
    with (qc_dir / "ppi_similarity_stats.json").open("w") as handle:
        json.dump(statistics, handle, indent=2)
    if node_jaccard:
        _plot_similarity_histogram(node_jaccard, "Pairwise node Jaccard similarity", qc_dir / "node_jaccard_hist.png")
    if edge_jaccard:
        _plot_similarity_histogram(edge_jaccard, "Pairwise edge Jaccard similarity", qc_dir / "edge_jaccard_hist.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tabula", "hbca"], required=True)
    parser.add_argument("--reliable-genes")
    parser.add_argument("--pseudobulk")
    parser.add_argument("--global-ppi", default=GLOBAL_PPI)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    base_dir = TABULA_INTERMEDIATE if args.dataset == "tabula" else HBCA_INTERMEDIATE
    genes_path = _resolve_reliable_gene_file(base_dir, args.reliable_genes)
    pseudobulk_path = args.pseudobulk or os.path.join(base_dir, "pseudobulk.h5ad")
    output_dir = args.output_dir or os.path.join(base_dir, "ppi")
    for path in (genes_path, pseudobulk_path, args.global_ppi):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run_config.json").open("w") as handle:
        json.dump({"method": "simple_lcc_extraction", "dataset": args.dataset}, handle, indent=2)

    with open(genes_path) as handle:
        gene_sets = json.load(handle)
    pseudobulk = sc.read_h5ad(pseudobulk_path)
    gene_mapping = map_genes(
        pseudobulk,
        HBCA_GENE_METADATA if args.dataset == "hbca" else None,
    )
    global_ppi = load_global_ppi(args.global_ppi)
    ppi_nodes, ppi_graphs, summary, thresholds = build_ppi_networks(
        gene_sets,
        gene_mapping,
        global_ppi,
    )
    write_outputs(output_dir, ppi_nodes, ppi_graphs, summary)
    generate_qc(output_dir, summary, global_ppi, thresholds)
    analyze_similarity(ppi_graphs, summary, output_dir)
    logger.info("Constructed %d PPI networks", len(ppi_nodes))


if __name__ == "__main__":
    main()
