#!/usr/bin/env python
"""Step 9: summarize and plot the processed PPI, CCI, and metagraph outputs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    ALS_BULK_DIR,
    GLOBAL_PPI,
    HBCA_INTERMEDIATE,
    OUTPUT_ROOT,
    TABULA_INTERMEDIATE,
)
from .step8_merge_datasets import normalize_cl_label, read_edgelist_any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MERGED_OUTPUT = os.path.join(os.path.dirname(HBCA_INTERMEDIATE), "networks_bulk")
EVAL_OUTPUT_BASE = OUTPUT_ROOT / "data_processing_bulk" / "evaluation_reports"

TISSUE_COLOR = "#8FBBD9"
CELL_COLOR = "#7DC67D"
PPI_MEDIAN_COLOR = "#B07CC6"
SIMILARITY_COLOR = "#555555"

def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_tissue_node(node: str) -> bool:
    """Check if a node is a tissue (BTO) node."""
    return isinstance(node, str) and node.startswith("BTO")


def load_metagraph(dataset_path: Path) -> nx.Graph:
    """Load and normalize metagraph."""
    mg_file = dataset_path / "mg_edgelist.txt"
    if not mg_file.exists():
        mg_file = dataset_path / "metagraph.txt"

    if not mg_file.exists():
        raise FileNotFoundError(f"Metagraph not found under {dataset_path}")

    g_raw = nx.read_edgelist(str(mg_file), delimiter="\t")
    g = nx.Graph()
    for u, v in g_raw.edges():
        g.add_edge(normalize_cl_label(u), normalize_cl_label(v))

    return g


def load_ppi_networks(dataset_path: Path) -> Dict[str, nx.Graph]:
    """Load all PPI networks for a dataset."""
    ppi_dir = dataset_path / "ppi" / "ppi_edgelists"
    if not ppi_dir.exists():
        ppi_dir = dataset_path / "ppi_edgelists"
    if not ppi_dir.exists():
        raise FileNotFoundError(f"PPI directory not found under {dataset_path}")

    ppi_networks = {}
    for ppi_file in sorted(ppi_dir.glob("*.txt")):
        if ppi_file.stem.lower() in {"inventory", "readme", "metadata"}:
            continue

        celltype = ppi_file.stem
        ppi_networks[celltype] = read_edgelist_any(ppi_file)

    return ppi_networks


def load_cci_network(dataset_path: Path) -> nx.Graph:
    """Load CCI network."""
    cci_file = dataset_path / "cci_edgelist.txt"
    if not cci_file.exists():
        raise FileNotFoundError(f"CCI network not found: {cci_file}")

    g_raw = nx.read_edgelist(str(cci_file), delimiter="\t")
    g = nx.Graph()
    for u, v in g_raw.edges():
        g.add_edge(normalize_cl_label(u), normalize_cl_label(v))

    return g


def _normalize_gene_list(values: object) -> List[str]:
    """Return a clean list of string gene symbols from arbitrary iterables."""
    if isinstance(values, str):
        return [values] if values else []
    if isinstance(values, (list, tuple, set)):
        return [str(gene) for gene in values if isinstance(gene, str) and gene]
    return []


def _parse_reliable_gene_payload(payload: object) -> Tuple[Dict[str, List[str]], Set[str]]:
    """Normalize reliable gene JSON payloads into per-cell and union outputs."""
    per_cell: Dict[str, List[str]] = {}
    union: Set[str] = set()
    if isinstance(payload, dict):
        for celltype, genes in payload.items():
            gene_list = _normalize_gene_list(genes)
            per_cell[celltype] = gene_list
            union.update(gene_list)
    elif isinstance(payload, list):
        gene_list = _normalize_gene_list(payload)
        union.update(gene_list)
    return per_cell, union


def _read_reliable_genes_from_dir(dir_path: Path) -> Tuple[Dict[str, List[str]], Set[str]]:
    """Load reliable genes from a directory, preferring Step2b-specific outputs when present."""
    per_cell_candidates = [
        dir_path / "reliable_genes_specific_per_celltype.json",
        dir_path / "reliable_genes_per_celltype.json",
    ]
    union_candidates = [
        dir_path / "reliable_genes_specific.json",
        dir_path / "reliable_genes.json",
    ]

    for per_cell_file in per_cell_candidates:
        if per_cell_file.exists():
            with open(per_cell_file, "r") as handle:
                payload = json.load(handle)
            per_cell, union = _parse_reliable_gene_payload(payload)
            if per_cell or union:
                return per_cell, union

    for genes_file in union_candidates:
        if genes_file.exists():
            with open(genes_file, "r") as handle:
                payload = json.load(handle)
            per_cell, union = _parse_reliable_gene_payload(payload)
            if per_cell or union:
                return per_cell, union

    return {}, set()


def load_reliable_genes(dataset_path: Path) -> Tuple[Dict[str, List[str]], Set[str]]:
    """Load reliable genes per cell type (if available) and union set."""
    per_cell, union = _read_reliable_genes_from_dir(dataset_path)
    if per_cell or union:
        return per_cell, union

    aggregated_union: Set[str] = set()
    aggregated_map: Dict[str, List[str]] = {}
    aggregated_seen: Dict[str, Set[str]] = {}
    found = False

    for subdir in sorted(dataset_path.iterdir()):
        if not subdir.is_dir():
            continue
        sub_map, sub_union = _read_reliable_genes_from_dir(subdir)
        if not sub_map and not sub_union:
            continue
        found = True
        aggregated_union.update(sub_union)
        for celltype, genes in sub_map.items():
            existing_list = aggregated_map.setdefault(celltype, [])
            existing_seen = aggregated_seen.setdefault(celltype, set())
            for gene in genes:
                if gene not in existing_seen:
                    existing_seen.add(gene)
                    existing_list.append(gene)

    if found:
        return aggregated_map, aggregated_union

    logger.warning(f"Reliable genes not found under {dataset_path}")
    return {}, set()


def analyze_metagraph_degrees(metagraph: nx.Graph) -> Dict[str, Dict[int, int]]:
    """
    Analyze degree distribution of metagraph, separated by tissue and cell type nodes.

    Returns:
        Dictionary with 'tissue' and 'cell_type' keys, each containing degree: count mappings
    """
    tissue_degrees = []
    cell_type_degrees = []

    for node in metagraph.nodes():
        degree = metagraph.degree(node)
        if is_tissue_node(node):
            tissue_degrees.append(degree)
        else:
            cell_type_degrees.append(degree)

    tissue_degree_dist = Counter(tissue_degrees)
    cell_type_degree_dist = Counter(cell_type_degrees)

    return {
        "tissue": dict(tissue_degree_dist),
        "cell_type": dict(cell_type_degree_dist)
    }


def extract_subgraph_by_node_type(metagraph: nx.Graph, keep_tissue: bool = True) -> nx.Graph:
    """Extract tissue-tissue or cell-cell subgraph."""
    subgraph = nx.Graph()

    for u, v in metagraph.edges():
        u_is_tissue = is_tissue_node(u)
        v_is_tissue = is_tissue_node(v)

        if keep_tissue:
            # Keep only tissue-tissue edges
            if u_is_tissue and v_is_tissue:
                subgraph.add_edge(u, v)
        else:
            # Keep only cell-cell edges
            if not u_is_tissue and not v_is_tissue:
                subgraph.add_edge(u, v)

    return subgraph


def compute_ppi_statistics(ppi_networks: Dict[str, nx.Graph]) -> Dict[str, object]:
    """Compute statistics for PPI networks."""
    node_counts = []
    edge_counts = []
    median_degrees = []
    lcc_sizes = []

    for g in ppi_networks.values():
        if g.number_of_nodes() == 0:
            continue

        node_counts.append(g.number_of_nodes())
        edge_counts.append(g.number_of_edges())

        # Median node degree
        degrees = [g.degree(n) for n in g.nodes()]
        median_degrees.append(np.median(degrees) if degrees else 0)

        # Largest connected component size
        if g.number_of_edges() > 0:
            lcc = max(nx.connected_components(g), key=len)
            lcc_sizes.append(len(lcc))
        else:
            lcc_sizes.append(g.number_of_nodes())

    unique_proteins = set()
    for g in ppi_networks.values():
        unique_proteins.update(g.nodes())

    return {
        "num_networks": len(ppi_networks),
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "median_degrees": median_degrees,
        "lcc_sizes": lcc_sizes,
        "mean_nodes": np.mean(node_counts) if node_counts else 0,
        "median_nodes": np.median(node_counts) if node_counts else 0,
        "std_nodes": np.std(node_counts, ddof=0) if node_counts else 0,
        "mean_edges": np.mean(edge_counts) if edge_counts else 0,
        "median_edges": np.median(edge_counts) if edge_counts else 0,
        "std_edges": np.std(edge_counts, ddof=0) if edge_counts else 0,
        "unique_proteins": len(unique_proteins),
    }


def generate_label_aliases(label: str) -> List[str]:
    """Return variants of a label to help match CL IDs across files."""
    if not isinstance(label, str):
        return []
    value = label.strip()
    if not value:
        return []
    aliases: List[str] = []

    def _add(candidate: Optional[str]) -> None:
        if candidate and candidate not in aliases:
            aliases.append(candidate)

    _add(value)
    _add(value.lower())
    _add(value.replace(" ", "_"))
    _add(value.replace(" ", "_").lower())
    normalized = normalize_cl_label(value)
    _add(normalized)
    _add(normalized.lower())
    cl_match = re.search(r"(CL[:_]\d+)", value, re.IGNORECASE)
    if cl_match:
        cl_id = cl_match.group(1).replace(":", "_").upper()
        _add(cl_id)
        _add(cl_id.lower())
    return aliases


def count_cci_degree_per_celltype(
    cci_network: nx.Graph,
    ppi_networks: Dict[str, nx.Graph],
) -> Dict[str, int]:
    """Return each PPI-backed cell type's degree in the CCI graph."""
    cci_degrees: Dict[str, int] = {celltype: 0 for celltype in ppi_networks}
    if cci_network.number_of_nodes() == 0 or not ppi_networks:
        return cci_degrees

    degree_lookup: Dict[str, int] = {}
    for node, degree in cci_network.degree():
        for alias in generate_label_aliases(node):
            degree_lookup[alias] = max(degree_lookup.get(alias, 0), degree)

    for celltype in ppi_networks:
        for alias in generate_label_aliases(celltype):
            if alias in degree_lookup:
                cci_degrees[celltype] = degree_lookup[alias]
                break
    return cci_degrees


def build_gene_count_lookup(per_cell_genes: Dict[str, List[str]]) -> Dict[str, int]:
    """Create a flexible lookup from cell type labels to reliable gene counts."""
    lookup: Dict[str, int] = {}
    for raw_label, genes in per_cell_genes.items():
        count = len(genes)
        label = str(raw_label).strip()
        if not label:
            continue
        for cand in generate_label_aliases(label):
            lookup.setdefault(cand, count)
    return lookup


def _edge_set(g: nx.Graph) -> Set[Tuple[str, str]]:
    """Return a normalized set of undirected edges."""
    return {tuple(sorted((u, v))) for u, v in g.edges()}


def compute_ppi_distance_matrix(
    ppi_networks: Dict[str, nx.Graph],
    *,
    min_edges: int = 1,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """
    Compute pairwise Jaccard distances between cell type-specific PPIs.

    Returns a (distance_matrix_df, summary_stats) tuple or (None, None) if insufficient data.
    """
    if not ppi_networks:
        return None, None

    candidates = [
        (celltype, g)
        for celltype, g in ppi_networks.items()
        if g.number_of_edges() >= min_edges
    ]
    if len(candidates) < 2:
        return None, None

    candidates.sort(key=lambda item: item[1].number_of_nodes(), reverse=True)

    celltypes = [ct for ct, _ in candidates]
    edge_sets = [_edge_set(g) for _, g in candidates]

    n = len(celltypes)
    dist_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        dist_matrix[i, i] = 0.0
        for j in range(i + 1, n):
            a = edge_sets[i]
            b = edge_sets[j]
            if not a and not b:
                dist = 0.0
            else:
                inter = len(a & b)
                union = len(a | b)
                dist = 1.0 - (inter / union if union > 0 else 0.0)
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    df = pd.DataFrame(dist_matrix, index=celltypes, columns=celltypes)
    upper_vals = dist_matrix[np.triu_indices(n, k=1)]
    summary = {
        "mean_distance": float(np.mean(upper_vals)) if upper_vals.size else 0.0,
        "median_distance": float(np.median(upper_vals)) if upper_vals.size else 0.0,
        "min_distance": float(np.min(upper_vals)) if upper_vals.size else 0.0,
        "max_distance": float(np.max(upper_vals)) if upper_vals.size else 0.0,
        "num_pairs": int(upper_vals.size),
    }
    return df, summary


def compute_ppi_node_similarity_values(
    ppi_networks: Dict[str, nx.Graph],
    selected_celltypes: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute pairwise Jaccard similarities between node sets of selected PPIs."""
    if not ppi_networks:
        return np.array([])

    celltypes = selected_celltypes or sorted(ppi_networks.keys())
    node_sets: List[Set[str]] = []
    for celltype in celltypes:
        graph = ppi_networks.get(celltype)
        if graph is None:
            continue
        nodes = set(graph.nodes())
        if not nodes:
            continue
        node_sets.append(nodes)

    n = len(node_sets)
    if n < 2:
        return np.array([])

    similarities: List[float] = []
    for i in range(n):
        a = node_sets[i]
        for j in range(i + 1, n):
            b = node_sets[j]
            union = len(a | b)
            if union == 0:
                similarities.append(0.0)
            else:
                similarities.append(len(a & b) / union)
    return np.array(similarities, dtype=float)


def build_celltype_summary_rows(
    ppi_networks: Dict[str, nx.Graph],
    gene_lookup: Dict[str, int],
) -> List[Dict[str, object]]:
    """Return per-cell-type rows with REG counts and PPI sizes."""
    rows: List[Dict[str, object]] = []
    for celltype, graph in sorted(ppi_networks.items()):
        label = str(celltype).strip()
        gene_count = None
        if gene_lookup:
            candidates = [
                label,
                label.lower(),
                label.replace(" ", "_"),
                normalize_cl_label(label) if label else None,
            ]
            cl_match = re.search(r"(CL[:_]\d+)", label, re.IGNORECASE)
            if cl_match:
                candidates.append(cl_match.group(1).replace(":", "_").upper())
            for cand in candidates:
                if cand and cand in gene_lookup:
                    gene_count = gene_lookup[cand]
                    break

        lcc_size = 0
        lcc_edges = 0
        if graph.number_of_nodes() > 0:
            if graph.number_of_edges() > 0:
                lcc = max(nx.connected_components(graph), key=len)
                lcc_size = len(lcc)
                lcc_edges = graph.subgraph(lcc).number_of_edges()
            else:
                lcc_size = graph.number_of_nodes()
                lcc_edges = 0

        rows.append(
            {
                "cell_type": label,
                "reliable_genes": gene_count,
                "ppi_nodes": graph.number_of_nodes(),
                "ppi_edges": graph.number_of_edges(),
                "ppi_lcc_nodes": lcc_size,
                "ppi_lcc_edges": lcc_edges,
            }
        )
    return rows


def plot_metagraph_degree_distribution(
    degree_data: Dict[str, Dict[int, int]],
    output_path: Path,
    title: str = "Metagraph Degree Distribution"
):
    """Plot degree distribution with separate colors for tissue and cell type nodes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tissue_samples: List[int] = []
    cell_samples: List[int] = []
    for degree, count in degree_data.get("tissue", {}).items():
        tissue_samples.extend([degree] * count)
    for degree, count in degree_data.get("cell_type", {}).items():
        cell_samples.extend([degree] * count)

    if not tissue_samples and not cell_samples:
        logger.warning("No metagraph degree data to plot.")
        return

    combined = tissue_samples + cell_samples
    min_degree = min(combined) if combined else 0
    max_degree = max(combined) if combined else 1
    bins = np.arange(min_degree - 0.5, max_degree + 1.5, 1)

    if tissue_samples:
        ax.hist(
            tissue_samples,
            bins=bins,
            color=TISSUE_COLOR,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
            label="Tissue nodes",
        )
    if cell_samples:
        ax.hist(
            cell_samples,
            bins=bins,
            color=CELL_COLOR,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.8,
            label="Cell type nodes",
        )

    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    _apply_publication_axes_style(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved metagraph degree distribution plot to {output_path}")


def plot_degree_distribution(
    graph: nx.Graph,
    output_path: Path,
    title: str,
    xlabel: str = "Degree",
    ylabel: str = "Count"
):
    """Plot degree distribution for a graph."""
    if graph.number_of_nodes() == 0:
        logger.warning(f"Empty graph, skipping plot: {title}")
        return

    degrees = [graph.degree(n) for n in graph.nodes()]
    fig, ax = plt.subplots(figsize=(10, 6))
    min_degree = min(degrees)
    max_degree = max(degrees)
    degree_span = max_degree - min_degree

    # Use integer bins for small ranges; otherwise auto-bin to avoid clutter
    if degree_span <= 40:
        bins = np.arange(min_degree - 0.5, max_degree + 1.5, 1)
    else:
        bins = np.histogram_bin_edges(degrees, bins="auto")

    ax.hist(
        degrees,
        bins=bins,
        color=TISSUE_COLOR,
        edgecolor='white',
        linewidth=0.8,
        alpha=0.9,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _apply_publication_axes_style(ax)

    # Avoid overplotting x-ticks on wide ranges
    if degree_span <= 25:
        ax.set_xticks(range(min_degree, max_degree + 1))
    else:
        tick_count = min(10, len(bins))
        if tick_count > 1:
            ticks = np.linspace(min_degree, max_degree, tick_count)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.0f}" for t in ticks], rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_path}")


def plot_ppi_metric_distribution(
    values: List[float],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str = "Count"
):
    """Plot distribution of a PPI metric."""
    if not values:
        logger.warning(f"No values, skipping plot: {title}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        values,
        bins=30,
        alpha=0.9,
        color=PPI_MEDIAN_COLOR,
        edgecolor='white',
        linewidth=0.8,
        rwidth=0.95,  # ensure bars touch each other (no visible gaps)
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _apply_publication_axes_style(ax)

    # Add median line (changed from mean)
    median_val = np.median(values)
    ax.axvline(median_val, color='#4B2C72', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_path}")


def plot_ppi_size_scatter(
    node_counts: List[int],
    edge_counts: List[int],
    output_path: Path,
    title: str = "PPI Network Size: Nodes vs Edges"
):
    """Plot network size with average degree lines."""
    if not node_counts or not edge_counts:
        logger.warning("No PPI data, skipping scatter plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(node_counts, edge_counts, alpha=0.6, s=50, color='steelblue')

    # Add average degree lines
    max_nodes = max(node_counts) if node_counts else 1000
    x_range = np.linspace(0, max_nodes, 100)

    for avg_degree in [5, 10, 20, 50]:
        # For undirected graph: edges = (nodes * avg_degree) / 2
        y_range = (x_range * avg_degree) / 2
        ax.plot(x_range, y_range, '--', alpha=0.5, label=f'Avg degree = {avg_degree}')

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Number of Edges")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved scatter plot to {output_path}")


def plot_ppi_distance_heatmap(
    distance_df: pd.DataFrame,
    output_path: Path,
    title: str = "Pairwise PPI Distances"
) -> None:
    """Plot heatmap of pairwise distances between PPIs."""
    if distance_df is None or distance_df.empty:
        logger.warning("No PPI distance matrix; skipping heatmap.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        distance_df,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "Jaccard distance"},
    )
    plt.title(title)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved PPI distance heatmap to {output_path}")


def plot_ppi_similarity_clustermap(
    distance_df: pd.DataFrame,
    output_path: Path,
    title: str = "PPI Similarity Clustermap",
    method: str = "average",
) -> None:
    """Plot clustered similarity matrix with dendrograms."""
    if distance_df is None or distance_df.empty:
        logger.warning("No PPI distance matrix; skipping clustered similarity plot.")
        return
    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
    except ImportError:  # pragma: no cover
        logger.warning("SciPy not available; skipping clustered similarity plot.")
        return

    try:
        dist_array = squareform(distance_df.values, checks=False)
    except ValueError as exc:
        logger.warning("Failed to convert distance matrix to condensed form: %s", exc)
        return

    linkage_mat = linkage(dist_array, method=method)
    similarity_df = 1.0 - distance_df

    tick_setting = similarity_df.shape[0] <= 80
    g = sns.clustermap(
        similarity_df,
        row_linkage=linkage_mat,
        col_linkage=linkage_mat,
        cmap="Greens",
        linewidths=0,
        xticklabels=tick_setting,
        yticklabels=tick_setting,
        figsize=(18, 18),
    )
    g.fig.suptitle(title, y=1.02)
    output_path = ensure_dir(output_path.parent) / output_path.name
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    logger.info("Saved PPI similarity clustermap to %s", output_path)


def plot_ppi_distance_distribution(
    distances: np.ndarray,
    output_path: Path,
    title: str = "Distribution of PPI Distances"
) -> None:
    """Plot histogram of pairwise distances."""
    if distances.size == 0:
        logger.warning("No PPI distance values; skipping histogram.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        distances,
        bins=20,
        color=PPI_MEDIAN_COLOR,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )
    ax.set_xlabel("Pairwise Jaccard distance")
    ax.set_ylabel("Number of cell-type pairs")
    ax.set_title(title)
    _apply_publication_axes_style(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved PPI distance distribution to {output_path}")


def plot_ppi_similarity_hist(
    similarities: np.ndarray,
    output_path: Path,
    xlabel: str,
    title: str,
) -> None:
    """Plot histogram of pairwise PPI similarity values."""
    if similarities.size == 0:
        logger.warning(f"No values available for {title}; skipping histogram.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        similarities,
        bins=40,
        color=SIMILARITY_COLOR,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.95,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    _apply_publication_axes_style(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved similarity histogram to {output_path}")


def plot_basic_hist(
    values: List[float],
    output_path: Path,
    *,
    bins: int = 30,
    color: str = CELL_COLOR,
    xlabel: str,
    title: str,
    ylabel: str = "Count",
):
    """Lightweight histogram helper with consistent styling."""
    if not values:
        logger.warning("No values to plot for %s", output_path.name)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        values,
        bins=bins,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _apply_publication_axes_style(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot to %s", output_path)


def _apply_publication_axes_style(ax: plt.Axes) -> None:
    """Apply a consistent, publication-style aesthetic to histogram axes."""
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=11, width=0.8)


def generate_summary_sentence(
    dataset_name: str,
    ppi_stats: Dict,
    metagraph: nx.Graph,
    global_ppi_count: int,
) -> str:
    """Generate the publication summary sentence for a dataset."""
    # Count tissue nodes
    tissue_nodes = [n for n in metagraph.nodes() if is_tissue_node(n)]

    # Count tissues directly connected to cell types
    tissues_connected_to_cells = set()
    for u, v in metagraph.edges():
        if is_tissue_node(u) and not is_tissue_node(v):
            tissues_connected_to_cells.add(u)
        elif is_tissue_node(v) and not is_tissue_node(u):
            tissues_connected_to_cells.add(v)

    # Count edge types
    cell_cell_edges = 0
    cell_tissue_edges = 0
    tissue_tissue_edges = 0

    for u, v in metagraph.edges():
        u_is_tissue = is_tissue_node(u)
        v_is_tissue = is_tissue_node(v)

        if not u_is_tissue and not v_is_tissue:
            cell_cell_edges += 1
        elif u_is_tissue and v_is_tissue:
            tissue_tissue_edges += 1
        else:
            cell_tissue_edges += 1

    summary = (
        f"[{dataset_name.upper()}] We have {ppi_stats['num_networks']} cell type-specific protein interaction networks, "
        f"which have, on median, {ppi_stats['median_nodes']:.0f} proteins per network "
        f"(median edges: {ppi_stats['median_edges']:.0f}). "
        f"The number of unique proteins across all cell type-specific protein interaction networks is "
        f"{ppi_stats['unique_proteins']} of the {global_ppi_count} proteins in the global reference protein interaction network. "
        f"In the metagraph, we have {len(tissue_nodes)} tissues (nodes), and {len(tissues_connected_to_cells)} are directly connected to cell types. "
        f"There are {cell_cell_edges} cell-cell interactions, {cell_tissue_edges} cell-tissue edges and {tissue_tissue_edges} tissue-tissue edges."
    )

    return summary


def evaluate_dataset(dataset_name: str, dataset_path: Path, output_dir: Path):
    """Evaluate a single dataset and generate all plots and statistics."""
    logger.info("Evaluating dataset: %s", dataset_name.upper())

    dataset_output_dir = ensure_dir(output_dir / dataset_name)

    logger.info("Loading data...")
    metagraph = load_metagraph(dataset_path)
    ppi_networks = load_ppi_networks(dataset_path)
    cci_network = load_cci_network(dataset_path)
    reliable_gene_map, reliable_gene_union = load_reliable_genes(dataset_path)
    global_ppi_path = dataset_path / "global_ppi_edgelist.txt"
    if not global_ppi_path.exists():
        global_ppi_path = Path(GLOBAL_PPI)
    global_ppi_count = read_edgelist_any(global_ppi_path).number_of_nodes()

    logger.info(f"  Metagraph: {metagraph.number_of_nodes()} nodes, {metagraph.number_of_edges()} edges")
    logger.info(f"  PPI networks: {len(ppi_networks)}")
    logger.info(f"  CCI network: {cci_network.number_of_nodes()} nodes, {cci_network.number_of_edges()} edges")
    logger.info(f"  Reliable genes (union): {len(reliable_gene_union)}")

    logger.info("Analyzing metagraph...")
    metagraph_degrees = analyze_metagraph_degrees(metagraph)
    tissue_tissue_graph = extract_subgraph_by_node_type(metagraph, keep_tissue=True)
    cell_cell_graph = extract_subgraph_by_node_type(metagraph, keep_tissue=False)

    logger.info("Analyzing PPI networks...")
    ppi_stats = compute_ppi_statistics(ppi_networks)
    ppi_distance_df, ppi_distance_stats = compute_ppi_distance_matrix(ppi_networks)

    logger.info("Counting CCI degree per cell type...")
    cci_degrees = count_cci_degree_per_celltype(cci_network, ppi_networks)

    logger.info("Generating plots...")

    # 1. Metagraph degree distribution
    plot_metagraph_degree_distribution(
        metagraph_degrees,
        dataset_output_dir / "metagraph_degree_distribution.png",
        title=f"Metagraph Degree Distribution ({dataset_name.upper()})"
    )

    # 2. Tissue-tissue degree distribution
    plot_degree_distribution(
        tissue_tissue_graph,
        dataset_output_dir / "tissue_tissue_degree_distribution.png",
        title=f"Tissue-Tissue Graph Degree Distribution ({dataset_name.upper()})"
    )

    # 3. Cell-cell degree distribution
    plot_degree_distribution(
        cell_cell_graph,
        dataset_output_dir / "cell_cell_degree_distribution.png",
        title=f"Cell Type-Cell Type Graph Degree Distribution ({dataset_name.upper()})"
    )

    # 4. Median node degree of PPI networks
    if ppi_stats['median_degrees']:
        plot_ppi_metric_distribution(
            ppi_stats['median_degrees'],
            dataset_output_dir / "ppi_median_degree_distribution.png",
            title=f"Median Node Degree of Cell Type-Specific PPI Networks ({dataset_name.upper()})",
            xlabel="Median Node Degree"
        )

    # 5. CCI degree per PPI-backed cell type
    if cci_degrees:
        plot_ppi_metric_distribution(
            list(cci_degrees.values()),
            dataset_output_dir / "cci_degree_per_celltype.png",
            title=f"CCI Degree per Cell Type PPI Network ({dataset_name.upper()})",
            xlabel="CCI Degree",
        )

    # 6. Pairwise PPI distances (all datasets)
    upper_vals = None
    if ppi_distance_df is not None:
        plot_ppi_distance_heatmap(
            ppi_distance_df,
            dataset_output_dir / "ppi_distance_heatmap.png",
            title=f"Pairwise Cell-Type PPI Distances ({dataset_name.upper()})",
        )
        upper_vals = ppi_distance_df.values[np.triu_indices_from(ppi_distance_df.values, k=1)]
        plot_ppi_distance_distribution(
            upper_vals,
            dataset_output_dir / "ppi_distance_distribution.png",
            title="Distribution of Pairwise PPI Distances",
        )
        if dataset_name == "merged":
            plot_ppi_similarity_clustermap(
                ppi_distance_df,
                dataset_output_dir / "ppi_distance_clustermap.png",
                title="Pairwise PPIN Similarity (All Cell Types)",
            )
        if ppi_distance_stats:
            logger.info(
                "PPI distance stats (mean=%.3f, median=%.3f, min=%.3f, max=%.3f, pairs=%d)",
                ppi_distance_stats["mean_distance"],
                ppi_distance_stats["median_distance"],
                ppi_distance_stats["min_distance"],
                ppi_distance_stats["max_distance"],
                ppi_distance_stats["num_pairs"],
            )

    # Additional plots for merged dataset
    if dataset_name == "merged":
        logger.info("Generating additional plots for merged dataset...")

        # PPI size distribution (LCC)
        if ppi_stats['lcc_sizes']:
            plot_ppi_metric_distribution(
                ppi_stats['lcc_sizes'],
                dataset_output_dir / "ppi_lcc_size_distribution.png",
                title="PPI Network Size Distribution (Largest Connected Component)",
                xlabel="LCC Size (number of nodes)"
            )

        # Network size scatter plot
        if ppi_stats['node_counts'] and ppi_stats['edge_counts']:
            plot_ppi_size_scatter(
                ppi_stats['node_counts'],
                ppi_stats['edge_counts'],
                dataset_output_dir / "ppi_nodes_vs_edges.png"
            )

        # Manuscript-style similarity histograms (all PPIs)
        edge_similarities = 1.0 - upper_vals if upper_vals is not None else np.array([])
        if edge_similarities.size:
            plot_ppi_similarity_hist(
                edge_similarities,
                dataset_output_dir / "ppi_edge_similarity_distribution.png",
                xlabel="Pairwise PPIN Edge Jaccard Similarity",
                title="Pairwise PPIN Edge Jaccard Similarity",
            )
            logger.info(
                "Edge similarity stats (min=%.3f, median=%.3f, mean=%.3f, max=%.3f)",
                edge_similarities.min(),
                np.median(edge_similarities),
                edge_similarities.mean(),
                edge_similarities.max(),
            )
        node_similarities = compute_ppi_node_similarity_values(ppi_networks)
        if node_similarities.size:
            plot_ppi_similarity_hist(
                node_similarities,
                dataset_output_dir / "ppi_node_similarity_distribution.png",
                xlabel="Pairwise PPIN Node Jaccard Similarity",
                title="Pairwise PPIN Node Jaccard Similarity",
            )
            logger.info(
                "Node similarity stats (min=%.3f, median=%.3f, mean=%.3f, max=%.3f)",
                node_similarities.min(),
                np.median(node_similarities),
                node_similarities.mean(),
                node_similarities.max(),
            )
        # Cell-cell (CCI) degree distribution
        if cci_network.number_of_nodes():
            cci_node_degrees = [cci_network.degree(n) for n in cci_network.nodes()]
            plot_basic_hist(
                cci_node_degrees,
                dataset_output_dir / "cci_degree_distribution.png",
                bins=60,
                color=CELL_COLOR,
                xlabel="Degree of Cell type-Cell type Graph",
                title="Degree of Cell type-Cell type Graph (Merged)",
            )

    celltype_rows: List[Dict[str, object]] = []
    celltype_summary_path: Optional[Path] = None
    if ppi_networks:
        gene_lookup = build_gene_count_lookup(reliable_gene_map) if reliable_gene_map else {}
        celltype_rows = build_celltype_summary_rows(ppi_networks, gene_lookup)
        if celltype_rows:
            celltype_summary_path = dataset_output_dir / "celltype_reg_ppi_summary.csv"
            pd.DataFrame(celltype_rows).to_csv(celltype_summary_path, index=False)
            logger.info("Saved cell-type REG/PPI summary to %s", celltype_summary_path)

    summary_sentence = generate_summary_sentence(
        dataset_name,
        ppi_stats,
        metagraph,
        global_ppi_count,
    )

    statistics = {
        "dataset": dataset_name,
        "summary_sentence": summary_sentence,
        "ppi_statistics": {
            "num_networks": ppi_stats['num_networks'],
            "mean_nodes": ppi_stats['mean_nodes'],
            "median_nodes": ppi_stats['median_nodes'],
            "std_nodes": ppi_stats['std_nodes'],
            "mean_edges": ppi_stats['mean_edges'],
            "median_edges": ppi_stats['median_edges'],
            "std_edges": ppi_stats['std_edges'],
            "unique_proteins": ppi_stats['unique_proteins'],
            "total_networks_with_cci": sum(degree > 0 for degree in cci_degrees.values()),
        },
        "metagraph_statistics": {
            "total_nodes": metagraph.number_of_nodes(),
            "total_edges": metagraph.number_of_edges(),
            "tissue_nodes": len([n for n in metagraph.nodes() if is_tissue_node(n)]),
            "cell_type_nodes": len([n for n in metagraph.nodes() if not is_tissue_node(n)]),
            "tissue_tissue_edges": tissue_tissue_graph.number_of_edges(),
            "cell_cell_edges": cell_cell_graph.number_of_edges(),
        },
        "celltype_summary": {
            "rows": len(celltype_rows),
            "file": celltype_summary_path.name if celltype_summary_path else None,
        },
        "ppi_distance_stats": ppi_distance_stats,
        "reliable_genes": len(reliable_gene_union),
    }

    stats_file = dataset_output_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)

    logger.info(f"Saved statistics to {stats_file}")

    logger.info("Summary: %s", summary_sentence)

    return statistics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate processed PPI, CCI, and metagraph outputs"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["hbca", "tabula", "als", "merged"],
        default=["hbca", "tabula", "als", "merged"],
        help="Datasets to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for evaluation reports",
    )
    parser.add_argument(
        "--hbca-dir",
        type=Path,
        default=Path(HBCA_INTERMEDIATE),
        help="Override path to HBCA intermediate outputs (supports Step 2b-filtered folders).",
    )
    parser.add_argument(
        "--tabula-dir",
        type=Path,
        default=Path(TABULA_INTERMEDIATE),
        help="Override path to Tabula intermediate outputs (supports Step 2b-filtered folders).",
    )
    parser.add_argument(
        "--als-dir",
        type=Path,
        default=Path(ALS_BULK_DIR),
        help="Override path to ALS outputs (combined or filtered).",
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=Path(MERGED_OUTPUT),
        help="Override path to merged dataset outputs.",
    )

    args = parser.parse_args()

    dataset_paths = {
        "hbca": args.hbca_dir.resolve(),
        "tabula": args.tabula_dir.resolve(),
        "als": args.als_dir.resolve(),
        "merged": args.merged_dir.resolve(),
    }
    if args.output_dir:
        output_dir = ensure_dir(args.output_dir.resolve())
    else:
        output_dir = ensure_dir(EVAL_OUTPUT_BASE)
    logger.info(f"Output directory: {output_dir}")
    logger.info("Dataset roots -> HBCA: %s | Tabula: %s | ALS: %s | Merged: %s",
                dataset_paths['hbca'], dataset_paths['tabula'], dataset_paths['als'], dataset_paths['merged'])

    all_statistics = {}

    for dataset_name in args.datasets:
        dataset_path = dataset_paths.get(dataset_name)
        if not dataset_path or not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found for {dataset_name}: {dataset_path}")
        all_statistics[dataset_name] = evaluate_dataset(
            dataset_name,
            dataset_path,
            output_dir,
        )

    aggregate_file = output_dir / "aggregate_summary.json"
    with open(aggregate_file, 'w') as f:
        json.dump(all_statistics, f, indent=2)

    logger.info("Evaluation results saved to %s", output_dir)


if __name__ == "__main__":
    main()
