#!/usr/bin/env python
"""Step 2b: rank genes by specificity and select a PPI-informed k per context."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse

from joblib import Parallel, delayed

from .config import (
    ALS_ASTRO_INTERMEDIATE,
    ALS_MN_INTERMEDIATE,
    GLOBAL_PPI,
    HBCA_GENE_METADATA,
    HBCA_INTERMEDIATE,
    TABULA_INTERMEDIATE,
)
from .step3_construct_ppi import map_genes
from .utils import pairwise_jaccard


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ALS_DATASETS = {"als_mn", "als_astro"}

CONTEXT_CANDIDATES = [
    "cell_type",
    "cell_type_label",
    "cell_type_ontology_term_id",
    "hbca_cell_type_ontology_term_id",
    "cell_type_name",
    "cell_type_id",
]

GENE_LOOKUP_COLUMNS = [
    "gene_symbol",
    "symbol",
    "gene_name",
    "GeneSymbol",
    "hgnc_symbol",
    "hgnc",
    "ensembl_id",
    "ensembl_gene_id",
    "ensembl",
    "gene_identifier",
    "gene_id",
    "GeneID",
    "original_id",
]

EPS = 1e-6
MAD_SCALE = 1.4826
CL_ID_PATTERN = re.compile(r"^(CL)[_:](\d{7})(.*)$")
N_JOBS = max(1, int(os.environ.get("PIPELINE_THREADS", 4)))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Read pseudobulk inputs and the candidate network-size grid."""
    parser = argparse.ArgumentParser(
        description="Step 2b: rank reliably expressed genes by context enrichment and select PPI size."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["tabula", "hbca", "als_mn", "als_astro"],
        help="Dataset to process.",
    )
    parser.add_argument("--pseudobulk", help="Override path to pseudobulk h5ad.")
    parser.add_argument("--reliable-genes", help="Override reliable_genes*.json from Step 2.")
    parser.add_argument("--output-dir", help="Override dataset intermediate directory.")
    parser.add_argument("--context-col", help="Override context column (obs).")
    parser.add_argument("--global-ppi", default=GLOBAL_PPI, help="Path to global PPI edgelist.")
    parser.add_argument("--log-offset", type=float, default=1.0, help="Offset for log2(count + offset).")
    parser.add_argument("--k-grid", help="Comma-separated list of k values (optional).")
    return parser.parse_args(argv)


def resolve_paths(dataset: str, args: argparse.Namespace) -> Tuple[str, str, str]:
    """Return the output directory, pseudobulk file and reliable-gene JSON for a dataset."""
    dataset = dataset.lower()
    if dataset == "tabula":
        default_root = TABULA_INTERMEDIATE
        default_pb = os.path.join(default_root, "pseudobulk.h5ad")
        default_reg = os.path.join(default_root, "reliable_genes.json")
    elif dataset == "hbca":
        default_root = HBCA_INTERMEDIATE
        default_pb = os.path.join(default_root, "pseudobulk.h5ad")
        default_reg = os.path.join(default_root, "reliable_genes.json")
    elif dataset == "als_mn":
        default_root = ALS_MN_INTERMEDIATE
        default_pb = os.path.join(default_root, "als_motor_neurons.h5ad")
        default_reg = os.path.join(default_root, "reliable_genes_per_celltype.json")
    elif dataset == "als_astro":
        default_root = ALS_ASTRO_INTERMEDIATE
        default_pb = os.path.join(default_root, "als_astrocytes.h5ad")
        default_reg = os.path.join(default_root, "reliable_genes_per_celltype.json")
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    base_dir = os.path.abspath(args.output_dir or default_root)
    pseudobulk = args.pseudobulk or default_pb
    reliable = args.reliable_genes or default_reg
    return base_dir, pseudobulk, reliable


def load_pseudobulk(path: str) -> sc.AnnData:
    """Load counts and make sample/gene identifiers unique for expression lookup."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pseudobulk AnnData not found: {path}")
    logger.info("Loading pseudobulk AnnData from %s", path)
    adata = sc.read_h5ad(path)
    if "counts" not in adata.layers:
        logger.warning("Counts layer missing; assuming X already stores raw counts.")
        adata.layers["counts"] = adata.X.copy()
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
    if not adata.var_names.is_unique:
        logger.warning("Gene identifiers are not unique; making var_names unique.")
        adata.var_names_make_unique()
    logger.info("Pseudobulk shape: %d samples × %d genes", adata.n_obs, adata.n_vars)
    return adata


def load_reliable_genes(path: str) -> Dict[str, List[str]]:
    """Read a gene union or per-context lists, merging equivalent CL labels."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reliable genes file not found: {path}")
    with open(path, "r") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {"default": list(dict.fromkeys(map(str, data)))}
    if isinstance(data, dict):
        normalized = {str(k): list(dict.fromkeys(map(str, v))) for k, v in data.items()}
        canonical: Dict[str, List[str]] = {}
        for raw_key, genes in normalized.items():
            key = canonical_cell_label(raw_key) or raw_key
            if key not in canonical:
                canonical[key] = list(genes)
            else:
                existing = set(canonical[key])
                canonical[key].extend(gene for gene in genes if gene not in existing)
        logger.info(
            "Loaded reliable genes for %d canonical contexts (from %d raw keys).",
            len(canonical),
            len(normalized),
        )
        return canonical
    raise ValueError(f"Unexpected format in {path}")




def resolve_context_column(
    adata: sc.AnnData,
    dataset: str,
    override: Optional[str],
) -> str:
    """Select the requested cell-type metadata column or a supported dataset default."""
    if override:
        if override not in adata.obs:
            raise KeyError(f"Context column '{override}' is not present in AnnData.obs")
        return override
    if dataset in ALS_DATASETS:
        if "cell_type_id" not in adata.obs.columns:
            adata.obs["cell_type_id"] = adata.obs.index.astype(str)
        return "cell_type_id"
    for candidate in CONTEXT_CANDIDATES:
        if candidate in adata.obs.columns:
            return candidate
    raise KeyError(f"No cell-type column found; tried {CONTEXT_CANDIDATES}")


def canonical_cell_label(value: object) -> str:
    """Normalize context labels so ALS keys match REG outputs."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if "|" in text:
        text = text.split("|", 1)[0]
    match = CL_ID_PATTERN.match(text)
    if match:
        prefix, digits, rest = match.groups()
        rest = rest or ""
        return f"{prefix}:{digits}{rest}"
    return text




def normalize_gene_id(value: object) -> str:
    """Strip version suffixes and uppercase identifiers before matching genes."""
    if value is None:
        return ""
    gene = str(value).strip()
    if not gene:
        return ""
    if "." in gene:
        gene = gene.split(".", 1)[0]
    return gene.upper()


def build_gene_lookup(adata: sc.AnnData) -> Dict[str, int]:
    """Map gene identifiers and annotation aliases to expression columns; first match wins."""
    lookup: Dict[str, int] = {}
    var_df = adata.var
    columns = [col for col in GENE_LOOKUP_COLUMNS if col in var_df.columns]
    var_names = adata.var_names.astype(str)

    identifiers = [var_names, *(var_df[col].astype(str).values for col in columns)]
    for values in identifiers:
        for idx, value in enumerate(values):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            text = str(value).strip()
            if not text or text.lower() in {"nan", "none"}:
                continue
            for key in (text, normalize_gene_id(text)):
                if key and key not in lookup:
                    lookup[key] = idx
    return lookup


SYMBOL_PRIORITY = [
    "symbol",
    "gene_symbol",
    "GeneSymbol",
    "hgnc_symbol",
    "hgnc",
    "gene_name",
]


def match_reliable_genes(
    adata: sc.AnnData,
    genes: Sequence[str],
    *,
    symbol_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Return matched raw counts, input gene IDs and corresponding PPI symbols in the same order."""
    lookup = build_gene_lookup(adata)
    positions: List[int] = []
    for gene in genes:
        idx = next(
            (lookup[key] for key in (gene, normalize_gene_id(gene)) if key in lookup),
            -1,
        )
        positions.append(idx)

    missing = sum(pos < 0 for pos in positions)
    if missing:
        logger.warning("Dropping %d genes absent from the pseudobulk matrix.", missing)
    else:
        logger.info("Matched all %d genes to the pseudobulk matrix.", len(genes))

    valid_positions = [pos for pos in positions if pos >= 0]
    valid_genes = [gene for gene, pos in zip(genes, positions) if pos >= 0]
    if not valid_positions:
        logger.error("No genes matched the pseudobulk matrix.")
        return np.empty((adata.n_obs, 0), dtype=float), [], []

    counts = adata.layers.get("counts", adata.X)
    if sparse.issparse(counts):
        counts = counts.tocsr()[:, valid_positions].toarray()
    else:
        counts = np.asarray(counts[:, valid_positions])
    # Kallisto estimated counts and replicate averages are fractional; retain
    # those values when computing context-specific expression ranks.
    counts = counts.astype(np.float64, copy=False)
    logger.info("Proceeding with %d/%d genes after mapping.", len(valid_positions), len(genes))

    ppi_names: List[str] = []
    symbol_col = next((col for col in SYMBOL_PRIORITY if col in adata.var.columns), None)
    symbols = adata.var[symbol_col].astype(str).values if symbol_col else None

    for pos, gene in zip(valid_positions, valid_genes):
        mapped = None
        if symbol_mapping:
            norm = normalize_gene_id(gene)
            mapped = symbol_mapping.get(gene) or symbol_mapping.get(norm)
        if not mapped and symbols is not None:
            candidate = symbols[pos].strip()
            if candidate and candidate.lower() not in {"nan", "none"}:
                mapped = candidate
        if not mapped:
            mapped = normalize_gene_id(gene)
        if mapped:
            ppi_names.append(str(mapped).strip().upper())
        else:
            ppi_names.append(normalize_gene_id(gene))

    return counts, valid_genes, ppi_names


def average_context_expression(
    log_expr: np.ndarray,
    labels: np.ndarray,
    unique_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average replicate log-expression profiles into a genes-by-context matrix."""
    n_cells = unique_labels.size
    collapsed = np.zeros((n_cells, log_expr.shape[1]), dtype=float)
    sample_counts = np.zeros(n_cells, dtype=int)
    for idx, cell in enumerate(unique_labels):
        mask = labels == cell
        sample_counts[idx] = int(mask.sum())
        collapsed[idx] = log_expr[mask].mean(axis=0)
    return collapsed.T, sample_counts


def context_enrichment_scores(context_expression: np.ndarray) -> np.ndarray:
    """Compare each context to the others using their median and scaled MAD."""
    n_cells = context_expression.shape[1]
    if n_cells < 2:
        raise ValueError("One-versus-rest enrichment requires at least two contexts.")
    if not np.isfinite(context_expression).all():
        raise ValueError("Context expression values must be finite.")

    def score_context(j: int) -> np.ndarray:
        """Compute one context’s enrichment over the median and scaled MAD of the others."""
        others = np.delete(context_expression, j, axis=1)
        median = np.median(others, axis=1)
        mad = MAD_SCALE * np.median(np.abs(others - median[:, None]), axis=1)
        return (context_expression[:, j] - median) / (mad + EPS)

    n_jobs = min(n_cells, N_JOBS)
    columns = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(score_context)(j) for j in range(n_cells)
    )
    return np.column_stack(columns)


def rank_reliable_genes(
    enrichment_scores: np.ndarray,
    gene_names: Sequence[str],
    cell_types: Sequence[str],
    reg_sets: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """Rank each context's REGs by enrichment, breaking exact ties by gene ID."""
    gene_names = np.asarray(gene_names)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    score_order = {}
    for c_idx, cell in enumerate(cell_types):
        genes = reg_sets.get(cell, reg_sets.get("default"))
        if genes is None:
            raise KeyError(f"Missing reliably expressed genes for context '{cell}'.")
        allowed = np.array(
            sorted({gene_to_idx[g] for g in genes if g in gene_to_idx}), dtype=int
        )
        order = np.lexsort((gene_names[allowed], -enrichment_scores[allowed, c_idx]))
        score_order[cell] = allowed[order]
    return score_order


def candidate_network_sizes(max_genes: int, override: Optional[str]) -> List[int]:
    """Use explicit k values or an evenly spaced grid ending at the available gene count."""
    if override:
        grid = sorted({max(1, int(k.strip())) for k in override.split(",") if k.strip()})
        return [k for k in grid if k <= max_genes]
    if max_genes <= 0:
        return [1]
    if max_genes <= 500:
        return [max_genes]

    start = 500
    step = max(100, int((max_genes - start) / 15) or 100)
    grid = list(range(start, max_genes + 1, step))
    if grid[-1] != max_genes:
        grid.append(max_genes)
    return sorted(set(grid))


def largest_connected_component_nodes(ppi: nx.Graph, genes: Iterable[str]) -> List[str]:
    """Return the largest connected part of the PPI induced by these genes."""
    genes = [g for g in genes if g in ppi]
    if not genes:
        return []
    return list(max(nx.connected_components(ppi.subgraph(genes)), key=len))


def evaluate_network_sizes(
    score_order: Dict[str, np.ndarray],
    ppi_gene_names: np.ndarray,
    global_ppi: nx.Graph,
    k_grid: List[int],
) -> Tuple[
    List[Dict[str, float]],
    Dict[str, List[float]],
    Dict[str, List[float]],
]:
    """
    For each k in the grid, compute:
      * median Jaccard similarity of LCC node sets across cell types
      * median LCC coverage per cell type
      * per-cell coverage and Jaccard curves (for later per-cell k selection)
    """
    cells = list(score_order.keys())

    def evaluate_network_size(k_val: int) -> Tuple[
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
    ]:
        """Measure connectivity and between-context overlap for the top k genes in each context."""
        node_sets: Dict[str, set] = {}
        coverage: List[float] = []
        coverage_by_cell: Dict[str, float] = {}

        for cell, order in score_order.items():
            top_idx = order[: min(k_val, len(order))]
            ppi_genes = ppi_gene_names[top_idx]
            lcc_nodes = largest_connected_component_nodes(global_ppi, ppi_genes)
            node_sets[cell] = set(lcc_nodes)

            cov = len(lcc_nodes) / max(k_val, 1)
            coverage.append(cov)
            coverage_by_cell[cell] = cov

        jac = pairwise_jaccard(node_sets)

        per_cell_jacc: Dict[str, float] = {}
        for cell, cell_nodes in node_sets.items():
            scores: List[float] = []
            for other, other_nodes in node_sets.items():
                if other == cell:
                    continue
                union = len(cell_nodes | other_nodes)
                if union == 0:
                    continue
                inter = len(cell_nodes & other_nodes)
                scores.append(inter / union)
            per_cell_jacc[cell] = float(np.median(scores)) if scores else 0.0

        combined_scores = [
            coverage_by_cell[cell] * (1.0 - per_cell_jacc.get(cell, 0.0))
            for cell in node_sets.keys()
        ]

        metrics_entry = {
            "k": k_val,
            "median_jaccard_nodes": float(np.median(jac)) if jac else 0.0,
            "median_coverage": float(np.median(coverage)) if coverage else 0.0,
            "median_combined_score": float(np.median(combined_scores)) if combined_scores else 0.0,
        }

        return metrics_entry, coverage_by_cell, per_cell_jacc

    results: List[
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
    ] = []
    if len(k_grid) > 1:
        n_jobs = min(len(k_grid), N_JOBS)
        logger.info("  Evaluating k grid in parallel with %d jobs.", n_jobs)
        results = Parallel(n_jobs=n_jobs)(delayed(evaluate_network_size)(k) for k in k_grid)
    else:
        total = len(k_grid)
        for idx, k_val in enumerate(k_grid, start=1):
            results.append(evaluate_network_size(k_val))
            if idx % 5 == 0 or idx == total:
                logger.info("  Evaluated %d/%d k values (last k=%d).", idx, total, k_val)

    metrics: List[Dict[str, float]] = []
    coverage_series: Dict[str, List[float]] = {cell: [] for cell in cells}
    jaccard_series: Dict[str, List[float]] = {cell: [] for cell in cells}
    for metric_entry, cov_cell, jac_cell in results:
        metrics.append(metric_entry)
        for cell in cells:
            coverage_series[cell].append(cov_cell.get(cell, 0.0))
            jaccard_series[cell].append(jac_cell.get(cell, 0.0))

    return metrics, coverage_series, jaccard_series


def compute_global_hubs(ppi: nx.Graph, percentile: float = 95.0) -> set[str]:
    """Return proteins at or above the requested global PPI degree percentile."""
    degrees = dict(ppi.degree())
    if not degrees:
        return set()
    threshold = np.percentile(list(degrees.values()), percentile)
    return {node for node, deg in degrees.items() if deg >= threshold}


def select_context_gene_sets(
    score_order: Dict[str, np.ndarray],
    enrichment_scores: np.ndarray,
    gene_names: np.ndarray,
    ppi_gene_names: np.ndarray,
    cell_types: np.ndarray,
    sample_counts: np.ndarray,
    global_ppi: nx.Graph,
    k_by_cell: Dict[str, int],
    hub_nodes: set[str],
) -> Tuple[Dict[str, List[str]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Select each context’s top-k genes and return network summaries and ranked gene scores."""
    gene_sets: Dict[str, List[str]] = {}
    summary_rows: List[Dict[str, object]] = []
    score_records: List[Dict[str, object]] = []

    for c_idx, cell in enumerate(cell_types):
        k_cell = int(k_by_cell[cell])
        order = score_order[cell]
        top_idx = order[: min(k_cell, len(order))]
        selected = list(gene_names[top_idx])
        ppi_selected = list(ppi_gene_names[top_idx])
        gene_sets[cell] = selected

        lcc_nodes = largest_connected_component_nodes(global_ppi, ppi_selected)
        lcc_graph = global_ppi.subgraph(lcc_nodes).copy()
        lcc_size = lcc_graph.number_of_nodes()
        lcc_edges = lcc_graph.number_of_edges()
        coverage = lcc_size / max(k_cell, 1)
        if lcc_edges > 0 and hub_nodes:
            hub_edge_count = sum(1 for u, v in lcc_graph.edges() if u in hub_nodes or v in hub_nodes)
            hub_edge_pct = hub_edge_count / lcc_edges
        else:
            hub_edge_pct = 0.0

        summary_rows.append(
            {
                "cell_type": cell,
                "n_samples": int(sample_counts[c_idx]),
                "top_k": k_cell,
                "genes_selected": len(selected),
                "lcc_nodes": lcc_size,
                "lcc_edges": lcc_edges,
                "lcc_coverage": coverage,
                "hub_edge_fraction": hub_edge_pct,
            }
        )

        for rank, idx in enumerate(top_idx, start=1):
            score_records.append(
                {
                    "cell_type": cell,
                    "gene": gene_names[idx],
                    "rank": rank,
                    "z_robust": float(enrichment_scores[idx, c_idx]),
                }
            )

    return gene_sets, summary_rows, score_records


def plot_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Save distributions of selected network size, connectivity and hub-edge coverage."""
    qc_dir = output_dir / "specificity_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(summary_df["genes_selected"], bins=30, color="steelblue", ax=ax)
    ax.set_xlabel("Genes selected (top-k*)")
    ax.set_ylabel("Cell types")
    ax.set_title("Distribution of selected genes per cell type")
    fig.tight_layout()
    fig.savefig(qc_dir / "selected_genes_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=summary_df,
        x="genes_selected",
        y="lcc_nodes",
        hue="n_samples",
        palette="viridis",
        ax=ax,
    )
    ax.set_xlabel("Genes selected")
    ax.set_ylabel("LCC size (nodes)")
    ax.set_title("Selected genes vs. LCC nodes")
    fig.tight_layout()
    fig.savefig(qc_dir / "genes_vs_lcc.png", dpi=150)
    plt.close(fig)


def plot_k_metrics(
    metrics: List[Dict[str, float]],
    k_star: int,
    output_dir: Path,
) -> None:
    """Plot connectivity, overlap and their combined score across candidate k values."""
    if not metrics:
        return
    qc_dir = output_dir / "specificity_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    ks = [m["k"] for m in metrics]
    j_values = [m["median_jaccard_nodes"] for m in metrics]
    c_values = [m["median_coverage"] for m in metrics]
    comb_values = [m["median_combined_score"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, j_values, marker="o", color="tab:blue", label="Median node Jaccard")
    ax.plot(ks, c_values, marker="s", color="tab:green", label="Median LCC coverage")
    ax.plot(ks, comb_values, marker="^", color="tab:gray", label="Median combined score")

    ax.axvline(k_star, color="red", linestyle="--", linewidth=2, label=f"k* = {k_star}")
    ax.set_xlabel("k (top genes per cell)")
    ax.set_ylabel("Metric value")
    ax.set_title("k-grid diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(qc_dir / "k_grid_metrics.png", dpi=150)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Rank reliable genes by context enrichment and choose k to balance connectivity and specificity."""
    args = parse_args(argv)
    dataset_key = args.dataset.lower()
    base_dir, pseudobulk_path, reliable_path = resolve_paths(args.dataset, args)
    adata = load_pseudobulk(pseudobulk_path)
    context_col = resolve_context_column(adata, dataset_key, args.context_col)
    reliable = load_reliable_genes(reliable_path)

    symbol_mapping: Dict[str, str] = {}
    if dataset_key == "hbca":
        metadata_path = HBCA_GENE_METADATA if os.path.exists(HBCA_GENE_METADATA) else None
        symbol_mapping = map_genes(adata, metadata_path)

    union_genes = sorted({gene for genes in reliable.values() for gene in genes})
    if not union_genes:
        raise ValueError("Reliable gene list is empty. Run Step 2 first.")

    logger.info("Mapping %d reliable genes onto the pseudobulk matrix...", len(union_genes))
    counts, gene_names, ppi_gene_names = match_reliable_genes(
        adata,
        union_genes,
        symbol_mapping=symbol_mapping or None,
    )
    if counts.size == 0:
        raise ValueError("None of the reliable genes matched the pseudobulk matrix.")
    logger.info("Gene mapping complete: %d genes retained.", counts.shape[1])

    log_expr = np.log2(counts + args.log_offset)
    logger.info(
        "Computed log2(count + %.2f) matrix: %d genes × %d samples",
        args.log_offset,
        log_expr.shape[1],
        log_expr.shape[0],
    )

    raw_labels = adata.obs[context_col].astype(str).values
    canon_labels = np.array(
        [canonical_cell_label(label) or str(label).strip() for label in raw_labels]
    )
    if not np.array_equal(raw_labels, canon_labels):
        diff_idx = np.where(raw_labels != canon_labels)[0]
        sample_example = diff_idx[0] if diff_idx.size else 0
        logger.info(
            "Canonicalized %d/%d context labels (e.g., %s -> %s).",
            diff_idx.size,
            raw_labels.size,
            raw_labels[sample_example],
            canon_labels[sample_example],
        )
    unique_cells = np.array(sorted(pd.unique(canon_labels)))
    logger.info("Collapsing expression per context (averaging replicates)...")
    context_expression, sample_counts = average_context_expression(log_expr, canon_labels, unique_cells)
    logger.info(
        "Collapsed expression to %d contexts (median %d samples each).",
        len(unique_cells),
        int(np.median(sample_counts)),
    )

    logger.info("Computing robust one-vs-rest z-scores for %d contexts...", len(unique_cells))
    enrichment_scores = context_enrichment_scores(context_expression)
    logger.info("Robust z computation complete.")
    logger.info("Ranking reliably expressed genes by decreasing enrichment...")
    gene_names_arr = np.asarray(gene_names)
    score_order = rank_reliable_genes(enrichment_scores, gene_names_arr, unique_cells, reliable)

    global_ppi = nx.read_edgelist(args.global_ppi, nodetype=str)
    k_grid = candidate_network_sizes(len(gene_names_arr), args.k_grid)
    if not k_grid:
        raise ValueError("The k grid contains no values within the available gene count.")
    logger.info(
        "Evaluating k grid from %d to %d (%d values).",
        k_grid[0],
        k_grid[-1],
        len(k_grid),
    )
    ppi_gene_names_arr = np.asarray(ppi_gene_names)
    (
        k_metrics,
        coverage_per_cell,
        jaccard_per_cell,
    ) = evaluate_network_sizes(score_order, ppi_gene_names_arr, global_ppi, k_grid)

    # The grid is sorted, so ties choose the smaller network.
    ks = np.array([m["k"] for m in k_metrics])
    combined_scores = [m["median_combined_score"] for m in k_metrics]
    k_combined = int(ks[np.argmax(combined_scores)])
    k_by_cell = {}
    for cell in score_order:
        coverage = np.asarray(coverage_per_cell[cell])
        overlap = np.asarray(jaccard_per_cell[cell])
        k_by_cell[cell] = int(ks[np.argmax(coverage * (1.0 - overlap))])

    k_values = list(k_by_cell.values())
    k_star = int(np.median(k_values))
    logger.info(
        "Selected global k=%d; per-cell k: min=%d, median=%d, max=%d.",
        k_combined,
        min(k_values),
        k_star,
        max(k_values),
    )

    hub_nodes = compute_global_hubs(global_ppi)
    ppi_gene_names_arr = np.asarray(ppi_gene_names)
    gene_sets, summary_rows, score_records = select_context_gene_sets(
        score_order,
        enrichment_scores,
        gene_names_arr,
        ppi_gene_names_arr,
        unique_cells,
        sample_counts,
        global_ppi,
        k_by_cell,
        hub_nodes,
    )

    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(
        output_dir,
        gene_sets,
        summary_rows,
        score_records,
        k_metrics,
        k_star,
    )
    logger.info("Step 2b complete for dataset %s.", args.dataset)


def write_outputs(
    output_dir: Path,
    gene_sets: Dict[str, List[str]],
    summary_rows: List[Dict[str, object]],
    score_records: List[Dict[str, object]],
    k_metrics: List[Dict[str, float]],
    k_star: int,
) -> None:
    """Save selected gene unions, per-context lists, ranks and network-size summaries."""
    with open(output_dir / "reliable_genes_specific.json", "w") as handle:
        json.dump(gene_sets, handle, indent=2)
    with open(output_dir / "reliable_genes_specific_per_celltype.json", "w") as handle:
        json.dump(gene_sets, handle, indent=2)
    union = sorted({gene for genes in gene_sets.values() for gene in genes})
    with open(output_dir / "reliable_genes_specific_union.json", "w") as handle:
        json.dump(union, handle, indent=2)

    summary_df = pd.DataFrame(summary_rows).sort_values("cell_type")
    summary_df["k_star"] = k_star
    summary_df.to_csv(output_dir / "specific_gene_summary.csv", index=False)

    scores_df = pd.DataFrame(score_records)
    scores_df.to_csv(output_dir / "specific_gene_scores.csv.gz", index=False, compression="gzip")

    metrics_df = pd.DataFrame(k_metrics)
    metrics_df["k_star"] = k_star
    metrics_df.to_csv(output_dir / "k_selection_metrics.csv", index=False)

    plot_summary(summary_df, output_dir)
    plot_k_metrics(k_metrics, k_star, output_dir)


if __name__ == "__main__":
    main()
