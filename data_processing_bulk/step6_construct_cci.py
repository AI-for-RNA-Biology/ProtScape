#!/usr/bin/env python
"""Step 6: construct CCI networks from significant CellPhoneDB interactions."""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .config import (
    CELLPHONEDB_MIN_LR,
    CELLPHONEDB_MIN_LR_BY_DATASET,
    CELLPHONEDB_PVALUE,
    CELLPHONEDB_THRESHOLD,
    HBCA_CCI_EDGELIST,
    HBCA_CELLPHONEDB_OUTPUT,
    TABULA_CCI_EDGELIST,
    TABULA_CELLPHONEDB_OUTPUT,
    TABULA_INTERMEDIATE,
    ALS_BULK_DIR,
    MERGED_CCI_EDGELIST,
    MERGED_CELLPHONEDB_OUTPUT,
)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


PairCounts = Dict[Tuple[str, str], int]


def normalize_cl_node(label: str) -> str:
    """Normalize Cell Ontology identifiers to CL_XXXX format while preserving suffixes."""
    if not isinstance(label, str):
        return label
    cleaned = label.strip()
    if not cleaned:
        return cleaned
    if cleaned[:3].lower() == "cl:":
        cleaned = "CL_" + cleaned[3:]
    elif cleaned[:3].lower() == "cl_":
        cleaned = "CL_" + cleaned[3:]
    return cleaned


def normalize_graph_nodes(graph: nx.Graph) -> nx.Graph:
    """Return a copy of the graph with CL node identifiers normalized."""
    mapping = {}
    for node in graph.nodes():
        normalized = normalize_cl_node(node)
        if normalized != node:
            mapping[node] = normalized
    if not mapping:
        return graph
    return nx.relabel_nodes(graph, mapping, copy=True)


def is_nuclear_context(node: str) -> bool:
    """Return True for an ALS nuclear pseudo-bulk context."""
    if not isinstance(node, str):
        return False
    label = normalize_cl_node(node).lower()
    return bool(label.startswith("cl_") and re.search(r"(^|_)nuc($|_)", label))


def is_cytoplasmic_context(node: str) -> bool:
    """Return True for an ALS cytoplasmic pseudo-bulk context."""
    if not isinstance(node, str):
        return False
    label = normalize_cl_node(node).lower()
    return bool(label.startswith("cl_") and re.search(r"(^|_)cyto($|_)", label))


def remove_invalid_compartment_edges(graph: nx.Graph) -> int:
    """Remove only nuc--nuc and nuc--cyto CCIs."""
    edges_to_remove = [
        (u, v)
        for u, v in graph.edges()
        if (
            is_nuclear_context(u)
            and (is_nuclear_context(v) or is_cytoplasmic_context(v))
        )
        or (
            is_nuclear_context(v)
            and is_cytoplasmic_context(u)
        )
    ]
    if edges_to_remove:
        graph.remove_edges_from(edges_to_remove)
    return len(edges_to_remove)


def build_human_to_cl_mapping(ppi_dir: str) -> Dict[str, str]:
    """Map human-readable Tabula labels to CL identifiers from PPI filenames."""

    human_to_cl_id = {}
    ppi_path = Path(ppi_dir)
    edgelist_dir = ppi_path / "ppi_edgelists"

    if not edgelist_dir.exists():
        logger.warning("PPI edgelist directory not found: %s", edgelist_dir)
        return human_to_cl_id

    for filename in edgelist_dir.glob("*.txt"):
        match = re.search(r"\[cl[_:](\d+)\]", filename.stem, re.IGNORECASE)
        if match:
            cl_id_underscore = normalize_cl_node(f"CL_{match.group(1)}")
            human_name = re.sub(r"_?\[cl[_:]\d+\]", "", filename.stem, flags=re.IGNORECASE)
            human_name = human_name.replace("_", " ").strip()
            human_to_cl_id[human_name.lower()] = cl_id_underscore

    logger.info(
        "Built human name -> CL ID mapping for %d cell types from PPI filenames",
        len(human_to_cl_id),
    )
    return human_to_cl_id


def parse_cpdb_output(
    pvalues_file: str,
    pair_counts: PairCounts,
    pvalue_cutoff: float,
    lr_cutoff: int,
    *,
    pvalue_store: List[np.ndarray] | None = None,
    lr_store: List[int] | None = None,
    human_to_cl_mapping: Dict[str, str] | None = None,
) -> None:
    """Parse a single CellPhoneDB p-values matrix and update pair counts."""
    logger.info("Parsing %s", pvalues_file)
    df = pd.read_csv(pvalues_file, sep="\t")
    significant_pairs_in_run: set[Tuple[str, str]] = set()

    value_cols = [column for column in df.columns if "|" in column]
    for column in value_cols:
        source, target = (part.strip() for part in column.split("|", 1))
        if not source or not target:
            continue

        if human_to_cl_mapping:
            source = human_to_cl_mapping.get(source.lower(), source)
            target = human_to_cl_mapping.get(target.lower(), target)

        source = normalize_cl_node(source)
        target = normalize_cl_node(target)

        significant = df[column][df[column] < pvalue_cutoff].dropna()
        if not significant.empty:
            if pvalue_store is not None:
                pvalue_store.append(significant.to_numpy(dtype=float))
            if lr_store is not None:
                lr_store.append(significant.shape[0])
        if significant.shape[0] < lr_cutoff:
            continue

        pair = tuple(sorted((source, target)))
        significant_pairs_in_run.add(pair)

    for pair in significant_pairs_in_run:
        pair_counts[pair] = pair_counts.get(pair, 0) + 1


def count_majority(
    pair_counts: PairCounts,
    num_runs: int,
    threshold: float,
) -> nx.Graph:
    """Keep interactions reproduced in the requested fraction of result files."""
    logger.info(
        "Applying majority voting with threshold %.2f across %d runs.",
        threshold,
        num_runs,
    )
    required = max(1, math.ceil(threshold * num_runs))

    edges = [
        (a, b)
        for (a, b), count in pair_counts.items()
        if count >= required
    ]

    graph = nx.Graph()
    graph.add_edges_from(edges)
    logger.info("Resulting graph: %d nodes, %d edges.", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _plot_histogram(data: np.ndarray, title: str, xlabel: str, output_path: Path, bins: int = 50) -> None:
    if data.size == 0:
        logger.warning("Skipping histogram %s because data array is empty.", output_path.name)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color="steelblue", edgecolor="black", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved histogram to %s", output_path)


def construct_cci(
    cpdb_dir: str,
    output_file: str,
    pvalue_cutoff: float,
    lr_cutoff: int,
    threshold: float,
    ppi_dir: str | None = None,
) -> nx.Graph:
    """Construct a CCI network from all CellPhoneDB runs in the directory."""
    human_to_cl_mapping = build_human_to_cl_mapping(ppi_dir) if ppi_dir else None

    pattern = os.path.join(cpdb_dir, "**", "*pvalues*.txt")
    pvalue_files = sorted(glob.glob(pattern, recursive=True))
    if not pvalue_files:
        raise FileNotFoundError(
            f"No CellPhoneDB p-values files found under {cpdb_dir}. "
            "Ensure Step 5 completed successfully."
        )

    logger.info("Found %d p-values files.", len(pvalue_files))

    pair_counts: PairCounts = {}
    pvalue_store: List[np.ndarray] = []
    lr_store: List[int] = []
    for pfile in pvalue_files:
        parse_cpdb_output(
            pfile,
            pair_counts,
            pvalue_cutoff=pvalue_cutoff,
            lr_cutoff=lr_cutoff,
            pvalue_store=pvalue_store,
            lr_store=lr_store,
            human_to_cl_mapping=human_to_cl_mapping,
        )

    if len(pvalue_files) == 1:
        logger.info("One p-values file found; recurrence filtering reduces to presence in that file.")
    graph = count_majority(pair_counts, num_runs=len(pvalue_files), threshold=threshold)
    graph = normalize_graph_nodes(graph)
    removed_compartment = remove_invalid_compartment_edges(graph)
    if removed_compartment:
        logger.info(
            "Removed %d nuc--nuc or nuc--cyto CCI edges.",
            removed_compartment,
        )
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            graph.remove_nodes_from(isolated_nodes)
            logger.info(
                "Removed %d isolated CCI nodes after compartment-edge filtering.",
                len(isolated_nodes),
            )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing CCI edgelist to %s", output_file)
    nx.write_edgelist(graph, output_file, data=False, delimiter="\t")

    diag_dir = Path(cpdb_dir) / "diagnostics"
    if pvalue_store:
        pvals = np.concatenate(pvalue_store)
        neg_log_p = -np.log10(np.clip(pvals, 1e-16, None))
        _plot_histogram(
            neg_log_p,
            title="CellPhoneDB -log10(p) distribution",
            xlabel="-log10(p)",
            output_path=diag_dir / "pvalues_hist.png",
        )
    if lr_store:
        lr_counts = np.array(lr_store)
        _plot_histogram(
            lr_counts,
            title="CellPhoneDB LR entries per cell pair",
            xlabel="Number of LR entries",
            output_path=diag_dir / "lr_counts_hist.png",
        )
    return graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct cell-cell interaction networks from CellPhoneDB outputs."
    )
    parser.add_argument(
        "--dataset",
        choices=["tabula", "hbca", "als", "merged", "all"],
        default="all",
        help="Dataset to process: tabula, hbca, als (all ALS cell types together), merged (HBCA+Tabula), or all (default: all).",
    )
    parser.add_argument(
        "--pvalue",
        type=float,
        default=CELLPHONEDB_PVALUE,
        help=f"P-value threshold for significant interactions (default: {CELLPHONEDB_PVALUE}).",
    )
    parser.add_argument(
        "--min-lr",
        type=int,
        default=None,
        help="Minimum number of significant ligand-receptor pairs per run (default: dataset-specific from config).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CELLPHONEDB_THRESHOLD,
        help=f"Majority threshold for retaining edges (default: {CELLPHONEDB_THRESHOLD}).",
    )
    parser.add_argument(
        "--tabula-output",
        default=TABULA_CCI_EDGELIST,
        help="Output edgelist path for Tabula Sapiens.",
    )
    parser.add_argument(
        "--hbca-output",
        default=HBCA_CCI_EDGELIST,
        help="Output edgelist path for HBCA.",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    parsed = build_parser().parse_args(args)

    def resolve_min_lr(dataset: str) -> int:
        if parsed.min_lr is not None:
            return parsed.min_lr
        return CELLPHONEDB_MIN_LR_BY_DATASET.get(dataset, CELLPHONEDB_MIN_LR)

    datasets = {
        "tabula": (
            TABULA_CELLPHONEDB_OUTPUT,
            parsed.tabula_output,
            os.path.join(TABULA_INTERMEDIATE, "ppi"),
        ),
        "hbca": (HBCA_CELLPHONEDB_OUTPUT, parsed.hbca_output, None),
        "als": (
            os.path.join(ALS_BULK_DIR, "cellphonedb_results"),
            os.path.join(ALS_BULK_DIR, "cci_edgelist.txt"),
            None,
        ),
        "merged": (MERGED_CELLPHONEDB_OUTPUT, MERGED_CCI_EDGELIST, None),
    }
    selected = datasets if parsed.dataset == "all" else (parsed.dataset,)
    results = {}
    for dataset in selected:
        cpdb_dir, output_file, ppi_dir = datasets[dataset]
        logger.info("Constructing %s CCI network", dataset)
        results[dataset] = construct_cci(
            cpdb_dir=cpdb_dir,
            output_file=output_file,
            pvalue_cutoff=parsed.pvalue,
            lr_cutoff=resolve_min_lr(dataset),
            threshold=parsed.threshold,
            ppi_dir=ppi_dir,
        )

    for name, graph in results.items():
        logger.info(
            "%s: %d nodes, %d edges (density %.4f)",
            name.upper(),
            graph.number_of_nodes(),
            graph.number_of_edges(),
            nx.density(graph),
        )


if __name__ == "__main__":
    main()
