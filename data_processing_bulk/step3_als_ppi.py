#!/usr/bin/env python
"""Step 3 (ALS): induce one PPI network for each ALS bulk context."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from .config import (
    ALS_ASTRO_INTERMEDIATE,
    ALS_INTERMEDIATE,
    ALS_MN_INTERMEDIATE,
    GLOBAL_PPI,
)
from .step3_construct_ppi import analyze_similarity


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DATASETS = {
    "motor_neuron": {
        "directory": ALS_MN_INTERMEDIATE,
        "label": "ALS motor neuron",
    },
    "astrocyte": {
        "directory": ALS_ASTRO_INTERMEDIATE,
        "label": "ALS astrocyte",
    },
}


def _resolve_per_cell_file(directory: Path) -> Path:
    for filename in (
        "reliable_genes_specific_per_celltype.json",
        "reliable_genes_per_celltype.json",
    ):
        path = directory / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"No per-context reliable-gene file found in {directory}")


def load_global_ppi() -> nx.Graph:
    graph = nx.read_edgelist(GLOBAL_PPI, nodetype=str)
    logger.info(
        "Loaded global PPI: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def _safe_context_name(context: str) -> str:
    return (
        context.replace(":", "_")
        .replace("/", "_")
        .replace("|", "_")
        .replace(" ", "_")
    )


def construct_ppi_per_celltype(
    reliable_genes_per_celltype: dict[str, list[str]],
    global_ppi: nx.Graph,
    output_dir: str,
    dataset_label: str | None = None,
) -> tuple[list[dict], list[dict], dict[str, nx.Graph]]:
    """Induce the global PPI on each context's genes and retain its LCC."""
    ppi_dir = Path(output_dir) / "ppi"
    edgelist_dir = ppi_dir / "ppi_edgelists"
    if edgelist_dir.exists():
        shutil.rmtree(edgelist_dir)
    edgelist_dir.mkdir(parents=True)

    network_stats = []
    celltype_lines = []
    similarity_rows = []
    similarity_graphs = {}

    for index, (context, genes) in enumerate(sorted(reliable_genes_per_celltype.items())):
        genes_in_ppi = [gene for gene in genes if gene in global_ppi]
        if not genes_in_ppi:
            logger.warning("%s: no reliable genes occur in the global PPI", context)
            continue

        subgraph = global_ppi.subgraph(genes_in_ppi).copy()
        lcc_nodes = max(nx.connected_components(subgraph), key=len)
        lcc = subgraph.subgraph(lcc_nodes).copy()
        nx.write_edgelist(
            lcc,
            edgelist_dir / f"{_safe_context_name(context)}.txt",
            delimiter="\t",
            data=False,
        )

        network_stats.append(
            {
                "cell_type_id": context,
                "num_reliable_genes": len(genes),
                "num_genes_in_ppi": len(genes_in_ppi),
                "num_nodes": lcc.number_of_nodes(),
                "num_edges": lcc.number_of_edges(),
            }
        )
        celltype_lines.append(f"{index}\t{context}\t{','.join(sorted(lcc.nodes()))}\n")

        graph_key = context if dataset_label is None else f"{dataset_label}::{context}"
        similarity_graphs[graph_key] = lcc
        similarity_rows.append(
            {
                "cell_type": graph_key,
                "original_cell_type": context,
                "dataset": dataset_label or "",
                "status": "kept",
                "lcc_size": lcc.number_of_nodes(),
                "lcc_edges": lcc.number_of_edges(),
            }
        )
        logger.info("%s: %d nodes, %d edges", context, lcc.number_of_nodes(), lcc.number_of_edges())

    if network_stats:
        pd.DataFrame(network_stats).to_csv(ppi_dir / "network_stats.csv", index=False)
    with (ppi_dir / "ppi_celltype_list.csv").open("w") as handle:
        handle.writelines(celltype_lines)
    return network_stats, similarity_rows, similarity_graphs


def generate_ppi_qc(output_dir: str, network_stats: list[dict], global_ppi: nx.Graph) -> None:
    """Write the compact QC summary and plots used by the ALS pipeline."""
    if not network_stats:
        logger.warning("No retained networks; skipping PPI QC")
        return

    qc_dir = Path(output_dir) / "ppi_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    stats = pd.DataFrame(network_stats)
    summary = {
        "num_networks": len(stats),
        "median_reliable_genes": float(stats["num_reliable_genes"].median()),
        "median_genes_in_ppi": float(stats["num_genes_in_ppi"].median()),
        "median_lcc_nodes": float(stats["num_nodes"].median()),
        "median_lcc_edges": float(stats["num_edges"].median()),
        "global_ppi_nodes": global_ppi.number_of_nodes(),
        "global_ppi_edges": global_ppi.number_of_edges(),
    }
    with (qc_dir / "summary_statistics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(stats["num_nodes"], bins=25, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(
        stats["num_nodes"].median(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Median = {stats["num_nodes"].median():.0f}',
    )
    ax.set(xlabel="LCC size (nodes)", ylabel="Frequency", title="Distribution of PPI Network Sizes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(qc_dir / "lcc_size_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(stats["num_reliable_genes"], stats["num_genes_in_ppi"], alpha=0.6, s=100)
    limit = max(stats["num_reliable_genes"].max(), stats["num_genes_in_ppi"].max())
    ax.plot([0, limit], [0, limit], "r--", alpha=0.5, label="100% coverage")
    ax.set(
        xlabel="Number of Reliable Genes",
        ylabel="Genes in Global PPI",
        title="PPI Coverage of Reliable Genes",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(qc_dir / "ppi_coverage.png", dpi=150)
    plt.close(fig)

    sorted_stats = stats.sort_values("num_nodes", ascending=False)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(range(len(sorted_stats)), sorted_stats["num_nodes"], color="steelblue", alpha=0.7)
    ax.set(
        xlabel="Cell Type (sorted by network size)",
        ylabel="Number of Nodes",
        title="PPI Network Sizes by Cell Type",
    )
    fig.tight_layout()
    fig.savefig(qc_dir / "network_sizes_by_celltype.png", dpi=150)
    plt.close(fig)


def _run_similarity(rows: list[dict], graphs: dict[str, nx.Graph], output_dir: Path, label: str) -> None:
    if not graphs:
        logger.warning("No %s PPIs available for similarity analysis", label)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    analyze_similarity(graphs, pd.DataFrame(rows), str(output_dir))


def process_dataset(dataset: str, global_ppi: nx.Graph) -> dict:
    config = DATASETS[dataset]
    directory = Path(config["directory"])
    with _resolve_per_cell_file(directory).open() as handle:
        reliable_genes = json.load(handle)

    stats, rows, graphs = construct_ppi_per_celltype(
        reliable_genes,
        global_ppi,
        str(directory),
        dataset_label=dataset,
    )
    generate_ppi_qc(str(directory), stats, global_ppi)
    _run_similarity(rows, graphs, directory / "ppi", config["label"])
    return {"summary": rows, "graphs": graphs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-type",
        choices=["motor_neuron", "astrocyte", "both"],
        default="both",
    )
    args = parser.parse_args()

    global_ppi = load_global_ppi()
    selected = DATASETS if args.cell_type == "both" else [args.cell_type]
    results = [process_dataset(dataset, global_ppi) for dataset in selected]

    combined_rows = [row for result in results for row in result["summary"]]
    combined_graphs = {
        name: graph
        for result in results
        for name, graph in result["graphs"].items()
    }
    _run_similarity(
        combined_rows,
        combined_graphs,
        Path(ALS_INTERMEDIATE) / "ppi_combined",
        "combined ALS",
    )


if __name__ == "__main__":
    main()
