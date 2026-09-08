#!/usr/bin/env python
"""Compute bulk-network statistics and plots."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import obonet
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from sklearn.manifold import MDS



REPO_ROOT = Path(__file__).resolve().parents[2]
with (REPO_ROOT / "configs" / "paths.yaml").open(encoding="utf-8") as handle:
    PATHS = yaml.safe_load(handle)

# Inputs produced by the bulk-data pipeline.
OUTPUT_ROOT = Path(PATHS["output_root"]).expanduser()
PROCESSED_ROOT = OUTPUT_ROOT / "data_processing_bulk"
NETWORKS_BULK = Path(PATHS["networks_bulk"]).expanduser()
TABULA_RAW_H5AD = Path(PATHS["tabula_h5ad"]).expanduser()
HBCA_GENE_METADATA = Path(PATHS["hbca_gene_metadata"]).expanduser()
ALS_GENE_METADATA = Path(PATHS["als_gene_metadata"]).expanduser()

# Biological annotations used for cell-class and tissue summaries. The
# cell-class table is curated metadata for the 207 contexts.
CELL_CLASS_MAPPING = Path(PATHS["celltype_class_mapping"]).expanduser()
BTO_OBO = Path(PATHS["tissue_ontology_obo"]).expanduser()

# Tables and plots from this analysis live together.
ANALYSIS_DIR = OUTPUT_ROOT / "analysis" / "data_bulk_statistics"

# Reliable, selected and PPI-LCC counts produced by the bulk-data pipeline.
DATASETS = [
    {
        "source": "Tabula Sapiens",
        "detail": "Tabula Sapiens",
        "reliable_csv": PROCESSED_ROOT
        / "tabula_pseudobulk/reliable_gene_summary.csv",
        "selected_csv": PROCESSED_ROOT
        / "tabula_pseudobulk/specific_gene_summary.csv",
    },
    {
        "source": "HBCA",
        "detail": "HBCA",
        "reliable_csv": PROCESSED_ROOT
        / "hbca_pseudobulk/reliable_gene_summary.csv",
        "selected_csv": PROCESSED_ROOT
        / "hbca_pseudobulk/specific_gene_summary.csv",
    },
    {
        "source": "ALS",
        "detail": "motor neurons",
        "reliable_json": PROCESSED_ROOT
        / "als_bulk/motor_neurons/reliable_genes_per_celltype.json",
        "selected_csv": PROCESSED_ROOT
        / "als_bulk/motor_neurons/specific_gene_summary.csv",
    },
    {
        "source": "ALS",
        "detail": "astrocytes",
        "reliable_json": PROCESSED_ROOT
        / "als_bulk/astrocytes/reliable_genes_per_celltype.json",
        "selected_csv": PROCESSED_ROOT
        / "als_bulk/astrocytes/specific_gene_summary.csv",
    },
]


CNS_TISSUES = {
    "midbrain",
    "cerebellum",
    "cerebral cortex",
    "basal ganglion",
    "hippocampus",
    "hypothalamus",
    "myelencephalon",
    "pons",
    "spinal cord",
    "thalamus",
}
IMMUNE_TISSUES = {"blood", "bone marrow", "lymph node", "spleen", "thymus"}
INTESTINE_TISSUES = {"small intestine", "large intestine"}
AIRWAY_TISSUES = {"lung", "trachea"}
NAMED_SINGLE_TISSUES = {"eye", "pancreas", "salivary gland", "muscle"}


def require_inputs(paths):
    """Fail early with a readable list of missing inputs."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing input files:\n{formatted}")


def reliable_counts(spec):
    """Read the reliable-gene counts for one expression dataset."""
    if "reliable_csv" in spec:
        return pd.read_csv(spec["reliable_csv"])[["cell_type", "genes_kept"]]

    with spec["reliable_json"].open() as handle:
        genes_by_context = json.load(handle)
    return pd.DataFrame(
        {
            "cell_type": list(genes_by_context),
            "genes_kept": [len(genes) for genes in genes_by_context.values()],
        }
    )


def h5ad_feature_count(path):
    """Read the number of gene features without loading the expression matrix."""
    with h5py.File(path, "r") as handle:
        var = handle["var"]
        index_name = var.attrs.get("_index", "_index")
        if isinstance(index_name, bytes):
            index_name = index_name.decode()
        index = var[index_name]
        # AnnData can store a categorical index as a group containing codes.
        return len(index["codes"]) if isinstance(index, h5py.Group) else len(index)


def measured_gene_counts():
    """Count the raw measured genes used as gene-retention denominators."""
    tabula = h5ad_feature_count(TABULA_RAW_H5AD)
    hbca = pd.read_csv(HBCA_GENE_METADATA, usecols=["ensembl_id"])[
        "ensembl_id"
    ].nunique()
    als = pd.read_csv(ALS_GENE_METADATA, usecols=["gene_identifier"])[
        "gene_identifier"
    ].nunique()
    counts = {"Tabula Sapiens": tabula, "HBCA": hbca, "ALS": als}
    sources = {
        "Tabula Sapiens": TABULA_RAW_H5AD,
        "HBCA": HBCA_GENE_METADATA,
        "ALS": ALS_GENE_METADATA,
    }
    for source, count in counts.items():
        print(f"[INPUT] {source}: {count:,} measured genes in {sources[source]}")
    return counts


def prepare_gene_retention():
    """Summarize reliable genes, selected genes and PPI LCC sizes."""
    measured_by_source = measured_gene_counts()
    rows = []
    for spec in DATASETS:
        measured_genes = measured_by_source[spec["source"]]
        reliable = reliable_counts(spec)
        selected = pd.read_csv(spec["selected_csv"])[
            ["cell_type", "genes_selected", "lcc_nodes"]
        ]
        data = reliable.merge(selected, on="cell_type", how="outer")
        data["source"] = spec["source"]
        data["dataset_detail"] = spec["detail"]
        data["total_measured_genes"] = measured_genes
        data["reliable_gene_fraction"] = (
            data["genes_kept"] / data["total_measured_genes"]
        )
        data["selected_gene_fraction"] = (
            data["genes_selected"] / data["total_measured_genes"]
        )
        data["selected_among_reliable_fraction"] = (
            data["genes_selected"] / data["genes_kept"]
        )
        data["lcc_among_selected_fraction"] = (
            data["lcc_nodes"] / data["genes_selected"]
        )
        rows.append(data)

    return pd.concat(rows, ignore_index=True)


def read_edges(path):
    """Read a simple undirected edge list, removing loops and duplicates."""
    nodes = set()
    edges = set()
    with path.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            source, target = parts[0].strip().upper(), parts[1].strip().upper()
            if not source or not target or source == target:
                continue
            nodes.update((source, target))
            edges.add(tuple(sorted((source, target))))
    return nodes, edges


def is_true(value):
    """Interpret booleans read from CSV without treating 'False' as true."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def source_label(row):
    if is_true(row["has_condition"]):
        return "ALS"
    if "HBCA__" in str(row["source_files"]):
        return "HBCA"
    return "Tabula Sapiens"


def database_label(row):
    if is_true(row["has_condition"]):
        return "ALS"
    sources = str(row["source_files"])
    has_tabula = "TABULA__" in sources
    has_hbca = "HBCA__" in sources
    if has_tabula and has_hbca:
        return "Tabula Sapiens + HBCA"
    if has_hbca:
        return "HBCA"
    if has_tabula:
        return "Tabula Sapiens"
    return "Unknown"


def prepare_ppi_tables(metadata):
    """Summarize PPI sizes and pairwise overlap."""
    global_nodes, global_edges = read_edges(NETWORKS_BULK / "global_ppi_edgelist.txt")

    metadata_by_context = metadata.set_index("edgelist")
    contexts = []
    metric_rows = []
    node_to_index = {}
    edge_to_index = {}
    node_rows, node_columns = [], []
    edge_rows, edge_columns = [], []

    ppi_paths = sorted((NETWORKS_BULK / "ppi_edgelists").glob("*.txt"))
    for path in ppi_paths:
        context = path.stem
        nodes, edges = read_edges(path)
        if not nodes:
            continue
        if context not in metadata_by_context.index:
            raise KeyError(f"No metadata found for PPI context: {context}")

        row_index = len(contexts)
        contexts.append(context)
        metadata_row = metadata_by_context.loc[context]
        metric_rows.append(
            {
                "context": context,
                "source": source_label(metadata_row),
                "n_proteins": len(nodes),
                "n_edges": len(edges),
                "global_node_fraction": len(nodes) / len(global_nodes),
                "global_edge_fraction": len(edges) / len(global_edges),
                "mean_degree": 2 * len(edges) / len(nodes),
            }
        )

        for node in nodes:
            node_id = node_to_index.setdefault(node, len(node_to_index))
            node_rows.append(row_index)
            node_columns.append(node_id)
        for edge in edges:
            edge_id = edge_to_index.setdefault(edge, len(edge_to_index))
            edge_rows.append(row_index)
            edge_columns.append(edge_id)

    metrics = pd.DataFrame(metric_rows)
    node_counts = metrics["n_proteins"].to_numpy(dtype=float)
    edge_counts = metrics["n_edges"].to_numpy(dtype=float)
    node_membership = csr_matrix(
        (np.ones(len(node_rows), dtype=np.int32), (node_rows, node_columns)),
        shape=(len(contexts), len(node_to_index)),
    )
    edge_membership = csr_matrix(
        (np.ones(len(edge_rows), dtype=np.int32), (edge_rows, edge_columns)),
        shape=(len(contexts), len(edge_to_index)),
    )

    node_intersections = (node_membership @ node_membership.T).toarray().astype(float)
    edge_intersections = (edge_membership @ edge_membership.T).toarray().astype(float)
    node_unions = node_counts[:, None] + node_counts[None, :] - node_intersections
    edge_unions = edge_counts[:, None] + edge_counts[None, :] - edge_intersections
    node_jaccard = np.divide(
        node_intersections,
        node_unions,
        out=np.zeros_like(node_intersections),
        where=node_unions > 0,
    )
    edge_jaccard = np.divide(
        edge_intersections,
        edge_unions,
        out=np.zeros_like(edge_intersections),
        where=edge_unions > 0,
    )
    np.fill_diagonal(edge_jaccard, 1.0)

    upper_i, upper_j = np.triu_indices(len(contexts), k=1)
    pairwise = pd.DataFrame(
        {
            "context1": [contexts[index] for index in upper_i],
            "context2": [contexts[index] for index in upper_j],
            "node_jaccard": node_jaccard[upper_i, upper_j],
            "edge_jaccard": edge_jaccard[upper_i, upper_j],
        }
    )
    return metrics, pairwise, contexts, edge_counts.astype(int), edge_jaccard


def read_cell_to_tissues():
    """Read direct context-to-tissue links from the final metagraph."""
    cell_to_tissues = {}
    with (NETWORKS_BULK / "mg_edgelist.txt").open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            source, target = parts[:2]
            if source.startswith("CL_") and target.startswith("BTO_"):
                cell, tissue = source, target
            elif target.startswith("CL_") and source.startswith("BTO_"):
                cell, tissue = target, source
            else:
                continue
            cell_to_tissues.setdefault(cell, set()).add(tissue)
    return cell_to_tissues


def prepare_tissue_tables(metadata):
    """Summarize the context-to-tissue assignments used by the model."""
    bto_graph = obonet.read_obo(str(BTO_OBO))
    bto_names = {
        str(node): str(attributes["name"])
        for node, attributes in bto_graph.nodes(data=True)
        if attributes.get("name")
    }
    cell_to_tissues = read_cell_to_tissues()
    compact_rows, long_rows = [], []

    for row in metadata.itertuples(index=False):
        tissues = sorted(cell_to_tissues.get(str(row.edgelist), set()))
        names = [
            bto_names.get(tissue.replace("BTO_", "BTO:", 1), tissue)
            for tissue in tissues
        ]
        compact_rows.append(
            {
                "edgelist": row.edgelist,
                "n_tissues": len(tissues),
                "tissue_names": ";".join(names),
            }
        )
        long_rows.extend(
            {"edgelist": row.edgelist, "tissue_name": name} for name in names
        )

    compact = pd.DataFrame(compact_rows).sort_values(
        ["n_tissues", "edgelist"], ascending=[False, True]
    )
    tissue_counts = (
        pd.DataFrame(long_rows)
        .groupby("tissue_name", as_index=False)
        .agg(
            n_contexts=("edgelist", "nunique"),
            n_context_tissue_links=("edgelist", "size"),
        )
        .sort_values(["n_contexts", "tissue_name"], ascending=[False, True])
    )
    tissue_counts["fraction_context_tissue_links"] = (
        tissue_counts["n_context_tissue_links"]
        / tissue_counts["n_context_tissue_links"].sum()
    )
    return compact.reset_index(drop=True), tissue_counts.reset_index(drop=True)


def tissue_plot_group(value):
    tissues = {name for name in str(value).split(";") if name and name != "nan"}
    if not tissues:
        return "unknown"
    if tissues == {"spinal cord"}:
        return "spinal cord"
    if tissues.issubset(CNS_TISSUES):
        return "CNS regions"
    if tissues.issubset(IMMUNE_TISSUES):
        return "immune/hematopoietic"
    if tissues.issubset(INTESTINE_TISSUES):
        return "intestine"
    if tissues.issubset(AIRWAY_TISSUES):
        return "airway"
    if len(tissues) == 1:
        tissue = next(iter(tissues))
        return tissue if tissue in NAMED_SINGLE_TISSUES else "other single-organ"
    return "broad multi-organ"


def prepare_mds_table(contexts, edge_counts, edge_jaccard, metadata, tissue_mapping):
    """Run deterministic metric MDS on edge-Jaccard distances."""
    distance = 1.0 - edge_jaccard
    np.fill_diagonal(distance, 0.0)
    coordinates = MDS(
        n_components=2,
        metric=True,
        dissimilarity="precomputed",
        random_state=7,
        n_init=4,
        max_iter=300,
        eps=1e-6,
    ).fit_transform(distance)

    result = pd.DataFrame(
        {
            "edgelist": contexts,
            "n_ppi_edges": edge_counts,
            "mds_1": coordinates[:, 0],
            "mds_2": coordinates[:, 1],
        }
    )
    classes = pd.read_csv(CELL_CLASS_MAPPING)[["edgelist", "cell_type_class"]]
    result = result.merge(classes, on="edgelist", how="left", validate="one_to_one")
    result = result.merge(
        metadata[["edgelist", "primary_dataset", "has_condition", "datasets", "source_files"]],
        on="edgelist",
        how="left",
        validate="one_to_one",
    )
    result["database"] = result.apply(database_label, axis=1)
    result = result.merge(
        tissue_mapping, on="edgelist", how="left", validate="one_to_one"
    )
    result["n_tissues"] = result["n_tissues"].fillna(0).astype(int)
    result["tissue_group"] = np.where(
        result["n_tissues"].eq(1),
        result["tissue_names"].fillna("unknown"),
        np.where(result["n_tissues"].eq(0), "unknown", "multi-tissue"),
    )
    result["tissue_plot_group"] = result["tissue_names"].apply(tissue_plot_group)

    if result["cell_type_class"].isna().any():
        missing = result.loc[result["cell_type_class"].isna(), "edgelist"].tolist()
        raise ValueError(f"Missing curated cell-class annotations for: {missing}")
    return result


def prepare_metagraph_edge_counts():
    """Count cell-cell and cell-tissue metagraph edges."""
    cell_cell_edges = set()
    context_tissue_edges = set()
    with (NETWORKS_BULK / "mg_edgelist.txt").open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            source, target = parts[:2]
            source_is_tissue = source.startswith("BTO_")
            target_is_tissue = target.startswith("BTO_")
            edge = tuple(sorted((source, target)))
            if not source_is_tissue and not target_is_tissue:
                cell_cell_edges.add(edge)
            elif source_is_tissue != target_is_tissue:
                context_tissue_edges.add(edge)
    return pd.DataFrame(
        [
            {
                "edge_type": "Cell-cell communication",
                "n_edges": len(cell_cell_edges),
            },
            {"edge_type": "Context-to-tissue", "n_edges": len(context_tissue_edges)},
        ]
    )


def main():
    inputs = [
        NETWORKS_BULK / "celltype_metadata.csv",
        NETWORKS_BULK / "global_ppi_edgelist.txt",
        NETWORKS_BULK / "mg_edgelist.txt",
        TABULA_RAW_H5AD,
        HBCA_GENE_METADATA,
        ALS_GENE_METADATA,
        CELL_CLASS_MAPPING,
        BTO_OBO,
    ]
    for spec in DATASETS:
        inputs.append(spec.get("reliable_csv", spec.get("reliable_json")))
        inputs.append(spec["selected_csv"])
    require_inputs(inputs)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(NETWORKS_BULK / "celltype_metadata.csv")

    gene_retention = prepare_gene_retention()
    ppi_metrics, pairwise, contexts, edge_counts, edge_jaccard = prepare_ppi_tables(
        metadata
    )
    context_tissues, tissue_counts = prepare_tissue_tables(metadata)
    mds = prepare_mds_table(
        contexts, edge_counts, edge_jaccard, metadata, context_tissues
    )
    metagraph_edges = prepare_metagraph_edge_counts()

    tables = {
        "gene_retention.csv": gene_retention,
        "ppi_metrics.csv": ppi_metrics,
        "pairwise_jaccard.csv": pairwise,
        "edge_jaccard_mds.csv": mds,
        "metagraph_edges.csv": metagraph_edges,
        "context_tissue_counts.csv": context_tissues,
        "tissue_context_counts.csv": tissue_counts,
    }
    for filename, table in tables.items():
        table.to_csv(ANALYSIS_DIR / filename, index=False)
        print(f"[OK] {filename}: {len(table):,} rows")

    from exploration.plotting.data_bulk_statistics_plots import plot_all

    plot_all(source=ANALYSIS_DIR, output=ANALYSIS_DIR)


if __name__ == "__main__":
    main()
