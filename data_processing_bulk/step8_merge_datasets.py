#!/usr/bin/env python
"""Step 8: merge the HBCA+Tabula and ALS networks used for model training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .config import (
    ALS_ASTRO_INTERMEDIATE,
    ALS_BULK_DIR,
    ALS_MN_INTERMEDIATE,
    GLOBAL_PPI,
    HBCA_INTERMEDIATE,
    MERGED_INTERMEDIATE,
    PINNACLE_BASE,
    TABULA_INTERMEDIATE,
)
from .ensembl_to_hgnc_converter import fetch_symbols_from_gprofiler

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_SPECIFIC_SUFFIX = {
    "reliable_genes.json": "reliable_genes_specific.json",
    "reliable_genes_per_celltype.json": "reliable_genes_specific_per_celltype.json",
}
MERGED_OUTPUT = os.path.join(PINNACLE_BASE, "networks_bulk")

ALS_MN_REL = os.path.relpath(ALS_MN_INTERMEDIATE, ALS_BULK_DIR)
ALS_ASTRO_REL = os.path.relpath(ALS_ASTRO_INTERMEDIATE, ALS_BULK_DIR)


def prefer_specific_file(path: str) -> str:
    """If a specificity-filtered variant exists beside the requested file, use it."""
    directory, name = os.path.split(path)
    candidate_name = _SPECIFIC_SUFFIX.get(name)
    if candidate_name:
        candidate = os.path.join(directory, candidate_name)
        if os.path.exists(candidate):
            return candidate
    return path


@lru_cache(None)
def get_global_ppi_node_count(path: str = GLOBAL_PPI) -> int:
    if not os.path.exists(path):
        logger.warning("Global PPI not found at %s when counting nodes.", path)
        return 0
    G = read_edgelist_any(path)
    return G.number_of_nodes()


def read_edgelist_any(path: str | Path) -> nx.Graph:
    """Read an edgelist handling either tab or whitespace delimiters."""
    path_str = str(path)
    try:
        graph = nx.read_edgelist(path_str, delimiter="\t")
    except Exception as exc:  # pragma: no cover - format issues fallback
        logger.debug("Tab-delimited read failed for %s (%s); retrying with default delimiter.", path_str, exc)
        graph = nx.read_edgelist(path_str)
        return graph

    if graph.number_of_edges() == 0:
        alt_graph = nx.read_edgelist(path_str)
        if alt_graph.number_of_edges() > 0:
            logger.debug("Edgelist %s contained no tab-delimited edges; using whitespace-delimited parse.", path_str)
            graph = alt_graph
    return graph


def write_count_edge_dict(global_ppi_path: Path, ppi_root: Path, output_file: Path) -> None:
    """Count how many context-specific networks contain each global PPI edge."""
    global_graph = read_edgelist_any(global_ppi_path)
    edge_counts = {edge: 0 for edge in global_graph.edges()}

    for ppi_file in (ppi_root / "ppi_edgelists").glob("*.txt"):
        for source, target in read_edgelist_any(ppi_file).edges():
            edge = (source, target)
            if edge not in edge_counts:
                edge = (target, source)
            if edge not in edge_counts:
                raise ValueError(f"Context PPI edge is absent from the global PPI: {source} {target}")
            edge_counts[edge] += 1

    with output_file.open("wb") as handle:
        pickle.dump(edge_counts, handle)


def summarize_ppi_dir(ppi_dir: str) -> Dict[str, object]:
    list_path = Path(ppi_dir) / "ppi_celltype_list.csv"
    if list_path.exists():
        df = pd.read_csv(
            list_path,
            sep="\t",
            header=None,
            names=["index", "cell_type", "genes"],
        )
        df = df.dropna(subset=["genes"])
        genes_series = df["genes"].astype(str)

        node_counts = [
            len([gene for gene in genes.split(",") if gene])
            for genes in genes_series
        ]
        unique_proteins = {
            gene
            for genes in genes_series
            for gene in genes.split(",")
            if gene
        }

        arr = np.array(node_counts, dtype=float)
        mean_nodes = float(arr.mean()) if len(arr) else 0.0
        std_nodes = float(arr.std(ddof=0)) if len(arr) else 0.0

        return {
            "num_networks": len(node_counts),
            "node_counts": node_counts,
            "mean_nodes": mean_nodes,
            "std_nodes": std_nodes,
            "unique_proteins": unique_proteins,
        }

    edgelist_dir = Path(ppi_dir) / "ppi_edgelists"
    files = [
        f
        for f in edgelist_dir.glob("*.txt")
        if f.stem.lower() not in {"inventory", "readme", "metadata"}
    ]
    if not files:
        raise FileNotFoundError(f"No PPI edgelists found in {edgelist_dir}")

    node_counts: List[int] = []
    unique_proteins: set = set()
    for f in files:
        G = read_edgelist_any(f)
        node_counts.append(G.number_of_nodes())
        unique_proteins.update(G.nodes())

    arr = np.array(node_counts, dtype=float)
    mean_nodes = float(arr.mean()) if len(arr) else 0.0
    std_nodes = float(arr.std(ddof=0)) if len(arr) else 0.0

    return {
        "num_networks": len(files),
        "node_counts": node_counts,
        "mean_nodes": mean_nodes,
        "std_nodes": std_nodes,
        "unique_proteins": unique_proteins,
    }


def normalize_cl_label(label: str) -> str:
    """Normalize Cell Ontology identifiers and remove dataset prefixes."""
    if not isinstance(label, str):
        return label
    value = label.strip()
    if not value:
        return value

    for prefix in ("HBCA__", "TABULA__"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            if value.upper().startswith("CL:"):
                return "CL_" + value[3:]
            return value

    if value.upper().startswith("CL:"):
        return "CL_" + value[3:]
    return value


def normalize_bto_label(label: str) -> str:
    """Standardize BTO identifiers to use underscores instead of colons."""
    if not isinstance(label, str):
        return label
    value = label.strip()
    if not value:
        return value
    if value.upper().startswith("BTO:"):
        return "BTO_" + value.split(":", 1)[1]
    if value.upper().startswith("BTO_"):
        return "BTO_" + value.split("_", 1)[1]
    return value


def normalize_node_label(label: str) -> str:
    """Normalize a generic metagraph node (cell type or BTO)."""
    label = normalize_bto_label(label)
    label = normalize_cl_label(label)
    return label


def is_nuclear_compartment_node(node: str) -> bool:
    """Return True for ALS-style cell nodes with a '_nuc_' compartment token."""
    if not isinstance(node, str):
        return False
    label = normalize_cl_label(node).lower()
    return bool(label.startswith("cl_") and re.search(r"(^|_)nuc($|_)", label))


def normalize_graph_labels(graph: nx.Graph) -> nx.Graph:
    """Return a new graph with normalized node labels (CL/BTO colon -> underscore)."""
    normalized = nx.Graph()
    for u, v in graph.edges():
        normalized.add_edge(normalize_node_label(u), normalize_node_label(v))

    for node in graph.nodes():
        if graph.degree(node) == 0:
            normalized.add_node(normalize_node_label(node))
    return normalized


def summarize_metagraph(mg_graph: nx.Graph) -> Dict[str, int]:
    mg_simple = nx.Graph(mg_graph)

    def is_tissue(node: str) -> bool:
        return isinstance(node, str) and node.startswith("BTO")

    tissues = {n for n in mg_simple.nodes if is_tissue(n)}
    tissues_connected = {t for t in tissues if any(not is_tissue(nb) for nb in mg_simple.neighbors(t))}

    cell_cell_edges = sum(1 for u, v in mg_simple.edges if not is_tissue(u) and not is_tissue(v))
    cell_tissue_edges = sum(1 for u, v in mg_simple.edges if is_tissue(u) ^ is_tissue(v))
    tissue_tissue_edges = sum(1 for u, v in mg_simple.edges if is_tissue(u) and is_tissue(v))

    return {
        "tissues_total": len(tissues),
        "tissues_connected": len(tissues_connected),
        "cell_cell_edges": cell_cell_edges,
        "cell_tissue_edges": cell_tissue_edges,
        "tissue_tissue_edges": tissue_tissue_edges,
    }


def log_final_summary(ppi_summary: Dict[str, object], mg_summary: Dict[str, int], gene_count: int, global_ppi_path: str) -> None:
    global_total = get_global_ppi_node_count(global_ppi_path)
    unique_proteins = len(ppi_summary["unique_proteins"])

    logger.info("")
    logger.info("Final dataset")
    logger.info(
        "We have %d cell type-specific protein interaction networks, which have, on average, %.0f +/- %.0f proteins per network.",
        ppi_summary["num_networks"],
        ppi_summary["mean_nodes"],
        ppi_summary["std_nodes"],
    )
    logger.info(
        "The number of unique proteins across all cell type-specific protein interaction networks is %d of the %d proteins in the global reference protein interaction network.",
        unique_proteins,
        global_total,
    )
    logger.info(
        "In the metagraph, we have %d tissues (nodes), and %d are directly connected to cell types.",
        mg_summary["tissues_total"],
        mg_summary["tissues_connected"],
    )
    logger.info(
        "There are %d cell-cell interactions, %d cell-tissue edges and %d tissue-tissue edges.",
        mg_summary["cell_cell_edges"],
        mg_summary["cell_tissue_edges"],
        mg_summary["tissue_tissue_edges"],
    )
    logger.info("Reliable genes: %d", gene_count)
    logger.info("")


def extract_cl_id_from_filename(filename: str) -> str | None:
    """Extract a base or ALS context-specific CL identifier from a filename."""
    match = re.match(r'(?:HBCA|TABULA)__(CL[:_]\d+)', filename, re.IGNORECASE)
    if match:
        return normalize_cl_label(match.group(1))

    match = re.search(r'\[cl[_:](\d+)\]', filename, re.IGNORECASE)
    if match:
        return f"CL_{match.group(1)}"

    match = re.match(r'(CL_\d+_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)', filename)
    if match:
        return match.group(1)

    match = re.match(r'CL_(\d+)', filename)
    if match:
        return f"CL_{match.group(1)}"

    match = re.search(r'CL:(\d+)', filename)
    if match:
        return f"CL_{match.group(1)}"

    return None


def standardize_celltype_name(cl_id: str) -> str:
    """Return a filesystem-safe CL or ALS context identifier."""
    if not cl_id:
        raise ValueError("CL ID cannot be empty")

    cl_clean = cl_id.strip()

    if cl_clean.upper().startswith("CL:"):
        cl_clean = "CL_" + cl_clean[3:]

    match = re.match(r"(CL_\d+)(.*)", cl_clean, re.IGNORECASE)
    if match:
        base_id = match.group(1).upper()
        suffix_raw = match.group(2)
    else:
        return cl_clean

    suffix = re.sub(r"[^0-9a-zA-Z_]+", "_", suffix_raw)
    suffix = re.sub(r"_+", "_", suffix).strip("_")
    return f"{base_id}_{suffix}" if suffix else base_id


def extract_base_cl_id(cl_id: str) -> str:
    """
    Extract the base CL identifier (CL_XXXXXXX) from an extended ID.

    Examples:
        CL_0000100_CTRL_cyto_d0 -> CL_0000100
        CL:0000236 -> CL_0000236
    """
    if not cl_id:
        return ""
    cl_clean = cl_id.strip()
    if cl_clean.upper().startswith("CL:"):
        cl_clean = "CL_" + cl_clean[3:]
    base_match = re.match(r"(CL_\d+)", cl_clean, re.IGNORECASE)
    if base_match:
        return base_match.group(1).upper()
    return cl_clean.upper()


def lookup_cl_name(cl_id: str, cl_id_to_name: Dict[str, str]) -> Optional[str]:
    """
    Retrieve the canonical Cell Ontology name for a given identifier.

    Handles CL IDs written with underscores (CL_0000000), colons (CL:0000000),
    or extended ALS-style suffixes (CL_0000000_CTRL_cyto_d0).
    """
    if not cl_id or not cl_id_to_name:
        return None

    cl_id_clean = cl_id.strip()
    candidates = [
        cl_id_clean,
        normalize_cl_label(cl_id_clean),
    ]

    # Base identifier without suffix (if present)
    base_match = re.match(r"(CL[_:]\d+)", cl_id_clean)
    if base_match:
        base_id = base_match.group(1)
        candidates.append(base_id)
        candidates.append(normalize_cl_label(base_id).replace("CL_", "CL:", 1))
        candidates.append(base_id.replace("CL_", "CL:", 1))

    # Add colon/underscore swapped variants explicitly
    if cl_id_clean.upper().startswith("CL:"):
        candidates.append(cl_id_clean.replace("CL:", "CL_", 1))
    if cl_id_clean.upper().startswith("CL_"):
        candidates.append(cl_id_clean.replace("CL_", "CL:", 1))

    for candidate in dict.fromkeys(candidates):
        name = cl_id_to_name.get(candidate)
        if name:
            return name
    return None


def merge_ppi_networks(
    hbca_ppi_root: Optional[str] = None,
    tabula_ppi_root: Optional[str] = None,
    output_root: str = "",
    als_ppi_roots: List[str] = None,
    merged_ppi_root: Optional[str] = None,
) -> Dict[str, int]:
    """Union PPI networks that share a Cell Ontology identifier."""
    logger.info("Merging PPI networks by CL ID")

    hbca_root = Path(hbca_ppi_root) if hbca_ppi_root else None
    tabula_root = Path(tabula_ppi_root) if tabula_ppi_root else None
    out_root = Path(output_root)
    out_edgelists = out_root / "ppi_edgelists"
    out_root.mkdir(parents=True, exist_ok=True)
    out_edgelists.mkdir(parents=True, exist_ok=True)

    def load_ppi_with_cl_ids(root: Path, dataset_name: str) -> Dict[str, List[tuple]]:
        """Load PPIs and map by CL ID. Returns {cl_id: [(file_path, original_name), ...]}"""
        edgelist_dir = root / "ppi_edgelists"
        if not edgelist_dir.exists():
            return {}

        cl_id_map = {}
        for f in sorted(edgelist_dir.glob("*.txt")):
            if f.stem.lower() in {"inventory", "readme", "metadata"}:
                continue

            cl_id = extract_cl_id_from_filename(f.stem)
            if cl_id:
                cl_id = normalize_cl_label(cl_id)
                if cl_id not in cl_id_map:
                    cl_id_map[cl_id] = []
                cl_id_map[cl_id].append((f, f.stem, dataset_name))
                logger.debug(f"{dataset_name}: {f.stem} → {cl_id}")
            else:
                logger.warning(f"Could not extract CL ID from: {f.stem} (dataset: {dataset_name})")

        return cl_id_map

    if hbca_root:
        logger.info("Loading HBCA PPIs...")
        hbca_cl_map = load_ppi_with_cl_ids(hbca_root, "HBCA")
        logger.info(f"  Found {len(hbca_cl_map)} unique CL IDs in HBCA")
    else:
        hbca_cl_map = {}

    if tabula_root:
        logger.info("Loading Tabula PPIs...")
        tabula_cl_map = load_ppi_with_cl_ids(tabula_root, "Tabula")
        logger.info(f"  Found {len(tabula_cl_map)} unique CL IDs in Tabula")
    else:
        tabula_cl_map = {}

    if merged_ppi_root:
        logger.info("Loading Merged PPIs...")
        merged_cl_map = load_ppi_with_cl_ids(Path(merged_ppi_root), "Merged")
        logger.info(f"  Found {len(merged_cl_map)} unique CL IDs in Merged")
    else:
        merged_cl_map = {}

    als_cl_maps = []
    if als_ppi_roots:
        for i, als_root in enumerate(als_ppi_roots):
            logger.info(f"Loading ALS PPI from: {als_root}")
            als_cl_map = load_ppi_with_cl_ids(Path(als_root), f"ALS_{i}")
            als_cl_maps.append(als_cl_map)
            logger.info(f"  Found {len(als_cl_map)} unique CL IDs in ALS_{i}")

    all_cl_ids = set(hbca_cl_map.keys()) | set(tabula_cl_map.keys()) | set(merged_cl_map.keys())
    for als_cl_map in als_cl_maps:
        all_cl_ids.update(als_cl_map.keys())

    logger.info(f"Total unique CL IDs across all datasets: {len(all_cl_ids)}")

    try:
        from .config import CL_PATH
        import obonet
        cl_graph = obonet.read_obo(CL_PATH)
        cl_id_to_name = {node_id: data.get('name', '') for node_id, data in cl_graph.nodes(data=True)}
        logger.info(f"Loaded Cell Ontology with {len(cl_id_to_name)} terms")
    except Exception as e:
        logger.warning(f"Could not load Cell Ontology: {e}. Using generic names.")
        cl_id_to_name = {}

    merged_networks = {}
    merged_genes = {}
    celltype_metadata: Dict[str, Dict[str, object]] = {}

    for cl_id in sorted(all_cl_ids):
        files_to_merge = []

        if cl_id in hbca_cl_map:
            files_to_merge.extend(hbca_cl_map[cl_id])
        if cl_id in tabula_cl_map:
            files_to_merge.extend(tabula_cl_map[cl_id])
        if cl_id in merged_cl_map:
            files_to_merge.extend(merged_cl_map[cl_id])
        for als_cl_map in als_cl_maps:
            if cl_id in als_cl_map:
                files_to_merge.extend(als_cl_map[cl_id])

        if not files_to_merge:
            continue

        canonical_name = lookup_cl_name(cl_id, cl_id_to_name)
        standardized_name = standardize_celltype_name(cl_id)

        if len(files_to_merge) > 1:
            logger.info(f"  Merging {len(files_to_merge)} networks for {cl_id} ({canonical_name or 'unknown'})")
            logger.info(f"    Sources: {', '.join(f'{ds}:{name}' for _, name, ds in files_to_merge)}")

        g_merged = nx.Graph()
        all_genes = set()

        for file_path, original_name, dataset_name in files_to_merge:
            try:
                g = read_edgelist_any(file_path)
            except Exception as e:
                logger.warning(f"    Failed to read {file_path.name}: {e}")
                continue
            if g.number_of_edges() == 0:
                logger.warning(f"    Empty network in {dataset_name}: {file_path.name}")
                continue
            g_merged = nx.compose(g_merged, g)
            all_genes.update(g.nodes())

        if g_merged.number_of_edges() == 0:
            logger.warning(f"  Skipping {cl_id} - no valid edges after merge")
            continue

        output_filename = f"{standardized_name}.txt"
        output_path = out_edgelists / output_filename
        nx.write_edgelist(g_merged, output_path, data=False)

        merged_networks[standardized_name] = g_merged
        merged_genes[standardized_name] = all_genes

        source_datasets = sorted({dataset_name for _, _, dataset_name in files_to_merge})
        metadata_key = Path(output_filename).stem
        base_cl_id = extract_base_cl_id(cl_id)
        has_condition = bool(base_cl_id and cl_id != base_cl_id)
        celltype_metadata[metadata_key] = {
            "cl_id": cl_id,
            "base_cl_id": base_cl_id,
            "canonical_name": canonical_name or "",
            "datasets": source_datasets,
            "num_source_networks": len(files_to_merge),
            "source_files": [file_path.name for file_path, _, _ in files_to_merge],
            "merged_nodes": g_merged.number_of_nodes(),
            "merged_edges": g_merged.number_of_edges(),
            "edgelist_path": f"ppi_edgelists/{output_filename}",
            "has_condition": has_condition,
            "primary_dataset": source_datasets[0] if source_datasets else "unknown",
        }

    list_path = out_root / "ppi_celltype_list.csv"
    with list_path.open("w") as f:
        for idx, (celltype_name, genes) in enumerate(sorted(merged_genes.items())):
            genes_str = ",".join(sorted(genes)) if genes else ""
            f.write(f"{idx}\t{celltype_name}\t{genes_str}\n")

    inventory_path = out_root / "inventory.txt"
    with inventory_path.open("w") as inv:
        for celltype_name in sorted(merged_networks.keys()):
            inv.write(f"ppi_edgelists/{celltype_name}.txt\n")

    if celltype_metadata:
        metadata_path = out_root / "celltype_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(celltype_metadata, fh, indent=2)

        metadata_df = pd.DataFrame.from_dict(celltype_metadata, orient="index")
        metadata_df.index.name = "edgelist"
        metadata_df_reset = metadata_df.reset_index()
        for col in ("datasets", "source_files"):
            if col in metadata_df_reset.columns:
                metadata_df_reset[col] = metadata_df_reset[col].apply(
                    lambda x: ";".join(x) if isinstance(x, (list, tuple, set)) else x
                )
        metadata_df_reset.to_csv(out_root / "celltype_metadata.csv", index=False)

    stats = {
        "total": len(merged_networks),
        "hbca_unique": len(hbca_cl_map),
        "tabula_unique": len(tabula_cl_map),
        "merged_unique": len(merged_cl_map),
        "als_unique": sum(len(m) for m in als_cl_maps),
        "shared_cl_ids": len([cl_id for cl_id in all_cl_ids
                              if sum([cl_id in m for m in [hbca_cl_map, tabula_cl_map, merged_cl_map] + als_cl_maps]) > 1]),
    }

    logger.info(f"Total merged PPI networks: {stats['total']}")
    logger.info(f"  - HBCA unique CL IDs: {stats['hbca_unique']}")
    logger.info(f"  - Tabula unique CL IDs: {stats['tabula_unique']}")
    logger.info(f"  - Merged unique CL IDs: {stats['merged_unique']}")
    logger.info(f"  - ALS unique CL IDs: {stats['als_unique']}")
    logger.info(f"  - Shared CL IDs (merged from multiple datasets): {stats['shared_cl_ids']}")

    return stats


def merge_cci_networks(
    cci_files: List[str],
    output_file: str,
) -> nx.Graph:
    """Merge CCI networks from multiple datasets (union of edges)."""
    logger.info("Merging CCI networks")

    g_merged = nx.Graph()
    for cci_file in cci_files:
        if os.path.exists(cci_file):
            logger.info(f"  Loading: {cci_file}")
            g_raw = nx.read_edgelist(cci_file, delimiter="\t")
            g = normalize_graph_labels(g_raw)
            if g.number_of_nodes() != g_raw.number_of_nodes() or g.number_of_edges() != g_raw.number_of_edges():
                logger.info(
                    "    %d nodes, %d edges (normalized from %d nodes, %d edges)",
                    g.number_of_nodes(),
                    g.number_of_edges(),
                    g_raw.number_of_nodes(),
                    g_raw.number_of_edges(),
                )
            else:
                logger.info(f"    {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
            g_merged = nx.compose(g_merged, g)
        else:
            logger.warning(f"  CCI file not found: {cci_file}")

    removed_nuc = remove_nuclear_cell_edges(g_merged)
    if removed_nuc:
        logger.info("Removed %d merged CCI edges involving ALS nuclear compartments", removed_nuc)
        isolated_nodes = list(nx.isolates(g_merged))
        if isolated_nodes:
            g_merged.remove_nodes_from(isolated_nodes)
            logger.info("Removed %d isolated CCI nodes after nuclear-edge filtering", len(isolated_nodes))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    nx.write_edgelist(g_merged, output_file, data=False, delimiter="\t")
    logger.info(f"Merged CCI: {g_merged.number_of_nodes()} nodes, {g_merged.number_of_edges()} edges")
    logger.info(f"Saved merged CCI to: {output_file}")
    return g_merged


def load_ppi_celltypes(ppi_dir: str) -> Set[str]:
    """Load cell type identifiers from ppi_celltype_list.csv if present."""
    ppi_list_file = Path(ppi_dir) / "ppi_celltype_list.csv"
    if not ppi_list_file.exists():
        return set()
    cell_types: Set[str] = set()
    with ppi_list_file.open("r") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1]:
                cell_types.add(parts[1])
    return cell_types


def check_celltype_tissue_coverage(mg_graph: nx.Graph, ppi_dir: str) -> Dict[str, object]:
    """
    Verify that every PPI-backed cell type exists in the metagraph and has a tissue edge.
    """
    if not ppi_dir or not os.path.exists(ppi_dir):
        return {}

    cell_types_with_ppi = load_ppi_celltypes(ppi_dir)
    if not cell_types_with_ppi:
        return {}

    def is_tissue(node: str) -> bool:
        return isinstance(node, str) and node.startswith("BTO")

    mg_celltypes = {n for n in mg_graph.nodes if not is_tissue(n)}
    missing_nodes = sorted(cell_types_with_ppi - mg_celltypes)

    missing_tissue_edges: List[str] = []
    for ct in (cell_types_with_ppi & mg_celltypes):
        if not any(is_tissue(nb) for nb in mg_graph.neighbors(ct)):
            missing_tissue_edges.append(ct)

    stats = {
        "ppi_celltypes": len(cell_types_with_ppi),
        "present_in_metagraph": len(cell_types_with_ppi & mg_celltypes),
        "missing_in_metagraph": len(missing_nodes),
        "missing_in_metagraph_examples": missing_nodes[:10],
        "missing_tissue_edges": len(missing_tissue_edges),
        "missing_tissue_edges_examples": sorted(missing_tissue_edges)[:10],
    }

    logger.info(
        "Metagraph coverage: %d/%d PPI cell types present; %d missing tissue edges",
        stats["present_in_metagraph"],
        stats["ppi_celltypes"],
        stats["missing_tissue_edges"],
    )
    if stats["missing_in_metagraph"]:
        logger.warning("  %d PPI cell types missing from metagraph (examples: %s)",
                       stats["missing_in_metagraph"],
                       ", ".join(stats["missing_in_metagraph_examples"]))
    if stats["missing_tissue_edges"]:
        logger.warning("  %d PPI cell types in metagraph without tissue edge (examples: %s)",
                       stats["missing_tissue_edges"],
                       ", ".join(stats["missing_tissue_edges_examples"]))
    return stats


def remove_nuclear_cell_edges(graph: nx.Graph) -> int:
    """
    Remove cell-cell edges where either endpoint is an ALS nuclear compartment.

    Mirrors the cleaning done in the notebook utility.
    """
    def is_tissue(node: str) -> bool:
        return isinstance(node, str) and node.startswith("BTO")

    edges_to_remove = [
        (u, v)
        for u, v in graph.edges()
        if not is_tissue(u) and not is_tissue(v)
        and (is_nuclear_compartment_node(u) or is_nuclear_compartment_node(v))
    ]
    if edges_to_remove:
        graph.remove_edges_from(edges_to_remove)
    return len(edges_to_remove)


def summarize_cci_vs_metagraph(cci_graph: nx.Graph, mg_graph: nx.Graph) -> Dict[str, object]:
    """Compute CCI coverage for cell types in the metagraph."""
    def is_tissue(node: str) -> bool:
        return isinstance(node, str) and node.startswith("BTO")

    mg_celltypes = {n for n in mg_graph.nodes if not is_tissue(n)}
    cci_celltypes = {n for n in cci_graph.nodes if not is_tissue(n)}

    degrees = []
    with_cci = 0
    for ct in mg_celltypes:
        deg = cci_graph.degree(ct) if ct in cci_graph else 0
        degrees.append(deg)
        if deg > 0:
            with_cci += 1

    median_deg = float(np.median(degrees)) if degrees else 0.0
    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    max_deg = int(np.max(degrees)) if degrees else 0
    missing_cci = len(mg_celltypes) - with_cci
    cci_only = len(cci_celltypes - mg_celltypes)

    stats = {
        "metagraph_celltypes": len(mg_celltypes),
        "metagraph_with_cci": with_cci,
        "metagraph_without_cci": missing_cci,
        "median_cci_degree": median_deg,
        "mean_cci_degree": mean_deg,
        "max_cci_degree": max_deg,
        "cci_celltypes_not_in_metagraph": cci_only,
    }

    logger.info(
        "CCI coverage: %d/%d metagraph cell types have CCI edges (median degree %.1f, mean %.1f, max %d)",
        stats["metagraph_with_cci"],
        stats["metagraph_celltypes"],
        stats["median_cci_degree"],
        stats["mean_cci_degree"],
        stats["max_cci_degree"],
    )
    if stats["metagraph_without_cci"]:
        logger.warning("  %d metagraph cell types lack CCI interactions", stats["metagraph_without_cci"])
    if stats["cci_celltypes_not_in_metagraph"]:
        logger.warning("  %d CCI cell types are absent from metagraph", stats["cci_celltypes_not_in_metagraph"])
    return stats


def merge_metagraphs(
    mg_files: List[str],
    output_file: str,
    ppi_dir: Optional[str] = None,
) -> nx.Graph:
    """
    Merge metagraphs from multiple datasets (union of edges).

    If ppi_dir is provided, filters out edges involving cell types that don't have a PPI network.
    """
    logger.info("Merging metagraphs")

    g_merged = nx.Graph()
    edges_seen = set()
    in_file_duplicates = 0
    cross_file_duplicates = 0
    all_nodes = set()
    for mg_file in mg_files:
        if os.path.exists(mg_file):
            logger.info(f"  Loading: {mg_file}")
            g_raw_multi = nx.read_edgelist(mg_file, delimiter="\t", create_using=nx.MultiGraph())
            raw_edges = g_raw_multi.number_of_edges()
            g_simple = nx.Graph(g_raw_multi)
            in_file_duplicates += raw_edges - g_simple.number_of_edges()

            g = normalize_graph_labels(g_simple)
            all_nodes.update(g.nodes())

            for u, v in g.edges():
                edge = tuple(sorted((u, v)))
                if edge in edges_seen:
                    cross_file_duplicates += 1
                    continue
                edges_seen.add(edge)
                g_merged.add_edge(*edge)
        else:
            logger.warning(f"  Metagraph file not found: {mg_file}")

    if all_nodes:
        g_merged.add_nodes_from(all_nodes)

    if in_file_duplicates:
        logger.info(f"  Deduplicated {in_file_duplicates} repeated edges inside input metagraphs")
    if cross_file_duplicates:
        logger.info(f"  Deduplicated {cross_file_duplicates} repeated edges across metagraphs")

    if ppi_dir and os.path.exists(ppi_dir):
        cell_types_with_ppi = load_ppi_celltypes(ppi_dir)
        if cell_types_with_ppi:
            logger.info(f"Found {len(cell_types_with_ppi)} cell types with PPI networks")

            def is_tissue(node: str) -> bool:
                return isinstance(node, str) and node.startswith("BTO")

            edges_to_remove = []
            for u, v in g_merged.edges():
                is_u_tissue = is_tissue(u)
                is_v_tissue = is_tissue(v)

                if is_u_tissue and is_v_tissue:
                    continue

                if not is_u_tissue and u not in cell_types_with_ppi:
                    edges_to_remove.append((u, v))
                elif not is_v_tissue and v not in cell_types_with_ppi:
                    edges_to_remove.append((u, v))

            if edges_to_remove:
                logger.info(f"Removing {len(edges_to_remove)} edges from metagraph (cell types without PPI)")
                g_merged.remove_edges_from(edges_to_remove)

    removed_nuc = remove_nuclear_cell_edges(g_merged)
    if removed_nuc:
        logger.info(f"Removed {removed_nuc} cell-cell edges containing 'nuc'")

    isolated_nodes = list(nx.isolates(g_merged))
    if isolated_nodes:
        logger.info(f"Removing {len(isolated_nodes)} isolated nodes")
        g_merged.remove_nodes_from(isolated_nodes)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    nx.write_edgelist(g_merged, output_file, data=False, delimiter="\t")

    bto_nodes = [n for n in g_merged.nodes() if n.startswith("BTO")]
    logger.info(f"Merged metagraph: {g_merged.number_of_nodes()} nodes ({len(bto_nodes)} tissues), {g_merged.number_of_edges()} edges")
    logger.info(f"Saved merged metagraph to: {output_file}")
    return g_merged


def merge_reliable_genes_multi(
    genes_files: List[str],
    output_file: str,
    per_cell_output: Optional[str] = None,
) -> Set[str]:
    """Merge reliable genes from multiple datasets."""
    logger.info("Merging reliable genes")

    def _canonicalize_celltype(label: str) -> str:
        text = (label or "").strip()
        if not text:
            return "unknown_cell_type"
        full_match = re.match(r"(CL[:_]\d+)([A-Za-z0-9_:.-]*)", text, re.IGNORECASE)
        if full_match:
            base = normalize_cl_label(full_match.group(1))
            suffix = full_match.group(2) or ""
            suffix = suffix.replace(":", "_")
            suffix = re.sub(r"\s+", "_", suffix)
            suffix = re.sub(r"_+", "_", suffix).strip("_")
            return f"{base}_{suffix}" if suffix else base
        match = re.search(r"(CL[:_]\d+)", text, re.IGNORECASE)
        if match:
            return normalize_cl_label(match.group(1))
        return normalize_cl_label(text)

    def _load_gene_payload(path: Path) -> Tuple[Set[str], Dict[str, List[str]]]:
        if not path.exists():
            return set(), {}
        with open(path, "r") as handle:
            payload = json.load(handle)
        per_cell: Dict[str, List[str]] = {}
        union: Set[str] = set()
        if isinstance(payload, dict):
            for cell_type, genes in payload.items():
                gene_list = [str(g).strip() for g in (genes or []) if isinstance(g, str) and str(g).strip()]
                if not gene_list:
                    continue
                canonical = _canonicalize_celltype(str(cell_type))
                per_cell.setdefault(canonical, []).extend(gene_list)
                union.update(gene_list)
        elif isinstance(payload, list):
            union.update(str(g).strip() for g in payload if isinstance(g, str) and str(g).strip())
        return union, per_cell

    def _load_dataset_genes(path_str: str) -> Tuple[Set[str], Dict[str, List[str]]]:
        path = Path(prefer_specific_file(path_str))
        union, per_cell = _load_gene_payload(path)
        if per_cell:
            return union, per_cell
        per_cell_path = Path(
            prefer_specific_file(path.with_name("reliable_genes_per_celltype.json"))
        )
        if per_cell_path.exists():
            per_union, per_mapping = _load_gene_payload(per_cell_path)
            if per_mapping:
                union.update(per_union)
                return union, per_mapping
        return union, per_cell

    def _build_gene_symbol_lookup(genes: Set[str]) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        ensembl_map: Dict[str, str] = {}
        for gene in genes:
            token = str(gene).strip()
            if not token:
                continue
            token_clean = token.split(".", 1)[0]
            if token_clean.upper().startswith("ENSG"):
                ensembl_map[token] = token_clean.upper()
            else:
                symbol = token.upper()
                lookup[token] = symbol
                lookup[token.upper()] = symbol
        if ensembl_map:
            try:
                gp_mapping = fetch_symbols_from_gprofiler(list(set(ensembl_map.values())))
            except Exception as exc:  # pragma: no cover - external service errors
                logger.warning("Failed to query g:Profiler for gene symbols: %s", exc)
                gp_mapping = {}
            for original, cleaned in ensembl_map.items():
                symbol = gp_mapping.get(cleaned) or gp_mapping.get(original)
                final_symbol = symbol.upper() if isinstance(symbol, str) and symbol else cleaned
                lookup[original] = final_symbol
                lookup[original.upper()] = final_symbol
        return lookup

    def _normalize_gene(gene: str, lookup: Dict[str, str]) -> Optional[str]:
        if not gene:
            return None
        token = gene.strip()
        if not token:
            return None
        return lookup.get(token) or lookup.get(token.upper()) or token.upper()

    all_genes_raw: Set[str] = set()
    per_cell_raw: Dict[str, Set[str]] = {}

    for genes_file in genes_files:
        union, per_cell = _load_dataset_genes(genes_file)
        if not union and not per_cell:
            logger.warning("  %s: no reliable genes found", Path(genes_file).parent.name)
            continue
        logger.info("  %s: %d genes", Path(genes_file).parent.name, len(union))
        all_genes_raw.update(union)
        for cell_type, genes in per_cell.items():
            per_cell_raw.setdefault(cell_type, set()).update(genes)

    symbol_lookup = _build_gene_symbol_lookup(all_genes_raw)
    normalized_union: Set[str] = set()
    normalized_per_cell: Dict[str, List[str]] = {}

    for gene in all_genes_raw:
        normalized = _normalize_gene(gene, symbol_lookup)
        if normalized:
            normalized_union.add(normalized)

    for cell_type, genes in per_cell_raw.items():
        normalized_genes = {
            normalized for g in genes if (normalized := _normalize_gene(g, symbol_lookup))
        }
        if normalized_genes:
            normalized_per_cell[cell_type] = sorted(normalized_genes)

    logger.info("  Total union after normalization: %d genes", len(normalized_union))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(sorted(normalized_union), f, indent=2)
    logger.info("Saved merged reliable genes to: %s", output_file)

    if per_cell_output and normalized_per_cell:
        with open(per_cell_output, "w") as handle:
            json.dump(
                dict(sorted(normalized_per_cell.items())),
                handle,
                indent=2,
            )
        logger.info("Saved per-cell reliable genes to: %s", per_cell_output)

    return normalized_union


def create_summary(
    output_dir: str,
    ppi_stats: Dict[str, int],
    cci_graph: nx.Graph,
    mg_graph: nx.Graph,
    genes: Set[str],
    ppi_summary: Dict[str, object],
    mg_summary: Dict[str, int],
    global_ppi_path: str,
    coverage_stats: Optional[Dict[str, object]] = None,
    cci_cell_stats: Optional[Dict[str, object]] = None,
) -> None:
    logger.info("Creating dataset summary")

    summary = {
        "ppi_networks": {
            "total_merged": ppi_stats["total"],
            "hbca_unique_cl_ids": ppi_stats.get("hbca_unique", 0),
            "tabula_unique_cl_ids": ppi_stats.get("tabula_unique", 0),
            "als_unique_cl_ids": ppi_stats.get("als_unique", 0),
            "shared_cl_ids": ppi_stats.get("shared_cl_ids", 0),
            "mean_nodes_per_network": ppi_summary["mean_nodes"],
            "std_nodes_per_network": ppi_summary["std_nodes"],
            "unique_proteins_total": len(ppi_summary["unique_proteins"]),
            "global_ppi_proteins": get_global_ppi_node_count(global_ppi_path),
        },
        "cci_network": {
            "nodes": cci_graph.number_of_nodes(),
            "edges": cci_graph.number_of_edges(),
            "density": nx.density(cci_graph),
        },
        "metagraph": {
            "nodes": mg_graph.number_of_nodes(),
            "edges": mg_graph.number_of_edges(),
            "density": nx.density(mg_graph),
            "tissues_total": mg_summary["tissues_total"],
            "tissues_connected": mg_summary["tissues_connected"],
            "cell_cell_edges": mg_summary["cell_cell_edges"],
            "cell_tissue_edges": mg_summary["cell_tissue_edges"],
            "tissue_tissue_edges": mg_summary["tissue_tissue_edges"],
        },
        "coverage": coverage_stats or {},
        "cci_by_celltype": cci_cell_stats or {},
        "genes": {
            "reliable_genes": len(genes),
        },
    }

    summary_file = Path(output_dir) / "merge_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to: %s", summary_file)


def _validate_output_path(output: Path, input_roots: List[Path]) -> None:
    def overlaps(left: Path, right: Path) -> bool:
        return left == right or left in right.parents or right in left.parents

    repository = Path(__file__).resolve().parents[1]
    home = Path.home().resolve()
    pinnacle = Path(PINNACLE_BASE).resolve()
    unsafe = (
        output == Path("/")
        or output == home
        or output in home.parents
        or output == pinnacle
        or output in pinnacle.parents
        or overlaps(output, repository)
        or any(overlaps(output, root) for root in input_roots)
    )
    if unsafe:
        raise ValueError(f"Refusing to replace unsafe output directory: {output}")


def _require_inputs(paths: List[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Step 8 inputs:\n" + "\n".join(missing))


def _als_gene_files(als_root: Path) -> List[str]:
    combined = als_root / "reliable_genes.json"
    if combined.exists():
        return [str(combined)]
    return [
        prefer_specific_file(str(als_root / ALS_MN_REL / "reliable_genes.json")),
        prefer_specific_file(str(als_root / ALS_ASTRO_REL / "reliable_genes.json")),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(MERGED_OUTPUT))
    parser.add_argument("--hbca-root", type=Path, default=Path(HBCA_INTERMEDIATE))
    parser.add_argument("--tabula-root", type=Path, default=Path(TABULA_INTERMEDIATE))
    parser.add_argument("--als-root", type=Path, default=Path(ALS_BULK_DIR))
    parser.add_argument("--merged-root", type=Path, default=Path(MERGED_INTERMEDIATE))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output_dir.resolve()
    hbca = args.hbca_root.resolve()
    tabula = args.tabula_root.resolve()
    als = args.als_root.resolve()
    merged = args.merged_root.resolve()
    input_roots = [hbca, tabula, als, merged]
    _validate_output_path(output, input_roots)

    gene_files = [
        prefer_specific_file(str(hbca / "reliable_genes.json")),
        prefer_specific_file(str(tabula / "reliable_genes.json")),
        *_als_gene_files(als),
    ]
    _require_inputs(
        [
            Path(GLOBAL_PPI),
            *(Path(path) for path in gene_files),
            hbca / "ppi" / "ppi_edgelists",
            tabula / "ppi" / "ppi_edgelists",
            merged / "cci_edgelist.txt",
            merged / "metagraph.txt",
            als / "ppi" / "ppi_edgelists",
            als / "cci_edgelist.txt",
            als / "metagraph.txt",
        ]
    )

    if output.exists():
        logger.info("Removing existing output directory: %s", output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    ppi_stats = merge_ppi_networks(
        output_root=str(output),
        hbca_ppi_root=str(hbca / "ppi"),
        tabula_ppi_root=str(tabula / "ppi"),
        als_ppi_roots=[str(als / "ppi")],
    )
    cci_graph = merge_cci_networks(
        [str(merged / "cci_edgelist.txt"), str(als / "cci_edgelist.txt")],
        str(output / "cci_edgelist.txt"),
    )
    mg_graph = merge_metagraphs(
        [str(merged / "metagraph.txt"), str(als / "metagraph.txt")],
        str(output / "mg_edgelist.txt"),
        str(output),
    )
    genes = merge_reliable_genes_multi(
        gene_files,
        str(output / "reliable_genes.json"),
        str(output / "reliable_genes_per_celltype.json"),
    )

    output_global_ppi = output / "global_ppi_edgelist.txt"
    shutil.copy(GLOBAL_PPI, output_global_ppi)
    write_count_edge_dict(output_global_ppi, output, output / "count_edge_dict.pkl")
    ppi_summary = summarize_ppi_dir(str(output))
    mg_summary = summarize_metagraph(mg_graph)
    coverage = check_celltype_tissue_coverage(mg_graph, str(output))
    cci_summary = summarize_cci_vs_metagraph(cci_graph, mg_graph)
    create_summary(
        str(output),
        ppi_stats,
        cci_graph,
        mg_graph,
        genes,
        ppi_summary,
        mg_summary,
        str(output_global_ppi),
        coverage,
        cci_summary,
    )
    log_final_summary(ppi_summary, mg_summary, len(genes), str(output_global_ppi))
    logger.info("Merged data saved to %s", output)


if __name__ == "__main__":
    main()
