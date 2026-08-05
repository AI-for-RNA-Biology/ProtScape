#!/usr/bin/env python
"""Step 0: merge and harmonize the two HBCA single-cell releases."""

import argparse
import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from .config import (
    HBCA_INTERMEDIATE,
    HBCA_NEURONS_PATH,
    HBCA_NONNEURONS_PATH,
    HBCA_SUPERCLUSTER_TO_CL,
)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


ESSENTIAL_COLUMNS = {
    "supercluster_term": "hbca_supercluster",
    "cluster_id": "hbca_cluster_id",
    "subcluster_id": "hbca_subcluster_id",
    "cell_type": "hbca_cell_type_original",
    "cell_type_ontology_term_id": "hbca_cell_type_ontology_term_id",
    "tissue": "tissue",
    "tissue_ontology_term_id": "tissue_ontology_term_id",
    "donor_id": "donor_id",
    "roi": "roi",
    "ROIGroupCoarse": "ROIGroupCoarse",
}


def _ensure_columns(adata: anndata.AnnData, subset: str) -> anndata.AnnData:
    adata = adata.copy()
    adata.obs["hbca_subset"] = subset
    for source, target in ESSENTIAL_COLUMNS.items():
        if source in adata.obs:
            adata.obs[target] = adata.obs[source].astype(str)
        else:
            logger.warning("%s is missing from the %s release", source, subset)
            adata.obs[target] = pd.NA

    cleaned = (
        adata.obs["hbca_supercluster"]
        .fillna("unknown")
        .str.strip()
        .str.replace(r"[^\w\s-]+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )
    adata.obs["hbca_supercluster_clean"] = cleaned.str.replace(" ", "_")
    if "total_UMIs" in adata.obs and "total_counts" not in adata.obs:
        adata.obs["total_counts"] = adata.obs["total_UMIs"].astype(np.float64)
    return adata


def _apply_cl_overrides(adata: anndata.AnnData) -> anndata.AnnData:
    """Replace generic neuronal CL IDs using the curated supercluster mapping."""
    mapping_path = Path(HBCA_SUPERCLUSTER_TO_CL)
    if not mapping_path.exists():
        logger.warning("CL mapping not found: %s", mapping_path)
        return adata

    overrides = pd.read_csv(mapping_path, encoding="utf-8-sig")
    required = {"supercluster", "cl_id", "cl_name"}
    missing = required - set(overrides.columns)
    if missing:
        raise KeyError(f"HBCA CL mapping is missing columns: {sorted(missing)}")

    mapping = dict(
        zip(
            overrides["supercluster"].astype(str).str.lower().str.strip(),
            overrides["cl_id"],
        )
    )
    superclusters = adata.obs["hbca_supercluster"].astype(str).str.lower().str.strip()
    mapped_ids = superclusters.map(mapping)
    mapped = mapped_ids.notna()
    cl_column = "hbca_cell_type_ontology_term_id"
    adata.obs.loc[mapped, cl_column] = mapped_ids[mapped]

    unmapped_generic = (~mapped) & (adata.obs[cl_column] == "CL:0000540")
    if unmapped_generic.any():
        logger.info("Removing %d generic neurons without a curated mapping", unmapped_generic.sum())
        adata = adata[~unmapped_generic].copy()
    logger.info("Assigned curated CL IDs to %d neuronal cells", mapped.sum())
    return adata


def merge_hbca_single_cell(
    neurons_path: str,
    nonneurons_path: str,
    output_dir: str,
) -> anndata.AnnData:
    neurons = _apply_cl_overrides(_ensure_columns(sc.read_h5ad(neurons_path), "neurons"))
    nonneurons = _ensure_columns(sc.read_h5ad(nonneurons_path), "nonneurons")
    logger.info("Loaded %d neurons and %d non-neuronal cells", neurons.n_obs, nonneurons.n_obs)

    merged = anndata.concat(
        [neurons, nonneurons],
        join="inner",
        keys=["neurons", "nonneurons"],
        index_unique="-",
    )
    merged.obs["cell_type"] = merged.obs["hbca_supercluster"].fillna("unknown")
    merged.obs["cell_type_clean"] = merged.obs["hbca_supercluster_clean"].fillna("unknown")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(output / "all_single_cell.h5ad")

    metadata_columns = [
        "hbca_subset",
        "cell_type",
        "cell_type_clean",
        "hbca_supercluster",
        "hbca_supercluster_clean",
        "hbca_cluster_id",
        "hbca_subcluster_id",
        "hbca_cell_type_original",
        "hbca_cell_type_ontology_term_id",
        "donor_id",
        "tissue",
        "tissue_ontology_term_id",
        "roi",
        "ROIGroupCoarse",
    ]
    merged.obs[metadata_columns].to_csv(output / "all_single_cell_metadata.csv")
    (
        merged.obs.groupby(["hbca_supercluster", "hbca_subset", "tissue", "hbca_cluster_id"])
        .size()
        .reset_index(name="n_cells")
        .sort_values("n_cells", ascending=False)
        .to_csv(output / "celltype_summary.csv", index=False)
    )
    logger.info("Saved merged HBCA data: %d cells x %d genes", merged.n_obs, merged.n_vars)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", default=HBCA_NEURONS_PATH)
    parser.add_argument("--nonneurons", default=HBCA_NONNEURONS_PATH)
    parser.add_argument("--output-dir", default=HBCA_INTERMEDIATE)
    args = parser.parse_args()
    merge_hbca_single_cell(args.neurons, args.nonneurons, args.output_dir)


if __name__ == "__main__":
    main()
