#!/usr/bin/env python
"""Step 1 (ALS): map genes to HGNC and store one bulk profile per context."""

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .config import ALS_ASTRO_INTERMEDIATE, ALS_MN_INTERMEDIATE
from .ensembl_to_hgnc_converter import convert_expression_matrix


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DATASETS = {
    "motor_neuron": {
        "directory": ALS_MN_INTERMEDIATE,
        "counts_file": "als_motor_neurons_raw_counts.csv",
        "metadata_file": "als_motor_neurons_raw_metadata.csv",
        "h5ad_file": "als_motor_neurons.h5ad",
    },
    "astrocyte": {
        "directory": ALS_ASTRO_INTERMEDIATE,
        "counts_file": "als_astrocytes_raw_counts.csv",
        "metadata_file": "als_astrocytes_raw_metadata.csv",
        "h5ad_file": "als_astrocytes.h5ad",
    },
}


def create_anndata(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    dataset: str,
) -> ad.AnnData:
    """Create an AnnData with log2(count + 1) in X and averaged counts in a layer."""
    raw_values = counts.T.to_numpy()
    log_values = np.log2(raw_values + 1.0)
    adata = ad.AnnData(
        X=log_values,
        obs=metadata.set_index("cell_type_id"),
        var=pd.DataFrame(index=counts.index),
    )
    adata.layers["counts"] = raw_values.copy()
    adata.var["gene_id"] = adata.var_names
    adata.var["gene_name"] = adata.var_names
    adata.obs["dataset"] = dataset
    adata.obs["source"] = "als_bulk"
    return adata


def process_dataset(dataset: str) -> ad.AnnData:
    config = DATASETS[dataset]
    directory = Path(config["directory"])
    counts = pd.read_csv(directory / config["counts_file"], index_col=0)
    metadata = pd.read_csv(directory / config["metadata_file"])

    logger.info("Mapping %s Ensembl IDs to HGNC symbols", dataset)
    counts = convert_expression_matrix(counts)
    adata = create_anndata(counts, metadata, dataset)

    output_path = directory / config["h5ad_file"]
    adata.write_h5ad(output_path)
    logger.info("Saved %s (%d contexts x %d genes) to %s", dataset, adata.n_obs, adata.n_vars, output_path)
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-type",
        choices=["motor_neuron", "astrocyte", "both"],
        default="both",
    )
    args = parser.parse_args()

    selected = DATASETS if args.cell_type == "both" else [args.cell_type]
    for dataset in selected:
        process_dataset(dataset)


if __name__ == "__main__":
    main()
