#!/usr/bin/env python
"""Step 2 (ALS): select reliably expressed genes in each bulk context."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .config import ALS_ASTRO_INTERMEDIATE, ALS_MN_INTERMEDIATE
from .utils import compute_reliable_genes, plot_reg_histogram


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DATASETS = {
    "motor_neuron": {
        "directory": ALS_MN_INTERMEDIATE,
        "h5ad_file": "als_motor_neurons.h5ad",
    },
    "astrocyte": {
        "directory": ALS_ASTRO_INTERMEDIATE,
        "h5ad_file": "als_astrocytes.h5ad",
    },
}


def _safe_filename(label: str) -> str:
    return re.sub(r"[^\w.-]+", "_", label.strip()) or "als_context"


def _load_log_expression(path: Path) -> pd.DataFrame:
    """Load the log2(count + 1) matrix written by ALS step 1."""
    adata = sc.read_h5ad(path)
    values = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    return pd.DataFrame(values.T, index=adata.var_names, columns=adata.obs_names)


def compute_reliable_genes_bulk(
    log_expression: pd.DataFrame,
    *,
    quantile: float = 0.99,
    min_fraction: float = 0.7,
    context: str | None = None,
    qc_dir: Path | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Apply the shared GMM threshold to a gene-by-context log matrix."""
    log_values = log_expression.T.to_numpy(dtype=float)
    sample_ids = log_expression.columns.astype(str).tolist()
    reliable_genes, thresholds, _, plotted_values = compute_reliable_genes(
        counts_matrix=log_values,
        genes=log_expression.index.astype(str),
        sample_ids=sample_ids,
        quantile=quantile,
        min_fraction=min_fraction,
        log_matrix=log_values,
    )

    if context is not None and qc_dir is not None:
        plot_reg_histogram(
            plotted_values,
            thresholds,
            quantile=quantile,
            output_path=qc_dir / f"{_safe_filename(context)}_gmm_hist.png",
            title=f"{context} log2(count + 1) distribution",
            log_offset=1.0,
        )
    return reliable_genes, thresholds


def filter_dataset(dataset: str) -> None:
    config = DATASETS[dataset]
    directory = Path(config["directory"])
    expression = _load_log_expression(directory / config["h5ad_file"])
    qc_dir = directory / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    reliable_by_context = {}
    thresholds_by_context = {}
    for context in sorted(expression.columns):
        reliable, thresholds = compute_reliable_genes_bulk(
            expression[[context]],
            context=context,
            qc_dir=qc_dir,
        )
        reliable_by_context[context] = reliable
        thresholds_by_context[context] = thresholds
        logger.info("%s: %d reliable genes", context, len(reliable))

    with (directory / "reliable_genes_per_celltype.json").open("w") as handle:
        json.dump(reliable_by_context, handle, indent=2)
    with (directory / "thresholds_per_celltype.json").open("w") as handle:
        json.dump(thresholds_by_context, handle, indent=2)

    union = sorted({gene for genes in reliable_by_context.values() for gene in genes})
    with (directory / "reliable_genes.json").open("w") as handle:
        json.dump(union, handle, indent=2)
    logger.info("%s: %d contexts, %d genes in union", dataset, len(reliable_by_context), len(union))


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
        filter_dataset(dataset)


if __name__ == "__main__":
    main()
