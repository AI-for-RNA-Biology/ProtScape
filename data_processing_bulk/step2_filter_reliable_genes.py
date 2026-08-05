#!/usr/bin/env python
"""Step 2: select reliably expressed genes (REGs) for each cell type."""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .config import HBCA_INTERMEDIATE, TABULA_INTERMEDIATE, resolve_params
from .utils import compute_reliable_genes, plot_reg_histogram


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_").lower()


def _load_pseudobulk(dataset, path=None):
    if path is None:
        root = TABULA_INTERMEDIATE if dataset == "tabula" else HBCA_INTERMEDIATE
        path = Path(root) / "pseudobulk.h5ad"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pseudobulk file not found: {path}")

    logger.info("Loading pseudobulk data from %s", path)
    adata = sc.read_h5ad(path)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    if "total_counts" not in adata.obs:
        adata.obs["total_counts"] = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
    if "library_size" not in adata.obs:
        adata.obs["library_size"] = adata.obs["total_counts"]
    return adata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hbca", "tabula"), required=True)
    parser.add_argument("--pseudobulk", help="Pseudobulk h5ad; defaults to the dataset intermediate directory.")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--min-fraction", type=float)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--quantile", type=float, help="Quantile of the GMM background component.")
    parser.add_argument("--log-offset", type=float)
    parser.add_argument("--fallback-threshold", type=float, default=0.5)
    parser.add_argument("--qc-dir", help="Optional directory for per-cell-type histograms.")
    parser.add_argument("--output-dir", help="Override the dataset intermediate directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    params = resolve_params(
        args.dataset,
        min_donor_fraction=args.min_fraction,
        min_donor_count=args.min_samples,
        gmm_quantile=args.quantile,
        log_offset=args.log_offset,
    )
    min_fraction = params.min_donor_fraction
    min_samples = params.min_donor_count
    quantile = params.gmm_quantile
    log_offset = params.log_offset
    logger.info(
        "REG parameters: quantile=%.3f, min_fraction=%.2f, min_samples=%d, "
        "log_offset=%.3f, fallback_threshold=%.2f",
        quantile,
        min_fraction,
        min_samples,
        log_offset,
        args.fallback_threshold,
    )

    adata = _load_pseudobulk(args.dataset, args.pseudobulk)
    if args.cell_type_col not in adata.obs:
        raise KeyError(f"Column '{args.cell_type_col}' not found in pseudobulk obs")

    counts = adata.layers["counts"]
    genes = adata.var_names.to_list()
    qc_dir = Path(args.qc_dir) if args.qc_dir else None
    if qc_dir:
        qc_dir.mkdir(parents=True, exist_ok=True)

    reliable_genes = {}
    threshold_rows = []
    summary_rows = []

    for cell_type, sample_names in adata.obs.groupby(args.cell_type_col).groups.items():
        positions = adata.obs.index.get_indexer(sample_names)
        n_samples = len(positions)
        cell_obs = adata.obs.iloc[positions]
        n_cells = int(cell_obs["n_cells"].sum()) if "n_cells" in cell_obs else 0
        median_library_size = (
            float(cell_obs["total_counts"].median()) if "total_counts" in cell_obs else float("nan")
        )

        if n_samples < min_samples:
            logger.warning(
                "Skipping %s: %d pseudobulk samples (minimum %d)",
                cell_type,
                n_samples,
                min_samples,
            )
            summary_rows.append(
                {
                    "cell_type": cell_type,
                    "n_samples": n_samples,
                    "genes_kept": 0,
                    "n_cells_total": n_cells,
                    "median_libsize": median_library_size,
                    "status": "skipped_low_samples",
                }
            )
            continue

        cell_counts = counts[positions]
        cell_counts = cell_counts.toarray() if sparse.issparse(cell_counts) else np.asarray(cell_counts)
        cell_log = np.log2(cell_counts + log_offset).astype(np.float32)
        sample_ids = cell_obs.index.to_list()
        selected, thresholds, _, _ = compute_reliable_genes(
            cell_counts,
            genes,
            sample_ids,
            quantile=quantile,
            min_fraction=min_fraction,
            min_count=min_samples,
            log_offset=log_offset,
            log_matrix=cell_log,
            fallback_threshold=args.fallback_threshold,
        )

        reliable_genes[cell_type] = selected
        threshold_rows.extend(
            {"cell_type": cell_type, "sample_id": sample_id, "threshold": threshold}
            for sample_id, threshold in thresholds.items()
        )
        summary_rows.append(
            {
                "cell_type": cell_type,
                "n_samples": n_samples,
                "genes_kept": len(selected),
                "n_cells_total": n_cells,
                "median_libsize": median_library_size,
                "status": "kept",
            }
        )
        logger.info("%s: kept %d genes", cell_type, len(selected))

        if qc_dir:
            plot_reg_histogram(
                cell_log,
                thresholds,
                quantile=quantile,
                output_path=qc_dir / f"{_safe_name(cell_type)}_hist.png",
                title=f"{cell_type} log2(count + {log_offset}) distribution",
                log_offset=log_offset,
            )

    output_dir = Path(
        args.output_dir
        or (TABULA_INTERMEDIATE if args.dataset == "tabula" else HBCA_INTERMEDIATE)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "reliable_genes.json").open("w") as handle:
        json.dump(reliable_genes, handle)
    pd.DataFrame(threshold_rows).to_csv(output_dir / "thresholds.csv", index=False)
    pd.DataFrame(summary_rows).sort_values("cell_type").to_csv(
        output_dir / "reliable_gene_summary.csv", index=False
    )

    logger.info("Saved Step 2 outputs to %s", output_dir)


if __name__ == "__main__":
    main()
