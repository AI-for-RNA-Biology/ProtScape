"""Small helpers shared by the bulk-data preparation steps."""

import logging
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.mixture import BayesianGaussianMixture


logger = logging.getLogger(__name__)


def format_cl_label(name: str, cl_id: str) -> str:
    """Combine a cell-type name and Cell Ontology identifier."""
    name = (name or "").strip()
    cl_id = (cl_id or "").strip()
    if not name:
        return cl_id
    return f"{name} [{cl_id}]" if cl_id else name


def standardize_gene_annotations(adata: sc.AnnData) -> sc.AnnData:
    """Store original IDs, version-free Ensembl IDs, and gene symbols."""
    if adata.n_vars == 0:
        return adata

    original_ids = pd.Index(adata.var_names.astype(str), name="original_id")
    original_series = original_ids.to_series().astype(str)
    ensembl_ids = original_series.str.replace(r"\.\d+$", "", regex=True)
    ensembl_ids = ensembl_ids.where(ensembl_ids.astype(bool), original_series)

    adata.var["original_id"] = original_ids.astype(str)
    adata.var["ensembl_id"] = ensembl_ids.astype(str)
    symbol_col = next(
        (column for column in ("gene_symbol", "symbol", "gene_name", "GeneSymbol") if column in adata.var),
        None,
    )
    if symbol_col:
        symbols = adata.var[symbol_col].astype(str).replace({"nan": np.nan, "None": np.nan})
    else:
        symbols = pd.Series(np.nan, index=adata.var.index, dtype=object)
    adata.var["symbol"] = symbols.fillna(adata.var["ensembl_id"])

    adata.var_names = pd.Index(adata.var["ensembl_id"].astype(str))
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    return adata


def read_h5ad(path: str) -> sc.AnnData:
    """Read an AnnData object into memory and ensure unique gene identifiers."""
    logger.info("Loading AnnData: %s", path)
    adata = sc.read_h5ad(path)
    if not adata.var_names.is_unique:
        logger.warning("Gene identifiers are not unique in %s; making them unique", path)
        adata.var_names_make_unique()
    return adata


def aggregate_pseudobulk(
    adata: sc.AnnData,
    *,
    dataset: str,
    cell_type_col: str,
    donor_col: str = None,
    tissue_col: Optional[str] = None,
    min_cells: int = 50,
    group_by_donor: bool = False,
) -> Tuple[sc.AnnData, pd.DataFrame]:
    """Sum raw counts per cell type, optionally retaining donor and tissue groups."""
    min_cells = max(0, int(min_cells or 0))
    if group_by_donor and donor_col is None:
        raise ValueError("donor_col is required when group_by_donor=True")

    group_cols = [cell_type_col]
    if group_by_donor:
        group_cols.append(donor_col)
        if tissue_col:
            group_cols.append(tissue_col)
    missing = set(group_cols) - set(adata.obs)
    if missing:
        raise KeyError(f"Missing required AnnData.obs columns: {sorted(missing)}")

    counts = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    grouping = adata.obs[group_cols].copy()
    grouping["cell_idx"] = np.arange(adata.n_obs)
    grouped = grouping.groupby(group_cols)
    logger.info("Aggregating %s by %s (%d groups)", dataset, ", ".join(group_cols), len(grouped))

    rows = []
    matrices = []
    skipped = 0
    for group_key, group in grouped:
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        cell_type = str(values[0])
        donor = str(values[1]) if group_by_donor else "pooled"
        tissue = str(values[2]) if group_by_donor and tissue_col else None
        cell_indices = group["cell_idx"].to_numpy()
        if len(cell_indices) < min_cells:
            skipped += 1
            continue

        cell_counts = np.asarray(counts[cell_indices].sum(axis=0)).ravel()
        total_counts = float(cell_counts.sum())
        sample_parts = [dataset, cell_type]
        if donor != "pooled":
            sample_parts.append(donor)
        if tissue is not None:
            sample_parts.append(tissue)
        rows.append(
            {
                "sample_id": "|".join(sample_parts),
                "dataset": dataset,
                "cell_type": cell_type,
                "donor": donor,
                "tissue": tissue,
                "n_cells": len(cell_indices),
                "total_counts": total_counts,
                "library_size": total_counts,
            }
        )
        matrices.append(cell_counts)

    if not matrices:
        raise ValueError("No pseudobulk samples generated; consider lowering min_cells")

    metadata = pd.DataFrame(rows)
    metadata["donor"] = metadata["donor"].fillna("pooled")
    metadata["tissue"] = metadata["tissue"].fillna("unknown" if group_by_donor else "pooled")
    for column in ("dataset", "cell_type", "donor", "tissue"):
        metadata[column] = metadata[column].astype(str)
    metadata = metadata.set_index("sample_id")

    pseudobulk = sc.AnnData(
        X=sparse.csr_matrix(np.vstack(matrices).astype(np.float32)),
        obs=metadata,
        var=adata.var.copy(),
    )
    logger.info(
        "Generated %d pseudobulk samples across %d genes%s",
        pseudobulk.n_obs,
        pseudobulk.n_vars,
        f"; skipped {skipped} small groups" if skipped else "",
    )
    return pseudobulk, metadata


# Keep these fixed GMM choices aligned with the frozen networks_bulk build:
# Bayesian two-component fit, seed 0, and the lower-mean background component.
def robust_gmm_threshold(
    values: np.ndarray,
    *,
    quantile: float = 0.9,
    fallback: float = 0.5,
) -> float:
    """Return a quantile of the lower-mean component of a two-component GMM."""
    vector = np.asarray(values, dtype=float)
    if vector.size == 0:
        return float(fallback)
    if np.allclose(vector, vector[0]):
        return float(vector[0])

    try:
        gmm = BayesianGaussianMixture(n_components=2, random_state=0, max_iter=1000)
        gmm.fit(vector.reshape(-1, 1))
        means = gmm.means_.ravel()
        variances = gmm.covariances_.ravel()
        if np.any(variances <= 0):
            raise ValueError("Non-positive GMM variance")

        order = np.argsort(means)
        if np.isclose(means[order[0]], means[order[1]], atol=1e-3):
            raise ValueError("GMM components collapsed")

        from scipy.stats import norm

        background = order[0]
        threshold = norm(
            loc=means[background], scale=np.sqrt(variances[background])
        ).ppf(quantile)
        if np.isnan(threshold):
            raise ValueError("GMM returned a NaN threshold")
        return float(threshold)
    except Exception as exc:  # noqa: BLE001
        logger.debug("GMM failed (%s); using %.3f", exc, fallback)
        return float(fallback)


def compute_reliable_genes(
    counts_matrix: np.ndarray,
    genes: Iterable[str],
    sample_ids: List[str],
    *,
    quantile: float = 0.9,
    min_fraction: float = 0.7,
    min_count: Optional[int] = None,
    log_offset: float = 1.0,
    log_matrix: Optional[np.ndarray] = None,
    fallback_threshold: float = 0.5,
) -> Tuple[List[str], Dict[str, float], np.ndarray, np.ndarray]:
    """Select genes above a per-sample GMM threshold in enough samples."""
    if counts_matrix.ndim != 2:
        raise ValueError("counts_matrix must be two-dimensional (samples x genes)")

    counts = np.asarray(counts_matrix, dtype=float)
    log_offset = 1.0 if log_offset is None else log_offset
    log_values = (
        np.log2(counts + log_offset)
        if log_matrix is None
        else np.asarray(log_matrix, dtype=float)
    )
    if log_values.shape != counts.shape:
        raise ValueError("log_matrix must have the same shape as counts_matrix")

    thresholds = {}
    expressed = np.zeros_like(log_values, dtype=bool)
    for row, sample_id in enumerate(sample_ids):
        threshold = robust_gmm_threshold(
            log_values[row], quantile=quantile, fallback=fallback_threshold
        )
        thresholds[sample_id] = threshold
        expressed[row] = log_values[row] >= threshold

    required = max(1, int(np.ceil(min_fraction * len(sample_ids))))
    if min_count is not None:
        required = max(required, min_count)
    keep = expressed.sum(axis=0) >= required
    return [gene for gene, selected in zip(genes, keep) if selected], thresholds, expressed, log_values


def plot_reg_histogram(
    log_matrix: np.ndarray,
    thresholds: Mapping[str, float],
    *,
    quantile: float,
    output_path: Path,
    title: str,
    log_offset: float = 1.0,
) -> None:
    """Plot the expression distribution and the thresholds actually used."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    values = np.asarray(log_matrix, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        logger.warning("No finite values available for QC plot: %s", title)
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=80, color="0.6", edgecolor="black", linewidth=0.5)
    for index, sample_id in enumerate(sorted(thresholds)):
        ax.axvline(
            thresholds[sample_id],
            color="tab:red",
            linestyle="--",
            linewidth=1,
            alpha=0.65,
            label="Per-sample GMM threshold" if index == 0 else None,
        )
    ax.set(
        xlabel=f"log2(count + {log_offset:g})",
        ylabel="Number of genes",
        title=title,
    )
    ax.text(
        0.98,
        0.95,
        f"Background-component quantile = {quantile:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="0.3",
    )
    if thresholds:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def pairwise_jaccard(values: Mapping[str, Collection]) -> List[float]:
    """Return pairwise Jaccard similarities between non-empty collections."""
    keys = list(values)
    scores = []
    for position, key in enumerate(keys[:-1]):
        left = set(values[key])
        if not left:
            continue
        for other_key in keys[position + 1 :]:
            right = set(values[other_key])
            if right:
                scores.append(len(left & right) / len(left | right))
    return scores
