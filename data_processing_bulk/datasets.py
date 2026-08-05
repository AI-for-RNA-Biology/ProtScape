"""Dataset loaders used by the single-cell pseudobulk pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import scanpy as sc

from . import utils
from .config import HBCA_GENE_METADATA, HBCA_INTERMEDIATE, TABULA_H5AD, TABULA_METADATA
from .ensembl_to_hgnc_converter import fetch_symbols_from_gprofiler as _fetch_symbols


logger = logging.getLogger(__name__)


def fetch_symbols_from_gprofiler(gene_ids) -> dict[str, str]:
    """Query HGNC mappings, returning an empty mapping if the optional client is absent."""
    try:
        return _fetch_symbols(gene_ids)
    except RuntimeError as exc:
        logger.warning("%s", exc)
        return {}


def load_tabula_sapiens(
    *,
    cell_type_col: str = "cell_ontology_class",
    donor_col: str = "donor",
    tissue_col: str = "organ_tissue",
) -> sc.AnnData:
    """Load Tabula Sapiens with the integer count matrix from ``raw.X``."""
    adata = utils.read_h5ad(TABULA_H5AD)
    if TABULA_METADATA:
        extra = pd.read_csv(TABULA_METADATA)
        if "cell_id" not in extra:
            raise KeyError("Tabula metadata must include 'cell_id'")
        extra = extra.set_index("cell_id")
        new_columns = extra.columns.difference(adata.obs.columns)
        adata.obs = adata.obs.join(extra[new_columns], how="left")

    required = {cell_type_col, donor_col, tissue_col}
    missing = required - set(adata.obs.columns)
    if missing:
        raise KeyError(f"Tabula AnnData is missing columns: {sorted(missing)}")
    if adata.raw is not None:
        adata.X = adata.raw.X
    return adata


def attach_hbca_gene_symbols(
    adata: sc.AnnData,
    *,
    gene_metadata_path: str | None = None,
) -> None:
    """Add ``adata.var['gene_symbol']`` from local metadata or g:Profiler."""
    if "gene_symbol" in adata.var:
        return

    metadata_path = Path(gene_metadata_path or HBCA_GENE_METADATA)
    symbols = None
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        id_column = next(
            (name for name in ("gene_identifier", "gene_id", "ensembl_id") if name in metadata),
            None,
        )
        symbol_column = next(
            (name for name in ("gene_symbol", "symbol", "gene_name", "feature_name") if name in metadata),
            None,
        )
        if id_column and symbol_column:
            lookup = (
                metadata.drop_duplicates(id_column)
                .set_index(id_column)[symbol_column]
            )
            symbols = lookup.reindex(adata.var_names)

    if symbols is None or symbols.isna().all():
        source_column = next(
            (name for name in ("symbol", "feature_name", "Gene", "gene_name") if name in adata.var),
            None,
        )
        if source_column:
            symbols = adata.var[source_column]
        else:
            names = adata.var_names.astype(str)
            names_are_symbols = sum(not name.startswith("ENS") for name in names[:100]) > 90
            if names_are_symbols:
                symbols = pd.Series(names, index=adata.var_names)
            else:
                lookup = fetch_symbols_from_gprofiler(names)
                if not lookup:
                    logger.warning("No HBCA gene-symbol mapping was available")
                    return
                symbols = pd.Series(names, index=adata.var_names).map(
                    lambda gene_id: lookup.get(gene_id) or lookup.get(gene_id.split(".", 1)[0])
                )

    adata.var["gene_symbol"] = pd.Series(symbols, index=adata.var_names).astype(str)


def load_hbca_merged(
    *,
    annotation_col: str = "hbca_supercluster",
    donor_col: str = "donor_id",
    tissue_col: str = "tissue",
) -> sc.AnnData:
    """Load the merged HBCA object produced by step 0."""
    path = Path(HBCA_INTERMEDIATE) / "all_single_cell.h5ad"
    if not path.exists():
        raise FileNotFoundError(f"Run step0_merge_single_cell.py first; missing {path}")

    adata = utils.read_h5ad(str(path))
    attach_hbca_gene_symbols(adata)
    required = {annotation_col, donor_col, tissue_col}
    missing = required - set(adata.obs.columns)
    if missing:
        raise KeyError(f"HBCA AnnData is missing columns: {sorted(missing)}")
    return adata
