"""Map Ensembl gene identifiers to HGNC symbols."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


logger = logging.getLogger(__name__)


def load_symbol_mapping(metadata_path: str | Path) -> Dict[str, str]:
    """Load a frozen Ensembl-to-HGNC mapping table."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen gene mapping: {path}")

    metadata = pd.read_csv(path)
    id_column = next(
        (name for name in ("gene_identifier", "gene_id", "ensembl_id") if name in metadata),
        None,
    )
    symbol_column = next(
        (name for name in ("gene_symbol", "symbol", "gene_name", "feature_name") if name in metadata),
        None,
    )
    if id_column is None or symbol_column is None:
        raise ValueError(f"Gene mapping must contain Ensembl ID and gene-symbol columns: {path}")

    rows = metadata[[id_column, symbol_column]].dropna().drop_duplicates(id_column)
    mapping = {}
    for gene_id, symbol in rows.itertuples(index=False):
        gene_id = str(gene_id)
        symbol = str(symbol).upper()
        mapping[gene_id] = symbol
        mapping[gene_id.split(".", 1)[0]] = symbol
    return mapping


def fetch_symbols_from_gprofiler(gene_ids: Iterable[str]) -> Dict[str, str]:
    """Return Ensembl-to-HGNC mappings for the supplied human gene IDs."""
    try:
        from gprofiler import GProfiler
    except ImportError as exc:
        raise RuntimeError("gprofiler-official is required for gene mapping") from exc

    original_ids = [gene_id for gene_id in gene_ids if isinstance(gene_id, str)]
    cleaned_ids = [gene_id.split(".", 1)[0] for gene_id in original_ids]
    cleaned_ids = [gene_id for gene_id in cleaned_ids if gene_id.upper().startswith("ENSG")]
    if not cleaned_ids:
        return {}

    try:
        result = GProfiler(return_dataframe=True).convert(
            organism="hsapiens",
            query=cleaned_ids,
            target_namespace="HGNC",
        )
    except Exception as exc:  # remote service errors
        logger.warning("g:Profiler lookup failed: %s", exc)
        return {}

    if result is None or result.empty:
        logger.warning("g:Profiler returned no HGNC mappings")
        return {}

    columns = set(result.columns.astype(str))
    input_col = next((column for column in ("input", "incoming", "query") if column in columns), None)
    symbol_col = next(
        (column for column in ("converted", "to", "name", "target_id") if column in columns),
        None,
    )
    if input_col is None or symbol_col is None:
        logger.warning("Unexpected g:Profiler columns: %s", sorted(columns))
        return {}

    mapping = dict(
        zip(
            result[input_col].astype(str),
            result[symbol_col].astype(str).str.upper(),
        )
    )
    for original_id in original_ids:
        cleaned_id = original_id.split(".", 1)[0]
        if cleaned_id in mapping:
            mapping[original_id] = mapping[cleaned_id]

    logger.info("Mapped %d gene identifiers to HGNC symbols", len(mapping))
    return mapping


def convert_ensembl_to_hgnc(
    ensembl_ids: pd.Series,
    metadata_path: str | Path | None = None,
) -> pd.Series:
    """Map a Series of Ensembl identifiers to HGNC symbols."""
    mapping = (
        load_symbol_mapping(metadata_path)
        if metadata_path is not None
        else fetch_symbols_from_gprofiler(ensembl_ids.tolist())
    )
    symbols = ensembl_ids.map(
        lambda gene_id: mapping.get(str(gene_id))
        or mapping.get(str(gene_id).split(".", 1)[0])
    )
    converted = int(symbols.notna().sum())
    logger.info("Mapped %d/%d Ensembl IDs to HGNC symbols", converted, len(ensembl_ids))
    return symbols


def convert_expression_matrix(
    expression: pd.DataFrame,
    metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    """Replace an Ensembl-indexed expression matrix with unique HGNC rows."""
    symbols = convert_ensembl_to_hgnc(
        pd.Series(expression.index),
        metadata_path=metadata_path,
    )
    converted = expression.copy()
    converted.index = symbols
    converted = converted.loc[converted.index.notna()]

    if converted.index.duplicated().any():
        converted["_mean"] = converted.mean(axis=1)
        converted = converted.sort_values("_mean", ascending=False)
        converted = converted.loc[~converted.index.duplicated(keep="first")]
        converted = converted.drop(columns="_mean")

    logger.info("Retained %d genes with HGNC symbols", len(converted))
    return converted
