"""Combine reliable-gene lists using common cell labels and gene symbols."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config import HBCA_GENE_METADATA
from .ensembl_to_hgnc_converter import load_symbol_mapping
from .network_merging import normalize_cl_label

logger = logging.getLogger(__name__)

SPECIFIC_FILENAMES = {
    "reliable_genes.json": "reliable_genes_specific.json",
    "reliable_genes_per_celltype.json": "reliable_genes_specific_per_celltype.json",
}


def prefer_specific_file(path: str) -> str:
    """Use the specificity-filtered gene list when it exists beside this file."""
    directory, name = os.path.split(path)
    candidate_name = SPECIFIC_FILENAMES.get(name)
    if candidate_name:
        candidate = os.path.join(directory, candidate_name)
        if os.path.exists(candidate):
            return candidate
    return path


def read_reliable_genes(path: Path) -> Tuple[Set[str], Dict[str, List[str]]]:
    """Read a gene list or cell-to-gene dictionary; normalize CL context labels.

    Context suffixes are retained, so nuclear and cytoplasmic lists stay distinct.
    Returns the gene union and any available per-cell lists.
    """
    if not path.exists():
        return set(), {}
    with path.open() as handle:
        payload = json.load(handle)
    union: Set[str] = set()
    per_cell: Dict[str, List[str]] = {}
    if isinstance(payload, list):
        union.update(gene.strip() for gene in payload if isinstance(gene, str) and gene.strip())
    elif isinstance(payload, dict):
        for cell_type, genes in payload.items():
            genes = [gene.strip() for gene in (genes or []) if isinstance(gene, str) and gene.strip()]
            if not genes:
                continue
            label = str(cell_type).strip()
            full_match = re.match(r"(CL[:_]\d+)([A-Za-z0-9_:.-]*)", label, re.IGNORECASE)
            if full_match:
                base = normalize_cl_label(full_match.group(1))
                suffix = (full_match.group(2) or "").replace(":", "_")
                suffix = re.sub(r"\s+", "_", suffix)
                suffix = re.sub(r"_+", "_", suffix).strip("_")
                label = f"{base}_{suffix}" if suffix else base
            elif label:
                match = re.search(r"(CL[:_]\d+)", label, re.IGNORECASE)
                label = normalize_cl_label(match.group(1) if match else label)
            else:
                label = "unknown_cell_type"
            per_cell.setdefault(label, []).extend(genes)
            union.update(genes)
    return union, per_cell


def merge_reliable_genes_multi(
    genes_files: List[str],
    output_file: str,
    per_cell_output: Optional[str] = None,
) -> Set[str]:
    """Save the merged gene union and optional per-cell lists; return the union.

    Prefer specificity-filtered lists and resolve Ensembl IDs with the configured
    HBCA annotation table. Unmapped Ensembl IDs are retained without versions.
    """
    all_genes: Set[str] = set()
    per_cell_genes: Dict[str, Set[str]] = {}
    for genes_file in genes_files:
        path = Path(prefer_specific_file(genes_file))
        union, per_cell = read_reliable_genes(path)
        if not per_cell:
            cell_path = Path(prefer_specific_file(path.with_name("reliable_genes_per_celltype.json")))
            cell_union, cell_mapping = read_reliable_genes(cell_path)
            if cell_mapping:
                union.update(cell_union)
                per_cell = cell_mapping
        if not union and not per_cell:
            logger.warning("No reliable genes found in %s", path.parent.name)
            continue
        all_genes.update(union)
        for cell_type, genes in per_cell.items():
            per_cell_genes.setdefault(cell_type, set()).update(genes)

    symbol_lookup = {}
    ensembl_ids = {}
    for gene in all_genes:
        clean_id = gene.split(".", 1)[0]
        if clean_id.upper().startswith("ENSG"):
            ensembl_ids[gene] = clean_id.upper()
        else:
            symbol_lookup[gene] = gene.upper()
            symbol_lookup[gene.upper()] = gene.upper()
    if ensembl_ids:
        annotation = load_symbol_mapping(HBCA_GENE_METADATA)
        for original, clean_id in ensembl_ids.items():
            symbol = annotation.get(clean_id) or annotation.get(original)
            symbol = symbol.upper() if isinstance(symbol, str) and symbol else clean_id
            symbol_lookup[original] = symbol
            symbol_lookup[original.upper()] = symbol

    normalized_union = {symbol_lookup[gene] for gene in all_genes}
    normalized_per_cell = {
        cell_type: sorted({symbol_lookup[gene] for gene in genes})
        for cell_type, genes in per_cell_genes.items() if genes
    }
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as handle:
        json.dump(sorted(normalized_union), handle, indent=2)
    if per_cell_output and normalized_per_cell:
        with open(per_cell_output, "w") as handle:
            json.dump(dict(sorted(normalized_per_cell.items())), handle, indent=2)
    logger.info("Saved %d merged reliable genes to %s", len(normalized_union), output_file)
    return normalized_union
