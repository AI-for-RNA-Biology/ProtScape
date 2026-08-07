"""Reliable-gene merging."""

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

_SPECIFIC_SUFFIX = {
    "reliable_genes.json": "reliable_genes_specific.json",
    "reliable_genes_per_celltype.json": "reliable_genes_specific_per_celltype.json",
}

def prefer_specific_file(path: str) -> str:
    """If a specificity-filtered variant exists beside the requested file, use it."""
    directory, name = os.path.split(path)
    candidate_name = _SPECIFIC_SUFFIX.get(name)
    if candidate_name:
        candidate = os.path.join(directory, candidate_name)
        if os.path.exists(candidate):
            return candidate
    return path


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
            frozen_mapping = load_symbol_mapping(HBCA_GENE_METADATA)
            for original, cleaned in ensembl_map.items():
                symbol = frozen_mapping.get(cleaned) or frozen_mapping.get(original)
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
