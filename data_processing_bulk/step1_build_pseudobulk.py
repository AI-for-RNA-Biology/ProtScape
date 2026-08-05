#!/usr/bin/env python
"""Step 1: sum raw counts into cell-type pseudobulks for Tabula and HBCA."""

from __future__ import annotations

import argparse
import logging
from functools import lru_cache
from pathlib import Path

import obonet
import pandas as pd

from . import datasets, utils
from .config import CL_PATH, HBCA_INTERMEDIATE, TABULA_INTERMEDIATE, resolve_params


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _save_outputs(output_dir: str, pseudobulk, metadata: pd.DataFrame) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pseudobulk.write_h5ad(output / "pseudobulk.h5ad")
    metadata.to_csv(output / "pseudobulk_metadata.csv")


@lru_cache(None)
def _load_cl_name_to_id_map(path: str) -> dict[str, str]:
    """Map Cell Ontology names and synonyms to CL identifiers."""
    graph = obonet.read_obo(path)
    mapping = {}
    for cl_id, attributes in graph.nodes(data=True):
        name = attributes.get("name")
        if isinstance(name, str) and name:
            mapping[name.lower()] = cl_id
        for synonym in attributes.get("synonym", []):
            if isinstance(synonym, str):
                parts = synonym.split('"')
                if len(parts) >= 2:
                    mapping.setdefault(parts[1].lower(), cl_id)
    return mapping


def run_tabula(args: argparse.Namespace) -> None:
    adata = datasets.load_tabula_sapiens(
        cell_type_col=args.cell_type_col,
        donor_col=args.donor_col,
        tissue_col=args.tissue_col,
    )
    adata = utils.standardize_gene_annotations(adata)

    cell_type_names = adata.obs[args.cell_type_col].astype(str)
    cl_ids = cell_type_names.str.lower().map(_load_cl_name_to_id_map(CL_PATH))
    unmapped = cl_ids.isna()
    if unmapped.any():
        logger.warning(
            "Removing %d Tabula cells from %d cell types without a CL mapping",
            unmapped.sum(),
            cell_type_names[unmapped].nunique(),
        )
        adata = adata[~unmapped].copy()
        cell_type_names = adata.obs[args.cell_type_col].astype(str)
        cl_ids = cell_type_names.str.lower().map(_load_cl_name_to_id_map(CL_PATH))

    adata.obs["cell_type_ontology_term_id"] = cl_ids
    adata.obs["cell_type_name"] = cell_type_names
    adata.obs["cell_type_label"] = pd.Series(
        [utils.format_cl_label(name, cl_id) for name, cl_id in zip(cell_type_names, cl_ids)],
        index=adata.obs_names,
        dtype="string",
    )

    label_map = pd.DataFrame(
        {
            "original_cell_ontology_class": cell_type_names.str.strip(),
            "cl_id": cl_ids,
        }
    ).dropna(subset=["cl_id"])
    mapping_path = Path(TABULA_INTERMEDIATE) / "tabula_celltype_to_cl_map.csv"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    label_map.drop_duplicates("original_cell_ontology_class").to_csv(mapping_path, index=False)

    pseudobulk, metadata = utils.aggregate_pseudobulk(
        adata,
        dataset="TabulaSapiens",
        cell_type_col="cell_type_label",
        donor_col=args.donor_col,
        tissue_col=args.tissue_col,
        min_cells=args.min_cells_resolved,
        group_by_donor=args.group_by_donor,
    )
    _save_outputs(TABULA_INTERMEDIATE, pseudobulk, metadata)


def run_hbca(args: argparse.Namespace) -> None:
    adata = datasets.load_hbca_merged(
        annotation_col=args.cell_type_col,
        donor_col=args.donor_col,
        tissue_col=args.tissue_col,
    )
    adata = utils.standardize_gene_annotations(adata)
    cl_column = "hbca_cell_type_ontology_term_id"
    if cl_column not in adata.obs:
        raise KeyError(f"HBCA AnnData is missing {cl_column}; run step 0 first")
    n_generic = int((adata.obs[cl_column] == "CL:0000540").sum())
    if n_generic:
        raise RuntimeError(f"Step 0 left {n_generic} cells with the generic neuronal CL ID")

    pseudobulk, metadata = utils.aggregate_pseudobulk(
        adata,
        dataset="HBCA",
        cell_type_col=cl_column,
        donor_col=args.donor_col,
        tissue_col=args.tissue_col,
        min_cells=args.min_cells_resolved,
        group_by_donor=args.group_by_donor,
    )
    _save_outputs(HBCA_INTERMEDIATE, pseudobulk, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tabula", "hbca"], required=True)
    parser.add_argument("--cell-type-col")
    parser.add_argument("--donor-col")
    parser.add_argument("--tissue-col")
    parser.add_argument("--min-cells", type=int)
    parser.add_argument(
        "--group-by-donor",
        action="store_true",
        help="Build donor-level rather than pooled cell-type pseudobulks.",
    )
    args = parser.parse_args()

    defaults = {
        "tabula": ("cell_ontology_class", "donor", "organ_tissue"),
        "hbca": ("hbca_supercluster", "donor_id", "tissue"),
    }
    default_cell_type, default_donor, default_tissue = defaults[args.dataset]
    args.cell_type_col = args.cell_type_col or default_cell_type
    args.donor_col = args.donor_col or default_donor
    args.tissue_col = args.tissue_col or default_tissue
    args.min_cells_resolved = resolve_params(
        args.dataset,
        min_cells_per_sample=args.min_cells,
    ).min_cells_per_sample

    if args.dataset == "tabula":
        run_tabula(args)
    else:
        run_hbca(args)


if __name__ == "__main__":
    main()
