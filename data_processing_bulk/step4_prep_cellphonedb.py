#!/usr/bin/env python
"""Step 4: prepare CellPhoneDB metadata and count matrices."""

from __future__ import annotations

import argparse
import logging
import pickle
import random
import re
import unicodedata
from pathlib import Path

import obonet
import pandas as pd
import scanpy as sc

from .config import (
    ALS_ASTRO_INTERMEDIATE,
    ALS_BULK_DIR,
    ALS_MN_INTERMEDIATE,
    CL_PATH,
    HBCA_CELLPHONEDB_INPUT,
    HBCA_GENE_METADATA,
    HBCA_INTERMEDIATE,
    MERGED_CELLPHONEDB_INPUT,
    TABULA_CELLPHONEDB_INPUT,
    TABULA_H5AD,
    TABULA_INTERMEDIATE,
)
from .ensembl_to_hgnc_converter import convert_ensembl_to_hgnc


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 7

DATASETS = {
    "tabula": {
        "source": Path(TABULA_H5AD),
        "output": Path(TABULA_CELLPHONEDB_INPUT),
        "groupby": ["cell_type"],
    },
    "hbca": {
        "source": Path(HBCA_INTERMEDIATE) / "all_single_cell.h5ad",
        "output": Path(HBCA_CELLPHONEDB_INPUT),
        "groupby": [
            "hbca_cell_type_ontology_term_id",
            "hbca_celltype_label",
            "hbca_cl_id",
            "cell_type_ontology_term_id",
            "cell_type",
            "hbca_supercluster",
        ],
    },
    "als": {
        "output": Path(ALS_BULK_DIR) / "cellphonedb_input",
        "groupby": ["cell_type_id", "cell_type_name", "cl_id"],
    },
    "merged": {
        "output": Path(MERGED_CELLPHONEDB_INPUT),
        "groupby": ["cell_type"],
    },
}


def extract_cl_id(cell_type: object) -> str:
    """Standardize Cell Ontology identifiers to ``CL_...`` labels."""
    label = str(cell_type).strip()
    if re.match(r"^CL:\d+_[A-Z]+_", label):
        return label.replace("CL:", "CL_", 1)
    match = re.search(r"\[CL:(\d+)\]", label, flags=re.IGNORECASE)
    if match:
        return f"CL_{match.group(1)}"
    if re.match(r"^CL:\d+$", label):
        return label.replace(":", "_")
    return label


def normalize_tabula_label(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower().strip()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text)).strip("_")


def _convert_hbca_genes(adata: sc.AnnData) -> sc.AnnData:
    symbols = convert_ensembl_to_hgnc(
        pd.Series(adata.var_names),
        metadata_path=HBCA_GENE_METADATA,
    )
    valid = symbols.notna().to_numpy()
    converted = adata[:, valid].copy()
    converted.var_names = symbols[valid].to_numpy()
    logger.info("Mapped %d/%d HBCA genes to HGNC symbols", converted.n_vars, adata.n_vars)
    return converted


def _choose_groupby(adata: sc.AnnData, candidates: list[str], dataset: str) -> str:
    groupby = next((column for column in candidates if column in adata.obs), None)
    if groupby is None:
        raise KeyError(f"No cell-type column found for {dataset}; tried {candidates}")
    return groupby


def _ppi_cell_types(dataset: str, output_dir: Path) -> set[str]:
    paths = list((output_dir.parent / "ppi" / "ppi_edgelists").glob("*.txt"))
    if dataset == "hbca":
        return {
            f"CL:{match.group(1)}"
            for path in paths
            if (match := re.search(r"\[CL[_:](\d+)\]", path.stem, flags=re.IGNORECASE))
        }
    if dataset == "tabula":
        return {
            f"CL_{match.group(1)}"
            for path in paths
            if (match := re.search(r"\[CL[_:](\d+)\]", path.stem, flags=re.IGNORECASE))
        }
    return {
        re.sub(r"_\[cl[_:]\d+\]$", "", path.stem, flags=re.IGNORECASE).replace("_", " ")
        for path in paths
    }


def _filter_to_ppi_cell_types(
    adata: sc.AnnData,
    dataset: str,
    groupby: str,
    output_dir: Path,
) -> sc.AnnData:
    if dataset not in {"hbca", "tabula"}:
        return adata
    ppi_types = _ppi_cell_types(dataset, output_dir)
    if not ppi_types:
        logger.warning("No PPI edgelists found for %s; retaining all cells", dataset)
        return adata

    filtered = adata[adata.obs[groupby].isin(ppi_types)].copy()
    if filtered.n_obs == 0:
        observed = sorted(adata.obs[groupby].astype(str).unique())[:10]
        raise ValueError(
            f"PPI filtering removed every {dataset} cell; "
            f"PPI labels={sorted(ppi_types)[:10]}, observed labels={observed}"
        )
    logger.info("PPI filtering retained %d/%d %s cells", filtered.n_obs, adata.n_obs, dataset)
    return filtered


def _cap_cells(
    adata: sc.AnnData,
    groupby: str,
    maximum: int | None,
    rng: random.Random | None = None,
) -> sc.AnnData:
    if maximum is None:
        return adata

    rng = rng or random.Random(RANDOM_SEED)
    keep = []
    for positions in adata.obs.groupby(adata.obs[groupby]).indices.values():
        positions = list(positions)
        keep.extend(rng.sample(positions, maximum) if len(positions) > maximum else positions)
    capped = adata[sorted(keep)].copy()
    logger.info("Cell cap retained %d/%d cells", capped.n_obs, adata.n_obs)
    return capped


def _cell_type_labels(adata: sc.AnnData, dataset: str, groupby: str):
    labels = adata.obs[groupby].astype(str)
    if dataset == "tabula":
        return labels.to_numpy()
    mapping = {label: extract_cl_id(label) for label in labels.unique()}
    return labels.map(mapping).to_numpy()


def _write_cellphonedb_inputs(
    adata: sc.AnnData,
    dataset: str,
    groupby: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / f"meta_{dataset}_CellPhoneDB.txt"
    pd.DataFrame(
        {"Cell": adata.obs_names, "cell_type": _cell_type_labels(adata, dataset, groupby)}
    ).to_csv(meta_path, sep="\t", index=False)

    counts = adata.raw.to_adata() if dataset == "tabula" and adata.raw is not None else adata
    if counts is not adata:
        counts.obs = adata.obs.copy()
    pickle_path = output_dir / f"counts_{dataset}_CellPhoneDB.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(
            {
                "X": counts.X,
                "obs_index": counts.obs_names.tolist(),
                "var_index": counts.var_names.tolist(),
            },
            handle,
        )
    logger.info(
        "Prepared %s CellPhoneDB inputs: %d cells, %d genes",
        dataset,
        counts.n_obs,
        counts.n_vars,
    )
    return meta_path, pickle_path


def prepare_inputs(
    adata: sc.AnnData,
    *,
    dataset: str,
    groupby: str,
    output_dir: Path,
    max_cells_per_type: int | None,
) -> tuple[Path, Path]:
    """Filter, deterministically cap, and serialize one dataset for CellPhoneDB."""
    if groupby not in adata.obs:
        raise KeyError(f"Cell-type column '{groupby}' is absent")
    adata = _filter_to_ppi_cell_types(adata, dataset, groupby, output_dir)
    adata = _cap_cells(adata, groupby, max_cells_per_type)
    return _write_cellphonedb_inputs(adata, dataset, groupby, output_dir)


def _combine_als() -> sc.AnnData:
    motor_neurons = sc.read_h5ad(Path(ALS_MN_INTERMEDIATE) / "als_motor_neurons.h5ad")
    astrocytes = sc.read_h5ad(Path(ALS_ASTRO_INTERMEDIATE) / "als_astrocytes.h5ad")
    combined = sc.concat([motor_neurons, astrocytes], join="outer", merge="same")
    combined.obs["cell_type_id"] = combined.obs_names

    output = Path(ALS_BULK_DIR)
    output.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(output / "als_combined.h5ad")
    combined.obs.to_csv(output / "als_combined_metadata.csv", index=False)
    logger.info("Combined ALS data: %d contexts, %d genes", combined.n_obs, combined.n_vars)
    return combined


def _inventory_mapping(path: Path) -> tuple[dict[str, str], set[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Tabula PPI inventory not found: {path}")

    mapping = {}
    cl_ids = set()
    with path.open() as handle:
        for line in handle:
            filename = Path(line.strip()).name
            match = re.match(r"(.+)\[(cl[_:]?\d+)\]\.txt", filename, flags=re.IGNORECASE)
            if not match:
                continue
            name = normalize_tabula_label(match.group(1).strip("_"))
            cl_id = match.group(2).upper().replace("_", ":")
            mapping[name] = cl_id
            cl_ids.add(cl_id)
    return mapping, cl_ids


def _ontology_mapping(cl_ids: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    graph = obonet.read_obo(CL_PATH)
    aliases = {}
    primary_names = {}
    for cl_id in sorted(cl_ids):
        if cl_id not in graph:
            continue
        attributes = graph.nodes[cl_id]
        candidates = []
        primary_name = attributes.get("name")
        if isinstance(primary_name, str):
            candidates.append(primary_name)
            primary_names[cl_id] = primary_name
        for synonym in attributes.get("synonym", []):
            match = re.match(r'"(.+)"', synonym) if isinstance(synonym, str) else None
            if match:
                candidates.append(match.group(1))
        for candidate in candidates:
            aliases.setdefault(normalize_tabula_label(candidate), cl_id)
    return aliases, primary_names


def _map_tabula_cell_types(adata: sc.AnnData) -> sc.AnnData:
    inventory_path = Path(TABULA_INTERMEDIATE) / "ppi" / "inventory.txt"
    name_to_cl_id, inventory_cl_ids = _inventory_mapping(inventory_path)
    ontology_aliases, cl_names = _ontology_mapping(inventory_cl_ids)
    for name, cl_id in ontology_aliases.items():
        name_to_cl_id.setdefault(name, cl_id)

    label_column = "cell_ontology_class"
    if label_column not in adata.obs:
        raise KeyError(f"Tabula AnnData is missing '{label_column}'")
    labels = adata.obs[label_column].astype(str)
    cl_column = next(
        (
            column
            for column in (
                "cell_ontology_term_id",
                "cell_ontology_id",
                "cell_type_ontology_term_id",
            )
            if column in adata.obs
        ),
        None,
    )

    if cl_column:
        mapped = adata.obs[cl_column].map(extract_cl_id)
        records = [
            {
                "original_cell_ontology_class": str(label),
                "normalized_name": normalize_tabula_label(label),
                "cl_id": str(cl_id),
                "cl_name": cl_names.get(str(cl_id), ""),
            }
            for label, cl_id in zip(labels, mapped)
        ]
    else:
        def map_name(label: str) -> str | None:
            upper = label.upper().strip()
            if upper.startswith(("CL:", "CL_")):
                return extract_cl_id(label)
            normalized = normalize_tabula_label(label)
            cl_id = name_to_cl_id.get(normalized)
            if cl_id is None:
                cl_id = name_to_cl_id.get(normalized.replace("_cell", ""))
            return extract_cl_id(cl_id) if cl_id else None

        unique_mapping = {label: map_name(label) for label in labels.unique()}
        mapped = labels.map(unique_mapping)
        records = [
            {
                "original_cell_ontology_class": str(label),
                "normalized_name": normalize_tabula_label(label),
                "cl_id": cl_id,
                "cl_name": cl_names.get(str(cl_id), "") if cl_id else "",
            }
            for label, cl_id in unique_mapping.items()
        ]

    adata.obs["cell_type"] = mapped
    adata = adata[adata.obs["cell_type"].notna()].copy()
    pd.DataFrame(records).to_csv(
        Path(TABULA_INTERMEDIATE) / "tabula_celltype_to_cl_map.csv",
        index=False,
    )
    adata.obs["cell_type"] = "TABULA__" + adata.obs["cell_type"].astype(str)
    return adata


def _sanitize_obs(adata: sc.AnnData) -> None:
    if "is_primary_data" in adata.obs:
        adata.obs["is_primary_data"] = adata.obs["is_primary_data"].astype(str)
    for column in adata.obs:
        if adata.obs[column].dtype == "object":
            adata.obs[column] = adata.obs[column].astype(str)


def _combine_merged(max_cells_per_type: int | None) -> sc.AnnData:
    hbca = sc.read_h5ad(DATASETS["hbca"]["source"])
    hbca_groupby = _choose_groupby(hbca, DATASETS["hbca"]["groupby"], "hbca")
    hbca.obs["cell_type"] = "HBCA__" + hbca.obs[hbca_groupby].map(extract_cl_id).astype(str)

    tabula = _map_tabula_cell_types(sc.read_h5ad(DATASETS["tabula"]["source"]))
    if max_cells_per_type is not None:
        rng = random.Random(RANDOM_SEED)
        hbca = _cap_cells(hbca, "cell_type", max_cells_per_type, rng)
        tabula = _cap_cells(tabula, "cell_type", max_cells_per_type, rng)

    hbca = _convert_hbca_genes(hbca)
    hbca.var_names_make_unique()
    tabula.var_names_make_unique()
    common_genes = sorted(set(hbca.var_names) & set(tabula.var_names))
    combined = sc.concat(
        [hbca[:, common_genes], tabula[:, common_genes]],
        join="outer",
        merge="same",
    )
    _sanitize_obs(combined)

    logger.info("Combined HBCA and Tabula: %d cells, %d genes", combined.n_obs, combined.n_vars)
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="tabula")
    parser.add_argument("--max-cells-per-type", type=int)
    return parser


def main(args: list[str] | None = None) -> None:
    parsed = build_parser().parse_args(args)
    config = DATASETS[parsed.dataset]

    if parsed.dataset == "als":
        adata = _combine_als()
    elif parsed.dataset == "merged":
        adata = _combine_merged(parsed.max_cells_per_type)
    else:
        adata = sc.read_h5ad(config["source"])
        if parsed.dataset == "hbca":
            groupby = _choose_groupby(adata, config["groupby"], parsed.dataset)
            adata = _filter_to_ppi_cell_types(adata, parsed.dataset, groupby, config["output"])
            adata = _cap_cells(adata, groupby, parsed.max_cells_per_type)
            adata = _convert_hbca_genes(adata)
            _write_cellphonedb_inputs(adata, parsed.dataset, groupby, config["output"])
            return
        elif parsed.dataset == "tabula":
            adata = _map_tabula_cell_types(adata)
            adata.obs["cell_type"] = adata.obs["cell_type"].str.replace(
                r"^TABULA__", "", regex=True
            )

    groupby = _choose_groupby(adata, config["groupby"], parsed.dataset)
    prepare_inputs(
        adata,
        dataset=parsed.dataset,
        groupby=groupby,
        output_dir=config["output"],
        max_cells_per_type=parsed.max_cells_per_type,
    )


if __name__ == "__main__":
    main()
