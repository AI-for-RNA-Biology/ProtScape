#!/usr/bin/env python
"""Step 7: connect CCI cell types to their BTO tissue hierarchy."""

import argparse
import logging
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import networkx as nx
import obonet
import pandas as pd

from .config import (
    ALS_ASTRO_INTERMEDIATE,
    ALS_BULK_DIR,
    ALS_MN_INTERMEDIATE,
    BTO_PATH,
    HBCA_CCI_EDGELIST,
    HBCA_INTERMEDIATE,
    MERGED_CCI_EDGELIST,
    MERGED_INTERMEDIATE,
    TABULA_CCI_EDGELIST,
    TABULA_INTERMEDIATE,
    TABULA_METADATA,
)

MANUAL_TISSUE_MAPPING = {
    "mammary": "mammary gland",
    "fat": "adipose tissue",
    "prostate": "prostate gland",
    "thalamic complex": "thalamus",
    "cerebral nuclei": "basal ganglion",
    "myelencephalon": "medulla oblongata",
}

SKIP_TISSUES = {
    "pooled",
    "mixed",
    "unknown",
    "na",
    "n/a",
}

PPI_COLUMNS = ["index", "cell_type", "genes"]


class DatasetFiles(NamedTuple):
    ppi_dir: Path
    cci_file: Path
    metadata_file: Path
    annotation_col: str
    tissue_col: str
    output_file: Path


def normalize_cl_label(label: str) -> str:
    """Normalize Cell Ontology identifiers to CL_XXXX format while preserving suffixes."""
    if not isinstance(label, str):
        return label
    cleaned = label.strip()
    return "CL_" + cleaned[3:] if cleaned.upper().startswith(("CL:", "CL_")) else cleaned


def load_celltype_ppi(ppi_dir: Path) -> List[Tuple[int, str]]:
    list_path = Path(ppi_dir) / "ppi_celltype_list.csv"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing PPI cell type list: {list_path}")

    df = pd.read_csv(list_path, sep="\t", header=None, names=PPI_COLUMNS)
    records = [
        (int(idx), normalize_cl_label(str(cell_type)))
        for idx, cell_type in zip(df["index"], df["cell_type"])
    ]
    print("Number of cell types with PPI networks:", len(records))
    return records


def filter_cci(cci_file: Path, celltype_ppi: Iterable[Tuple[int, str]]) -> nx.MultiGraph:
    cci = nx.read_edgelist(cci_file, delimiter="\t", create_using=nx.MultiGraph)
    node_mapping = {}
    for node in cci.nodes():
        normalized = normalize_cl_label(node)
        if normalized != node:
            node_mapping[node] = normalized
    if node_mapping:
        cci = nx.relabel_nodes(cci, node_mapping, copy=True)
    print(f"CCI before PPI filtering: {cci.number_of_nodes()} nodes, {cci.number_of_edges()} edges")

    celltypes_with_ppi = [normalize_cl_label(cell_type) for _, cell_type in celltype_ppi]
    celltypes_in_cci = [ct for ct in celltypes_with_ppi if ct in cci]
    cci = cci.subgraph(celltypes_in_cci)
    print(f"CCI after PPI filtering: {cci.number_of_nodes()} nodes, {cci.number_of_edges()} edges")

    return cci


def _load_tabula_label_map(ppi_dir: Optional[Path]) -> Dict[str, str]:
    if ppi_dir is None:
        return {}
    mapping_file = Path(ppi_dir).parent / "tabula_celltype_to_cl_map.csv"
    if not mapping_file.exists():
        return {}
    df_map = pd.read_csv(mapping_file)
    df_map = df_map.dropna(subset=["cl_id", "original_cell_ontology_class"])
    label_map = {}
    for _, row in df_map.iterrows():
        cl_id = normalize_cl_label(str(row["cl_id"]))
        label = str(row["original_cell_ontology_class"]).strip()
        if not cl_id or not label:
            continue
        label_map[label] = cl_id
        label_map[label.lower()] = cl_id
    print(f"Loaded {len(label_map)//2} Tabula label mappings from {mapping_file}")
    return label_map


def read_tissue_metadata(
    metadata_path: Path,
    annotation_col: str,
    celltypes: Iterable[str],
    tissue_col: str = "organ_tissue",
    ppi_dir: Optional[Path] = None,
    dataset: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], List[str]]:
    df = pd.read_csv(metadata_path, sep="\t" if Path(metadata_path).suffix == ".tsv" else ",")
    celltypes = sorted({normalize_cl_label(str(ct)) for ct in celltypes})
    dataset_name = (dataset or "").strip().lower()

    cl_id_to_human: Dict[str, str] = {}
    human_to_cl: Dict[str, str] = {}
    tabula_label_to_cl: Dict[str, str] = {}
    if ppi_dir:
        edgelist_dir = Path(ppi_dir) / "ppi_edgelists"
        if edgelist_dir.exists():
            for filename in edgelist_dir.glob("*.txt"):
                match = re.search(r'\[cl[_:](\d+)\]', filename.stem, re.IGNORECASE)
                if match:
                    cl_id_underscore = normalize_cl_label(f"CL_{match.group(1)}")
                    human_name = re.sub(r'_?\[cl[_:]\d+\]', '', filename.stem, flags=re.IGNORECASE)
                    human_name = human_name.replace('_', ' ').strip()
                    cl_id_to_human[cl_id_underscore] = human_name
                    if human_name:
                        human_to_cl.setdefault(human_name.lower(), cl_id_underscore)
            print(
                "Built CL_ID -> human name mapping for "
                f"{len(cl_id_to_human)} cell types from PPI filenames"
            )
        if dataset_name == "tabula":
            tabula_label_to_cl = _load_tabula_label_map(ppi_dir)

    if tissue_col not in df.columns:
        alternatives = ["tissue", "organ_tissue", "tissue_name", "roi"]
        for alt in alternatives:
            if alt in df.columns:
                logging.info("Using '%s' as tissue column ('%s' not found)", alt, tissue_col)
                tissue_col = alt
                break
        else:
            raise KeyError(
                f"Tissue column '{tissue_col}' not found in metadata. "
                f"Available columns: {df.columns.tolist()}"
            )

    df["_raw_celltype"] = df[annotation_col].astype(str).str.strip()
    metadata_to_cci: Dict[str, str] = {}

    if dataset_name == "merged":
        for cci_label in celltypes:
            metadata_to_cci[cci_label] = cci_label
            metadata_to_cci[cci_label.lower()] = cci_label
    elif dataset_name in {"tabula", "hbca"}:
        for cci_label in celltypes:
            metadata_to_cci[cci_label] = cci_label
            if cci_label.startswith("CL_"):
                metadata_to_cci[cci_label.replace("CL_", "CL:", 1)] = cci_label
            if dataset_name == "tabula":
                human_name = cl_id_to_human.get(cci_label)
                if human_name:
                    metadata_to_cci[human_name.lower()] = cci_label
        if tabula_label_to_cl:
            for label, cl_id in tabula_label_to_cl.items():
                metadata_to_cci[label] = cl_id
    elif dataset_name == "als":
        for cci_label in celltypes:
            metadata_to_cci[cci_label] = cci_label
            if cci_label.startswith("CL_"):
                metadata_to_cci[cci_label.replace("CL_", "CL:", 1)] = cci_label
    else:
        for cci_label in celltypes:
            metadata_to_cci[cci_label.lower()] = cci_label

    celltype2tissue: Dict[str, List[str]] = {}
    matched_count = 0
    unmatched_sample: List[str] = []

    fallback_added = 0

    for cell_type in df["_raw_celltype"].dropna().astype(str).unique():
        cell_type_stripped = cell_type.strip()
        normalized_cell_type = normalize_cl_label(cell_type_stripped)
        cell_type_lower = cell_type_stripped.lower()
        lookup_keys = [
            cell_type_stripped,
            normalized_cell_type,
            cell_type_lower,
            normalize_cl_label(cell_type_lower),
        ]
        if cell_type_stripped.startswith("CL:"):
            lookup_keys.append(cell_type_stripped.replace("CL:", "CL_", 1))
        if normalized_cell_type.startswith("CL_"):
            lookup_keys.append(normalized_cell_type.replace("CL_", "CL:", 1))
        lookup_keys = list(dict.fromkeys(key for key in lookup_keys if key))

        cci_label = None
        for key in lookup_keys:
            cci_label = metadata_to_cci.get(key)
            if cci_label:
                break

        if not cci_label and dataset_name == "tabula":
            base_name = cell_type_lower.split(",")[0].strip()
            base_candidates = [base_name, normalize_cl_label(base_name)]
            for key in base_candidates:
                cci_label = metadata_to_cci.get(key)
                if cci_label:
                    break
            if not cci_label:
                for cl_id, human_name in cl_id_to_human.items():
                    if human_name.lower() == cell_type_lower:
                        cci_label = metadata_to_cci.get(cl_id)
                        if cci_label:
                            break

        fallback_label = None
        if not cci_label and normalized_cell_type.startswith("CL_"):
            fallback_label = normalized_cell_type
        elif not cci_label and cell_type_lower in human_to_cl:
            fallback_label = human_to_cl[cell_type_lower]
        elif not cci_label:
            base_name = cell_type_lower.split(",")[0].strip()
            if base_name in human_to_cl:
                fallback_label = human_to_cl[base_name]

        if not cci_label and fallback_label:
            cci_label = fallback_label
            fallback_added += 1

        if not cci_label:
            if len(unmatched_sample) < 10:
                unmatched_sample.append(cell_type)
            continue

        matched_count += 1

        tissues = (
            df.loc[df["_raw_celltype"] == cell_type, tissue_col]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        if cci_label in celltype2tissue:
            celltype2tissue[cci_label] = list(
                dict.fromkeys(celltype2tissue[cci_label] + tissues)
            )
        else:
            celltype2tissue[cci_label] = tissues

    print(f"Matched {matched_count} unique cell types from metadata to CCI nodes")
    if unmatched_sample:
        sample_preview = sorted(set(unmatched_sample))
        print(f"Sample unmatched cell types: {sample_preview}")
    print(
        f"Total celltype2tissue mappings: {len(celltype2tissue)} "
        f"(fallback added: {fallback_added})"
    )

    unique_tissues = (
        df[tissue_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    return celltype2tissue, unique_tissues


def all_descendants(G: nx.DiGraph, term: str) -> set:
    """Return a BTO term and its ancestors (OBO edges point child to parent)."""
    ancestors = nx.descendants(G, term)
    if "BTO:0000000" in G and "BTO:0000000" not in ancestors:
        undirected = G.to_undirected()
        if nx.has_path(undirected, term, "BTO:0000000"):
            ancestors.add("BTO:0000000")
    return ancestors | {term}


def extract_bto(unique_tissues: Iterable[str]) -> Tuple[nx.MultiDiGraph, Dict[str, str]]:
    unique_tissues = list(unique_tissues)
    bto_G = obonet.read_obo(BTO_PATH)
    print(f"BTO: {bto_G.number_of_nodes()} nodes, {bto_G.number_of_edges()} edges")

    name2bto = {data.get("name", "").lower(): node_id for node_id, data in bto_G.nodes(data=True)}

    bto_nodes = set()
    tissue2bto: Dict[str, str] = {}
    skipped_tissues = []
    for tissue in unique_tissues:
        if pd.isna(tissue):
            continue
        orig = tissue
        key = " ".join(str(tissue).split("_")).lower()

        if key in SKIP_TISSUES:
            skipped_tissues.append(orig)
            continue

        if key not in name2bto:
            key = MANUAL_TISSUE_MAPPING.get(key, key)
        if key not in name2bto:
            raise KeyError(f"Tissue '{orig}' not found in BTO (after manual mapping). "
                          f"Available tissues: {sorted(list(name2bto.keys())[:20])}")

        bto_id = name2bto[key]
        tissue2bto[orig] = bto_id
        bto_nodes.update(all_descendants(bto_G, bto_id))

    if skipped_tissues:
        print(f"Skipped {len(skipped_tissues)} non-anatomical tissues: {skipped_tissues}")

    bto_subgraph = bto_G.subgraph(bto_nodes).copy()

    bto_mapping = {node: node.replace(":", "_") for node in bto_subgraph.nodes()}
    bto_subgraph = nx.relabel_nodes(bto_subgraph, bto_mapping, copy=True)
    tissue2bto = {tissue: bto_mapping[bto_id] for tissue, bto_id in tissue2bto.items()}

    print("Number of tissues (input)", len(unique_tissues))
    print("Number of tissues (mapped)", len(tissue2bto))
    print("Number of BTO nodes", len(bto_subgraph.nodes()))
    print("Number of BTO edges", len(bto_subgraph.edges()))
    return bto_subgraph, tissue2bto


def create_ct_graph(
    celltype2tissue: Dict[str, List[str]],
    tissue2bto: Dict[str, str],
) -> List[Tuple[str, str]]:
    ct_edges: List[Tuple[str, str]] = []
    skipped_ct_tissue = 0
    duplicate_edges = 0
    dedup_set = set()
    for cell_type, tissues in celltype2tissue.items():
        for tissue in tissues:
            if tissue not in tissue2bto:
                skipped_ct_tissue += 1
                continue
            edge = (cell_type, tissue2bto[tissue])
            if edge in dedup_set:
                duplicate_edges += 1
                continue
            dedup_set.add(edge)
            ct_edges.append(edge)
    if skipped_ct_tissue > 0:
        print(f"Skipped {skipped_ct_tissue} cell-tissue edges with unmapped tissues")
    if duplicate_edges > 0:
        print(f"Deduplicated {duplicate_edges} repeated cell-tissue edges")
    print("Number of cell-tissue edges:", len(ct_edges))
    return ct_edges


def write_metagraph(
    cci: nx.MultiGraph,
    bto: nx.MultiGraph,
    ct_edges: List[Tuple[str, str]],
    outfile: Path,
) -> nx.Graph:
    metagraph = nx.compose(nx.Graph(cci), nx.Graph(bto))
    metagraph.add_edges_from(ct_edges)
    print(
        f"Metagraph: {metagraph.number_of_nodes()} nodes, "
        f"{metagraph.number_of_edges()} edges"
    )

    nx.write_edgelist(metagraph, outfile, data=False, delimiter="\t")
    print(f"Saved metagraph to {outfile}")
    return metagraph


def _reset_edgelist_dir(ppi_dir: Path) -> Path:
    ppi_dir.mkdir(parents=True, exist_ok=True)
    edgelist_dir = ppi_dir / "ppi_edgelists"
    if edgelist_dir.exists():
        shutil.rmtree(edgelist_dir)
    edgelist_dir.mkdir()
    return edgelist_dir


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _read_ppi_list(ppi_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        ppi_dir / "ppi_celltype_list.csv",
        sep="\t",
        header=None,
        names=PPI_COLUMNS,
    )


def _prepare_als_files() -> DatasetFiles:
    root = Path(ALS_BULK_DIR)
    ppi_dir = root / "ppi"
    edgelist_dir = _reset_edgelist_dir(ppi_dir)
    source_dirs = [
        Path(ALS_MN_INTERMEDIATE) / "ppi",
        Path(ALS_ASTRO_INTERMEDIATE) / "ppi",
    ]

    mn_df, astro_df = (_read_ppi_list(source_dir) for source_dir in source_dirs)
    astro_df["index"] += len(mn_df)
    pd.concat([mn_df, astro_df], ignore_index=True).to_csv(
        ppi_dir / "ppi_celltype_list.csv",
        sep="\t",
        header=False,
        index=False,
    )
    logging.info(
        "Combined %d motor neuron and %d astrocyte PPI networks",
        len(mn_df),
        len(astro_df),
    )

    for source_dir in source_dirs:
        for source in sorted((source_dir / "ppi_edgelists").glob("*.txt")):
            _link_or_copy(source, edgelist_dir / source.name)

    metadata = pd.read_csv(root / "als_combined_metadata.csv")
    if "tissue" not in metadata.columns:
        metadata["tissue"] = "spinal cord"
    metadata_file = root / "als_metagraph_metadata.csv"
    metadata.to_csv(metadata_file, index=False)

    return DatasetFiles(
        ppi_dir,
        root / "cci_edgelist.txt",
        metadata_file,
        "cell_type_id",
        "tissue",
        root / "metagraph.txt",
    )


def _ppi_filename_lookup(edgelist_dir: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for path in sorted(edgelist_dir.glob("*.txt")):
        match = re.search(r"\[cl[_:](\d+)\]", path.stem, re.IGNORECASE)
        if match:
            lookup[normalize_cl_label(f"CL_{match.group(1)}")] = path.name
            human_name = re.sub(
                r"_?\[cl[_:]\d+\]",
                "",
                path.stem,
                flags=re.IGNORECASE,
            ).replace("_", " ").strip()
            if human_name:
                lookup[human_name] = path.name
        elif path.stem.startswith(("CL_", "CL:")):
            lookup[normalize_cl_label(path.stem)] = path.name
        else:
            lookup[path.stem] = path.name
            lookup[path.stem.replace("_", " ")] = path.name
    return lookup


def _collect_merged_ppi_records(
    source_ppi_dir: Path,
    prefix: str,
    output_edgelist_dir: Path,
) -> List[Tuple[str, object]]:
    source_list = source_ppi_dir / "ppi_celltype_list.csv"
    if not source_list.exists():
        raise FileNotFoundError(f"PPI list not found for {prefix}: {source_list}")

    records: List[Tuple[str, object]] = []
    source_edgelists = source_ppi_dir / "ppi_edgelists"
    filename_lookup = _ppi_filename_lookup(source_edgelists)
    for _, row in _read_ppi_list(source_ppi_dir).iterrows():
        original = str(row["cell_type"])
        normalized = normalize_cl_label(original)
        source_filename = next(
            (
                filename_lookup[candidate]
                for candidate in (normalized, original, original.replace("_", " "))
                if candidate in filename_lookup
            ),
            None,
        )

        target_celltype = normalized
        if source_filename:
            match = re.search(r"\[cl[_:](\d+)\]", source_filename, re.IGNORECASE)
            if match:
                target_celltype = normalize_cl_label(f"CL_{match.group(1)}")

        merged_celltype = f"{prefix}__{target_celltype}"
        records.append((merged_celltype, row["genes"]))
        if source_filename:
            source = source_edgelists / source_filename
            if source.exists():
                _link_or_copy(source, output_edgelist_dir / f"{merged_celltype}.txt")
            else:
                logging.warning("Missing PPI edgelist for %s: %s", original, source)
        else:
            logging.warning("No PPI edgelist matched %s cell type '%s'", prefix, original)
    return records


def _prepare_merged_ppis(ppi_dir: Path) -> None:
    output_edgelists = _reset_edgelist_dir(ppi_dir)
    records: List[Tuple[str, object]] = []
    for source_dir, prefix in [
        (Path(HBCA_INTERMEDIATE) / "ppi", "HBCA"),
        (Path(TABULA_INTERMEDIATE) / "ppi", "TABULA"),
    ]:
        records.extend(_collect_merged_ppi_records(source_dir, prefix, output_edgelists))

    with (ppi_dir / "ppi_celltype_list.csv").open("w") as handle:
        for index, (celltype, genes) in enumerate(records):
            handle.write(f"{index}\t{celltype}\t{genes}\n")


def _normalize_tabula_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    cleaned = unicodedata.normalize("NFKD", str(value))
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii").lower().strip()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", cleaned)).strip("_")


def _create_merged_metadata() -> Path:
    hbca = pd.read_csv(Path(HBCA_INTERMEDIATE) / "all_single_cell_metadata.csv")
    hbca["merged_cell_type"] = "HBCA__" + hbca[
        "hbca_cell_type_ontology_term_id"
    ].apply(normalize_cl_label)
    hbca["merged_tissue"] = hbca["ROIGroupCoarse"]

    tabula = pd.read_csv(TABULA_METADATA)
    mapping_file = Path(TABULA_INTERMEDIATE) / "tabula_celltype_to_cl_map.csv"
    tabula_mapping: Dict[str, str] = {}
    if mapping_file.exists():
        mapping = pd.read_csv(mapping_file)
        if {"normalized_name", "cl_id"}.issubset(mapping.columns):
            tabula_mapping = {
                str(row["normalized_name"]): normalize_cl_label(str(row["cl_id"]))
                for _, row in mapping.iterrows()
                if pd.notna(row["normalized_name"]) and pd.notna(row["cl_id"])
            }
    else:
        logging.warning("Tabula mapping file not found: %s", mapping_file)

    tabula["__norm_name"] = tabula["cell_ontology_class"].apply(_normalize_tabula_label)
    tabula["__cl_from_mapping"] = tabula["__norm_name"].map(tabula_mapping)
    if "cell_ontology_term_id" in tabula.columns:
        tabula["__cl_from_meta"] = tabula["cell_ontology_term_id"].apply(
            lambda value: normalize_cl_label(value) if pd.notna(value) else ""
        )
    else:
        tabula["__cl_from_meta"] = ""
    tabula["__final_cl"] = tabula["__cl_from_meta"]
    tabula.loc[tabula["__final_cl"] == "", "__final_cl"] = tabula["__cl_from_mapping"]
    tabula = tabula[tabula["__final_cl"] != ""].copy()
    tabula["merged_cell_type"] = "TABULA__" + tabula["__final_cl"]
    tabula["merged_tissue"] = tabula["organ_tissue"]

    merged = pd.concat(
        [
            hbca[["merged_cell_type", "merged_tissue"]],
            tabula[["merged_cell_type", "merged_tissue"]],
        ],
        ignore_index=True,
    )
    metadata_file = Path(MERGED_INTERMEDIATE) / "merged_metadata.csv"
    merged.to_csv(metadata_file, index=False)
    return metadata_file


def _prepare_merged_files() -> DatasetFiles:
    root = Path(MERGED_INTERMEDIATE)
    ppi_dir = root / "ppi"
    _prepare_merged_ppis(ppi_dir)
    return DatasetFiles(
        ppi_dir,
        Path(MERGED_CCI_EDGELIST),
        _create_merged_metadata(),
        "merged_cell_type",
        "merged_tissue",
        root / "metagraph.txt",
    )


def _prepare_dataset_files(dataset: str) -> DatasetFiles:
    if dataset == "tabula":
        root = Path(TABULA_INTERMEDIATE)
        return DatasetFiles(
            root / "ppi",
            Path(TABULA_CCI_EDGELIST),
            Path(TABULA_METADATA),
            "cell_ontology_class",
            "organ_tissue",
            root / "metagraph.txt",
        )
    if dataset == "hbca":
        root = Path(HBCA_INTERMEDIATE)
        metadata_file = root / "all_single_cell_metadata.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(f"HBCA single-cell metadata not found: {metadata_file}")
        return DatasetFiles(
            root / "ppi",
            Path(HBCA_CCI_EDGELIST),
            metadata_file,
            "hbca_cell_type_ontology_term_id",
            "ROIGroupCoarse",
            root / "metagraph.txt",
        )
    if dataset == "als":
        return _prepare_als_files()
    if dataset == "merged":
        return _prepare_merged_files()
    raise ValueError(f"Unknown dataset {dataset}")


def process_dataset(dataset: str) -> None:
    files = _prepare_dataset_files(dataset)
    logging.info("Processing %s metagraph", dataset.upper())

    celltype_ppi = load_celltype_ppi(files.ppi_dir)
    ppi_celltypes = sorted({normalize_cl_label(cell_type) for _, cell_type in celltype_ppi})
    cci = filter_cci(files.cci_file, celltype_ppi)

    celltype2tissue, unique_tissues = read_tissue_metadata(
        files.metadata_file,
        files.annotation_col,
        ppi_celltypes,
        files.tissue_col,
        ppi_dir=files.ppi_dir,
        dataset=dataset,
    )
    missing_celltypes = sorted(ct for ct in ppi_celltypes if not celltype2tissue.get(ct))
    if missing_celltypes:
        logging.warning(
            "Missing tissue assignments for %d PPI-backed cell types. Examples: %s",
            len(missing_celltypes),
            ", ".join(missing_celltypes[:10]) + (" ..." if len(missing_celltypes) > 10 else ""),
        )

    bto_subgraph, tissue2bto = extract_bto(unique_tissues)
    ct_edges = create_ct_graph(celltype2tissue, tissue2bto)

    write_metagraph(cci, bto_subgraph, ct_edges, files.output_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct metagraphs from PPIs, CCIs, and tissue mappings."
    )
    parser.add_argument(
        "--dataset",
        choices=["tabula", "hbca", "als", "merged", "all"],
        default="tabula",
        help="Dataset to process (or 'all' to run sequentially).",
    )
    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args)
    datasets = ["tabula", "hbca", "als", "merged"] if parsed.dataset == "all" else [parsed.dataset]
    for ds in datasets:
        process_dataset(ds)


if __name__ == "__main__":
    main()
