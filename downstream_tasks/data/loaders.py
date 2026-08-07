# loaders.py
# Embedding loaders for sequence and HC embeddings

import gc
import ast
import pickle
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_esm_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """
    Load sequence embeddings from pickle file.

    Args:
        path: Path to sequence embeddings pickle file (ESM or ProstT5)

    Returns:
        Dictionary mapping gene names (uppercase) to embedding vectors
    """
    with open(path, "rb") as f:
        esm_raw = pickle.load(f)

    esm_dict = {}
    if isinstance(esm_raw, pd.DataFrame):
        if "gene_name" not in esm_raw.columns:
            raise KeyError("Missing required column 'gene_name' in sequence embedding dataframe")

        # Support ESM and ProstT5 output table formats.
        path_name = str(path).lower()
        if "prostt5" in path_name:
            candidates = ("ProstT5-Embeddings", "ESM2-Embeddings")
        else:
            candidates = ("ESM2-Embeddings", "ProstT5-Embeddings")

        emb_col = None
        for candidate in candidates:
            if candidate in esm_raw.columns:
                emb_col = candidate
                break
        if emb_col is None:
            raise KeyError(
                "Could not find embedding column. Expected one of: "
                "'ESM2-Embeddings', 'ProstT5-Embeddings'"
            )

        for _, row in esm_raw.iterrows():
            emb = row[emb_col]
            if emb is None:
                continue
            arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            esm_dict[str(row["gene_name"]).upper()] = arr
    else:
        esm_dict = {str(k).upper(): np.array(v, dtype=np.float32) for k, v in esm_raw.items()}

    return esm_dict


@lru_cache(maxsize=4)
def load_pinnacle_paper_labels(
    labels_path: str,
) -> Tuple[Dict[int, str], Dict[str, List[str]], Tuple[Tuple[int, int, str], ...]]:
    """Parse original PINNACLE paper labels stored as a Python-literal text dict."""
    labels = ast.literal_eval(Path(labels_path).read_text())
    cell_types = [str(x) for x in labels["Cell Type"]]
    names = [str(x) for x in labels["Name"]]

    mg_end_idx = next(
        (
            i
            for i, ct in enumerate(cell_types)
            if not (ct.startswith("CCI") or ct.startswith("BTO"))
        ),
        len(cell_types),
    )

    cell_rows = []
    for label_idx, cell_type in enumerate(cell_types[:mg_end_idx]):
        if cell_type.startswith("CCI"):
            clean_name = cell_type.replace("CCI_", "", 1)
            cell_rows.append((len(cell_rows), label_idx, clean_name))

    celltype_to_proteins: Dict[str, List[str]] = defaultdict(list)
    for cell_type, name in zip(cell_types[mg_end_idx:], names[mg_end_idx:]):
        if not cell_type.startswith("Sanity"):
            celltype_to_proteins[cell_type].append(name.upper())

    idx_to_celltype = {cell_id: cell_name for cell_id, _, cell_name in cell_rows}
    return idx_to_celltype, dict(celltype_to_proteins), tuple(cell_rows)


def load_pinnacle_paper_gene_universe(labels_path: Path) -> Set[str]:
    """Return the original PINNACLE paper protein universe."""
    _, celltype_to_proteins, _ = load_pinnacle_paper_labels(str(labels_path))
    return {
        protein
        for proteins in celltype_to_proteins.values()
        for protein in proteins
    }


def load_pinnacle_paper_protein_embeddings(embed_path: Path, labels_path: Path) -> Dict:
    """
    Adapt original PINNACLE paper embeddings to the HC embedding schema.

    The paper files store embeddings as {cell_idx: tensor} and labels as a
    flattened text dictionary. Downstream loaders expect protein_names by cell.
    """
    raw_embed = torch.load(embed_path, map_location="cpu", weights_only=True)
    idx_to_celltype, celltype_to_proteins, _ = load_pinnacle_paper_labels(str(labels_path))

    embeddings = {}
    protein_names = {}
    cell_ids = []
    cell_names = []

    def _cell_sort_key(raw_cell_id):
        try:
            return int(raw_cell_id)
        except (TypeError, ValueError):
            return raw_cell_id

    for raw_cell_id in sorted(raw_embed, key=_cell_sort_key):
        try:
            cell_id = int(raw_cell_id)
        except (TypeError, ValueError):
            continue
        embs = raw_embed[raw_cell_id]

        cell_name = idx_to_celltype.get(cell_id)
        if cell_name is None:
            continue

        proteins = celltype_to_proteins.get(cell_name, [])
        if not proteins:
            continue

        if isinstance(embs, torch.Tensor) and embs.ndim == 1:
            embs = embs.unsqueeze(0)
        elif not isinstance(embs, torch.Tensor):
            embs = np.asarray(embs)
            if embs.ndim == 1:
                embs = embs[None, :]

        n = min(int(embs.shape[0]), len(proteins))
        if n == 0:
            continue

        embeddings[cell_id] = embs[:n]
        protein_names[cell_id] = proteins[:n]
        cell_ids.append(cell_id)
        cell_names.append(cell_name)

    return {
        "embeddings": embeddings,
        "protein_names": protein_names,
        "cell_ids": cell_ids,
        "cell_names": cell_names,
        "dataset_mode": "legacy",
        "model_type": "pinnacle_paper",
    }


def load_pinnacle_paper_cell_embeddings(embed_path: Path, labels_path: Path) -> Dict:
    """Adapt original PINNACLE metagraph embeddings to the HC cell schema."""
    raw_embed = torch.load(embed_path, map_location="cpu", weights_only=True)
    mg_embeddings = raw_embed.get("embeddings") if isinstance(raw_embed, dict) else raw_embed
    if mg_embeddings is None:
        raise ValueError(f"No metagraph embeddings found in {embed_path}")

    _, _, cell_rows = load_pinnacle_paper_labels(str(labels_path))
    cell_ids = []
    cell_names = []
    cell_vecs = []

    for cell_id, label_idx, cell_name in cell_rows:
        if label_idx >= len(mg_embeddings):
            continue
        cell_ids.append(cell_id)
        cell_names.append(cell_name)
        cell_vecs.append(mg_embeddings[label_idx])

    if isinstance(mg_embeddings, torch.Tensor):
        embeddings = torch.stack(cell_vecs, dim=0) if cell_vecs else mg_embeddings[:0]
        embed_dim = int(embeddings.shape[1]) if embeddings.ndim > 1 else 0
    else:
        embeddings = (
            np.stack(cell_vecs, axis=0)
            if cell_vecs
            else np.empty((0, 0), dtype=np.float32)
        )
        embed_dim = int(embeddings.shape[1]) if embeddings.ndim > 1 else 0

    return {
        "embeddings": embeddings,
        "cell_ids": cell_ids,
        "cell_names": cell_names,
        "embed_dim": embed_dim,
        "n_cells": len(cell_ids),
        "dataset_mode": "legacy",
        "model_type": "pinnacle_paper",
    }


def build_gene_contexts_from_hc(
    hc_protein_embed: Dict,
    target_genes: Optional[Set[str]] = None,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[str]]]:
    """
    Build gene context vectors from HC (Hierarchical Cell) embeddings.

    Args:
        hc_protein_embed: HC protein embeddings dictionary with 'embeddings' and 'protein_names'
        target_genes: Optional set of target genes to filter

    Returns:
        Tuple of (gene_to_vecs, gene_to_cells) where gene_to_cells tracks cell IDs for stratification
    """
    gene_to_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    gene_to_cells: Dict[str, List[str]] = defaultdict(list)

    embs_by_cell = hc_protein_embed.get("embeddings", {})
    names_by_cell = hc_protein_embed.get("protein_names", {})

    for cell_id, embs in tqdm(embs_by_cell.items(), desc="Indexing HC contexts"):
        names = names_by_cell.get(cell_id) or names_by_cell.get(str(cell_id))
        if names is None:
            continue

        arr = embs.detach().cpu().numpy() if isinstance(embs, torch.Tensor) else np.asarray(embs)
        n = min(arr.shape[0], len(names))

        for i in range(n):
            gene = str(names[i]).upper()
            if target_genes is None or gene in target_genes:
                gene_to_vecs[gene].append(arr[i].astype(np.float32))
                gene_to_cells[gene].append(str(cell_id))

    return dict(gene_to_vecs), dict(gene_to_cells)


def load_cell_tissue_mapping(data_root: Path) -> Tuple[Dict[str, List[int]], List[str]]:
    """Load cell context to tissue ids from the bulk metagraph metadata."""
    mapping_path = data_root / "networks_bulk" / "celltype_tissue_mapping_long.csv"
    if not mapping_path.exists():
        class_path = data_root / "networks_bulk" / "celltype_class_mapping.csv"
        if not class_path.exists():
            raise FileNotFoundError(f"No cell tissue/class mapping found under {data_root}")
        df = pd.read_csv(class_path).fillna("")
        group_col = "cell_type_class"
        cell_col = "edgelist"
    else:
        df = pd.read_csv(mapping_path).fillna("")
        group_col = "tissue_name"
        cell_col = "edgelist"

    group_names = sorted({str(x).strip() for x in df[group_col] if str(x).strip()})
    group_to_id = {name: idx for idx, name in enumerate(group_names)}
    cell_to_groups: Dict[str, List[int]] = defaultdict(list)
    for _, row in df.iterrows():
        cell = str(row[cell_col]).strip()
        group = str(row[group_col]).strip()
        if not cell or not group:
            continue
        group_id = group_to_id[group]
        if group_id not in cell_to_groups[cell]:
            cell_to_groups[cell].append(group_id)

    return dict(cell_to_groups), group_names


def build_gene_contexts_from_hc_with_tissue_groups(
    hc_protein_embed: Dict,
    data_root: Path,
    target_genes: Optional[Set[str]] = None,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[int]], List[str]]:
    """Build gene context vectors and duplicate each context into its tissue groups."""
    cell_to_groups, group_names = load_cell_tissue_mapping(data_root)

    cell_ids = hc_protein_embed.get("cell_ids", [])
    cell_names = hc_protein_embed.get("cell_names", [])
    cell_id_to_name = {str(cell_id): str(name) for cell_id, name in zip(cell_ids, cell_names)}

    gene_to_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    gene_to_groups: Dict[str, List[int]] = defaultdict(list)

    embs_by_cell = hc_protein_embed.get("embeddings", {})
    names_by_cell = hc_protein_embed.get("protein_names", {})
    unknown_group = len(group_names)
    group_names = group_names + ["Unknown"]

    for cell_id, embs in tqdm(embs_by_cell.items(), desc="Indexing HC tissue contexts"):
        names = names_by_cell.get(cell_id) or names_by_cell.get(str(cell_id))
        if names is None:
            continue

        cell_name = cell_id_to_name.get(str(cell_id), str(cell_id))
        tissue_ids = cell_to_groups.get(cell_name, [unknown_group])

        arr = embs.detach().cpu().numpy() if isinstance(embs, torch.Tensor) else np.asarray(embs)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        n = min(arr.shape[0], len(names))

        for i in range(n):
            gene = str(names[i]).upper()
            if target_genes is None or gene in target_genes:
                vec = arr[i]
                for tissue_id in tissue_ids:
                    gene_to_vecs[gene].append(vec)
                    gene_to_groups[gene].append(tissue_id)

    return dict(gene_to_vecs), dict(gene_to_groups), group_names


def build_gene_contexts_from_hc_with_cell_and_tissue(
    hc_protein_embed: Dict,
    hc_cell_embed: Dict,
    data_root: Path,
    target_genes: Optional[Set[str]] = None,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[int]], List[str]]:
    """Build protein+cell+tissue-one-hot contexts and tissue ids."""
    cell_to_groups, group_names = load_cell_tissue_mapping(data_root)

    cell_ids = hc_protein_embed.get("cell_ids", [])
    cell_names = hc_protein_embed.get("cell_names", [])
    cell_id_to_name = {str(cell_id): str(name) for cell_id, name in zip(cell_ids, cell_names)}

    unknown_group = len(group_names)
    group_names = group_names + ["Unknown"]
    tissue_eye = np.eye(len(group_names), dtype=np.float32)

    cell_emb_tensor = hc_cell_embed.get("embeddings")
    cell_embed_ids = hc_cell_embed.get("cell_ids", [])
    if isinstance(cell_emb_tensor, torch.Tensor):
        cell_emb_tensor = cell_emb_tensor.detach().cpu().numpy()
    cell_id_to_idx = {str(cell_id): idx for idx, cell_id in enumerate(cell_embed_ids)}

    gene_to_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    gene_to_groups: Dict[str, List[int]] = defaultdict(list)

    embs_by_cell = hc_protein_embed.get("embeddings", {})
    names_by_cell = hc_protein_embed.get("protein_names", {})

    for cell_id, embs in tqdm(embs_by_cell.items(), desc="Indexing HC+Cell+Tissue contexts"):
        names = names_by_cell.get(cell_id) or names_by_cell.get(str(cell_id))
        if names is None:
            continue

        cell_idx = cell_id_to_idx.get(str(cell_id))
        if cell_idx is None:
            continue
        cell_vec = cell_emb_tensor[cell_idx].astype(np.float32, copy=False)

        cell_name = cell_id_to_name.get(str(cell_id), str(cell_id))
        tissue_ids = cell_to_groups.get(cell_name, [unknown_group])

        arr = embs.detach().cpu().numpy() if isinstance(embs, torch.Tensor) else np.asarray(embs)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        n = min(arr.shape[0], len(names))

        for i in range(n):
            gene = str(names[i]).upper()
            if target_genes is None or gene in target_genes:
                for tissue_id in tissue_ids:
                    gene_to_vecs[gene].append(
                        np.concatenate([arr[i], cell_vec, tissue_eye[tissue_id]], axis=0)
                    )
                    gene_to_groups[gene].append(tissue_id)

    return dict(gene_to_vecs), dict(gene_to_groups), group_names


def build_gene_means_from_hc(
    hc_protein_embed: Dict,
    target_genes: Optional[Set[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Build mean gene context vectors from HC embeddings without storing all contexts.

    Args:
        hc_protein_embed: HC protein embeddings dictionary with 'embeddings' and 'protein_names'
        target_genes: Optional set of target genes to filter

    Returns:
        Tuple of (gene_to_mean, gene_to_cells) where gene_to_cells tracks cell IDs
    """
    gene_sum: Dict[str, np.ndarray] = {}
    gene_count: Dict[str, int] = {}
    gene_to_cells: Dict[str, List[str]] = defaultdict(list)

    embs_by_cell = hc_protein_embed.get("embeddings", {})
    names_by_cell = hc_protein_embed.get("protein_names", {})

    for cell_id, embs in tqdm(embs_by_cell.items(), desc="Indexing HC means"):
        names = names_by_cell.get(cell_id) or names_by_cell.get(str(cell_id))
        if names is None:
            continue

        arr = embs.detach().cpu().numpy() if isinstance(embs, torch.Tensor) else np.asarray(embs)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        n = min(arr.shape[0], len(names))

        for i in range(n):
            gene = str(names[i]).upper()
            if target_genes is None or gene in target_genes:
                vec = arr[i]
                if gene in gene_sum:
                    gene_sum[gene] += vec
                    gene_count[gene] += 1
                else:
                    gene_sum[gene] = vec.copy()
                    gene_count[gene] = 1
                gene_to_cells[gene].append(str(cell_id))

    gene_to_mean = {g: (gene_sum[g] / gene_count[g]).astype(np.float32, copy=False) for g in gene_sum}
    return gene_to_mean, dict(gene_to_cells)


def build_gene_contexts_from_hc_with_cell(
    hc_protein_embed: Dict,
    hc_cell_embed: Dict,
    target_genes: Optional[Set[str]] = None,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[str]]]:
    """
    Build gene contexts by concatenating each protein embedding with its corresponding cell embedding.
    This adds cell-level context to each protein instance in the bag.

    Args:
        hc_protein_embed: HC protein embeddings dictionary
        hc_cell_embed: HC cell embeddings dictionary with 'embeddings' and 'cell_ids'
        target_genes: Optional set of target genes to filter

    Returns:
        Tuple of (gene_to_vecs, gene_to_cells) with concatenated protein+cell embeddings
    """
    gene_to_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    gene_to_cells: Dict[str, List[str]] = defaultdict(list)

    embs_by_cell = hc_protein_embed.get("embeddings", {})
    names_by_cell = hc_protein_embed.get("protein_names", {})

    # Get cell embeddings tensor and mapping
    cell_emb_tensor = hc_cell_embed.get("embeddings")
    cell_ids = hc_cell_embed.get("cell_ids", [])

    if cell_emb_tensor is None or len(cell_ids) == 0:
        print("[WARN] No cell embeddings found, falling back to protein-only")
        return build_gene_contexts_from_hc(hc_protein_embed, target_genes)

    # Convert to numpy if needed
    if isinstance(cell_emb_tensor, torch.Tensor):
        cell_emb_tensor = cell_emb_tensor.detach().cpu().numpy()

    # Create cell_id -> cell_embedding index mapping
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

    for cell_id, embs in tqdm(embs_by_cell.items(), desc="Indexing HC+Cell contexts"):
        names = names_by_cell.get(cell_id) or names_by_cell.get(str(cell_id))
        if names is None:
            continue

        # Get cell embedding for this cell type
        cell_idx = cell_id_to_idx.get(cell_id)
        if cell_idx is None:
            cell_idx = cell_id_to_idx.get(str(cell_id))
        if cell_idx is None:
            continue
        cell_vec = cell_emb_tensor[cell_idx].astype(np.float32)

        arr = embs.detach().cpu().numpy() if isinstance(embs, torch.Tensor) else np.asarray(embs)
        n = min(arr.shape[0], len(names))

        for i in range(n):
            gene = str(names[i]).upper()
            if target_genes is None or gene in target_genes:
                prot_vec = arr[i].astype(np.float32)
                # Concatenate protein embedding with cell embedding
                combined = np.concatenate([prot_vec, cell_vec], axis=0)
                gene_to_vecs[gene].append(combined)
                gene_to_cells[gene].append(str(cell_id))

    return dict(gene_to_vecs), dict(gene_to_cells)


def build_gene_means_from_hc_with_cell(
    hc_protein_embed: Dict,
    hc_cell_embed: Dict,
    target_genes: Optional[Set[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Build mean gene contexts by concatenating protein and cell embeddings.

    Args:
        hc_protein_embed: HC protein embeddings dictionary
        hc_cell_embed: HC cell embeddings dictionary with 'embeddings' and 'cell_ids'
        target_genes: Optional set of target genes to filter

    Returns:
        Tuple of (gene_to_mean, gene_to_cells) with mean of protein+cell concatenated
    """
    gene_sum_prot: Dict[str, np.ndarray] = {}
    gene_sum_cell: Dict[str, np.ndarray] = {}
    gene_count: Dict[str, int] = {}
    gene_to_cells: Dict[str, List[str]] = defaultdict(list)

    embs_by_cell = hc_protein_embed.get("embeddings", {})
    names_by_cell = hc_protein_embed.get("protein_names", {})

    cell_emb_tensor = hc_cell_embed.get("embeddings")
    cell_ids = hc_cell_embed.get("cell_ids", [])

    if cell_emb_tensor is None or len(cell_ids) == 0:
        print("[WARN] No cell embeddings found, falling back to protein-only means")
        return build_gene_means_from_hc(hc_protein_embed, target_genes)

    if isinstance(cell_emb_tensor, torch.Tensor):
        cell_emb_tensor = cell_emb_tensor.detach().cpu().numpy()

    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

    for cell_id, embs in tqdm(embs_by_cell.items(), desc="Indexing HC+Cell means"):
        names = names_by_cell.get(cell_id) or names_by_cell.get(str(cell_id))
        if names is None:
            continue

        cell_idx = cell_id_to_idx.get(cell_id)
        if cell_idx is None:
            cell_idx = cell_id_to_idx.get(str(cell_id))
        if cell_idx is None:
            continue
        cell_vec = cell_emb_tensor[cell_idx]
        if cell_vec.dtype != np.float32:
            cell_vec = cell_vec.astype(np.float32, copy=False)

        arr = embs.detach().cpu().numpy() if isinstance(embs, torch.Tensor) else np.asarray(embs)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        n = min(arr.shape[0], len(names))

        for i in range(n):
            gene = str(names[i]).upper()
            if target_genes is None or gene in target_genes:
                prot_vec = arr[i]
                if gene in gene_sum_prot:
                    gene_sum_prot[gene] += prot_vec
                    gene_sum_cell[gene] += cell_vec
                    gene_count[gene] += 1
                else:
                    gene_sum_prot[gene] = prot_vec.copy()
                    gene_sum_cell[gene] = cell_vec.copy()
                    gene_count[gene] = 1
                gene_to_cells[gene].append(str(cell_id))

    gene_to_mean = {
        g: np.concatenate(
            [(gene_sum_prot[g] / gene_count[g]), (gene_sum_cell[g] / gene_count[g])],
            axis=0,
        ).astype(np.float32, copy=False)
        for g in gene_sum_prot
    }
    return gene_to_mean, dict(gene_to_cells)


class EmbeddingLoader:
    """
    Unified loader for sequence and HC embeddings.
    """

    def __init__(
        self,
        esm_path: Path,
        hc_protein_path: Optional[Path] = None,
        hc_cell_path: Optional[Path] = None,
        hc_protein_labels_path: Optional[Path] = None,
        hc_cell_labels_path: Optional[Path] = None,
    ):
        """
        Args:
            esm_path: Path to ESM embeddings pickle
            hc_protein_path: Path to HC protein embeddings (optional)
            hc_cell_path: Path to HC cell embeddings (optional)
            hc_protein_labels_path: Optional original PINNACLE paper labels
            hc_cell_labels_path: Optional original PINNACLE paper labels for cell embeddings
        """
        self.esm_path = esm_path
        self.hc_protein_path = hc_protein_path
        self.hc_cell_path = hc_cell_path
        self.hc_protein_labels_path = hc_protein_labels_path
        self.hc_cell_labels_path = hc_cell_labels_path

        self._esm_dict: Optional[Dict[str, np.ndarray]] = None
        self._hc_protein_embed: Optional[Dict] = None
        self._hc_cell_embed: Optional[Dict] = None

    def load_esm(self) -> Dict[str, np.ndarray]:
        """Load sequence embeddings (cached)."""
        if self._esm_dict is None:
            print(f"Loading sequence embeddings from {self.esm_path}")
            self._esm_dict = load_esm_embeddings(self.esm_path)
            print(f"  Loaded {len(self._esm_dict)} sequence embeddings")
        return self._esm_dict

    def _load_hc_protein_embed(self) -> Dict:
        if self.hc_protein_path is None:
            raise ValueError("HC protein path not configured")

        if self._hc_protein_embed is None:
            print(f"Loading HC protein embeddings from {self.hc_protein_path}")
            if self.hc_protein_labels_path is not None:
                self._hc_protein_embed = load_pinnacle_paper_protein_embeddings(
                    self.hc_protein_path,
                    self.hc_protein_labels_path,
                )
            else:
                self._hc_protein_embed = torch.load(
                    self.hc_protein_path,
                    map_location="cpu",
                    weights_only=True,
                )
        return self._hc_protein_embed

    def _load_hc_cell_embed(self) -> Dict:
        if self.hc_cell_path is None:
            raise ValueError("HC cell path not configured")

        if self._hc_cell_embed is None:
            print(f"Loading HC cell embeddings from {self.hc_cell_path}")
            if self.hc_cell_labels_path is not None:
                self._hc_cell_embed = load_pinnacle_paper_cell_embeddings(
                    self.hc_cell_path,
                    self.hc_cell_labels_path,
                )
            else:
                self._hc_cell_embed = torch.load(
                    self.hc_cell_path,
                    map_location="cpu",
                    weights_only=True,
                )
        return self._hc_cell_embed

    def load_hc(self, target_genes: Optional[Set[str]] = None) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[str]]]:
        """
        Load HC context embeddings.

        Args:
            target_genes: Optional set of target genes to filter

        Returns:
            Tuple of (gene_to_vecs, gene_to_cells)
        """
        return build_gene_contexts_from_hc(self._load_hc_protein_embed(), target_genes)

    def load_hc_tissue_groups(
        self,
        data_root: Path,
        target_genes: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[int]], List[str]]:
        """Load HC context embeddings grouped by bulk tissue metadata."""
        return build_gene_contexts_from_hc_with_tissue_groups(
            self._load_hc_protein_embed(),
            data_root,
            target_genes,
        )

    def load_hc_cell_tissue_features(
        self,
        data_root: Path,
        target_genes: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[int]], List[str]]:
        """Load HC contexts with cell embeddings and tissue one-hot features."""
        return build_gene_contexts_from_hc_with_cell_and_tissue(
            self._load_hc_protein_embed(),
            self._load_hc_cell_embed(),
            data_root,
            target_genes,
        )

    def load_hc_mean(self, target_genes: Optional[Set[str]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
        """
        Load mean HC context embeddings without storing all contexts.

        Args:
            target_genes: Optional set of target genes to filter

        Returns:
            Tuple of (gene_to_mean, gene_to_cells)
        """
        return build_gene_means_from_hc(self._load_hc_protein_embed(), target_genes)

    def load_hc_with_cell(self, target_genes: Optional[Set[str]] = None) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[str]]]:
        """
        Load HC context embeddings with cell embeddings concatenated.

        Args:
            target_genes: Optional set of target genes to filter

        Returns:
            Tuple of (gene_to_vecs, gene_to_cells) with protein+cell concatenated
        """
        return build_gene_contexts_from_hc_with_cell(
            self._load_hc_protein_embed(),
            self._load_hc_cell_embed(),
            target_genes,
        )

    def load_hc_with_cell_mean(self, target_genes: Optional[Set[str]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
        """
        Load mean HC context embeddings with cell embeddings concatenated.

        Args:
            target_genes: Optional set of target genes to filter

        Returns:
            Tuple of (gene_to_mean, gene_to_cells) with protein+cell concatenated
        """
        return build_gene_means_from_hc_with_cell(
            self._load_hc_protein_embed(),
            self._load_hc_cell_embed(),
            target_genes,
        )

    def clear_cache(self, keep_esm: bool = False) -> None:
        """Release cached embedding objects once downstream features are materialized."""
        cleared = False

        if not keep_esm and self._esm_dict is not None:
            self._esm_dict = None
            cleared = True

        if self._hc_protein_embed is not None:
            self._hc_protein_embed = None
            cleared = True

        if self._hc_cell_embed is not None:
            self._hc_cell_embed = None
            cleared = True

        if cleared:
            gc.collect()

    def get_esm_dim(self) -> int:
        """Get ESM embedding dimension."""
        esm = self.load_esm()
        if esm:
            return next(iter(esm.values())).shape[0]
        raise ValueError("No ESM embeddings loaded")

    def get_hc_dim(self) -> int:
        """Get HC embedding dimension."""
        embs = self._load_hc_protein_embed().get("embeddings", {})
        for cell_id, e in embs.items():
            arr = e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else np.asarray(e)
            return arr.shape[1] if arr.ndim > 1 else arr.shape[0]
        raise ValueError("No HC embeddings found")
