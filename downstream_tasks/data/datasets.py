# datasets.py
# Dataset classes for downstream tasks

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class ABMILDataset(Dataset):
    """
    Dataset for ABMIL models with context bags and optional ESM.

    Each sample consists of:
    - A context bag: variable-size set of cell-type specific embeddings for a gene
    - An ESM embedding: fixed-size sequence-based embedding for the gene
    - Labels: multi-label classification targets
    """

    def __init__(
        self,
        ctx_bags: List[np.ndarray],
        esm_vecs: Optional[np.ndarray],
        labels: np.ndarray,
    ):
        """
        Args:
            ctx_bags: List of numpy arrays, each of shape (n_cells, ctx_dim)
            esm_vecs: Optional array of shape (n_samples, esm_dim), can be None for context-only
            labels: Array of shape (n_samples, n_classes)
        """
        self.ctx_bags = [torch.as_tensor(bag, dtype=torch.float32) for bag in ctx_bags]
        self.esm_vecs = (
            torch.as_tensor(esm_vecs, dtype=torch.float32) if esm_vecs is not None else None
        )
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict, torch.Tensor]:
        ctx = self.ctx_bags[idx]
        item = {"ctx": ctx}
        if self.esm_vecs is not None:
            item["esm"] = self.esm_vecs[idx]
        else:
            item["esm"] = None
        return item, self.labels[idx]


def collate_abmil(batch: List[Tuple[Dict, torch.Tensor]]) -> Tuple[Dict, torch.Tensor]:
    """
    Collate function for ABMIL batches.

    Handles variable-size context bags by keeping them as a list.

    Args:
        batch: List of (x_dict, label) tuples

    Returns:
        Tuple of (collated_dict, labels_tensor)
    """
    items = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])

    ctx_list = [i["ctx"] for i in items]
    bag_sizes = torch.tensor([ctx.shape[0] for ctx in ctx_list], dtype=torch.long)
    ctx_ptr = torch.empty(len(ctx_list) + 1, dtype=torch.long)
    ctx_ptr[0] = 0
    ctx_ptr[1:] = torch.cumsum(bag_sizes, dim=0)
    ctx_batch = torch.repeat_interleave(torch.arange(len(ctx_list), dtype=torch.long), bag_sizes)
    ctx = torch.cat(ctx_list, dim=0)

    if items[0]["esm"] is not None:
        esm = torch.stack([i["esm"] for i in items])
    else:
        esm = None

    out = {"ctx": ctx, "ctx_batch": ctx_batch, "ctx_ptr": ctx_ptr, "esm": esm}

    return out, labels


class LinearDataset(Dataset):
    """
    Dataset for linear probe models.

    Each sample consists of:
    - Features: fixed-size feature vector (pooled embeddings)
    - Labels: multi-label classification targets
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: Array of shape (n_samples, feature_dim)
            labels: Array of shape (n_samples, n_classes)
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def subset_bags(bags: List[np.ndarray], indices: np.ndarray) -> List[np.ndarray]:
    """
    Subset a list of bags by indices.

    Args:
        bags: List of numpy arrays
        indices: Array of indices to select

    Returns:
        Subsetted list of bags
    """
    return [bags[i] for i in indices]
