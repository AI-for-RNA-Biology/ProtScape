"""Context-free ProtScape baseline on the released global interactome."""

from __future__ import annotations

import hashlib
import io
import pickle
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch_geometric.data import Data

from .data_handler.generate_input import read_data
from .models.protein_modules import prot_module
from .s2gae_utils import (
    S2GAEDecoder,
    edge_mask_per_graph,
    s2gae_loss,
    structured_negative_sampling_k,
    undirected_decoder_logits,
)

GLOBAL_K_VALUES = (1, 10, 50, 100, 500)
GLOBAL_NEGATIVE_BANK_SIZE = 500
PROTOCOL_VERSION = "global_s2gae_unique_ppi_v3"


def protocol_metadata() -> dict:
    return {
        "version": PROTOCOL_VERSION,
        "selection_metric": "global_val_ap",
        "negative_bank_size": GLOBAL_NEGATIVE_BANK_SIZE,
        "reported_k_values": list(GLOBAL_K_VALUES),
        "negative_scope": "global_reference_nonedge",
        "validation_negative_bank": "fixed_seeded_bank500",
        "optimization": "full_batch_global_ppi",
        "split_unit": "unique_undirected_context_observed_pair",
        "evaluation_unit": "unique_undirected_global_pair",
        "reference_only_edges": "train_only",
        "validation_topology": "train",
        "test_topology": "train_plus_validation",
        "primary_mask_type": "dm",
    }


@dataclass
class GlobalPPIData:
    features: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    protein_names: list[str]
    all_edge_index: torch.Tensor
    train_edge_index: torch.Tensor
    train_val_edge_index: torch.Tensor
    val_edge_index: torch.Tensor
    test_edge_index: torch.Tensor
    source_context_count: int
    graph_fingerprint: str
    feature_fingerprint: str
    split_fingerprint: str
    split_counts: dict[str, int]


def _symmetric_edges(edge_index: torch.Tensor) -> torch.Tensor:
    return torch.cat([edge_index, edge_index.flip(0)], dim=1).contiguous()


def _canonical_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long()
    left = torch.minimum(edge_index[0], edge_index[1])
    right = torch.maximum(edge_index[0], edge_index[1])
    return left * int(num_nodes) + right


def _fingerprint(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _assign_context_splits(
    unique_edges: torch.Tensor,
    contexts: dict[int, Data],
    num_nodes: int,
) -> np.ndarray:
    keys = _canonical_keys(unique_edges, num_nodes).numpy()
    order = np.argsort(keys)
    sorted_keys = keys[order]
    split_codes = np.full(keys.size, -1, dtype=np.int8)

    for graph in contexts.values():
        local_to_global = graph.feature_index.detach().cpu().numpy()
        local_edges = graph.edge_index.detach().cpu().numpy()
        global_left = local_to_global[local_edges[0]]
        global_right = local_to_global[local_edges[1]]
        local_keys = (
            np.minimum(global_left, global_right) * num_nodes
            + np.maximum(global_left, global_right)
        )

        for code, mask_name in enumerate(("train_mask", "val_mask", "test_mask")):
            mask = getattr(graph, mask_name).detach().cpu().numpy().astype(bool)
            selected = np.unique(local_keys[mask])
            positions = np.searchsorted(sorted_keys, selected)
            if np.any(positions >= sorted_keys.size) or not np.array_equal(
                sorted_keys[positions], selected
            ):
                raise ValueError(f"{mask_name} contains an edge outside the global PPI.")
            global_positions = order[positions]
            previous = split_codes[global_positions]
            if np.any((previous >= 0) & (previous != code)):
                raise ValueError("An undirected PPI pair occurs in more than one split.")
            split_codes[global_positions] = code

    return split_codes


def _verify_edge_counts(
    count_path: Path,
    protein_names: Sequence[str],
    unique_edges: torch.Tensor,
    split_codes: np.ndarray,
) -> None:
    with count_path.open("rb") as handle:
        edge_counts = pickle.load(handle)

    name_to_index = {name: index for index, name in enumerate(protein_names)}
    expected = []
    num_nodes = len(protein_names)
    for (left, right), count in edge_counts.items():
        if count <= 0 or left not in name_to_index or right not in name_to_index:
            continue
        left_index = name_to_index[left]
        right_index = name_to_index[right]
        expected.append(
            min(left_index, right_index) * num_nodes + max(left_index, right_index)
        )

    observed_mask = torch.from_numpy(split_codes >= 0)
    observed = _canonical_keys(unique_edges[:, observed_mask], num_nodes).numpy()
    if not np.array_equal(np.sort(np.unique(expected)), np.sort(np.unique(observed))):
        raise ValueError("Context-observed edges do not match count_edge_dict.pkl.")


def load_global_ppi_data(
    networks_dir: Path,
    esm2_embeddings: Path,
    *,
    seed: int = 0,
    verbose: bool = True,
) -> GlobalPPIData:
    """Load released inputs and preserve the paper's leakage-controlled split."""
    networks_dir = Path(networks_dir).expanduser()
    esm2_embeddings = Path(esm2_embeddings).expanduser()
    required = {
        "global PPI": networks_dir / "global_ppi_edgelist.txt",
        "context PPIs": networks_dir / "ppi_edgelists",
        "metagraph": networks_dir / "mg_edgelist.txt",
        "edge counts": networks_dir / "count_edge_dict.pkl",
        "ESM2 embeddings": esm2_embeddings,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing released input(s): " + "; ".join(missing))

    with ExitStack() as stack:
        if not verbose:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
        loaded = read_data(
            required["global PPI"],
            required["context PPIs"],
            required["metagraph"],
            get_CT_map=True,
            ppi_feat_dir=esm2_embeddings,
            symmetric_ppi=True,
            dataset_mode="bulk",
            split_mode="global",
            count_edge_path=required["edge counts"],
            weighted_ppi_loss=False,
            defer_ppi_features=True,
            seed=seed,
            verbose=verbose,
        )
    contexts, mg_data, _, _, _, _, _, _ = loaded
    protein_names = [str(name) for name in mg_data.global_protein_names]
    features = mg_data.global_protein_features.float().contiguous()
    feature_mean = mg_data.global_feature_mean.float().contiguous()
    feature_std = mg_data.global_feature_std.float().contiguous()

    graph = nx.read_edgelist(required["global PPI"])
    retained = set(protein_names)
    graph.remove_nodes_from([name for name in graph if name not in retained])
    if list(graph.nodes()) != protein_names:
        raise ValueError("Global protein order differs from the released ESM2 feature order.")

    name_to_index = {name: index for index, name in enumerate(protein_names)}
    unique_edges = torch.tensor(
        [(name_to_index[left], name_to_index[right]) for left, right in graph.edges()],
        dtype=torch.long,
    ).t().contiguous()
    if unique_edges.numel() == 0:
        raise ValueError("The released global PPI contains no retained edges.")

    split_codes = _assign_context_splits(unique_edges, contexts, len(protein_names))
    _verify_edge_counts(
        required["edge counts"], protein_names, unique_edges, split_codes
    )

    train_mask = torch.from_numpy((split_codes == -1) | (split_codes == 0))
    val_mask = torch.from_numpy(split_codes == 1)
    test_mask = torch.from_numpy(split_codes == 2)
    train_val_mask = torch.from_numpy(split_codes != 2)
    train_unique = unique_edges[:, train_mask]
    val_unique = unique_edges[:, val_mask]
    test_unique = unique_edges[:, test_mask]
    train_val_unique = unique_edges[:, train_val_mask]

    graph_fingerprint = _fingerprint(
        [
            "\n".join(protein_names).encode(),
            np.sort(_canonical_keys(unique_edges, len(protein_names)).numpy()).tobytes(),
        ]
    )
    split_fingerprint = _fingerprint(
        [
            np.sort(_canonical_keys(train_unique, len(protein_names)).numpy()).tobytes(),
            np.sort(_canonical_keys(val_unique, len(protein_names)).numpy()).tobytes(),
            np.sort(_canonical_keys(test_unique, len(protein_names)).numpy()).tobytes(),
        ]
    )
    split_counts = {
        "reference_only_train": int((split_codes == -1).sum()),
        "shared_train": int((split_codes == 0).sum()),
        "shared_val": int((split_codes == 1).sum()),
        "shared_test": int((split_codes == 2).sum()),
    }

    return GlobalPPIData(
        features=features,
        feature_mean=feature_mean,
        feature_std=feature_std,
        protein_names=protein_names,
        all_edge_index=_symmetric_edges(unique_edges),
        train_edge_index=_symmetric_edges(train_unique),
        train_val_edge_index=_symmetric_edges(train_val_unique),
        val_edge_index=val_unique,
        test_edge_index=test_unique,
        source_context_count=len(contexts),
        graph_fingerprint=graph_fingerprint,
        feature_fingerprint=_tensor_fingerprint(features),
        split_fingerprint=split_fingerprint,
        split_counts=split_counts,
    )


class GlobalS2GAE(nn.Module):
    """The ProtScape protein backbone and decoder without contextual heads."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.4,
        decode_channels: int = 512,
        decoder_layers: int = 2,
        decoder_dropout: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.model_config = {
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "decode_channels": int(decode_channels),
            "decoder_layers": int(decoder_layers),
            "decoder_dropout": float(decoder_dropout),
        }
        encoder_config = {
            "input_dim": int(input_dim),
            "gnn_method": "ACM_RandomWalk",
            "hidden_dim": int(hidden_dim),
            "gnn_dropout": float(dropout),
            "gnn_batchnorm": True,
            "gnn_activation": "leaky_relu",
            "jumping_knowledge": "concat",
            "n_layers": int(num_layers),
            "gat_heads": 8,
            "use_virtual_node": False,
            "n_cells": None,
            "add_virtual_node": False,
        }
        self.encoder = prot_module(encoder_config, str(device), graph_saint_norm=False)
        self.decoder = S2GAEDecoder(
            hidden_channels=int(hidden_dim),
            decode_channels=int(decode_channels),
            num_encoder_layers=int(num_layers),
            num_decoder_layers=int(decoder_layers),
            dropout=float(decoder_dropout),
        )

    @property
    def embedding_dim(self) -> int:
        return self.model_config["hidden_dim"] * self.model_config["num_layers"]

    def encode(
        self,
        features: torch.Tensor,
        message_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        graph = Data(
            x=features,
            edge_index=message_edge_index,
            edge_weight=torch.ones(
                message_edge_index.size(1),
                dtype=features.dtype,
                device=message_edge_index.device,
            ),
            y=torch.ones(
                message_edge_index.size(1),
                dtype=torch.bool,
                device=message_edge_index.device,
            ),
        )
        embeddings, layer_embeddings = self.encoder.forward_batch_graph(
            graph, return_layer_outputs=True
        )
        return embeddings, layer_embeddings

    def score_edges(
        self,
        layer_embeddings: list[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return undirected_decoder_logits(self.decoder, layer_embeddings, edge_index)


def masked_reconstruction_step(
    model: GlobalS2GAE,
    data: GlobalPPIData,
    *,
    device: torch.device,
    mask_ratio: float = 0.5,
    mask_type: str = "dm",
    k_negatives: int = 1,
    validate_targets: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must lie strictly between 0 and 1.")
    if k_negatives < 1:
        raise ValueError("k_negatives must be positive.")
    if mask_type not in {"dm", "um"}:
        raise ValueError("mask_type must be 'dm' or 'um'.")

    message_edges, positive_edges, sampling_edges, mask_index = edge_mask_per_graph(
        data.train_edge_index,
        data.features.size(0),
        mask_ratio,
        device,
        mask_type=mask_type,
        add_self_loop=True,
    )
    negative_edges = structured_negative_sampling_k(
        sampling_edges,
        num_nodes=data.features.size(0),
        k=k_negatives,
        mask_idx=mask_index,
        exclude_edge_index=data.all_edge_index,
    ).to(device)
    expected_negatives = positive_edges.size(1) * k_negatives
    if negative_edges.size(1) != expected_negatives:
        raise RuntimeError(
            f"Sampled {negative_edges.size(1)} negatives; expected {expected_negatives}."
        )
    if validate_targets:
        validate_masked_targets(
            message_edges.cpu(),
            positive_edges.cpu(),
            negative_edges.cpu(),
            data.all_edge_index,
            data.features.size(0),
            require_pairwise_hidden=mask_type == "um",
        )

    _, layers = model.encode(data.features.to(device), message_edges)
    return s2gae_loss(model.decoder, layers, positive_edges, negative_edges)


def validate_masked_targets(
    message_edges: torch.Tensor,
    positive_edges: torch.Tensor,
    negative_edges: torch.Tensor,
    all_positive_edges: torch.Tensor,
    num_nodes: int,
    *,
    require_pairwise_hidden: bool,
) -> None:
    directional_message = (
        message_edges[0].detach().cpu().long() * num_nodes
        + message_edges[1].detach().cpu().long()
    )
    directional_positive = (
        positive_edges[0].detach().cpu().long() * num_nodes
        + positive_edges[1].detach().cpu().long()
    )
    non_loop_positive = positive_edges[0].detach().cpu() != positive_edges[1].detach().cpu()
    if torch.isin(
        directional_positive[non_loop_positive], directional_message
    ).any():
        raise ValueError("A masked directed edge remains in the message-passing graph.")

    message_keys = torch.unique(_canonical_keys(message_edges, num_nodes))
    positive_keys = torch.unique(_canonical_keys(positive_edges, num_nodes))
    negative_keys = _canonical_keys(negative_edges, num_nodes)
    all_positive_keys = torch.unique(_canonical_keys(all_positive_edges, num_nodes))

    non_loop_message = message_keys % num_nodes != torch.div(
        message_keys, num_nodes, rounding_mode="floor"
    )
    message_keys = message_keys[non_loop_message]
    non_loop_positive_keys = positive_keys % num_nodes != torch.div(
        positive_keys, num_nodes, rounding_mode="floor"
    )
    if require_pairwise_hidden and torch.isin(
        positive_keys[non_loop_positive_keys], message_keys
    ).any():
        raise ValueError("A masked interaction remains in the message-passing graph.")
    if torch.isin(negative_keys, all_positive_keys).any():
        raise ValueError("A sampled negative is a known global interaction.")
    if (negative_edges[0] == negative_edges[1]).any():
        raise ValueError("A sampled negative is a self-loop.")


def sample_negative_bank(
    positive_edges: torch.Tensor,
    all_positive_edges: torch.Tensor,
    num_nodes: int,
    bank_size: int,
    *,
    seed: int,
    return_k: int | None = None,
) -> torch.Tensor:
    """Fixed structured target-corruption bank used by the paper evaluator."""
    if bank_size < 1:
        raise ValueError("bank_size must be positive.")
    return_k = bank_size if return_k is None else int(return_k)
    if return_k < 1 or return_k > bank_size:
        raise ValueError("return_k must lie within the generated bank.")
    sources = positive_edges[0].detach().cpu().numpy().astype(np.int64, copy=False)
    positives = all_positive_edges.detach().cpu().numpy().astype(np.int64, copy=False)
    rng = np.random.RandomState(seed)

    known_targets: dict[int, list[int]] = {}
    for left, right in zip(positives[0], positives[1]):
        known_targets.setdefault(int(left), []).append(int(right))
        known_targets.setdefault(int(right), []).append(int(left))

    source_rows: dict[int, list[int]] = {}
    for row, source in enumerate(sources):
        source_rows.setdefault(int(source), []).append(row)

    negative_targets = np.empty((sources.size, bank_size), dtype=np.int64)
    for source, rows in source_rows.items():
        rows_array = np.asarray(rows, dtype=np.int64)
        forbidden = np.asarray(known_targets.get(source, ()), dtype=np.int64)
        needed = rows_array.size * bank_size
        draws = []
        collected = 0
        while collected < needed:
            candidates = rng.randint(
                0, num_nodes, size=max(4096, 2 * (needed - collected))
            )
            keep = candidates != source
            if forbidden.size:
                keep &= ~np.isin(candidates, forbidden)
            candidates = candidates[keep]
            take = min(candidates.size, needed - collected)
            draws.append(candidates[:take])
            collected += take
        negative_targets[rows_array] = np.concatenate(draws).reshape(
            rows_array.size, bank_size
        )

    return torch.from_numpy(
        np.stack(
            [
                np.repeat(sources, return_k),
                negative_targets[:, :return_k].reshape(-1),
            ],
            axis=0,
        )
    ).long()


def _normalize_k_values(k_values: Sequence[int]) -> list[int]:
    values = sorted({int(k) for k in k_values})
    if not values or not set(values).issubset(GLOBAL_K_VALUES):
        raise ValueError(f"k_values must be a subset of {GLOBAL_K_VALUES}.")
    return values


def build_global_negative_bank(
    data: GlobalPPIData,
    *,
    split: str,
    max_k: int,
    seed: int,
) -> torch.Tensor:
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'.")
    if max_k not in GLOBAL_K_VALUES:
        raise ValueError(f"max_k must be one of {GLOBAL_K_VALUES}.")
    positive_edges = data.val_edge_index if split == "val" else data.test_edge_index
    return sample_negative_bank(
        positive_edges,
        data.all_edge_index,
        data.features.size(0),
        GLOBAL_NEGATIVE_BANK_SIZE,
        seed=seed,
        return_k=max_k,
    ).reshape(2, positive_edges.size(1), max_k)


def binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)
    roc = 0.5 if np.unique(labels).size < 2 else roc_auc_score(labels, scores)
    return {
        "roc": float(roc),
        "ap": float(average_precision_score(labels, scores)),
        "acc": float(accuracy_score(labels, scores >= 0.5)),
        "f1": float(
            f1_score(
                labels,
                scores >= 0.5,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        ),
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
    }


@torch.no_grad()
def _score_in_chunks(
    model: GlobalS2GAE,
    layers: list[torch.Tensor],
    edges: torch.Tensor,
    device: torch.device,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    output = []
    for start in range(0, edges.size(1), chunk_size):
        edge_batch = edges[:, start : start + chunk_size].to(device)
        output.append(torch.sigmoid(model.score_edges(layers, edge_batch)).cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


@torch.no_grad()
def evaluate_global_edges(
    model: GlobalS2GAE,
    data: GlobalPPIData,
    *,
    split: str,
    k_values: Sequence[int],
    device: torch.device,
    seed: int = 0,
    layer_embeddings: list[torch.Tensor] | None = None,
    negative_bank: torch.Tensor | None = None,
) -> dict[int, dict[str, float | int]]:
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'.")
    k_values = _normalize_k_values(k_values)
    max_k = max(k_values)
    positive_edges = data.val_edge_index if split == "val" else data.test_edge_index
    message_edges = data.train_edge_index if split == "val" else data.train_val_edge_index
    if negative_bank is None:
        negative_bank = build_global_negative_bank(
            data, split=split, max_k=max_k, seed=seed
        )
    if negative_bank.shape[:2] != (2, positive_edges.size(1)):
        raise ValueError("Global negative bank does not match the positive targets.")
    if negative_bank.size(2) < max_k:
        raise ValueError("Global negative bank is smaller than the requested k.")

    model.eval()
    if layer_embeddings is None:
        _, layer_embeddings = model.encode(
            data.features.to(device), message_edges.to(device)
        )
    positive_scores = _score_in_chunks(
        model, layer_embeddings, positive_edges, device
    )
    scored_negative_bank = negative_bank[:, :, :max_k]
    all_negative_scores = _score_in_chunks(
        model, layer_embeddings, scored_negative_bank.reshape(2, -1), device
    ).reshape(positive_edges.size(1), max_k)
    output = {}
    for k in k_values:
        negative_scores = all_negative_scores[:, :k].reshape(-1)
        scores = np.concatenate([positive_scores, negative_scores])
        labels = np.concatenate(
            [
                np.ones(positive_scores.size, dtype=np.int8),
                np.zeros(negative_scores.size, dtype=np.int8),
            ]
        )
        output[k] = binary_metrics(scores, labels)
    return output
