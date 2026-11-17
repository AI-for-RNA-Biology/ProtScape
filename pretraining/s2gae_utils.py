"""
S2GAE (Self-Supervised Graph Autoencoders) utilities for multi-cell PPI training.

Based on: Tan et al., "S2GAE: Self-Supervised Graph Autoencoders Are Generalizable
Learners with Graph Masking", WSDM 2023: https://doi.org/10.1145/3539597.3570404
Original code: https://github.com/qiaoyu-tan/S2GAE.

This module provides edge masking utilities and a cross-layer link prediction decoder
adapted for the hierarchical multi-cell PPI graph setting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import (
    add_self_loops,
    to_undirected,
)

S2GAE_DECODER_CHUNK_SIZE = 50000


def sample_structured_negative_targets(
    pos_source,
    exclude_edge_index,
    num_nodes,
    return_valid_mask=False,
):
    """Sample one negative target per source, excluding known positives.

    Dense neighborhoods use an explicit list of allowed targets. With
    return_valid_mask=True, omit sources with no allowed target and return
    their validity mask; otherwise, raise if any source has no valid negative.
    """
    device = pos_source.device
    num_nodes = int(num_nodes)
    pos_source = pos_source.to(device=device, dtype=torch.long)
    if pos_source.numel() == 0:
        if return_valid_mask:
            return pos_source, torch.empty(0, dtype=torch.bool, device=device)
        return pos_source

    forbidden_keys = None
    positive_edges = None
    if exclude_edge_index is not None and exclude_edge_index.numel() > 0:
        positive_edges = exclude_edge_index.to(device=device, dtype=torch.long)
        positive_edges = torch.cat([positive_edges, positive_edges.flip(0)], dim=1)
        forbidden_keys = (
            positive_edges[0] * num_nodes + positive_edges[1]
        ).unique().sort().values

    def is_forbidden(src, dst):
        """Mark self-loops and known positive pairs among candidate negative edges."""
        blocked = src == dst
        if forbidden_keys is None or forbidden_keys.numel() == 0:
            return blocked

        candidate_keys = src * num_nodes + dst
        idx = torch.searchsorted(forbidden_keys, candidate_keys)
        in_bounds = idx < forbidden_keys.numel()
        hits = torch.zeros_like(in_bounds, dtype=torch.bool)
        if in_bounds.any():
            hits[in_bounds] = forbidden_keys[idx[in_bounds]] == candidate_keys[in_bounds]
        return blocked | hits

    neg_target = torch.randint(num_nodes, (pos_source.numel(),), device=device)
    bad = is_forbidden(pos_source, neg_target)

    for _ in range(20):
        if not bad.any():
            break
        n_bad = int(bad.sum().item())
        neg_target[bad] = torch.randint(num_nodes, (n_bad,), device=device)
        bad = is_forbidden(pos_source, neg_target)

    valid_mask = torch.ones(pos_source.numel(), dtype=torch.bool, device=device)
    if bad.any():
        # Sample from the remaining non-neighbors; omit sources with none.
        for src in torch.unique(pos_source[bad]).tolist():
            src = int(src)
            src_mask = bad & (pos_source == src)
            blocked = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            blocked[src] = True
            if positive_edges is not None and positive_edges.numel() > 0:
                targets = positive_edges[1, positive_edges[0] == src]
                blocked[targets] = True
            allowed = (~blocked).nonzero(as_tuple=False).view(-1)
            if allowed.numel() == 0:
                valid_mask[src_mask] = False
                continue
            draw = torch.randint(allowed.numel(), (int(src_mask.sum().item()),), device=device)
            neg_target[src_mask] = allowed[draw]

    if return_valid_mask:
        return neg_target[valid_mask], valid_mask
    if not valid_mask.all():
        raise RuntimeError("Structured negative sampling found no valid negative for at least one source.")
    return neg_target


def structured_negative_sampling_k(
    edge_index,
    num_nodes=None,
    k=1,
    mask_idx=None,
    exclude_edge_index=None,
):
    """Sample k negative targets per selected positive source, excluding known neighbors."""
    if edge_index is None or edge_index.numel() == 0 or k <= 0:
        dev = edge_index.device if edge_index is not None else torch.device("cpu")
        return torch.empty((2, 0), dtype=torch.long, device=dev)

    if num_nodes is None:
        max_edge = int(edge_index.max().item())
        max_exclude = -1
        if exclude_edge_index is not None and exclude_edge_index.numel() > 0:
            max_exclude = int(exclude_edge_index.max().item())
        num_nodes = max(max_edge, max_exclude) + 1
    pos_edge_index = edge_index
    if mask_idx is not None:
        pos_edge_index = edge_index[:, mask_idx.to(device=edge_index.device, dtype=torch.long)]
    if exclude_edge_index is None:
        exclude_edge_index = edge_index

    neg_chunks = []
    for _ in range(k):
        src = pos_edge_index[0]
        neg_dst, valid_mask = sample_structured_negative_targets(
            src,
            exclude_edge_index,
            num_nodes,
            return_valid_mask=True,
        )
        if neg_dst.numel() > 0:
            neg_edges = torch.stack([src[valid_mask], neg_dst], dim=0)
            neg_chunks.append(neg_edges)

    return torch.cat(neg_chunks, dim=1) if neg_chunks else torch.empty((2, 0), dtype=torch.long, device=edge_index.device)


def edge_mask_per_graph(edge_index, num_nodes, mask_ratio, device,
                        mask_type='um', add_self_loop=True):
    """Return message edges, masked positives, all positives and masked indices.
    Undirected masking removes both directions but reconstructs each pair once."""
    if edge_index.device != torch.device('cpu'):
        edge_index_cpu = edge_index.cpu()
    else:
        edge_index_cpu = edge_index

    if mask_type == 'um':
        # Undirected masking: work with undirected edges
        edge_index_undirected = to_undirected(edge_index_cpu)
        # Get unique edges (upper triangular)
        row, col = edge_index_undirected[0], edge_index_undirected[1]
        mask = row < col
        unique_edges = torch.stack([row[mask], col[mask]], dim=0)

        num_edges = unique_edges.size(1)
        indices = np.arange(num_edges)
        np.random.shuffle(indices)

        mask_num = int(num_edges * mask_ratio)
        split_at = num_edges - mask_num
        keep_idx = torch.from_numpy(indices[:split_at])
        mask_idx = torch.from_numpy(indices[split_at:])

        # Edges to keep for message passing (make undirected)
        remaining_edges = unique_edges[:, keep_idx]
        remaining_edge_index = to_undirected(remaining_edges)

        # Reconstruct each masked undirected pair once.
        masked_edges = unique_edges[:, mask_idx]
        # For structured negatives in undirected mode, exclude both directions
        # of each positive edge to avoid sampling reverse true edges as negatives.
        sampling_edge_index = torch.cat([unique_edges, unique_edges[[1, 0], :]], dim=1)

    else:  # dm - directed masking
        num_edges = edge_index_cpu.size(1)
        indices = np.arange(num_edges)
        np.random.shuffle(indices)

        mask_num = int(num_edges * mask_ratio)
        split_at = num_edges - mask_num
        keep_idx = torch.from_numpy(indices[:split_at])
        mask_idx = torch.from_numpy(indices[split_at:])

        remaining_edge_index = edge_index_cpu[:, keep_idx]
        masked_edges = edge_index_cpu[:, mask_idx]
        sampling_edge_index = edge_index_cpu

    # Add self-loops
    if add_self_loop:
        edge_index_with_loops, _ = add_self_loops(remaining_edge_index, num_nodes=num_nodes)
    else:
        edge_index_with_loops = remaining_edge_index

    return (
        edge_index_with_loops.to(device),
        masked_edges.to(device),
        sampling_edge_index,
        mask_idx,
    )


def batch_edge_mask(
    ppi_data_batch,
    mask_ratio,
    device,
    mask_type='um',
    k_negatives=1,
    exclude_edge_index_by_cell=None,
):
    """Mask each cell's positive edges and pair reconstruction targets with sampled negatives."""
    masked_edge_indices = {}
    reconstruction_targets = {}

    for celltype, data in ppi_data_batch.items():
        if hasattr(data, 'x'):
            num_nodes = data.x.size(0)
            edge_index = data.edge_index
        else:
            edge_index = data.get('total_edge_index', data.get('edge_index'))
            num_nodes = edge_index.max().item() + 1

        # Get only positive edges (y==1) for masking.
        if hasattr(data, 'y') and (data.y is not None):
            y = data.y.view(-1)
            if y.numel() == edge_index.size(1):
                pos_edge_index = edge_index[:, y.to(dtype=torch.bool)]
            else:
                pos_edge_index = edge_index
        elif isinstance(data, dict):
            y = data.get('y', None)
            if y is None:
                pos_edge_index = edge_index
            else:
                y = torch.as_tensor(y).view(-1)
                if y.numel() == edge_index.size(1):
                    pos_edge_index = edge_index[:, y.to(dtype=torch.bool)]
                else:
                    pos_edge_index = edge_index
        else:
            pos_edge_index = edge_index

        edge_index_masked, masked_pos_edges, sampling_edge_index, mask_idx = edge_mask_per_graph(
            pos_edge_index, num_nodes, mask_ratio, device, mask_type
        )

        masked_edge_indices[celltype] = edge_index_masked

        k_neg = max(1, int(k_negatives))
        exclude_edge_index = None
        if exclude_edge_index_by_cell is not None:
            exclude_edge_index = exclude_edge_index_by_cell.get(celltype)

        neg_edges = structured_negative_sampling_k(
            sampling_edge_index,
            num_nodes=num_nodes,
            k=k_neg,
            mask_idx=mask_idx,
            exclude_edge_index=exclude_edge_index,
        ).to(device)

        reconstruction_targets[celltype] = {
            'pos_edges': masked_pos_edges,
            'neg_edges': neg_edges
        }

    return masked_edge_indices, reconstruction_targets


class S2GAEDecoder(nn.Module):
    """Score edges with an MLP over products of source and target GNN-layer embeddings."""

    def __init__(
        self,
        hidden_channels,
        decode_channels,
        num_encoder_layers,
        num_decoder_layers=2,
        dropout=0.5,
        raw_input_dim=0,
    ):
        """Build the cross-layer MLP, optionally including products of raw protein features."""
        super(S2GAEDecoder, self).__init__()

        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.raw_input_dim = int(raw_input_dim)

        # Cross-layer: L*L combinations of source and target layer representations
        n_cross = num_encoder_layers * num_encoder_layers
        input_dim = hidden_channels * n_cross + self.raw_input_dim

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, decode_channels))
        for _ in range(num_decoder_layers - 2):
            self.lins.append(nn.Linear(decode_channels, decode_channels))
        self.lins.append(nn.Linear(decode_channels, 1))

    def reset_parameters(self):
        """Reinitialize the decoder MLP weights and biases."""
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer_features(self, layer_embeds_src, layer_embeds_dst, raw_src=None, raw_dst=None):
        """Concatenate source/target products for all layer pairs, with raw features first."""
        cross_products = []
        if raw_src is not None and raw_dst is not None:
            cross_products.append(raw_src * raw_dst)
        for src_emb in layer_embeds_src:
            for dst_emb in layer_embeds_dst:
                cross_products.append(src_emb * dst_emb)
        return torch.cat(cross_products, dim=-1)

    def forward(self, layer_embeddings, edge_index):
        """Return one logit per edge from GNN-layer embeddings and optional raw features."""
        raw_embeddings = None
        if isinstance(layer_embeddings, dict):
            raw_embeddings = layer_embeddings.get("raw")
            layer_embeddings = layer_embeddings.get("layers")

        # Get source and target embeddings from each layer
        src_embeds = [emb[edge_index[0]] for emb in layer_embeddings]
        dst_embeds = [emb[edge_index[1]] for emb in layer_embeddings]
        raw_src = None
        raw_dst = None

        if self.raw_input_dim > 0:
            if raw_embeddings is None:
                raise ValueError("S2GAEDecoder expected raw embeddings but none were provided.")
            raw_src = raw_embeddings[edge_index[0]]
            raw_dst = raw_embeddings[edge_index[1]]

        # Cross-layer features
        x = self.cross_layer_features(src_embeds, dst_embeds, raw_src=raw_src, raw_dst=raw_dst)

        # MLP
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x.squeeze(-1)


class S2GAEDecoderSimple(nn.Module):
    """Score edges by the dot product of concatenated GNN-layer embeddings."""

    def __init__(self, hidden_channels, num_encoder_layers, dropout=0.0):
        """Record the layer dimensions for the parameter-free dot-product decoder."""
        super(S2GAEDecoderSimple, self).__init__()

        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.total_dim = hidden_channels * num_encoder_layers

    def reset_parameters(self):
        """No parameters need resetting for the dot-product decoder."""
        pass

    def forward(self, layer_embeddings, edge_index):
        """Return one dot-product logit per edge using concatenated GNN-layer outputs."""
        # Concatenate all layer outputs
        concat_emb = torch.cat(layer_embeddings, dim=-1)

        # Get source and target embeddings
        src_emb = concat_emb[edge_index[0]]
        dst_emb = concat_emb[edge_index[1]]

        # Dot product
        dots = (src_emb * dst_emb).sum(dim=-1)

        return dots


def decode_edges(decoder, layer_embeddings, edge_index):
    """Decode edges, recomputing training activations during backward to reduce memory."""
    if (
        decoder.training
        and torch.is_grad_enabled()
        and (
            isinstance(layer_embeddings, dict)
            or any(x.requires_grad for x in layer_embeddings)
        )
    ):
        # Recompute decoder activations during backward to limit memory use.
        if isinstance(layer_embeddings, dict):
            layers = list(layer_embeddings["layers"])
            raw_embeddings = layer_embeddings.get("raw")
        else:
            layers = list(layer_embeddings)
            raw_embeddings = None

        def run_decoder(*args):
            """Rebuild layer/raw inputs from tensors supplied by activation checkpointing."""
            if raw_embeddings is None:
                local_layers = list(args[:-1])
                local_edge_index = args[-1]
                decoder_input = local_layers
            else:
                local_layers = list(args[:-2])
                local_raw = args[-2]
                local_edge_index = args[-1]
                decoder_input = {"layers": local_layers, "raw": local_raw}
            return decoder(decoder_input, local_edge_index)

        checkpoint_args = layers + ([raw_embeddings] if raw_embeddings is not None else []) + [edge_index]
        return checkpoint(run_decoder, *checkpoint_args, use_reentrant=False)

    return decoder(layer_embeddings, edge_index)


def decode_edges_in_chunks(decoder, layer_embeddings, edge_index, chunk_size=S2GAE_DECODER_CHUNK_SIZE):
    """Split large edge sets into chunks before running the decoder."""
    if edge_index.numel() == 0:
        return torch.empty(0, device=edge_index.device)

    if edge_index.shape[1] <= chunk_size:
        return decode_edges(decoder, layer_embeddings, edge_index)

    logits = []
    for start in range(0, edge_index.shape[1], chunk_size):
        end = min(start + chunk_size, edge_index.shape[1])
        logits.append(decode_edges(decoder, layer_embeddings, edge_index[:, start:end]))
    return torch.cat(logits, dim=0)


def undirected_decoder_logits(decoder, layer_embeddings, edge_index):
    """Average directional probabilities for an undirected PPI edge score."""
    logits = decode_edges_in_chunks(decoder, layer_embeddings, edge_index)
    rev_logits = decode_edges_in_chunks(decoder, layer_embeddings, edge_index.flip(0))
    # Stable logit(0.5 * (sigmoid(logits) + sigmoid(rev_logits))).
    log_prob = torch.logaddexp(
        F.logsigmoid(logits),
        F.logsigmoid(rev_logits),
    ) - np.log(2.0)
    log_not_prob = torch.logaddexp(
        F.logsigmoid(-logits),
        F.logsigmoid(-rev_logits),
    ) - np.log(2.0)
    return log_prob - log_not_prob


def s2gae_loss(decoder, layer_embeddings, pos_edges, neg_edges):
    """Return binary reconstruction loss, logits and labels for masked and negative edges."""
    # Compute logits
    pos_logits = undirected_decoder_logits(decoder, layer_embeddings, pos_edges)
    neg_logits = undirected_decoder_logits(decoder, layer_embeddings, neg_edges)

    # Labels: 1 for masked positives, 0 for sampled negatives
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([
        torch.ones_like(pos_logits, device=pos_logits.device),
        torch.zeros_like(neg_logits, device=neg_logits.device)
    ], dim=0)

    return F.binary_cross_entropy_with_logits(logits, labels), logits, labels
