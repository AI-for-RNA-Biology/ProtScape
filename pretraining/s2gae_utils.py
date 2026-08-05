"""
S2GAE (Self-Supervised Graph Autoencoders) utilities for multi-cell PPI training.

Based on: Tan et al., "S2GAE: Self-Supervised Graph Autoencoders Are Generalizable 
Learners with Graph Masking", WSDM 2023.

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
    negative_sampling,
)

S2GAE_DECODER_CHUNK_SIZE = 50000


def _make_exclusion_edge_index(edge_index):
    if edge_index is None or edge_index.numel() == 0:
        return edge_index
    # PPI edges are undirected; if (u, v) is known, (v, u) is known too.
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def sample_structured_negative_targets(
    pos_source,
    exclude_edge_index,
    num_nodes,
    return_valid_mask=False,
):
    """Sample one negative target per source, excluding known positives.

    We used to concatenate ``exclude_edge_index`` into PyG's structured sampler
    and keep only the first block. That made PyG also sample negatives for the
    exclusion edges, which can loop forever on dense GraphSAINT subgraphs.
    Here we keep the structured-negative semantics but only sample the sources
    we actually need.
    """
    device = pos_source.device
    num_nodes = int(num_nodes)
    pos_source = pos_source.to(device=device, dtype=torch.long)
    if pos_source.numel() == 0:
        if return_valid_mask:
            return pos_source, torch.empty(0, dtype=torch.bool, device=device)
        return pos_source

    forbidden_keys = None
    exclude_for_cleanup = None
    if exclude_edge_index is not None and exclude_edge_index.numel() > 0:
        exclude_for_cleanup = _make_exclusion_edge_index(
            exclude_edge_index.to(device=device, dtype=torch.long)
        )
        forbidden_keys = (
            exclude_for_cleanup[0] * num_nodes + exclude_for_cleanup[1]
        ).unique().sort().values

    def is_forbidden(src, dst):
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
        # Exact cleanup for the rare dense sources that random resampling did
        # not solve. If a source has no valid local negative, drop it.
        for src in torch.unique(pos_source[bad]).tolist():
            src = int(src)
            src_mask = bad & (pos_source == src)
            blocked = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            blocked[src] = True
            if exclude_for_cleanup is not None and exclude_for_cleanup.numel() > 0:
                targets = exclude_for_cleanup[1, exclude_for_cleanup[0] == src]
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
    """K structured negatives per positive edge."""
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
                        mask_type='dm', add_self_loop=True):
    """
    Apply edge masking to a single graph for S2GAE training.
    
    Args:
        edge_index: [2, E] tensor of edges (can be directed or undirected)
        num_nodes: number of nodes in the graph
        mask_ratio: fraction of edges to mask (0.5-0.7 typical)
        device: torch device
        mask_type: 'um' (undirected mask) or 'dm' (directed mask)
        add_self_loop: whether to add self-loops to the remaining edges
        
    Returns:
        edge_index_masked: edge_index after masking (for message passing)
        masked_edges: [2, M] the edges that were masked (targets for reconstruction)
        sampling_edge_index: [2, E_pos] full positive edge index used for structured negatives
        mask_idx: [M] indices of masked edges in sampling_edge_index
    """
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
        keep_idx = torch.from_numpy(indices[:-mask_num])
        mask_idx = torch.from_numpy(indices[-mask_num:])
        
        # Edges to keep for message passing (make undirected)
        remaining_edges = unique_edges[:, keep_idx]
        remaining_edge_index = to_undirected(remaining_edges)
        
        # Edges that were masked (make undirected for reconstruction targets)
        masked_unique = unique_edges[:, mask_idx]
        # Keep masked positives as unique undirected edges (single direction) to avoid
        # double counting in the reconstruction loss
        masked_edges = masked_unique
        # For structured negatives in undirected mode, exclude both directions
        # of each positive edge to avoid sampling reverse true edges as negatives.
        sampling_edge_index = torch.cat([unique_edges, unique_edges[[1, 0], :]], dim=1)
        
    else:  # dm - directed masking
        num_edges = edge_index_cpu.size(1)
        indices = np.arange(num_edges)
        np.random.shuffle(indices)
        
        mask_num = int(num_edges * mask_ratio)
        keep_idx = torch.from_numpy(indices[:-mask_num])
        mask_idx = torch.from_numpy(indices[-mask_num:])
        
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
    """
    Apply edge masking to a batch of cell-specific PPI graphs.
    
    Args:
        ppi_data_batch: dict mapping celltype -> {x, edge_index, ...} or Data objects
        mask_ratio: fraction of edges to mask
        device: torch device
        mask_type: 'um' or 'dm'
        
    Returns:
        masked_edge_indices: dict mapping celltype -> edge_index for message passing
        reconstruction_targets: dict mapping celltype -> {pos_edges, neg_edges}
    """
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


def generate_negative_edges(edge_index, num_nodes, num_neg, device):
    """
    Generate negative edges (non-existing edges) for contrastive learning.
    
    Args:
        edge_index: [2, E] existing edges
        num_nodes: number of nodes
        num_neg: number of negative edges to generate
        device: torch device
        
    Returns:
        neg_edges: [2, num_neg] negative edge indices
    """
    if num_neg <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index_cpu = (
        torch.empty((2, 0), dtype=torch.long)
        if edge_index is None or edge_index.numel() == 0
        else edge_index.detach().to(device='cpu', dtype=torch.long)
    )

    # Exclude both directions 
    if edge_index_cpu.numel() > 0:
        edge_index_exclude = torch.cat([edge_index_cpu, edge_index_cpu[[1, 0], :]], dim=1)
    else:
        edge_index_exclude = edge_index_cpu

    neg_chunks = []
    remaining = int(num_neg)
    while remaining > 0:
        neg = negative_sampling(
            edge_index=edge_index_exclude,
            num_nodes=num_nodes,
            num_neg_samples=remaining,
            method='sparse',
        )
        # Filter self-loops 
        if neg.numel() > 0:
            mask = neg[0] != neg[1]
            neg = neg[:, mask]

        if neg.numel() == 0:
            # Resample if no valid negatives (all were self-loops)
            cand_src = torch.randint(0, num_nodes, (remaining * 2,), device='cpu')
            cand_dst = torch.randint(0, num_nodes, (remaining * 2,), device='cpu')
            mask = cand_src != cand_dst
            neg = torch.stack([cand_src[mask], cand_dst[mask]], dim=0)[:, :remaining]

        neg_chunks.append(neg[:, :remaining])
        remaining -= neg_chunks[-1].size(1)

        # Avoid duplicates in further sampling
        if remaining > 0 and neg_chunks[-1].numel() > 0:
            edge_index_exclude = torch.cat(
                [edge_index_exclude, neg_chunks[-1], neg_chunks[-1][[1, 0], :]],
                dim=1
            )

    neg_edges = torch.cat(neg_chunks, dim=1)[:, :num_neg]
    return neg_edges.to(device)


class S2GAEDecoder(nn.Module):
    """
    Cross-layer link prediction decoder for S2GAE.
    
    Uses representations from all GNN layers to predict links,
    enabling the model to leverage both local and global structural information.
    
    The decoder computes pairwise products across all layer combinations
    (L x L combinations for L layers), then applies an MLP for final prediction.
    """
    
    def __init__(
        self,
        hidden_channels,
        decode_channels,
        num_encoder_layers,
        num_decoder_layers=2,
        dropout=0.5,
        raw_input_dim=0,
    ):
        """
        Args:
            hidden_channels: hidden dimension of each encoder layer
            decode_channels: hidden dimension in the decoder MLP
            num_encoder_layers: number of GNN layers (L)
            num_decoder_layers: number of MLP layers in decoder
            dropout: dropout rate
        """
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
        for lin in self.lins:
            lin.reset_parameters()
    
    def cross_layer_features(self, layer_embeds_src, layer_embeds_dst, raw_src=None, raw_dst=None):
        """
        Compute cross-layer pairwise features.
        
        Args:
            layer_embeds_src: list of [batch, d] embeddings for source nodes (one per layer)
            layer_embeds_dst: list of [batch, d] embeddings for target nodes (one per layer)
            
        Returns:
            cross_features: [batch, L*L*d] concatenated cross-layer products
        """
        cross_products = []
        if raw_src is not None and raw_dst is not None:
            cross_products.append(raw_src * raw_dst)
        for src_emb in layer_embeds_src:
            for dst_emb in layer_embeds_dst:
                cross_products.append(src_emb * dst_emb)
        return torch.cat(cross_products, dim=-1)
    
    def forward(self, layer_embeddings, edge_index):
        """
        Predict link probabilities for given edges.
        
        Args:
            layer_embeddings: list of [N, d] tensors, one per GNN layer
            edge_index: [2, E] edges to predict
            
        Returns:
            logits: [E] raw logits (apply sigmoid outside if needed)
        """
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
    """
    Simplified S2GAE decoder using only the concatenation of all layer outputs.
    
    This is a simpler variant that concatenates layer representations before
    computing the dot product, rather than doing cross-layer products.
    """
    
    def __init__(self, hidden_channels, num_encoder_layers, dropout=0.0):
        """
        Args:
            hidden_channels: hidden dimension of each encoder layer
            num_encoder_layers: number of GNN layers (L)
            dropout: dropout rate
        """
        super(S2GAEDecoderSimple, self).__init__()
        
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.total_dim = hidden_channels * num_encoder_layers
        
    def reset_parameters(self):
        pass 
    
    def forward(self, layer_embeddings, edge_index):
        """
        Predict link probabilities for given edges.
        
        Args:
            layer_embeddings: list of [N, d] tensors, one per GNN layer
            edge_index: [2, E] edges to predict
            
        Returns:
            logits: [E] raw logits (apply sigmoid outside if needed)
        """
        # Concatenate all layer outputs
        concat_emb = torch.cat(layer_embeddings, dim=-1)
        
        # Get source and target embeddings
        src_emb = concat_emb[edge_index[0]]
        dst_emb = concat_emb[edge_index[1]]
        
        # Dot product
        dots = (src_emb * dst_emb).sum(dim=-1)
        
        return dots


def _decoder_logits(decoder, layer_embeddings, edge_index):
    if (
        decoder.training
        and torch.is_grad_enabled()
        and (
            isinstance(layer_embeddings, dict)
            or any(x.requires_grad for x in layer_embeddings)
        )
    ):
        # kneg=10 can create very large decoder tensors. Checkpointing keeps the
        # same loss/decoder but recomputes decoder activations during backward.
        if isinstance(layer_embeddings, dict):
            layers = list(layer_embeddings["layers"])
            raw_embeddings = layer_embeddings.get("raw")
        else:
            layers = list(layer_embeddings)
            raw_embeddings = None

        def run_decoder(*args):
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


def _decoder_logits_chunked(decoder, layer_embeddings, edge_index, chunk_size=S2GAE_DECODER_CHUNK_SIZE):
    if edge_index.numel() == 0:
        return torch.empty(0, device=edge_index.device)

    if edge_index.shape[1] <= chunk_size:
        return _decoder_logits(decoder, layer_embeddings, edge_index)

    logits = []
    for start in range(0, edge_index.shape[1], chunk_size):
        end = min(start + chunk_size, edge_index.shape[1])
        logits.append(_decoder_logits(decoder, layer_embeddings, edge_index[:, start:end]))
    return torch.cat(logits, dim=0)


def undirected_decoder_logits(decoder, layer_embeddings, edge_index):
    """Average directional probabilities for an undirected PPI edge score."""
    logits = _decoder_logits_chunked(decoder, layer_embeddings, edge_index)
    rev_logits = _decoder_logits_chunked(decoder, layer_embeddings, edge_index.flip(0))
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
    """
    Compute S2GAE reconstruction loss.
    
    Args:
        decoder: S2GAEDecoder or S2GAEDecoderSimple
        layer_embeddings: list of [N, d] tensors from each GNN layer
        pos_edges: [2, P] positive (masked) edges to reconstruct
        neg_edges: [2, N] negative edges
        
    Returns:
        loss: scalar reconstruction loss
    """
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


def s2gae_loss_batch(decoder, layer_embeddings_dict, reconstruction_targets, device):
    """
    Compute S2GAE loss for a batch of cell-specific graphs.
    
    Args:
        decoder: S2GAEDecoder module
        layer_embeddings_dict: dict mapping celltype -> list of [N, d] layer embeddings
        reconstruction_targets: dict mapping celltype -> {pos_edges, neg_edges}
        device: torch device
        
    Returns:
        total_loss: averaged reconstruction loss across all cell types
    """
    total_loss = 0.0
    n_cells = 0
    
    ppi_preds = {}
    ppi_labels = {}
    for celltype, targets in reconstruction_targets.items():
        layer_embs = layer_embeddings_dict[celltype]
        pos_edges = targets['pos_edges']
        neg_edges = targets['neg_edges']
        
        loss, logits, labels = s2gae_loss(decoder, layer_embs, pos_edges, neg_edges)
        total_loss += loss
        n_cells += 1
        ppi_preds[celltype] = logits
        ppi_labels[celltype] = labels
    
    if n_cells > 0:
        return total_loss / n_cells, ppi_preds, ppi_labels
    else:
        return total_loss, ppi_preds, ppi_labels
