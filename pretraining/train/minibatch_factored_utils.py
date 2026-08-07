import os
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import GraphSAINTNodeSampler, GraphSAINTEdgeSampler
import torch.nn.functional as F

from ..losses.loss import el_dot, calc_uniformity_loss
from ..s2gae_utils import (
    batch_edge_mask,
    s2gae_loss,
    sample_structured_negative_targets,
    undirected_decoder_logits,
)
from ..data_handler.generate_input import graph_saint_dryrun

from tqdm import tqdm
from time import time

from torchmetrics import F1Score, Accuracy, AUROC, AveragePrecision
from torch_geometric.utils import scatter

import wandb

# Model-agnostic batch generation


def _prepare_s2gae_decoder_input(decoder, layer_embeddings, raw_embeddings=None):
    if getattr(decoder, "raw_input_dim", 0) > 0:
        return {"layers": layer_embeddings, "raw": raw_embeddings}
    return layer_embeddings


def _concat_batched_s2gae_edges(reconstruction_targets, batch_ptr, cell_names, device):
    empty_edges = torch.empty((2, 0), dtype=torch.long, device=device)
    pos_chunks = []
    neg_chunks = []

    for i, celltype in enumerate(cell_names):
        offset = int(batch_ptr[i].item())
        targets = reconstruction_targets[celltype]
        pos_edges = targets["pos_edges"]
        neg_edges = targets["neg_edges"]

        if pos_edges.numel() > 0:
            pos_chunks.append(pos_edges + offset)
        if neg_edges.numel() > 0:
            neg_chunks.append(neg_edges + offset)

    pos_edges = torch.cat(pos_chunks, dim=1) if pos_chunks else empty_edges
    neg_edges = torch.cat(neg_chunks, dim=1) if neg_chunks else empty_edges
    return pos_edges, neg_edges


def _ppi_phuber_ce_loss_from_prob(pred_prob, labels, tau=10.0, weight=None):
    labels = labels.to(dtype=pred_prob.dtype, device=pred_prob.device)
    pred_prob = pred_prob.clamp(min=1e-7, max=1.0 - 1e-7)
    p_true = torch.where(labels > 0.5, pred_prob, 1.0 - pred_prob)
    threshold = 1.0 / tau
    loss = torch.where(
        p_true > threshold,
        -torch.log(p_true),
        -tau * p_true + np.log(tau) + 1.0,
    )
    if weight is not None:
        loss = loss * weight.to(device=loss.device, dtype=loss.dtype)
    return loss.mean()


def _ppi_loss_from_prob(pred_prob, labels, loss_type="bce", phuber_tau=10.0, weight=None):
    labels = labels.to(dtype=pred_prob.dtype, device=pred_prob.device)
    if weight is not None:
        weight = weight.to(device=pred_prob.device, dtype=pred_prob.dtype)
    if loss_type == "bce":
        return F.binary_cross_entropy(pred_prob, labels, weight=weight, reduction='mean')
    if loss_type == "l1":
        loss = torch.abs(pred_prob - labels)
        if weight is not None:
            loss = loss * weight
        return loss.mean()
    if loss_type == "phuber":
        return _ppi_phuber_ce_loss_from_prob(pred_prob, labels, tau=phuber_tau, weight=weight)
    raise ValueError(f"Unknown PPI loss: {loss_type}")


def _ppi_loss_from_logits(logits, labels, loss_type="bce", phuber_tau=10.0):
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, labels)
    if loss_type == "l1":
        return _ppi_loss_from_prob(torch.sigmoid(logits), labels, loss_type="l1")
    if loss_type == "phuber":
        return _ppi_phuber_ce_loss_from_prob(torch.sigmoid(logits), labels, tau=phuber_tau)
    raise ValueError(f"Unknown PPI loss: {loss_type}")


def _ppi_loss_from_dict(ppi_preds, ppi_labels, loss_type="bce", phuber_tau=10.0):
    losses = []
    for celltype, pred_prob in ppi_preds.items():
        losses.append(_ppi_loss_from_prob(
            pred_prob, ppi_labels[celltype], loss_type=loss_type, phuber_tau=phuber_tau))
    return torch.stack(losses).mean()


def _batched_s2gae_loss(
        decoder,
        layer_outputs,
        reconstruction_targets,
        batch_ppi_x,
        cell_names,
        loss_type="bce",
        phuber_tau=10.0):
    raw_embeddings = batch_ppi_x.x if getattr(decoder, "raw_input_dim", 0) > 0 else None
    decoder_input = _prepare_s2gae_decoder_input(
        decoder,
        layer_outputs,
        raw_embeddings=raw_embeddings,
    )
    pos_edges, neg_edges = _concat_batched_s2gae_edges(
        reconstruction_targets,
        batch_ppi_x.ptr,
        cell_names,
        batch_ppi_x.x.device,
    )
    loss, logits, labels = s2gae_loss(decoder, decoder_input, pos_edges, neg_edges)
    if loss_type != "bce":
        loss = _ppi_loss_from_logits(logits, labels, loss_type=loss_type, phuber_tau=phuber_tau)
    return loss, logits, labels


def generate_batch(
    data_dict,
    edge_attr_dict,
    mask, 
    batch_size,
    device,
    ppi=False,
    loader_type="graphsaint",
    verbose=False,
    n_jobs:int=None,
    evaluation_mode:str='local',
    graph_saint_norm:bool=False,
    pin_memory:bool=True,
    gs_sample_coverage=100,
    gs_filter = 'RandomWalk',
    root_gsnorm_path = None,
    weighted_ppi_loss=False
    ):
    """ Generate a batch of data for training or evaluation.
    Args:
        data_dict (dict): Dictionary of data objects for each subnetwork.
        metapaths (list): List of metapaths to be used.
        edge_attr_dict (dict): Dictionary mapping edge attributes to indices.
        mask (str): Mask type ('train', 'val', 'test', or 'all').
        batch_size (int): Size of the batch.
        device (str): Device to which the data should be moved ('cpu' or 'cuda').
        ppi (bool): Whether the task is PPI or not.
        loader_type (str): Type of DataLoader to use ('neighbor' or 'graphsaint').
        num_layers (int): Number of layers for the DataLoader.
        n_jobs (int): number of CPUs to parallelize the dataloading on CPUs or not. Default is None = no parallelization.
        
    Returns:
        loader_dict (dict): Dictionary of DataLoaders for each subnetwork.
    """
    # Iterate through subnetworks
    
    if graph_saint_norm:
        assert mask == 'train'
        assert loader_type == 'graphsaintedge'
        
    def get_samples(data, root_gsnorm_path=None):
        full_positive_edge_index = data.edge_index
        if root_gsnorm_path is not None:
            # true path of saved gs statistics
            gsnorm_path = root_gsnorm_path + f'/graphsaintedgesampler_{gs_sample_coverage}.pt'
        else:
            gsnorm_path = None
            
        if mask == "train":
            pos_edge_index = data.edge_index[:, data.train_mask]
            eval_edges = None # all of them are assumed to be seen in the GAE framework
            edge_type = data.edge_attr[data.train_mask]
            if weighted_ppi_loss:
                loss_edge_weights = data.ppi_weights[data.train_mask]
                
        elif mask == "val":
            pos_train_edge_index = data.edge_index[:, data.train_mask]
            evaluated_edge_index = data.edge_index[:, data.val_mask]
            # contains all edges used for training and evaluation - so that the graphsaint sampler takes into account both for evaluation
            pos_edge_index = torch.cat([pos_train_edge_index, evaluated_edge_index], dim=-1)
            
            eval_edges = torch.ones(pos_edge_index.shape[-1], dtype=torch.int8)
            eval_edges[:pos_train_edge_index.shape[-1]] = 0  # mark training edges as 0, evaluation edges as 1
            
            train_edge_type = data.edge_attr[data.train_mask]
            val_edge_type = data.edge_attr[data.val_mask]
            edge_type = torch.cat([train_edge_type, val_edge_type], dim=0)
            if weighted_ppi_loss:
                loss_edge_weights_train = data.ppi_weights[data.train_mask]
                loss_edge_weights_val = data.ppi_weights[data.val_mask]
                loss_edge_weights = torch.cat([loss_edge_weights_train, loss_edge_weights_val])
                
                
        elif mask == "test":
            pos_train_edge_index = data.edge_index[:, data.train_mask]
            pos_val_edge_index = data.edge_index[:, data.val_mask]
            evaluated_edge_index = data.edge_index[:, data.test_mask]
            # contains all edges used for evaluation - so that the graphsaint sampler takes into account both for evaluation
            
            pos_edge_index = torch.cat([pos_train_edge_index, pos_val_edge_index, evaluated_edge_index], dim=-1)
            
            eval_edges = torch.ones(pos_edge_index.shape[-1], dtype=torch.int8)
            eval_edges[:pos_train_edge_index.shape[-1] + pos_val_edge_index.shape[-1]] = 0  # mark training edges as 0, evaluation edges as 1
            
            train_edge_type = data.edge_attr[data.train_mask]
            val_edge_type = data.edge_attr[data.val_mask]
            test_edge_type = data.edge_attr[data.test_mask]
            edge_type = torch.cat([train_edge_type, val_edge_type, test_edge_type], dim=0)

            if weighted_ppi_loss:
                loss_edge_weights = data.ppi_weights
                
        else:
            raise NotImplementedError(f'Unsupported mask: {mask}')
        
        # If PPI task --> build special DataLoader
        if ppi:
            if eval_edges is None: # correspond only to train
                data = Data(
                        x = data.x,
                        edge_index = pos_edge_index,
                        edge_attr=edge_type,
                        n_id = torch.arange(data.x.shape[0]),
                )
                if graph_saint_norm:
                    # Add the precomputed GraphSAINT edge and loss weights.
                    assert gsnorm_path is not None
                    if not os.path.exists(gsnorm_path):
                        print('compute gs statistics in :', gsnorm_path)
                        graph_saint_dryrun(
                            data, root_gsnorm_path,
                            batch_size,
                            num_steps=16,
                            sample_coverage=gs_sample_coverage, # to estimate sampling stats / rule : while total_sampled_nodes < self.N * self.sample_coverage:
                            num_workers=16,
                            pin_memory=True)
                    
                    #load normalization statistics
                    gs_node_norm, gs_edge_norm = torch.load(gsnorm_path, weights_only=False) # gs_node_norm (N,) tensor; gs_edge_norm (E,) tensor
                    ## Operations on pytorch geometric
                    # 1) node norm first corresponds to node_count across various sampling steps
                    #  then node_norm = num_sampled_subgraphs / node_count / self.N
                    # 2) edge norm comes from the following operations
                    # t = torch.empty_like(edge_count).scatter_(0, edge_idx, node_count[row])
                    # --> Assigns the node counts to the corresponding edges in t.
                    # edge_norm = (t / edge_count).clamp_(0, 1e4)
                    # edge_norm[torch.isnan(edge_norm)] = 0.1
                    # From pygeo exemple: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_saint.py
                    # they leverage node_norm and edge_norm as follows:
                    # edge_weight = data.edge_norm * data.edge_weight (-> 1/ deg)
                    # out = model(data.x, data.edge_index, edge_weight)
                    # loss = F.nll_loss(out, data.y, reduction='none')
                    # loss = (loss * data.node_norm)[data.train_mask].sum()
                    
                    """
                    data.edge_loss_weight = 1. / (data.edge_index.shape[1] * gs_edge_norm) # saved in data for edge loss normalization
                    """
                    data.edge_loss_weight = gs_edge_norm # saved in data for edge loss normalization
                    
                    if verbose:
                        print(f'gs_node_norm: min = {gs_node_norm.min()} / max = {gs_node_norm.max()} / mean = {gs_node_norm.mean()}')
                        print(f'gs_edge_norm: min = {gs_edge_norm.min()} / max = {gs_edge_norm.max()} / mean = {gs_edge_norm.mean()}')
                    
                    # edge weights must satisfie:
                    # W_{u,v} = Atilde_{v, u} * p_v / p_{u,v}, with Atilde_{v, u} = A_{v,u} / deg{v}
                    # considering that the message passing to update node v is:
                    # h_v <- sum_u W_{u, v} X_u
                    if gs_filter == 'RandomWalk':
                        # compute random walk matrix on the entire graph
                        ones_ = torch.ones(data.edge_index.shape[1])
                        
                        deg = scatter(ones_, data.edge_index[1], dim=0, dim_size=data.x.shape[0], reduce='sum')
                        deg_inv = 1. / deg
                        deg_inv.masked_fill_(deg_inv == float('inf'), 0)

                        """
                        --> vanilla version assuming that gs_edge_norm = p_{u,v} / gs_node_norm = p_v
                        which is not the case in pygeo implementation.
                        
                        edge_weights = data.edge_index / gs_edge_norm
                        node_prob_by_edge = gs_node_norm[data.edge_index[0,:]]
                        deg_inv_by_edge = deg_inv[data.edge_index[0, :]]
                        edge_weights *= node_prob_by_edge * deg_inv_by_edge
                        """
                        deg_inv_by_edge = deg_inv[data.edge_index[0, :]]
                        edge_weights = gs_edge_norm * deg_inv_by_edge
                        if verbose:
                            print(f'deg: min = {deg.min()} / max = {deg.max()} / mean = {deg.mean()}')
                            print(f'edge_weights: min = {edge_weights.min()} / max = {edge_weights.max()} / mean = {edge_weights.mean()}')
                    
                    else:
                        raise NotImplementedError(f'gs_filfer = {gs_filter} not implemented. Only supports RandomWalk filter')
                    
                    data.edge_weight = edge_weights
                
                if weighted_ppi_loss:
                    data.loss_edge_weights = loss_edge_weights
            else:
                # in evaluation mode
                # if graph_saint_norm = True: edge_weights are computed on the fly in the GNN layers keeping edge_weights=None beforehand for sampling.
                data = Data(
                    x = data.x,
                    edge_index = pos_edge_index,
                    edge_attr=edge_type,
                    eval_edges = eval_edges,
                    n_id = torch.arange(data.x.shape[0]),
                )
                if weighted_ppi_loss:
                    data.loss_edge_weights = loss_edge_weights
                    
            # Based on loader type 
            if evaluation_mode == 'local':
                if loader_type == "graphsaint":
                    loader = GraphSAINTEdgeSampler(
                        data,
                        batch_size = batch_size,
                        num_steps = 16,
                        pin_memory=pin_memory
                        )
                elif loader_type=='graphsaintnode':
                    # used by default in pinnacle and preliminary experiments with HC models 
                    loader = GraphSAINTNodeSampler(
                        data,
                        batch_size = batch_size,
                        num_steps = 16,
                        pin_memory=pin_memory
                        )
                
                else:
                    raise NotImplementedError(f'Loader type {loader_type} not implemented for PPI task.')
            
            else:
                loader = data    
            loader.full_positive_edge_index = full_positive_edge_index
            return loader
        
        else:
            raise NotImplementedError('Non-PPI batch generation not implemented yet.')
    
    # Initialize dictionaries
    loader_dict = dict()
    
    if n_jobs is None:
            
        for i, (key, data) in tqdm(enumerate(data_dict.items()), desc="Generating batches", disable=not verbose):
            if root_gsnorm_path is not None:
                local_root_gsnorm_path = root_gsnorm_path + f'/celltype{key}/'
                os.makedirs(local_root_gsnorm_path, exist_ok=True)
            else:
                local_root_gsnorm_path = None
            loader_dict[key] = get_samples(data, local_root_gsnorm_path)
        
    else:
        raise NotImplementedError('Parallel data loading not implemented yet.')

    return loader_dict


def batch_positive_exclusion_edge_index(batch, full_positive_edge_index, device):
    """Map all known positive edges into the sampled batch's local node ids."""
    if full_positive_edge_index is None or full_positive_edge_index.numel() == 0:
        return batch.edge_index

    edge_device = batch.edge_index.device
    full_positive_edge_index = full_positive_edge_index.to(edge_device)
    n_id = getattr(batch, "n_id", None)
    if n_id is None:
        if int(full_positive_edge_index.max().item()) < int(batch.num_nodes):
            return full_positive_edge_index.to(device)
        return batch.edge_index

    n_id = n_id.to(edge_device, dtype=torch.long)
    if n_id.numel() == 0:
        return batch.edge_index

    map_size = int(max(n_id.max().item(), full_positive_edge_index.max().item())) + 1
    global_to_local = torch.full((map_size,), -1, dtype=torch.long, device=edge_device)
    global_to_local[n_id] = torch.arange(n_id.numel(), dtype=torch.long, device=edge_device)
    local_edges = global_to_local[full_positive_edge_index]
    valid = (local_edges[0] >= 0) & (local_edges[1] >= 0)
    local_edges = local_edges[:, valid]
    if local_edges.numel() == 0:
        return batch.edge_index
    return local_edges.to(device)


# Negative Sample function ++
def negative_sampler(
    pos_edge_index,
    edge_type,
    edge_attr_dict,
    k_negatives=1,
    exclude_edge_index=None,
    num_nodes=None,
    verbose=False,
):
    """ Generate negative samples for link prediction.
    Args:
        pos_edge_index (torch.Tensor): Positive edge indices of shape (2, num_edges).
        edge_type (torch.Tensor): Edge types corresponding to the positive edges.
        edge_attr_dict (dict): Dictionary mapping edge attributes to indices.
        k_negatives (int): Number of structured negatives per positive edge.
    Returns:
        neg_edge_index (torch.Tensor): Negative edge indices of shape (2, num_neg_edges).
        neg_edge_type (torch.Tensor): Edge types corresponding to the negative edges.
    """
    # If no edges
    if len(edge_type) == 0:
        return pos_edge_index, edge_type
    
    neg_edge_index = None
    neg_edge_type = []
    k_neg = max(1, int(k_negatives))
    if num_nodes is None:
        max_pos = int(pos_edge_index.max().item()) if pos_edge_index.numel() > 0 else -1
        max_exclude = (
            int(exclude_edge_index.max().item())
            if exclude_edge_index is not None and exclude_edge_index.numel() > 0
            else -1
        )
        num_nodes = max(max_pos, max_exclude) + 1
    if exclude_edge_index is None:
        exclude_edge_index = pos_edge_index
    # Loop through each edge type
    for attr, idx in edge_attr_dict.items():
        # Select edges of this type
        mask = (edge_type == idx)
        # If there are none, skip
        if mask.sum() == 0: continue
        # Create structured negative samples
        pos_rel_edge_index = pos_edge_index.T[mask].T
        for _ in range(k_neg):
            neg_source = pos_rel_edge_index[0]
            neg_rand, valid_mask = sample_structured_negative_targets(
                neg_source,
                exclude_edge_index,
                num_nodes,
                return_valid_mask=True,
            )
            if neg_rand.numel() == 0:
                continue
            # Form negative edges
            neg_rel_edge_index = torch.stack((neg_source[valid_mask], neg_rand), dim=0)

            # Merge into total negatives
            if neg_edge_index is None:
                neg_edge_index = neg_rel_edge_index
            else:
                neg_edge_index = torch.cat((neg_edge_index, neg_rel_edge_index), 1)
            # Save edge types
            neg_edge_type.extend([idx] * int(neg_rel_edge_index.size(1)))

    if neg_edge_index is None:
        neg_edge_index = pos_edge_index[:, :0]
    return neg_edge_index, torch.tensor(neg_edge_type, dtype=edge_type.dtype, device=edge_type.device)


def build_cci_edge_index(mg_data, edge_attr_dict, cell_ids=None, device=None):
    """Return CCI edges as a symmetric (bidirectional) edge index."""
    if mg_data is None:
        return None
    edge_index = mg_data.edge_index
    edge_type = mg_data.edge_attr
    cci_type = edge_attr_dict["cell_cell"]
    mask = edge_type == cci_type
    edge_index = edge_index[:, mask]
    if cell_ids is not None:
        cell_ids_t = torch.as_tensor(cell_ids, device=edge_index.device, dtype=torch.long)
        if cell_ids_t.numel() == 0:
            return edge_index[:, :0]
        cell_mask = torch.zeros(mg_data.num_nodes, dtype=torch.bool, device=edge_index.device)
        cell_mask[cell_ids_t] = True
        keep = cell_mask[edge_index[0]] & cell_mask[edge_index[1]]
        edge_index = edge_index[:, keep]

    # Make CCI edges symmetric
    edge_index = _make_undirected_edge_index(edge_index)
    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


def _normalize_split_ratios(split_ratios):
    if split_ratios is None:
        split_ratios = (1.0, 0.0, 0.0)
    ratios = list(split_ratios)
    if len(ratios) != 3:
        raise ValueError(f"split_ratios must have length 3 (train/val/test), got {split_ratios!r}")
    total = float(sum(ratios))
    if total <= 0:
        return [1.0, 0.0, 0.0]
    return [r / total for r in ratios]


def _make_undirected_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return edge_index
    rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    return torch.unique(edge_index, dim=1)


def get_cci_edge_splits(
    mg_data,
    edge_attr_dict,
    cell_ids=None,
    split_ratios=(0.8, 0.1, 0.1),
    seed=0,
    undirected: bool = True,
):
    if mg_data is None:
        return None

    ratios = _normalize_split_ratios(split_ratios)
    cached = getattr(mg_data, "cci_edge_splits", None)
    cached_ids = getattr(mg_data, "cci_edge_split_cell_ids", None)
    cached_seed = getattr(mg_data, "cci_edge_split_seed", None)
    cached_ratios = getattr(mg_data, "cci_edge_split_ratios", None)
    cached_undirected = getattr(mg_data, "cci_edge_split_undirected", None)
    if cached is not None:
        ids_ok = True
        if cell_ids is not None:
            cell_ids_t = torch.as_tensor(cell_ids, dtype=torch.long).cpu()
            ids_ok = cached_ids is not None and torch.equal(cached_ids, cell_ids_t)
        if ids_ok and cached_seed == int(seed) and cached_ratios == ratios and cached_undirected == bool(undirected):
            return cached

    full_edge_index = build_cci_edge_index(
        mg_data, edge_attr_dict, cell_ids=cell_ids, device=None
    )
    if full_edge_index is None or full_edge_index.numel() == 0:
        return None
    full_edge_index = full_edge_index.cpu()

    src = full_edge_index[0].cpu()
    dst = full_edge_index[1].cpu()
    min_v = torch.minimum(src, dst)
    max_v = torch.maximum(src, dst)
    pairs = torch.stack([min_v, max_v], dim=0).t()
    unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    num_pairs = int(unique_pairs.size(0))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_pairs, generator=generator)

    n_train = int(num_pairs * ratios[0])
    n_val = int(num_pairs * ratios[1])
    if n_train == 0 and num_pairs > 0:
        n_train = 1
    if n_train + n_val > num_pairs:
        n_val = max(0, num_pairs - n_train)

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    pair_mask = torch.zeros(num_pairs, dtype=torch.bool)
    pair_mask[idx_train] = True
    train_mask = pair_mask[inverse]
    pair_mask[:] = False
    pair_mask[idx_val] = True
    val_mask = pair_mask[inverse]
    pair_mask[:] = False
    pair_mask[idx_test] = True
    test_mask = pair_mask[inverse]

    splits = {
        "train": full_edge_index[:, train_mask],
        "val": full_edge_index[:, val_mask],
        "test": full_edge_index[:, test_mask],
    }

    if undirected:
        for split_name, edges in splits.items():
            splits[split_name] = _make_undirected_edge_index(edges)

    mg_data.cci_edge_splits = splits
    if cell_ids is not None:
        mg_data.cci_edge_split_cell_ids = torch.as_tensor(cell_ids, dtype=torch.long).cpu()
    mg_data.cci_edge_split_seed = int(seed)
    mg_data.cci_edge_split_ratios = ratios
    mg_data.cci_edge_split_undirected = bool(undirected)
    return splits


def build_mg_edge_data(
    mg_data,
    edge_attr_dict,
    cell_ids=None,
    device=None,
    pos_edge_index=None,
    full_pos_edge_index=None,
):
    if mg_data is None:
        return None
    if pos_edge_index is None:
        pos_edge_index = build_cci_edge_index(
            mg_data, edge_attr_dict, cell_ids=cell_ids, device=device
        )
    elif device is not None:
        pos_edge_index = pos_edge_index.to(device)
    if pos_edge_index is None or pos_edge_index.numel() == 0:
        return None
    cci_type = edge_attr_dict["cell_cell"]
    pos_edge_type = torch.full(
        (pos_edge_index.size(1),),
        cci_type,
        device=pos_edge_index.device,
        dtype=mg_data.edge_attr.dtype,
    )
    num_nodes = int(mg_data.num_nodes)
    if full_pos_edge_index is None:
        full_pos_edge_index = build_cci_edge_index(
            mg_data, edge_attr_dict, cell_ids=cell_ids, device=device
        )
    if full_pos_edge_index is not None and device is not None:
        full_pos_edge_index = full_pos_edge_index.to(pos_edge_index.device)

    # Structured negative sampling for CCI edges (1:1), excluding self-loops
    # and every known positive CCI edge.
    neg_edge_index = pos_edge_index[:, :0]
    if full_pos_edge_index is not None and full_pos_edge_index.numel() > 0 and pos_edge_index.numel() > 0:
        if cell_ids is None:
            neg_dst, neg_valid = sample_structured_negative_targets(
                pos_edge_index[0],
                full_pos_edge_index,
                num_nodes,
                return_valid_mask=True,
            )
            pos_edge_index = pos_edge_index[:, neg_valid]
            pos_edge_type = pos_edge_type[neg_valid]
            neg_edge_index = torch.stack([pos_edge_index[0], neg_dst], dim=0)
        else:
            cell_ids_t = torch.as_tensor(cell_ids, device=pos_edge_index.device, dtype=torch.long)
            num_cells = int(cell_ids_t.numel())
            if num_cells > 0:
                map_global_to_local = torch.full(
                    (num_nodes,), -1, device=pos_edge_index.device, dtype=torch.long
                )
                map_global_to_local[cell_ids_t] = torch.arange(num_cells, device=pos_edge_index.device)

                full_local = map_global_to_local[full_pos_edge_index]
                pos_local = map_global_to_local[pos_edge_index]
                valid_full = (full_local[0] >= 0) & (full_local[1] >= 0)
                valid_pos = (pos_local[0] >= 0) & (pos_local[1] >= 0)
                full_local = full_local[:, valid_full]
                pos_local = pos_local[:, valid_pos]
                pos_edge_index = pos_edge_index[:, valid_pos]
                pos_edge_type = pos_edge_type[valid_pos]

                if full_local.numel() > 0 and pos_local.numel() > 0:
                    neg_dst, neg_valid = sample_structured_negative_targets(
                        pos_local[0],
                        full_local,
                        num_cells,
                        return_valid_mask=True,
                    )
                    pos_local = pos_local[:, neg_valid]
                    pos_edge_index = pos_edge_index[:, neg_valid]
                    pos_edge_type = pos_edge_type[neg_valid]
                    neg_edge_index = torch.stack(
                        (cell_ids_t[pos_local[0]], cell_ids_t[neg_dst]), dim=0
                    )
    neg_edge_type = torch.full(
        (neg_edge_index.size(1),),
        cci_type,
        device=pos_edge_index.device,
        dtype=mg_data.edge_attr.dtype,
    )
    total_edge_index = torch.cat((pos_edge_index, neg_edge_index), dim=1)
    total_edge_type = torch.cat((pos_edge_type, neg_edge_type), dim=0)
    y = torch.zeros(total_edge_index.size(1), dtype=torch.float32, device=total_edge_index.device)
    y[:pos_edge_index.size(1)] = 1.0
    return {
        "total_edge_index": total_edge_index,
        "total_edge_type": total_edge_type,
        "y": y,
    }


def _cell_ids_cache_key(cell_ids):
    if cell_ids is None:
        return None
    return tuple(int(x) for x in cell_ids)


def _move_mg_data_dict(mg_data_dict, device):
    if mg_data_dict is None:
        return None
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in mg_data_dict.items()
    }


def get_cached_mg_edge_data_eval(
    mg_data,
    edge_attr_dict,
    *,
    cell_ids=None,
    device=None,
    pos_edge_index=None,
    split="val",
):
    if mg_data is None:
        return None
    cache = getattr(mg_data, "cci_eval_cache", None)
    key = (split, _cell_ids_cache_key(cell_ids))
    if cache is None:
        cache = {}
        mg_data.cci_eval_cache = cache
    mg_data_cached = cache.get(key)
    if mg_data_cached is None:
        pos_cpu = pos_edge_index
        if pos_cpu is not None and pos_cpu.device != torch.device("cpu"):
            pos_cpu = pos_cpu.cpu()
        mg_data_cached = build_mg_edge_data(
            mg_data,
            edge_attr_dict,
            cell_ids=cell_ids,
            device="cpu",
            pos_edge_index=pos_cpu,
        )
        cache[key] = mg_data_cached
    return _move_mg_data_dict(mg_data_cached, device) if device is not None else mg_data_cached

# PINNACLE batch generation

def pred_batch2dict(
    packed_batch: object,
    mg_x_ori: dict,
    cell_names: list,
    edge_attr_dict:dict,
    device: str,
    negative_sampling:bool=True,
    k_negatives:int=1,
    evaluation_mode='local') -> dict:
    """
    Re-initialize :code:`ppi_x`, :code:`metagraph`, and transform packed batches of all graphs to a dictionary of batches. Note that different from :code:`train_batch2dict`, we are also re-initializing full :code:`ppi_x` because we are feeding all nodes instead of the sampled nodes in the batch to the model during prediction.
    
    :param packed_batch: An iterable (tuple if directly following unpacking of the output of :code:`generatePPIBatch`) storing batches of :class:`Data` from all graphs in one round.
    :param mg_x_ori: metagraph original node embeddings.
    :param ppi_x_ori: Original all PPI node embeddings
    :param cell_type_order: Cell type order.
    :param device: A string indicating the device. Default is "cuda".
    
    :return: A dictionary of edge data storing batches from all graphs in one round, :code:`ppi_x_batch`, :code:`ppi_node_ind_batch` extracted from batches, and the re-initialized node embeddings :code:`mg_x_init`.
    """
    if mg_x_ori is not None:
        raise NotImplementedError('batch transformer cannot handle metagraph for now')
    
    ppi_dict = {}
    ppi_eval_edges = {}
    ppi_labels = {}
    
    def build_eval_edges(batch, cell_name):
        pos_eval_edges = batch.edge_index[:, batch.eval_edges == 1]
        assert pos_eval_edges.size(1) > 0, f'No eval edges found for cell type {cell_name} in the batch.'

        if negative_sampling:
            k_neg = max(1, int(k_negatives))
            pos_eval_edge_type = batch.edge_attr[batch.eval_edges == 1]
            neg_eval_edges, _ = negative_sampler(
                pos_eval_edges,
                pos_eval_edge_type,
                edge_attr_dict,
                k_negatives=k_neg,
                exclude_edge_index=batch.edge_index,
                num_nodes=batch.num_nodes,
            )

            eval_edges = torch.cat([pos_eval_edges, neg_eval_edges], dim=-1)
            labels = torch.zeros(eval_edges.size(1), dtype=torch.float32, device=eval_edges.device)
            labels[:pos_eval_edges.size(1)] = 1.0
        else:
            eval_edges = pos_eval_edges
            labels = torch.ones(pos_eval_edges.size(1), dtype=torch.float32, device=pos_eval_edges.device)

        ppi_eval_edges[cell_name] = eval_edges
        ppi_labels[cell_name] = labels

    if evaluation_mode == 'local':
        for ind, batch in enumerate(packed_batch):
            cell_name = cell_names[ind]
            build_eval_edges(batch, cell_name)

            # Keep only training edges for message passing (avoid leakage).
            batch.edge_index = batch.edge_index[:, batch.eval_edges == 0]
            batch.eval_edges = None
            ppi_dict[cell_name] = batch.to(device)
        
    elif evaluation_mode == 'global':
        for cell_name in packed_batch.keys():
            batch = packed_batch[cell_name]
            build_eval_edges(batch, cell_name)

            batch.edge_index = batch.edge_index[:, batch.eval_edges == 0]
            batch.eval_edges = None
            ppi_dict[cell_name] = batch.to(device)

    else:
        raise ValueError(f"Unknown evaluation_mode={evaluation_mode}. Expected 'local' or 'global'.")
        
    return ppi_dict, ppi_eval_edges, ppi_labels



# Hierarchical-model batch generation


def train_batch2dict(
    packed_batch: object,
    mg_x_ori: dict,
    cell_names: list,
    device: str,
    verbose:bool=False) -> dict:
    """
    
    
    Re-initialize :code:`metagraph`, and transform packed batches of all graphs to a dictionary of batches.
    Note that different from :code:`pred_batch2dict`, we are not re-initializing :code:`ppi_x` here because we are only feeding sampled nodes in the batch to the model during training.
    
    :param packed_batch: An iterable (tuple if directly following unpacking of the output of :code:`generatePPIBatch`) storing batches of :class:`Data` from all graphs in one round.
    :param mg_x_ori: metagraph original node embeddings.
    :param cell_type_order: Cell type order.
    :param device: A string indicating the device. Default is "cuda".
    
    :return: A dictionary of edge data from all graphs in one round, :code:`ppi_x_batch`, :code:`ppi_node_ind_batch` and :code:`ppi_metapaths_batch` extracted from batches, and the re-initialized node embeddings :code:`mg_x_init`.
    """
    # Unpack batches
    if verbose:
        print('-- call train_batch2dict')
    if mg_x_ori is not None:
        raise NotImplementedError('batch transformer cannot handle metagraph for now')
    
    ppi_dict = {}
    
    for ind, batch in enumerate(packed_batch):
        if verbose:
            print(f' ind = {ind} / batch:', batch)
        cell_name = cell_names[ind]
        ppi_dict[cell_name] = batch.to(device)
        
    return ppi_dict


def iterate_train_batch_hierarchical_model(
    cfg: dict,
    ppi_train_loader_dict: dict,
    CT_map: dict,
    mg_data,
    model: torch.nn.Module,
    edge_attr_dict:dict,
    device: str,
    optimizer: torch.optim=None,
    current_epoch: int=0,
    verbose=False):
    """
    
    Args:
        cfg (dict): configuration parameters
        ppi_train_loader_dict (dict): A dictionary of PPI DataLoaders for each cell type. For instance each is a graphsaint-based loader.
        ppi_x_ori (dict): Original PPI node embeddings.
        ppi_metapaths_ori (dict): Original PPI metapaths.
        CT_map (dict): cell to tissues assignments as a one-hot encoded label, see generate_input::compute_CT_map
        model (torch.nn.Module): The model to train.
        device (str): Device to use for training ('cpu' or 'cuda').
        optimizer (torch.optim, optional): Optimizer for training. Defaults to None.
    Returns:
        tuple: 
            A tuple containing updated PPI node embeddings, metagraph node embeddings,
            metagraph predictions, PPI predictions, PPI data labels, and the total loss.
    """
    total_samples = total_loss = 0
    mg_data_train = None
    

    # we setup metrics of interest 
    train_roc_score = AUROC(task='binary').to(device)
    train_ap_score = AveragePrecision(task='binary').to(device)
    train_acc = Accuracy(task='binary').to(device)
    train_f1 = F1Score(task='binary', average='macro').to(device)
    metrics = {'roc': train_roc_score, 'ap': train_ap_score, 'acc': train_acc, 'f1':train_f1} 

    count = 0

    cell_names = list(ppi_train_loader_dict.keys())
    full_positive_edge_index = {
        celltype: getattr(loader, "full_positive_edge_index", None)
        for celltype, loader in ppi_train_loader_dict.items()
    }
    if getattr(cfg, "use_metagraph", False):
        cci_splits = getattr(mg_data, "cci_edge_splits", None)
        pos_edges = None
        if cci_splits is not None:
            pos_edges = cci_splits.get("train", None)
        mg_data_train = build_mg_edge_data(
            mg_data,
            edge_attr_dict,
            cell_ids=cell_names,
            device=device,
            pos_edge_index=pos_edges,
        )
        if getattr(model, "cci_edge_index", None) is None:
            cci_edge_index = None
            if cci_splits is not None:
                cci_edge_index = cci_splits.get("train", None)
            if cci_edge_index is None:
                cci_edge_index = build_cci_edge_index(mg_data, edge_attr_dict, cell_ids=cell_names, device=device)
            if cci_edge_index is not None:
                model.set_cci_graph(cci_edge_index.to(device), mg_data.num_nodes)
        if mg_data_train is not None:
            metrics.update({
                'roc_meta': AUROC(task='binary').to(device),
                'ap_meta': AveragePrecision(task='binary').to(device),
                'acc_meta': Accuracy(task='binary').to(device),
                'f1_meta': F1Score(task='binary', average='macro').to(device)
            })

    # START BATCH FOR LOOP
    if cfg.hierarchical_mode == 'CTassignment':
        missing_ct = [c for c in cell_names if c not in CT_map]
        if missing_ct:
            raise KeyError(
                f"CT_map is missing {len(missing_ct)} cell types from the loader keys "
                f"(e.g. {missing_ct[:5]})."
            )
        cell_labels = torch.stack(
            [CT_map[c]["onehot"] for c in cell_names],
            dim=0,
        ).to(device=device, dtype=torch.float32)
        if verbose:
            print('--- keys CT:', list(CT_map.keys()))
            print('cell_labels:', cell_labels.shape)
    
        metagraph_loss_nn = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device)
    
    start_epoch = time()
    batching = True
    mg_pred = None

    s2gae_cfg = getattr(cfg, "s2gae_config")
    s2gae_enabled = bool(s2gae_cfg.get("enabled")) if s2gae_cfg is not None else False
    s2gae_loss_weight = float(s2gae_cfg.get("loss_weight")) if s2gae_enabled else 0.0
    s2gae_mask_ratio = float(s2gae_cfg.get("mask_ratio")) if s2gae_enabled else 0.0
    s2gae_mask_type = str(s2gae_cfg.get("mask_type")) if s2gae_enabled else "um"
    ppi_loss_type = str(getattr(cfg, "ppi_loss", "bce")).lower()
    ppi_phuber_tau = float(getattr(cfg, "ppi_phuber_tau", 10.0))

    uniformity_cfg = getattr(cfg, "uniformity_config", None)
    uniformity_enabled = bool(uniformity_cfg.get("enabled")) if uniformity_cfg is not None else False
    uniformity_lambda_reg = float(uniformity_cfg.get("lambda_reg")) if uniformity_enabled else 0.0
    uniformity_t = float(uniformity_cfg.get("t")) if uniformity_enabled else 2.0
    
    ppi_metapaths_ori = None
    ppi_x_out, ppi_preds_all, ppi_data_y = None, None, None
    
    for packed_batch in tqdm(zip(*ppi_train_loader_dict.values()), desc='process training batch'):
        
        count += 1
        if verbose:
            print(f"Training batch {count}")
        optimizer.zero_grad()
        
        # Unpack batches to edges, nodes, and indices, and reinitialize mg_x
        
        if verbose:
            start_batch = time()
        # formate as dictionary
        ppi_dict = train_batch2dict(
            packed_batch, None, cell_names, device)
        del packed_batch
        exclude_edge_index_by_cell = {
            celltype: batch_positive_exclusion_edge_index(
                data,
                full_positive_edge_index.get(celltype),
                device,
            )
            for celltype, data in ppi_dict.items()
        }
        
        if verbose:
            print('train_batch2dict:', time() - start_batch)
        
        
        batch_size = sum([data.x.shape[0] for data in ppi_dict.values()])  # Number of all samples across all cell types
        if verbose:
            print('effective number of nodes in batch:', batch_size)
        
        # Generate PPI and metagraph embeddings and Compute predictions for metagraph
        if verbose:
            start_forward = time()
        if cfg.hierarchical_mode == 'CTassignment':
            assert cfg.cell_config['n_cells'] is not None
            # if batching = False
            # outputs coincide with : return ppi_emb_dict, None, cells_x, cells_pred
            # if batching = True
            # outputs coincide with : return emb_ppi_x, batch_ppi_x, cells_x, cells_pred 
            # -> (only node embeddings, batch graph object with original node features,...)

            s2gae_targets = None
            layer_outputs = None
            ppi_dict_forward = ppi_dict
            if s2gae_enabled:
                k_neg = int(getattr(cfg, 'k_negatives', 1))
                masked_edge_indices, s2gae_targets = batch_edge_mask(
                    ppi_dict, s2gae_mask_ratio, device, mask_type=s2gae_mask_type,
                    k_negatives=k_neg,
                    exclude_edge_index_by_cell=exclude_edge_index_by_cell,
                )
                #run message passing on the masked edges with edge_index_mp
                ppi_dict_forward = {}
                for celltype, data in ppi_dict.items():
                    ppi_dict_forward[celltype] = Data(
                        x=data.x,
                        edge_index=data.edge_index,
                        edge_index_mp=masked_edge_indices[celltype],
                    )

                emb_ppi_x, batch_ppi_x, cells_x, cells_pred, layer_outputs = model(
                    ppi_dict_forward,
                    batching=batching,
                    return_layer_outputs=True,
                )
                batch_ppi_x_full = batch_ppi_x
            else:
                emb_ppi_x, batch_ppi_x, cells_x, cells_pred = model(
                    ppi_dict,
                    batching=batching,
                )
                batch_ppi_x_full = batch_ppi_x
        
        else:
            raise NotImplementedError('Model forward is not implemented for this hierarchical mode.')

        # Metagraph predictions (uses cell/tissue embeddings only)
        mg_pred = None
        mg_logits = None
        if getattr(cfg, "use_metagraph", False) and (mg_data_train is not None):
            mg_logits = model.predict_cci_edges(
                cells_x,
                list(ppi_dict.keys()),
                mg_data_train["total_edge_index"],
                assume_cci_encoded=True,
            )
            if mg_logits is not None:
                mg_pred = torch.sigmoid(mg_logits)
            if metrics is not None and 'roc_meta' in metrics and mg_pred is not None:
                mg_labels_int = mg_data_train["y"].to(device=device, dtype=torch.int32)
                for m in ['roc_meta', 'ap_meta', 'acc_meta', 'f1_meta']:
                    _ = metrics[m](mg_pred, mg_labels_int)
            
        # Compute predictions for PPI layers
        if verbose:
            print('time forward:', time() - start_forward)
            print('--- compute edge predictions for ppi graphs')
        
        # select negative edges for evaluation
        
        s2gae_loss_val = None
        if s2gae_enabled:
            assert cfg.graph_saint_norm is False, "S2GAE is not compatible with GraphSAINT normalization."
            assert cfg.weighted_ppi_loss is False, "S2GAE is not compatible with weighted PPI loss."
            
            if getattr(model, "s2gae_decoder", None) is None or layer_outputs is None:
                raise ValueError("S2GAE enabled but decoder/layer outputs missing.")
            s2gae_loss_val, s2gae_logits_batch, ppi_labels_batch = _batched_s2gae_loss(
                model.s2gae_decoder,
                layer_outputs,
                s2gae_targets,
                batch_ppi_x,
                list(ppi_dict.keys()),
                loss_type=ppi_loss_type,
                phuber_tau=ppi_phuber_tau,
            )
            ppi_preds_batch = torch.sigmoid(s2gae_logits_batch)
                
            with torch.no_grad():
                for m in ["roc", "ap", "acc", "f1"]:
                    if m in metrics:
                        _ = metrics[m](ppi_preds_batch, ppi_labels_batch.to(torch.long))
                
        else:
            # we perform here the negative sampling for the GAE model whose inference does not depend on it
            # contrary to the S2GAE model.
        
            with torch.no_grad():
                ppi_neg_edges = dict()
                k_neg = max(1, int(getattr(cfg, "k_negatives", 1)))
                for cell_type in ppi_dict.keys():
                    neg_edge_index, neg_edge_type = negative_sampler(
                        ppi_dict[cell_type].edge_index,
                        ppi_dict[cell_type].edge_attr,
                        edge_attr_dict,
                        k_negatives=k_neg,
                        exclude_edge_index=exclude_edge_index_by_cell.get(cell_type),
                        num_nodes=ppi_dict[cell_type].num_nodes)
                    ppi_neg_edges[cell_type] = {
                        'edge_index': neg_edge_index,
                        'edge_type': neg_edge_type
                    }
                    
            # get predictions and formate labels
            if not batching:
                ppi_preds = dict()
                ppi_preds_label = dict()
                ppi_preds_weights = []
                for celltype, x in ppi_dict.items():
                    all_edges = torch.cat((ppi_dict[celltype].edge_index, ppi_neg_edges[celltype]['edge_index']), dim=-1)
                    labels = torch.zeros(all_edges.shape[1], device=device, dtype=torch.float32)
                    labels[:ppi_dict[celltype].edge_index.shape[-1]] = 1  # positive edges first
                    
                    ppi_preds[celltype] = el_dot(x, all_edges, [])
                    ppi_preds_label[celltype] = labels           
                    if cfg.graph_saint_norm:
                        raise NotImplementedError('weighted loss computation for batching=False not implemented')
                    
            else:
                ppi_preds_pos_batch = el_dot(emb_ppi_x, batch_ppi_x_full.edge_index, [])
                
                ppi_preds_weights = None
                
                with torch.no_grad():
                    if cfg.graph_saint_norm:
                        ppi_preds_weights = torch.cat([batch_ppi_x.edge_loss_weight, batch_ppi_x.edge_loss_weight])
                    elif cfg.weighted_ppi_loss:
                        ppi_preds_weights = torch.cat([batch_ppi_x.loss_edge_weights, batch_ppi_x.loss_edge_weights])
                        ppi_preds_weights /= ppi_preds_weights.sum()
                    
                # same weights assumed to both positive and negative samples
                
                # construct a batch with only negative edges
                with torch.no_grad():
                    neg_batch = Batch().from_data_list([
                        Data(
                            num_nodes=int(batch_ppi_x.ptr[i + 1] - batch_ppi_x.ptr[i]),
                            edge_index=ppi_neg_edges[celltype]["edge_index"],
                        )
                        for i, celltype in enumerate(ppi_dict.keys())
                    ])
                    
                
                ppi_preds_neg_batch = el_dot(emb_ppi_x, neg_batch.edge_index, [])
                # combine pos and neg predictions
                ppi_preds_batch = torch.cat((ppi_preds_pos_batch, ppi_preds_neg_batch), dim=0)
            
                # construct labels
                batch_labels = torch.zeros(ppi_preds_batch.shape[0], device=device, dtype=torch.float32)
                batch_labels[:ppi_preds_pos_batch.shape[-1]] = 1  # positive edges first
                
                with torch.no_grad():
                    for m in ["roc", "ap", "acc", "f1"]:
                        if m in metrics:
                            _ = metrics[m](ppi_preds_batch, batch_labels.to(torch.long))
                    
        # Compute protein edge prediction loss on train set
        if verbose:
            print('time compute ppi predictions:', time() - start_metrics)
            print('--- compute metagraph loss')
            
        if cfg.hierarchical_mode == 'CTassignment':
            if s2gae_enabled:
                if s2gae_loss_val is None:
                    raise ValueError("S2GAE enabled but s2gae_loss_val is None.")
                link_loss = s2gae_loss_val
            else:
                
                if not batching:
                    link_loss = _ppi_loss_from_dict(
                        ppi_preds, ppi_preds_label,
                        loss_type=ppi_loss_type, phuber_tau=ppi_phuber_tau)
                else:
                    link_loss = _ppi_loss_from_prob(
                        ppi_preds_batch, batch_labels, loss_type=ppi_loss_type,
                        phuber_tau=ppi_phuber_tau, weight=ppi_preds_weights)
                    
            mg_loss = torch.tensor(0.0, device=device)
            if getattr(cfg, "use_metagraph", False) and (mg_logits is not None) and (mg_data_train is not None):
                mg_loss = F.binary_cross_entropy_with_logits(mg_logits, mg_data_train["y"], reduction='mean')
        else:
            #ppi_loss, mg_loss  = calc_link_pred_loss(mg_pred, mg_data_train, ppi_preds, ppi_data_batch, hparams['loss_type'])
            raise NotImplementedError('Model forward is not implemented for this hierarchical mode.')

        del batch_ppi_x
        
        # Compute CTassignment loss
        
        CT_loss = metagraph_loss_nn(cells_pred, cell_labels)
        uniformity_loss = torch.tensor(0.0, device=device)
        uniformity_reg = torch.tensor(0.0, device=device)
        if uniformity_enabled and uniformity_lambda_reg > 0.0:
            uniformity_terms = []
            uni_proj = getattr(model, 'uniformity_layer', None)
            if batching:
                for i, _celltype in enumerate(ppi_dict.keys()):
                    start_idx = int(batch_ppi_x_full.ptr[i])
                    end_idx = int(batch_ppi_x_full.ptr[i + 1])
                    local_embed = emb_ppi_x[start_idx:end_idx]
                    if local_embed.shape[0] < 2:
                        continue
                    if uni_proj is not None:
                        local_embed = F.relu(uni_proj(local_embed))
                    uniformity_terms.append(calc_uniformity_loss(local_embed, t=uniformity_t))
            else:
                for celltype in ppi_dict.keys():
                    local_embed = emb_ppi_x[celltype]
                    if local_embed.shape[0] < 2:
                        continue
                    if uni_proj is not None:
                        local_embed = F.relu(uni_proj(local_embed))
                    uniformity_terms.append(calc_uniformity_loss(local_embed, t=uniformity_t))

            if len(uniformity_terms) > 0:
                uniformity_loss = torch.stack(uniformity_terms).mean()
                uniformity_reg = uniformity_lambda_reg * uniformity_loss

        if verbose:
            print("Link Prediction: ", link_loss, "CT Loss:", CT_loss, "MG Loss:", mg_loss, "Uni Loss:", uniformity_loss, "Uni Reg:", uniformity_reg, "Uni t:", uniformity_t)
        wandb_log_dict = {"Link Prediction Loss": link_loss, "CT Loss": CT_loss}
        if s2gae_loss_val is not None:
            wandb_log_dict["S2GAE Loss"] = s2gae_loss_val
        if getattr(cfg, "use_metagraph", False):
            wandb_log_dict["Metagraph Loss"] = mg_loss
        if uniformity_enabled:
            wandb_log_dict["Uniformity Loss"] = uniformity_loss
            wandb_log_dict["Uniformity Reg"] = uniformity_reg
            wandb_log_dict["Uniformity t"] = uniformity_t
            wandb_log_dict["Uniformity Lambda Reg"] = uniformity_lambda_reg
        wandb.log(wandb_log_dict)
        metagraph_loss_weight = getattr(cfg, "metagraph_loss_weight", 1.0)
        if s2gae_enabled:
            combined_loss = (
                s2gae_loss_weight * link_loss
                + cfg.reg_CTassignment * CT_loss
                + metagraph_loss_weight * mg_loss
                + uniformity_reg
            )
        else:
            combined_loss = (
                link_loss
                + cfg.reg_CTassignment * CT_loss
                + metagraph_loss_weight * mg_loss
                + uniformity_reg
            )
        combined_loss.backward()
        
        optimizer.step()
        
        # Calculate loss
        total_samples += batch_size
        total_loss += float(combined_loss) * batch_size
        # Note that here for simplicity the total loss rather than only the link prediction BCEloss is weighted by edge batch size. 
    
    if verbose:
        print('epoch time:', time() - start_epoch) 
    
    total_loss = total_loss/total_samples  # Weighted total train loss
    
    with torch.no_grad():
        
        # compute averaged metrics across batches
        computed_metrics = {}
        for m in metrics.keys():
            computed_metrics[m] = metrics[m].compute().item()
    
        mg_pred_out = None
    return total_loss, computed_metrics
    

def iterate_predict_batch_hierarchical_model(
    cfg: dict,
    ppi_loader_dict: dict,
    CT_map: dict,
    mg_data,
    model: torch.nn.Module,
    edge_attr_dict:dict,
    device: str,
    verbose=False,
    eval_on_gpu=True,
    split: str = "val")-> tuple:
    """
    Iterate batches for prediction (val/test).
    """
    cell_names = list(ppi_loader_dict.keys())
    
    ppi_preds_all = {cell_name :[] for cell_name in cell_names} # key = cell type
    ppi_labels_all = {cell_name :[] for cell_name in cell_names}
    count = 0
    model.eval()
    batching = cfg.cell_config.get("n_cells", None) is not None

    s2gae_cfg = getattr(cfg, "s2gae_config", None)
    s2gae_enabled = bool(s2gae_cfg.get("enabled", False)) if s2gae_cfg is not None else False
    mg_data_eval = None
    mg_pred = None
    mg_logits = None
    mg_logits = None
    if getattr(cfg, "use_metagraph", False):
        cci_splits = getattr(mg_data, "cci_edge_splits", None)
        pos_edges = None
        if cci_splits is not None:
            pos_edges = cci_splits.get(split, None)
        mg_data_eval = get_cached_mg_edge_data_eval(
            mg_data,
            edge_attr_dict,
            cell_ids=cell_names,
            device=device,
            pos_edge_index=pos_edges,
            split=split,
        )
        if getattr(model, "cci_edge_index", None) is None:
            cci_edge_index = None
            if cci_splits is not None:
                cci_edge_index = cci_splits.get("train", None)
            if cci_edge_index is None:
                cci_edge_index = build_cci_edge_index(mg_data, edge_attr_dict, cell_ids=cell_names, device=device)
            if cci_edge_index is not None:
                model.set_cci_graph(cci_edge_index.to(device), mg_data.num_nodes)
    with torch.no_grad():
        for packed_batch in tqdm(zip(*ppi_loader_dict.values()), desc='process eval batch'):
            count += 1
            
            # Unpack batches and reinitialize mg_x
            # Includes structured negative sampling in ppi_eval_edges
            with torch.no_grad():
                k_neg = max(1, int(getattr(cfg, "k_negatives", 1)))
                ppi_dict, ppi_eval_edges, ppi_labels = pred_batch2dict(
                    packed_batch, None, cell_names, edge_attr_dict, device,
                    negative_sampling=True, k_negatives=k_neg, evaluation_mode='local')

            del packed_batch
            
            # Generate PPI and metagraph embeddings & Compute predictions for metagraph
            if cfg.hierarchical_mode == 'CTassignment':
                assert cfg.cell_config['n_cells'] is not None
                # if batching = False
                # outputs coincide with : return ppi_emb_dict, None, cells_x, cells_pred
                # if batching = True
                # outputs coincide with : return emb_ppi_x, batch_ppi_x, cells_x, cells_pred 
                # -> (only node embeddings, batch graph object with original node features,...)

                if s2gae_enabled:
                    emb_ppi_x, batch_ppi_x, cells_x, cells_pred, layer_outputs = model(
                        ppi_dict,
                        batching=batching,
                        return_layer_outputs=True,
                    )
                else:
                    emb_ppi_x, batch_ppi_x, cells_x, cells_pred = model(
                        ppi_dict,
                        batching=batching)
            
            else:
                raise NotImplementedError('Model forward is not implemented for this hierarchical mode.')
            
            if mg_data_eval is not None:
                mg_logits = model.predict_cci_edges(
                    cells_x,
                    list(ppi_dict.keys()),
                    mg_data_eval["total_edge_index"],
                    assume_cci_encoded=True,
                )
                if mg_logits is not None:
                    mg_pred = torch.sigmoid(mg_logits)
            
            # We systematically evaluate PPI predictions without batching
            if eval_on_gpu:
                saving_device = device
            else:
                saving_device = 'cpu'
            if not batching:
                for celltype in ppi_dict.keys():
                    eval_edges = ppi_eval_edges[celltype].to(device)
                    labels = ppi_labels[celltype].to(device)

                    if s2gae_enabled:
                        if getattr(model, "s2gae_decoder", None) is None:
                            raise ValueError("S2GAE enabled but model.s2gae_decoder is None.")
                        decoder_input = _prepare_s2gae_decoder_input(
                            model.s2gae_decoder,
                            layer_outputs[celltype],
                            raw_embeddings=ppi_dict[celltype].x,
                        )
                        logits = undirected_decoder_logits(model.s2gae_decoder, decoder_input, eval_edges)
                        preds = torch.sigmoid(logits)
                    else:
                        preds = el_dot(emb_ppi_x[celltype], eval_edges, [])

                    ppi_preds_all[celltype].append(preds.to(saving_device))
                    ppi_labels_all[celltype].append(labels.to(saving_device))
            else:
                for i, celltype in enumerate(ppi_dict.keys()):
                    start = batch_ppi_x.ptr[i]
                    end = batch_ppi_x.ptr[i+1]
                    eval_edges = ppi_eval_edges[celltype].to(device)
                    labels = ppi_labels[celltype].to(device)

                    if s2gae_enabled:
                        if getattr(model, "s2gae_decoder", None) is None:
                            raise ValueError("S2GAE enabled but model.s2gae_decoder is None.")
                        layer_outputs_local = [h[start:end] for h in layer_outputs]
                        decoder_input = _prepare_s2gae_decoder_input(
                            model.s2gae_decoder,
                            layer_outputs_local,
                            raw_embeddings=batch_ppi_x.x[start:end],
                        )
                        logits = undirected_decoder_logits(model.s2gae_decoder, decoder_input, eval_edges)
                        preds = torch.sigmoid(logits)
                    else:
                        local_emb = emb_ppi_x[start:end]
                        preds = el_dot(local_emb, eval_edges, [])

                    ppi_preds_all[celltype].append(preds.to(saving_device))
                    ppi_labels_all[celltype].append(labels.to(saving_device))
            
        for cell_type in ppi_preds_all.keys():
            ppi_preds_all[cell_type] = torch.concat(ppi_preds_all[cell_type], dim=0)    
            ppi_labels_all[cell_type] = torch.concat(ppi_labels_all[cell_type], dim=0)
            
    return mg_pred, mg_data_eval, ppi_preds_all, ppi_labels_all


def iterate_predict_hierarchical_model(
    cfg: dict,
    ppi_loader_dict: dict,
    CT_map: dict,
    mg_data,
    model: torch.nn.Module,
    edge_attr_dict:dict,
    device: str,
    verbose=False,
    eval_on_gpu=True,
    split: str = "val")-> tuple:

    cell_names = list(ppi_loader_dict.keys())
    model.eval()

    # Global evaluation uses full graphs (all cell types). Batching them into a single
    # mega-graph can easily OOM (especially with ACM / dense message passing). We keep
    # batching disabled here to run each cell graph sequentially inside the model.
    batching = False

    # Build eval edges + labels (includes negative sampling)
    with torch.no_grad():
        k_neg = max(1, int(getattr(cfg, "k_negatives", 1)))
        ppi_dict, ppi_eval_edges, ppi_labels = pred_batch2dict(
            ppi_loader_dict, None, cell_names, edge_attr_dict, device,
            negative_sampling=True, k_negatives=k_neg, evaluation_mode='global',
        )

    s2gae_cfg = getattr(cfg, "s2gae_config", None)
    s2gae_enabled = bool(s2gae_cfg.get("enabled", False)) if s2gae_cfg is not None else False
    mg_data_eval = None
    mg_pred = None
    if getattr(cfg, "use_metagraph", False):
        cci_splits = getattr(mg_data, "cci_edge_splits", None)
        pos_edges = None
        if cci_splits is not None:
            pos_edges = cci_splits.get(split, None)
        mg_data_eval = get_cached_mg_edge_data_eval(
            mg_data,
            edge_attr_dict,
            cell_ids=cell_names,
            device=device,
            pos_edge_index=pos_edges,
            split=split,
        )
        if getattr(model, "cci_edge_index", None) is None:
            cci_edge_index = None
            if cci_splits is not None:
                cci_edge_index = cci_splits.get("train", None)
            if cci_edge_index is None:
                cci_edge_index = build_cci_edge_index(mg_data, edge_attr_dict, cell_ids=cell_names, device=device)
            if cci_edge_index is not None:
                model.set_cci_graph(cci_edge_index.to(device), mg_data.num_nodes)

    # Forward
    with torch.no_grad():
        if s2gae_enabled:
            emb_ppi_x, batch_ppi_x, cells_x, cells_pred, layer_outputs = model(
                ppi_dict, batching=batching, return_layer_outputs=True
            )
        else:
            emb_ppi_x, batch_ppi_x, cells_x, cells_pred = model(
                ppi_dict, batching=batching, return_layer_outputs=False
            )
            layer_outputs = None
        if mg_data_eval is not None:
            mg_logits = model.predict_cci_edges(
                cells_x,
                list(ppi_dict.keys()),
                mg_data_eval["total_edge_index"],
                assume_cci_encoded=True,
            )
            if mg_logits is not None:
                mg_pred = torch.sigmoid(mg_logits)

    saving_device = device if eval_on_gpu else "cpu"

    ppi_preds_all = {}
    ppi_labels_all = {}

    if not batching:
        for celltype in ppi_dict.keys():
            eval_edges = ppi_eval_edges[celltype].to(device)
            labels = ppi_labels[celltype].to(device)

            if s2gae_enabled:
                if getattr(model, "s2gae_decoder", None) is None:
                    raise ValueError("S2GAE enabled but model.s2gae_decoder is None.")
                decoder_input = _prepare_s2gae_decoder_input(
                    model.s2gae_decoder,
                    layer_outputs[celltype],
                    raw_embeddings=ppi_dict[celltype].x,
                )
                logits = undirected_decoder_logits(model.s2gae_decoder, decoder_input, eval_edges)
                preds = torch.sigmoid(logits)
            else:
                preds = el_dot(emb_ppi_x[celltype], eval_edges, [])

            ppi_preds_all[celltype] = preds.to(saving_device)
            ppi_labels_all[celltype] = labels.to(saving_device)

    else:
        for i, celltype in enumerate(ppi_dict.keys()):
            start = batch_ppi_x.ptr[i]
            end = batch_ppi_x.ptr[i + 1]
            eval_edges = ppi_eval_edges[celltype].to(device)
            labels = ppi_labels[celltype].to(device)

            if s2gae_enabled:
                if getattr(model, "s2gae_decoder", None) is None or layer_outputs is None:
                    raise ValueError("S2GAE enabled but decoder/layer outputs missing.")
                layer_outputs_local = [h[start:end] for h in layer_outputs]
                decoder_input = _prepare_s2gae_decoder_input(
                    model.s2gae_decoder,
                    layer_outputs_local,
                    raw_embeddings=batch_ppi_x.x[start:end],
                )
                logits = undirected_decoder_logits(model.s2gae_decoder, decoder_input, eval_edges)
                preds = torch.sigmoid(logits)
            else:
                local_emb = emb_ppi_x[start:end]
                preds = el_dot(local_emb, eval_edges, [])

            ppi_preds_all[celltype] = preds.to(saving_device)
            ppi_labels_all[celltype] = labels.to(saving_device)

    return mg_pred, mg_data_eval, ppi_preds_all, ppi_labels_all
