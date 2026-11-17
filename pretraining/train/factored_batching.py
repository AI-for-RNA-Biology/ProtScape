# GraphSAINT: Graph Sampling Based Inductive Learning Method (Zeng et al., 2020).
# https://arxiv.org/abs/1907.04931

import os

import torch
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTEdgeSampler, GraphSAINTNodeSampler
from torch_geometric.utils import scatter
from tqdm import tqdm

from ..data_handler.generate_input import graph_saint_dryrun
from ..s2gae_utils import sample_structured_negative_targets


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
    """Build one PPI sampler per cell, keeping held-out edges marked for evaluation.
    Known positive edges are retained to exclude them during negative sampling."""
    if graph_saint_norm:
        assert mask == 'train'
        assert loader_type == 'graphsaint'

    def build_sampler(data, root_gsnorm_path=None):
        """Select split edges, attach GraphSAINT weights when requested and build the cell sampler."""
        full_positive_edge_index = data.edge_index
        if root_gsnorm_path is not None:
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
            # Sample training and validation edges together; mark validation edges below.
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
            # Sample all splits together; mark only test edges for scoring.

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
                            sample_coverage=gs_sample_coverage,
                            num_workers=16,
                            pin_memory=True)

                    # Load cached sampling weights.
                    gs_node_norm, gs_edge_norm = torch.load(gsnorm_path, weights_only=False) # gs_node_norm (N,) tensor; gs_edge_norm (E,) tensor
                    data.edge_loss_weight = gs_edge_norm # saved in data for edge loss normalization

                    if verbose:
                        print(f'gs_node_norm: min = {gs_node_norm.min()} / max = {gs_node_norm.max()} / mean = {gs_node_norm.mean()}')
                        print(f'gs_edge_norm: min = {gs_edge_norm.min()} / max = {gs_edge_norm.max()} / mean = {gs_edge_norm.mean()}')

                    # Correct random-walk edge weights for their sampling probabilities.
                    if gs_filter == 'RandomWalk':
                        # Compute full-graph random-walk weights.
                        ones_ = torch.ones(data.edge_index.shape[1])

                        deg = scatter(ones_, data.edge_index[1], dim=0, dim_size=data.x.shape[0], reduce='sum')
                        deg_inv = 1. / deg
                        deg_inv.masked_fill_(deg_inv == float('inf'), 0)

                        deg_inv_by_edge = deg_inv[data.edge_index[0, :]]
                        edge_weights = gs_edge_norm * deg_inv_by_edge
                        if verbose:
                            print(f'deg: min = {deg.min()} / max = {deg.max()} / mean = {deg.mean()}')
                            print(f'edge_weights: min = {edge_weights.min()} / max = {edge_weights.max()} / mean = {edge_weights.mean()}')

                    else:
                        raise NotImplementedError(f'GraphSAINT weighting requires RandomWalk, got {gs_filter}')

                    data.edge_weight = edge_weights

                if weighted_ppi_loss:
                    data.loss_edge_weights = loss_edge_weights
            else:
                # Evaluation convolutions compute their random-walk weights directly.
                data = Data(
                    x = data.x,
                    edge_index = pos_edge_index,
                    edge_attr=edge_type,
                    eval_edges = eval_edges,
                    n_id = torch.arange(data.x.shape[0]),
                )
                if weighted_ppi_loss:
                    data.loss_edge_weights = loss_edge_weights

            if evaluation_mode == 'local':
                if loader_type == "graphsaint":
                    loader = GraphSAINTEdgeSampler(
                        data,
                        batch_size = batch_size,
                        num_steps = 16,
                        pin_memory=pin_memory
                        )
                elif loader_type=='graphsaintnode':
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
    loader_dict = dict()

    if n_jobs is None:

        for i, (key, data) in tqdm(enumerate(data_dict.items()), desc="Generating batches", disable=not verbose):
            if root_gsnorm_path is not None:
                local_root_gsnorm_path = root_gsnorm_path + f'/celltype{key}/'
                os.makedirs(local_root_gsnorm_path, exist_ok=True)
            else:
                local_root_gsnorm_path = None
            loader_dict[key] = build_sampler(data, local_root_gsnorm_path)

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


def negative_sampler(
    pos_edge_index,
    edge_type,
    edge_attr_dict,
    k_negatives=1,
    exclude_edge_index=None,
    num_nodes=None,
    verbose=False,
):
    """Sample negative targets for each positive PPI edge, preserving its relation type."""
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
    for attr, idx in edge_attr_dict.items():
        mask = (edge_type == idx)
        if mask.sum() == 0: continue
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
            neg_rel_edge_index = torch.stack((neg_source[valid_mask], neg_rand), dim=0)

            if neg_edge_index is None:
                neg_edge_index = neg_rel_edge_index
            else:
                neg_edge_index = torch.cat((neg_edge_index, neg_rel_edge_index), 1)
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

    edge_index = make_undirected_edges(edge_index)
    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


def make_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Add reverse edges and remove duplicate directed pairs."""
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
    """Split CCI pairs once per seed; both directions of a pair stay in the same split."""
    if mg_data is None:
        return None

    ratios = list(split_ratios) if split_ratios is not None else [1.0, 0.0, 0.0]
    if len(ratios) != 3:
        raise ValueError(f"Expected train/val/test ratios, got {split_ratios!r}")
    total = float(sum(ratios))
    ratios = [r / total for r in ratios] if total > 0 else [1.0, 0.0, 0.0]
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
            splits[split_name] = make_undirected_edges(edges)

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
    """Pair positive CCI edges with sampled negatives and return edges, types and labels."""
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


def get_cached_mg_edge_data_eval(
    mg_data,
    edge_attr_dict,
    *,
    cell_ids=None,
    device=None,
    pos_edge_index=None,
    split="val",
):
    """Reuse the same CCI evaluation negatives for a split and set of cell IDs."""
    if mg_data is None:
        return None
    cache = getattr(mg_data, "cci_eval_cache", None)
    key = (split, None if cell_ids is None else tuple(int(x) for x in cell_ids))
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
    if mg_data_cached is None or device is None:
        return mg_data_cached
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in mg_data_cached.items()}


def pred_batch2dict(
    packed_batch: object,
    mg_x_ori: dict,
    cell_names: list,
    edge_attr_dict:dict,
    device: str,
    negative_sampling:bool=True,
    k_negatives:int=1,
    evaluation_mode='local') -> dict:
    """Return per-cell graphs, held-out edges and labels; remove held-out message edges."""
    if mg_x_ori is not None:
        raise NotImplementedError('batch transformer cannot handle metagraph for now')

    ppi_dict = {}
    ppi_eval_edges = {}
    ppi_labels = {}

    if evaluation_mode == 'local':
        batches = zip(cell_names, packed_batch)
    elif evaluation_mode == 'global':
        batches = packed_batch.items()
    else:
        raise ValueError(f"Unknown evaluation_mode={evaluation_mode!r}")

    for cell_name, batch in batches:
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

        # Held-out edges are scored but excluded from message passing.
        batch.edge_index = batch.edge_index[:, batch.eval_edges == 0]
        batch.eval_edges = None
        ppi_dict[cell_name] = batch.to(device)

    return ppi_dict, ppi_eval_edges, ppi_labels


def train_batch2dict(
    packed_batch: object,
    mg_x_ori: dict,
    cell_names: list,
    device: str,
    verbose:bool=False) -> dict:
    """Associate sampled graphs with their cell IDs and move them to the training device."""
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
