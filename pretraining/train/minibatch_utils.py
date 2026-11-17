import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, GraphSAINTEdgeSampler
from torch_geometric.utils import structured_negative_sampling
import torch.nn.functional as F

from ..utils import construct_metapath, get_embeddings
from ..losses.loss import el_dot, calc_link_pred_loss, calc_center_loss

from tqdm import tqdm
from time import time

from torchmetrics import F1Score, Accuracy, AUROC, AveragePrecision
import wandb
from joblib import Parallel, delayed


def sample_negative_targets(pos_source, exclude_edge_index, num_nodes):
    """Use PyG structured sampling to draw non-neighbors for the requested sources."""
    device = pos_source.device
    num_nodes = int(num_nodes)
    pos_source = pos_source.to(device=device, dtype=torch.long)
    if pos_source.numel() == 0:
        return pos_source

    sampler_edge_index = torch.stack([pos_source, pos_source], dim=0)
    if exclude_edge_index is not None and exclude_edge_index.numel() > 0:
        exclude_edge_index = exclude_edge_index.to(device=device, dtype=torch.long)
        exclude_edge_index = torch.cat([exclude_edge_index, exclude_edge_index.flip(0)], dim=1)
        sampler_edge_index = torch.cat([sampler_edge_index, exclude_edge_index], dim=1)

    _, _, neg_target = structured_negative_sampling(
        sampler_edge_index,
        num_nodes=num_nodes,
        contains_neg_self_loops=False,
    )
    return neg_target[:pos_source.numel()].to(device)


def generate_batch(
    data_dict,
    metapaths,
    edge_attr_dict,
    mask,
    batch_size,
    device,
    ppi=False,
    loader_type="graphsaint",
    num_layers=2,
    verbose=False,
    n_jobs:int=None,
    pin_memory:bool=True):
    """Build PINNACLE samplers, labeled edges, metapaths and node features for a split."""
    if verbose:
        print("Generating batches for metapaths:", metapaths)
    def get_samples(key, data):
        """Create one graph's sampled inputs, labeled edges, metapaths and original features."""
        if mask == "train":
            pos_edge_index = data.edge_index[:, data.train_mask]
            edge_type = data.edge_attr[data.train_mask]
        elif mask == "val":
            pos_edge_index = data.edge_index[:, data.val_mask]
            edge_type = data.edge_attr[data.val_mask]
        elif mask == "test":
            pos_edge_index = data.edge_index[:, data.test_mask]
            edge_type = data.edge_attr[data.test_mask]
        else:
            pos_edge_index = data.edge_index
            edge_type = data.edge_attr
        # Generate Negative edges
        neg_edge_index, neg_edge_type = negative_sampler(
            pos_edge_index,
            edge_type,
            edge_attr_dict,
            exclude_edge_index=data.edge_index,
            num_nodes=data.x.size(0),
        )
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        total_edge_type = torch.cat([edge_type, neg_edge_type], dim=-1)
        y = torch.zeros(total_edge_index.size(1)).float()
        y[:pos_edge_index.size(1)] = 1

        # Construct metapath adjacency matrices
        local_metapath_adjs_dict = construct_metapath(metapaths, pos_edge_index, edge_type, data.x.size(0), verbose=verbose)
        if verbose:
            print(f'[key = {key}] local_metapath_adjs_dict ', 'len ', len(metapath_adjs_dict[key]), ' - per entry:', [x.shape for x in local_metapath_adjs_dict])

        local_x = data.x
        local_masked_data_dict = dict()
        if ppi:
            data = Data(x = data.x, edge_index = total_edge_index, edge_attr = total_edge_type, y = y)
            data.n_id = torch.arange(data.num_nodes)

            if loader_type == "neighbor":
                loader = NeighborLoader(
                    data,
                    num_neighbors = [-1] * num_layers,
                    batch_size = batch_size,
                    input_nodes =
                    torch.arange(data.num_nodes),
                    shuffle = True,
                    pin_memory=pin_memory)
            elif loader_type == "graphsaint":
                loader = GraphSAINTEdgeSampler(
                    data,
                    batch_size = batch_size,
                    num_steps = 16,
                    pin_memory=pin_memory
                    )
            else:
                raise NotImplementedError
            local_loader = loader
            local_masked_data_dict["total_edge_type"] = total_edge_type
            return (
                key,
                local_metapath_adjs_dict,
                local_x,
                local_masked_data_dict,
                local_loader)
        else:
            local_masked_data_dict["total_edge_index"] = total_edge_index
            local_masked_data_dict["total_edge_type"] = total_edge_type
            local_masked_data_dict["y"] = y
            return (
                key,
                local_metapath_adjs_dict,
                local_x,
                local_masked_data_dict)
    masked_data_dict = dict()
    metapath_adjs_dict = dict()
    x_dict = dict()
    loader_dict = dict()
    if n_jobs is None:
        for i, (key, data) in tqdm(enumerate(data_dict.items()), desc="Generating batches", disable=not verbose):
            res = get_samples(key, data)
            metapath_adjs_dict[key] = [x.to(device) for x in res[1]]
            x_dict[key] = res[2].to(device)
            masked_data_dict[key] = {}
            for subkey in res[3].keys():
                masked_data_dict[key][subkey] = res[3][subkey].to(device)
            if ppi:
                loader_dict[key] = res[4]

    else:
        verbose_joblib = 10 if verbose else 0
        list_res = Parallel(n_jobs=n_jobs, verbose=verbose_joblib)(
            delayed(get_samples)(key, data) for key, data in tqdm(data_dict.items())
            )
        for res in list_res:
            key = res[0]
            metapath_adjs_dict[key] = res[1]
            x_dict[key] = res[2].to(device)
            masked_data_dict[key] = {}
            for subkey in res[3].keys():
                masked_data_dict[key][subkey] = res[3][subkey].to(device)
            if ppi:
                loader_dict[key] = res[4]

    return loader_dict, masked_data_dict, metapath_adjs_dict, x_dict


def generate_batch_corrected(
    data_dict,
    metapaths,
    edge_attr_dict,
    mask,
    batch_size,
    device,
    ppi=False,
    loader_type="graphsaint",
    num_layers=2,
    verbose=False,
    n_jobs:int=None,
    pin_memory:bool=True):
    """Build evaluation samplers while keeping held-out edges out of message passing."""
    if verbose:
        print("Generating batches for metapaths:", metapaths)
    def get_samples(key, data):
        """Prepare evaluation edges separately from the graph used for message passing."""
        if mask == "train":
            raise NotImplementedError('The corrected sampler does not support a training mask.')
        elif mask == "val":
            pos_train_edge_index = data.edge_index[:, data.train_mask]
            pos_edge_index_messagepassing = pos_train_edge_index

            evaluated_edge_index = data.edge_index[:, data.val_mask]
            # Sample training and validation edges together; mark validation edges below.
            pos_edge_index = torch.cat([pos_train_edge_index, evaluated_edge_index], dim=-1)

            eval_edges = torch.ones(pos_edge_index.shape[-1], dtype=torch.int8)
            eval_edges[:pos_train_edge_index.shape[-1]] = 0  # mark training edges as 0, evaluation edges as 1

            train_edge_type = data.edge_attr[data.train_mask]
            edge_type_messagepassing = train_edge_type
            val_edge_type = data.edge_attr[data.val_mask]
            edge_type = torch.cat([train_edge_type, val_edge_type], dim=0)
        elif mask == "test":
            pos_train_edge_index = data.edge_index[:, data.train_mask]
            pos_val_edge_index = data.edge_index[:, data.val_mask]
            pos_edge_index_messagepassing = torch.cat([pos_train_edge_index, pos_val_edge_index], dim=-1)

            evaluated_edge_index = data.edge_index[:, data.test_mask]
            # Sample all splits together; mark only test edges for scoring.

            pos_edge_index = torch.cat([pos_train_edge_index, pos_val_edge_index, evaluated_edge_index], dim=-1)

            eval_edges = torch.ones(pos_edge_index.shape[-1], dtype=torch.int8)
            eval_edges[:pos_train_edge_index.shape[-1] + pos_val_edge_index.shape[-1]] = 0  # mark training edges as 0, evaluation edges as 1

            train_edge_type = data.edge_attr[data.train_mask]
            val_edge_type = data.edge_attr[data.val_mask]
            edge_type_messagepassing = torch.cat([train_edge_type, val_edge_type], dim=0)
            test_edge_type = data.edge_attr[data.test_mask]
            edge_type = torch.cat([train_edge_type, val_edge_type, test_edge_type], dim=0)

        else:
            pos_edge_index = data.edge_index
            edge_type = data.edge_attr
            pos_edge_index_messagepassing = pos_edge_index
            edge_type_messagepassing=edge_type
        # Construct metapath adjacency matrices
        local_metapath_adjs_dict = construct_metapath(metapaths, pos_edge_index_messagepassing, edge_type_messagepassing, data.x.size(0), verbose=verbose)
        if verbose:
            print(f'[key = {key}] local_metapath_adjs_dict ', 'len ', len(metapath_adjs_dict[key]), ' - per entry:', [x.shape for x in local_metapath_adjs_dict])

        local_x = data.x
        local_masked_data_dict = dict()
        if ppi:
            if eval_edges is None:
                data = Data(
                    x = data.x,
                    edge_index = pos_edge_index,
                    edge_attr=edge_type,
                    n_id = torch.arange(data.x.shape[0]),
                )
            else:
                data = Data(
                    x = data.x,
                    edge_index = pos_edge_index,
                    edge_attr=edge_type,
                    eval_edges = eval_edges,
                    n_id = torch.arange(data.x.shape[0]),
                )

            local_loader = data
            local_masked_data_dict["total_edge_type"] = edge_type
            return (
                key,
                local_metapath_adjs_dict,
                local_x,
                local_masked_data_dict,
                local_loader)
        else:
            local_masked_data_dict["total_edge_index"] = pos_edge_index
            local_masked_data_dict["total_edge_type"] = edge_type
            return (
                key,
                local_metapath_adjs_dict,
                local_x,
                local_masked_data_dict)
    masked_data_dict = dict()
    metapath_adjs_dict = dict()
    x_dict = dict()
    loader_dict = dict()
    if n_jobs is None:
        for i, (key, data) in tqdm(enumerate(data_dict.items()), desc="Generating batches", disable=not verbose):
            res = get_samples(key, data)
            metapath_adjs_dict[key] = [x.to(device) for x in res[1]]
            x_dict[key] = res[2].to(device)
            masked_data_dict[key] = {}
            for subkey in res[3].keys():
                masked_data_dict[key][subkey] = res[3][subkey].to(device)
            if ppi:
                loader_dict[key] = res[4]
    else:
        verbose_joblib = 10 if verbose else 0
        list_res = Parallel(n_jobs=n_jobs, verbose=verbose_joblib)(
            delayed(get_samples)(key, data) for key, data in tqdm(data_dict.items())
            )
        for res in list_res:
            key = res[0]
            metapath_adjs_dict[key] = res[1]
            x_dict[key] = res[2].to(device)
            masked_data_dict[key] = {}
            for subkey in res[3].keys():
                masked_data_dict[key][subkey] = res[3][subkey].to(device)
            if ppi:
                loader_dict[key] = res[4]

    return loader_dict, masked_data_dict, metapath_adjs_dict, x_dict


def negative_sampler(pos_edge_index, edge_type, edge_attr_dict, exclude_edge_index=None, num_nodes=None):
    """Sample negative targets while preserving each positive edge's relation type."""
    if len(edge_type) == 0:
        return pos_edge_index, edge_type

    neg_edge_index = None
    neg_edge_type = []
    if num_nodes is None:
        max_pos = int(pos_edge_index.max().item()) if pos_edge_index.numel() > 0 else -1
        max_exclude = int(exclude_edge_index.max().item()) if exclude_edge_index is not None and exclude_edge_index.numel() > 0 else -1
        num_nodes = max(max_pos, max_exclude) + 1
    if exclude_edge_index is None:
        exclude_edge_index = pos_edge_index
    for attr, idx in edge_attr_dict.items():
        mask = (edge_type == idx)
        if mask.sum() == 0: continue
        pos_rel_edge_index = pos_edge_index.T[mask].T
        neg_source = pos_rel_edge_index[0]
        neg_rand = sample_negative_targets(
            neg_source,
            exclude_edge_index,
            num_nodes,
        )
        neg_rel_edge_index = torch.stack((neg_source, neg_rand), dim = 0)
        if neg_edge_index == None:
            neg_edge_index = neg_rel_edge_index
        else:
            neg_edge_index = torch.cat((neg_edge_index, neg_rel_edge_index), 1)
        neg_edge_type.extend([idx] * mask.sum())

    return neg_edge_index, torch.tensor(neg_edge_type, dtype=edge_type.dtype, device=edge_type.device)

def pred_batch2dict(
        packed_batch: object, mg_x_ori: dict, ppi_x_ori: dict, cell_type_order: list, device: str,
        negative_sampling:bool=True,
        evaluation_mode: str='standard') -> dict:
    """Unpack PINNACLE evaluation graphs into edges, node features, indices and metagraph features."""
    # Re-initalize mg_x from mg_x in each batch
    ppi_x_init = {key: x.clone().to(device) for key, x in ppi_x_ori.items()}
    if mg_x_ori is not None:
        mg_x_init = mg_x_ori.clone().to(device) if len(mg_x_ori)!=0 else []
    else:
        mg_x_init = None

    if evaluation_mode == 'standard':
        ppi_data_batch = {key: {} for key in cell_type_order}
        for ind, batch in enumerate(packed_batch):
            i = cell_type_order[ind]
            ori_map = batch.n_id
            ppi_data_batch[i]['total_edge_index'] = ori_map[batch.edge_index].to(device)
            ppi_data_batch[i]['total_edge_type'] = batch.edge_attr.to(device)
            ppi_data_batch[i]['y'] = batch.y.to(device)

    elif evaluation_mode == 'corrected':
        ppi_data_batch = {}
        # Full-graph inference uses the original features below.
        def build_eval_edges(batch, id_celltype):
            """Pair held-out positive edges with sampled negatives for this cell."""
            ppi_data_batch[id_celltype] = {}
            pos_eval_edges = batch.edge_index[:, batch.eval_edges == 1]
            assert pos_eval_edges.size(1) > 0, f'No eval edges found for cell type {id_celltype} in the batch.'
            pos_eval_edge_types = batch.edge_attr[batch.eval_edges == 1]
            if negative_sampling:
                neg_dst = sample_negative_targets(
                    pos_eval_edges[0],
                    batch.edge_index.to(device=pos_eval_edges.device, dtype=torch.long),
                    batch.num_nodes,
                )
                neg_eval_edges = torch.stack([pos_eval_edges[0], neg_dst], dim=0)
                neg_eval_edge_types = batch.edge_attr[batch.eval_edges == 1]

                eval_edges = torch.cat([pos_eval_edges, neg_eval_edges], dim=-1)
                eval_edge_types = torch.cat([pos_eval_edge_types, neg_eval_edge_types], dim=0)

                labels = torch.zeros(eval_edges.size(1), dtype=torch.float32)
                labels[:pos_eval_edges.size(1)] = 1.0
            else:
                eval_edges = pos_eval_edges
                labels = torch.ones(pos_eval_edges.size(1), dtype=torch.float32)

            ppi_data_batch[id_celltype]['total_edge_index'] = eval_edges.to(device)
            ppi_data_batch[id_celltype]['y'] = labels.to(device)
            ppi_data_batch[id_celltype]['total_edge_type'] = eval_edge_types.to(device)

        for celltype in packed_batch.keys():
            batch = packed_batch[celltype]
            build_eval_edges(batch, celltype)

    return ppi_data_batch, ppi_x_init, mg_x_init


def iterate_train_batch(
    ppi_train_loader_dict: dict,
    ppi_x_ori: dict,
    ppi_metapaths_ori: dict,
    mg_x_ori: dict,
    mg_metapaths_train: list,
    mg_data_train: dict,
    tissue_neighbors: dict,
    model: torch.nn.Module,
    hparams: dict,
    device: str,
    wandb: object=None,
    center_loss: torch.nn.Module=None,
    optimizer: torch.optim=None,
    mask_train_ori: list=None,
    store_global_pred_train:bool=False) -> tuple:
    """Train PINNACLE over paired cell batches and return embeddings, predictions and metrics."""
    total_samples = total_loss = 0
    count = 0
    if store_global_pred_train:
        ppi_preds_all = {}
        ppi_data_y = {key:{'y': torch.tensor([]), 'total_edge_type': torch.tensor([])} for key in ppi_x_ori.keys()}
        ppi_x_out = {key: torch.zeros((x.shape[0], model.output)) for key, x in ppi_x_ori.items()}
        metrics = None # computed out of this function
        computed_metrics = None
    else:
        # Accumulate edge metrics across sampled batches.
        train_roc_score_ppi = AUROC(task='binary').to(device)
        train_ap_score_ppi = AveragePrecision(task='binary').to(device)
        train_acc_score_ppi = Accuracy(task='binary').to(device)
        train_f1_score_ppi = F1Score(task='binary', average='macro').to(device)
        train_roc_score_meta = AUROC(task='binary').to(device)
        train_ap_score_meta = AveragePrecision(task='binary').to(device)
        train_acc_score_meta = Accuracy(task='binary').to(device)
        train_f1_score_meta = F1Score(task='binary', average='macro').to(device)

        ppi_preds_all = None
        ppi_data_y = None
        ppi_x_out = None
        metrics = {
            'roc_ppi': train_roc_score_ppi, 'ap_ppi': train_ap_score_ppi, 'acc_ppi': train_acc_score_ppi, 'f1_ppi': train_f1_score_ppi,
            'roc_meta': train_roc_score_meta, 'ap_meta': train_ap_score_meta, 'acc_meta': train_acc_score_meta, 'f1_meta': train_f1_score_meta
        }
    for packed_batch in zip(*ppi_train_loader_dict.values()):
        count += 1

        print(f"Training batch {count}")
        optimizer.zero_grad()

        # Unpack protein graphs and reset metagraph features for this batch.
        ppi_data_batch, ppi_x, ppi_node_ind_batch, ppi_metapaths_batch, _ = train_batch2dict(
            packed_batch, mg_x_ori, ppi_metapaths_ori, list(ppi_train_loader_dict.keys()), device)

        del packed_batch

        batch_size = sum([data['y'].shape[0] for data in ppi_data_batch.values()])  # Number of all samples across all cell types
        print('effective number of nodes in batch:', batch_size)

        # Encode protein and metagraph features.
        ppi_x, mg_x = model(
            ppi_x, mg_x_ori, ppi_metapaths_batch, mg_metapaths_train, ppi_data_batch,
            mg_data_train["total_edge_index"], tissue_neighbors)

        # Compute predictions for metagraph for train
        print('--- compute predictions for metagraph')

        mg_pred = el_dot(mg_x, mg_data_train["total_edge_index"], model.mg_relw[mg_data_train["total_edge_type"]])

        with torch.no_grad():
            if metrics is not None: # compute metrics on meta-graph per batch
                for m in ['roc_meta', 'ap_meta', 'acc_meta', 'f1_meta']:
                    _ = metrics[m](mg_pred, mg_data_train['y'].to(device=device, dtype=torch.int32))

        # Compute predictions for PPI layers
        print('--- compute predictions for ppi graphs')
        ppi_preds = dict()

        for celltype, x in ppi_x.items():
            ppi_preds[celltype] = el_dot(x, ppi_data_batch[celltype]['total_edge_index'], [])

            with torch.no_grad():
                if store_global_pred_train:
                    pred_cpu = ppi_preds[celltype].detach().cpu()
                    if celltype not in ppi_preds_all:
                        ppi_preds_all[celltype] = torch.empty((0, *pred_cpu.shape[1:])) if pred_cpu.ndim > 1 else torch.tensor([])
                    ppi_preds_all[celltype] = torch.cat([ppi_preds_all[celltype], pred_cpu])
                    y_cpu = ppi_data_batch[celltype]['y'].detach().cpu()
                    ppi_data_y[celltype]['y'] = torch.cat([ppi_data_y[celltype]['y'], y_cpu])
                    edge_type_cpu = ppi_data_batch[celltype]['total_edge_type'].detach().cpu()
                    ppi_data_y[celltype]['total_edge_type'] = torch.cat([
                        ppi_data_y[celltype]['total_edge_type'], edge_type_cpu
                    ])
                    indices_cpu = ppi_node_ind_batch[celltype].cpu()
                    ppi_x_out[celltype][indices_cpu] = x.detach().cpu()

        if metrics is not None: # compute metrics on ppi graphs per batch
            ppi_batch_preds = torch.cat([pred for pred in ppi_preds.values()], dim=0)
            ppi_batch_labels = torch.cat([data['y'] for data in ppi_data_batch.values()], dim=0).to(device=device, dtype=torch.int32)
            for m in ['roc_ppi', 'ap_ppi', 'acc_ppi', 'f1_ppi']:
                _ = metrics[m](ppi_batch_preds, ppi_batch_labels)
            del ppi_batch_preds
            del ppi_batch_labels

        # Compute train loss
        ppi_loss, mg_loss = calc_link_pred_loss(mg_pred, mg_data_train, ppi_preds, ppi_data_batch, hparams['loss_type'])
        link_loss = hparams['theta'] * ppi_loss + (1 - hparams['theta']) * mg_loss

        embed = torch.cat(list(ppi_x.values())) # Protein
        centers = mg_x[0:len(ppi_x)] # Cell type

        # Protein labels
        center_loss_labels = torch.cat([(torch.ones(x.shape[0]) * key).to(torch.long) for key, x in ppi_x.items()])  # Build center loss labels based on batched nodes to ensure consistency with embedding labels

        # Train mask
        train_mask = construct_batch_center_loss_mask(mask_train_ori, ppi_node_ind_batch, ppi_x_ori)

        # Center loss
        cent_loss = calc_center_loss(center_loss, embed, centers, center_loss_labels, train_mask)
        print("Link Prediction: ", link_loss, "Center Loss: ", cent_loss)
        wandb.log({"Link Prediction Loss": link_loss, "Center Loss": cent_loss})
        combined_loss = link_loss + (cent_loss * hparams["lambda"])
        combined_loss.backward()

        for param in center_loss.parameters():
            param.grad.data *= (hparams["lr_cent"] / (hparams["lambda"] * hparams["lr"]))
        if hparams['gradclip'] != -1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['gradclip'])
        optimizer.step()

        total_samples += batch_size
        total_loss += float(combined_loss) * batch_size
        # Weight the combined loss by the number of sampled protein nodes.

    print('count - number of different packed_batches')
    total_loss = total_loss/total_samples  # Weighted total train loss

    if metrics is not None:
        # aggregate metrics across batches for the epoch
        computed_metrics = {}
        for m in metrics.keys():
            computed_metrics[m] = metrics[m].compute().item()

    return ppi_x_out, mg_x, mg_pred, ppi_preds_all, ppi_data_y, total_loss, computed_metrics


def iterate_predict_batch(ppi_loader_dict: dict, ppi_x_ori: dict, ppi_metapaths_eval: dict, mg_x_ori: dict,  mg_metapaths: list, mg_data: dict, tissue_neighbors: dict, model: torch.nn.Module, hparams: dict, device: str) -> tuple:
    """Score PINNACLE PPI and metagraph edges over sampled evaluation batches."""
    ppi_preds_all = {}
    ppi_data_y = {key:{'y':torch.tensor([]), 'total_edge_type':torch.tensor([])} for key in ppi_x_ori.keys()}
    count = 0

    for packed_batch in zip(*ppi_loader_dict.values()):
        count += 1

        # Unpack the sampled graphs.
        ppi_data_batch, ppi_x, mg_x = pred_batch2dict(
            packed_batch, mg_x_ori, ppi_x_ori, list(ppi_loader_dict.keys()), device,
            evaluation_mode='standard')

        # Encode protein and metagraph features.
        if mg_data["total_edge_index"] !=  []:
            mg_data["total_edge_index"] = mg_data["total_edge_index"].to(device)
        ppi_x, mg_x = get_embeddings(model.to(device), ppi_x, mg_x, ppi_metapaths_eval, mg_metapaths, ppi_data_batch, mg_data["total_edge_index"], tissue_neighbors)

        # Compute predictions for metagraph for val/test only once
        if count == 1:
            mg_pred = el_dot(mg_x.to(device), mg_data["total_edge_index"], model.mg_relw[mg_data["total_edge_type"]])

        # Compute predictions for PPI layers
        ppi_preds = dict()
        for celltype, x in ppi_x.items():
            ppi_preds[celltype] = el_dot(x.to(device), ppi_data_batch[celltype]['total_edge_index'].to(device), [])
            ppi_preds_all[celltype] = torch.cat([ppi_preds_all.setdefault(celltype, torch.tensor([])), ppi_preds[celltype].detach().cpu()])
            ppi_data_y[celltype]['y'] = torch.cat([ppi_data_y[celltype]['y'], ppi_data_batch[celltype]['y'].detach().cpu()])
            ppi_data_y[celltype]['total_edge_type'] = torch.cat([ppi_data_y[celltype]['total_edge_type'], ppi_data_batch[celltype]['total_edge_type'].detach().cpu()])

    return ppi_x, mg_x, mg_pred, ppi_preds_all, ppi_data_y


def iterate_predict(
    ppi_loader_dict: dict,
    ppi_x_ori: dict,
    ppi_metapaths_eval: dict,
    mg_x_ori: dict,
    mg_metapaths: list,
    mg_data: dict,
    tissue_neighbors: dict,
    model: torch.nn.Module,
    hparams: dict,
    device: str) -> tuple:
    """Score PINNACLE edges using complete cell graphs."""
    ppi_preds_all = {}
    ppi_data_y = {key:{
        'y':None, 'total_edge_type':None
        } for key in ppi_x_ori.keys()}
    model.eval()

    with torch.no_grad():

        # Unpack the sampled graphs.
        ppi_data_batch, ppi_x, mg_x = pred_batch2dict(
            ppi_loader_dict, mg_x_ori, ppi_x_ori, list(ppi_loader_dict.keys()), device,
            evaluation_mode='corrected')

        # Encode protein and metagraph features.
        if mg_data["total_edge_index"] !=  []: mg_data["total_edge_index"] = mg_data["total_edge_index"].to(device)
        ppi_x, mg_x = get_embeddings(model.to(device), ppi_x, mg_x, ppi_metapaths_eval, mg_metapaths, ppi_data_batch, mg_data["total_edge_index"], tissue_neighbors)

        # Compute predictions for metagraph for val/test only once
        mg_pred = el_dot(mg_x.to(device), mg_data["total_edge_index"], model.mg_relw[mg_data["total_edge_type"]])

        # Compute predictions for PPI layers
        for celltype, x in ppi_x.items():
            ppi_preds_all[celltype] = el_dot(x.to(device), ppi_data_batch[celltype]['total_edge_index'].to(device), [])
            ppi_data_y[celltype]['y'] = ppi_data_batch[celltype]['y'].detach()
            ppi_data_y[celltype]['total_edge_type'] =  ppi_data_batch[celltype]['total_edge_type'].detach()

    return ppi_x, mg_x, mg_pred, ppi_preds_all, ppi_data_y


def construct_batch_center_loss_mask(original_mask: list, ppi_node_ind_batch: dict, ppi_x_ori: dict) -> list:
    """Map the original center-loss mask onto nodes sampled in the current batch."""
    ppi_size = [x.shape[0] for x in ppi_x_ori.values()]
    ppi_cum_size = np.append(0, np.cumsum(ppi_size))
    ppi_node_ind_batch_concat = torch.cat([value + ppi_cum_size[i] for i, value in enumerate(ppi_node_ind_batch.values())]).detach().cpu().numpy()
    original_mask_set = set(original_mask)
    train_mask_batch = [i for i, ind in enumerate(ppi_node_ind_batch_concat) if ind in original_mask_set]

    return train_mask_batch


def train_batch2dict(packed_batch: object, mg_x_ori: dict, ppi_metapaths: dict, cell_type_order: list, device: str) -> dict:
    """Unpack sampled graphs and gather their protein features and metapaths."""
    ppi_data_batch = {key:{} for key in cell_type_order}
    ppi_x_batch = {}
    ppi_node_ind_batch = {}
    ppi_metapaths_out = {}
    for ind, batch in enumerate(packed_batch):
        i = cell_type_order[ind]
        ppi_node_ind_batch[i] = batch.n_id.to(device)
        ppi_x_batch[i] = batch.x.to(device)
        ppi_data_batch[i]['total_edge_index'] = batch.edge_index.to(device)
        ppi_data_batch[i]['total_edge_type'] = batch.edge_attr.to(device)
        ppi_data_batch[i]['y'] = batch.y.to(device)

        ppi_metapaths_batch = construct_metapath(ppi_metapaths, batch.edge_index[:, batch.y.type(torch.bool)], batch.edge_attr[batch.y.type(torch.bool)], batch.x.shape[0])

        ppi_metapaths_out[i] = [ppi_metapaths_batch[0].to(device)]

    return ppi_data_batch, ppi_x_batch, ppi_node_ind_batch, ppi_metapaths_out, []#, mg_x_init
