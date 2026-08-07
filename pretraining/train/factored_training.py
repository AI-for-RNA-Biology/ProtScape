from time import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch_geometric.data import Batch, Data
from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score
from tqdm import tqdm

from ..losses.loss import calc_uniformity_loss, el_dot
from ..s2gae_utils import (
    batch_edge_mask,
    s2gae_loss,
    undirected_decoder_logits,
)
from .factored_batching import (
    batch_positive_exclusion_edge_index,
    build_cci_edge_index,
    build_mg_edge_data,
    get_cached_mg_edge_data_eval,
    negative_sampler,
    pred_batch2dict,
    train_batch2dict,
)


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
