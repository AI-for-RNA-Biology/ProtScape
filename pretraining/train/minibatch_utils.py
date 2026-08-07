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


def _make_exclusion_edge_index(edge_index):
    if edge_index is None or edge_index.numel() == 0:
        return edge_index
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _structured_negative_targets(pos_source, exclude_edge_index, num_nodes):
    device = pos_source.device
    num_nodes = int(num_nodes)
    pos_source = pos_source.to(device=device, dtype=torch.long)
    if pos_source.numel() == 0:
        return pos_source

    sampler_edge_index = torch.stack([pos_source, pos_source], dim=0)
    if exclude_edge_index is not None and exclude_edge_index.numel() > 0:
        exclude_edge_index = _make_exclusion_edge_index(
            exclude_edge_index.to(device=device, dtype=torch.long)
        )
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
        masked_data_dict (dict): Dictionary containing masked data for each subnetwork.
        metapath_adjs_dict (dict): Dictionary containing metapath adjacency matrices for each subnetwork.
        x_dict (dict): Dictionary containing node features for each subnetwork.
    """
    if verbose:
        print("Generating batches for metapaths:", metapaths)
    # Iterate through subnetworks
    
    def get_samples(key, data):
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
        # All edges and labels
        # Create a total edge set (Concatenate pos and negative edges)
        # 1 for positive edges and 0 for 
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) 
        total_edge_type = torch.cat([edge_type, neg_edge_type], dim=-1)
        y = torch.zeros(total_edge_index.size(1)).float() 
        y[:pos_edge_index.size(1)] = 1

        # Metapath adjs
        # Construct metapath adjacency matrices
        local_metapath_adjs_dict = construct_metapath(metapaths, pos_edge_index, edge_type, data.x.size(0), verbose=verbose)
        if verbose:
            print(f'[key = {key}] local_metapath_adjs_dict ', 'len ', len(metapath_adjs_dict[key]), ' - per entry:', [x.shape for x in local_metapath_adjs_dict])
        
        # Save information
        # Prepare node features (Save features (move to device, GPU/CPU).)
        local_x = data.x#.to(device)
        local_masked_data_dict = dict()
        # If PPI task --> build special DataLoader
        if ppi:
            data = Data(x = data.x, edge_index = total_edge_index, edge_attr = total_edge_type, y = y)
            data.n_id = torch.arange(data.num_nodes)
            
            # Based on loader type 
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
            # If not PPI, just store tensors
            local_masked_data_dict["total_edge_index"] = total_edge_index#.to(device)
            local_masked_data_dict["total_edge_type"] = total_edge_type#.to(device)
            local_masked_data_dict["y"] = y#.to(device)
            return (
                key,
                local_metapath_adjs_dict,
                local_x,
                local_masked_data_dict)
    
    # Initialize dictionaries
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
        masked_data_dict (dict): Dictionary containing masked data for each subnetwork.
        metapath_adjs_dict (dict): Dictionary containing metapath adjacency matrices for each subnetwork.
        x_dict (dict): Dictionary containing node features for each subnetwork.
    """
    if verbose:
        print("Generating batches for metapaths:", metapaths)
    # Iterate through subnetworks
    
    def get_samples(key, data):
        if mask == "train":
            raise NotImplementedError('The corrected sampler does not support a training mask.')
            pos_edge_index = data.edge_index[:, data.train_mask]
            pos_edge_index_messagepassing = pos_edge_index
            eval_edges = None # all of them are assumed to be seen in the GAE framework
            edge_type = data.edge_attr[data.train_mask]
            edge_type_messagepassing = edge_type
        elif mask == "val":
            pos_train_edge_index = data.edge_index[:, data.train_mask]
            pos_edge_index_messagepassing = pos_train_edge_index
            
            evaluated_edge_index = data.edge_index[:, data.val_mask]
            # contains all edges used for training and evaluation - so that the graphsaint sampler takes into account both for evaluation
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
            # contains all edges used for evaluation - so that the graphsaint sampler takes into account both for evaluation
            
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
        # Metapath adjs
        # Construct metapath adjacency matrices
        local_metapath_adjs_dict = construct_metapath(metapaths, pos_edge_index_messagepassing, edge_type_messagepassing, data.x.size(0), verbose=verbose)
        if verbose:
            print(f'[key = {key}] local_metapath_adjs_dict ', 'len ', len(metapath_adjs_dict[key]), ' - per entry:', [x.shape for x in local_metapath_adjs_dict])
        
        # Save information
        # Prepare node features (Save features (move to device, GPU/CPU).)
        local_x = data.x#.to(device)
        local_masked_data_dict = dict()
        # If PPI task --> build special DataLoader
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
            # If not PPI, just store tensors
            local_masked_data_dict["total_edge_index"] = pos_edge_index
            local_masked_data_dict["total_edge_type"] = edge_type
            #local_masked_data_dict["y"] = y#.to(device)
            return (
                key,
                local_metapath_adjs_dict,
                local_x,
                local_masked_data_dict)
    
    # Initialize dictionaries
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


# Negative Sample function ++
def negative_sampler(pos_edge_index, edge_type, edge_attr_dict, exclude_edge_index=None, num_nodes=None):
    """ Generate negative samples for link prediction.
    Args:
        pos_edge_index (torch.Tensor): Positive edge indices of shape (2, num_edges).
        edge_type (torch.Tensor): Edge types corresponding to the positive edges.
        edge_attr_dict (dict): Dictionary mapping edge attributes to indices.
    Returns:
        neg_edge_index (torch.Tensor): Negative edge indices of shape (2, num_neg_edges).
        neg_edge_type (torch.Tensor): Edge types corresponding to the negative edges.
    """
    # If no edges, just return input.
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
    # Loop through each edge type
    for attr, idx in edge_attr_dict.items():
        # Select edges of this type
        mask = (edge_type == idx)
        # If there are none, skip
        if mask.sum() == 0: continue
        # Create structured negative samples
        pos_rel_edge_index = pos_edge_index.T[mask].T        
        neg_source = pos_rel_edge_index[0]
        neg_rand = _structured_negative_targets(
            neg_source,
            exclude_edge_index,
            num_nodes,
        )
        # Form negative edges
        neg_rel_edge_index = torch.stack((neg_source, neg_rand), dim = 0)
        # Merge into total negatives
        if neg_edge_index == None: 
            neg_edge_index = neg_rel_edge_index 
        else: 
            neg_edge_index = torch.cat((neg_edge_index, neg_rel_edge_index), 1) 
        # Save edge types
        neg_edge_type.extend([idx] * mask.sum())

    return neg_edge_index, torch.tensor(neg_edge_type, dtype=edge_type.dtype, device=edge_type.device)

def pred_batch2dict(
        packed_batch: object, mg_x_ori: dict, ppi_x_ori: dict, cell_type_order: list, device: str,
        negative_sampling:bool=True,
        evaluation_mode: str='standard') -> dict:
    """
    Re-initialize :code:`ppi_x`, :code:`metagraph`, and transform packed batches of all graphs to a dictionary of batches. Note that different from :code:`train_batch2dict`, we are also re-initializing full :code:`ppi_x` because we are feeding all nodes instead of the sampled nodes in the batch to the model during prediction.
    
    :param packed_batch: An iterable (tuple if directly following unpacking of the output of :code:`generatePPIBatch`) storing batches of :class:`Data` from all graphs in one round.
    :param mg_x_ori: metagraph original node embeddings.
    :param ppi_x_ori: Original all PPI node embeddings
    :param cell_type_order: Cell type order.
    :param device: A string indicating the device. Default is "cuda".
    
    :return: A dictionary of edge data storing batches from all graphs in one round, :code:`ppi_x_batch`, :code:`ppi_node_ind_batch` extracted from batches, and the re-initialized node embeddings :code:`mg_x_init`.
    """
    # Re-initalize mg_x from mg_x in each batch
    ppi_x_init = {key: x.clone().to(device) for key, x in ppi_x_ori.items()}
    if mg_x_ori is not None:
        mg_x_init = mg_x_ori.clone().to(device) if len(mg_x_ori)!=0 else []
    else:
        mg_x_init = None

    # Unpack batches
    if evaluation_mode == 'standard':
        ppi_data_batch = {key: {} for key in cell_type_order}
        for ind, batch in enumerate(packed_batch):
            i = cell_type_order[ind]
            ori_map = batch.n_id
            ppi_data_batch[i]['total_edge_index'] = ori_map[batch.edge_index].to(device)
            ppi_data_batch[i]['total_edge_type'] = batch.edge_attr.to(device)
            ppi_data_batch[i]['y'] = batch.y.to(device)

    elif evaluation_mode == 'corrected':
        # full data as dict which does not contain negative samples
        ppi_data_batch = {}
        # we dont need to change batch as it is not used for inferring node embeddings
        # as pinnacle uses a copy of the correct graph in ppi_metapaths
        def build_eval_edges(batch, id_celltype):
            ppi_data_batch[id_celltype] = {}
            pos_eval_edges = batch.edge_index[:, batch.eval_edges == 1]
            assert pos_eval_edges.size(1) > 0, f'No eval edges found for cell type {id_celltype} in the batch.'
            pos_eval_edge_types = batch.edge_attr[batch.eval_edges == 1]
            if negative_sampling:
                neg_dst = _structured_negative_targets(
                    pos_eval_edges[0],
                    batch.edge_index.to(device=pos_eval_edges.device, dtype=torch.long),
                    batch.num_nodes,
                )
                neg_eval_edges = torch.stack([pos_eval_edges[0], neg_dst], dim=0)
                neg_eval_edge_types = batch.edge_attr[batch.eval_edges == 1]
                
                # combine pos and neg eval edges
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
    """
    Iterate batches for train. In each batch, only embeddings of nodes corresponding to
    the sampled edges (i.e., sampled nodes and their 2-hop neighbors) are attention-pooled
    to approximate the global embedding of a cell type's PPI, and used to update the node embedding in CCI. 
    
    Args:
        ppi_train_loader_dict (dict): A dictionary of PPI DataLoaders for each cell type.
        ppi_x_ori (dict): Original PPI node embeddings.
        ppi_metapaths_ori (dict): Original PPI metapaths.
        mg_x_ori (dict): Original metagraph node embeddings.
        mg_metapaths_train (list): Metapaths for training.
        mg_data_train (dict): Metagraph data for training.
        tissue_neighbors (dict): Tissue neighbors for the metagraph.
        model (torch.nn.Module): The model to train.
        hparams (dict): Hyperparameters for training.
        device (str): Device to use for training ('cpu' or 'cuda').
        wandb (object, optional): Weights & Biases object for logging. Defaults to None.
        center_loss (torch.nn.Module, optional): Center loss module. Defaults to None.
        optimizer (torch.optim, optional): Optimizer for training. Defaults to None.
        mask_train_ori (list, optional): Original train mask for nodes. Defaults to None.
        compute_metrics (bool, optional): Whether to compute metrics during training by batches. Defaults to True. Otherwise it is done on the global train PPI at the epoch end.
    Returns:
        tuple: 
            A tuple containing updated PPI node embeddings, metagraph node embeddings,
            metagraph predictions, PPI predictions, PPI data labels, and the total loss.
    """
    total_samples = total_loss = 0
    count = 0
    if store_global_pred_train:
        ppi_preds_all = {}
        ppi_data_y = {key:{'y': torch.tensor([]), 'total_edge_type': torch.tensor([])} for key in ppi_x_ori.keys()}
        ppi_x_out = {key: torch.zeros((x.shape[0], model.output)) for key, x in ppi_x_ori.items()}
        metrics = None # computed out of this function
        computed_metrics = None
    else:
        # we setup metrics of interest 
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

    # START BATCH FOR LOOP
    for packed_batch in zip(*ppi_train_loader_dict.values()):
        count += 1
        
        print(f"Training batch {count}")
        optimizer.zero_grad()
        
        # Unpack batches to edges, nodes, and indices, and reinitialize mg_x
        ppi_data_batch, ppi_x, ppi_node_ind_batch, ppi_metapaths_batch, _ = train_batch2dict(
            packed_batch, mg_x_ori, ppi_metapaths_ori, list(ppi_train_loader_dict.keys()), device)
        
        del packed_batch
        
        batch_size = sum([data['y'].shape[0] for data in ppi_data_batch.values()])  # Number of all samples across all cell types
        print('effective number of nodes in batch:', batch_size)

        # Generate PPI and metagraph embeddings & Compute predictions for metagraph
        ppi_x, mg_x = model(
            ppi_x, mg_x_ori, ppi_metapaths_batch, mg_metapaths_train, ppi_data_batch,
            mg_data_train["total_edge_index"], tissue_neighbors)

        # Compute predictions for metagraph for train
        print('--- compute predictions for metagraph')
        print('mg_data_train["total_edge_index"]:', mg_data_train["total_edge_index"].shape)
        print('model.mg_relw[mg_data_train["total_edge_type"]]:', model.mg_relw[mg_data_train["total_edge_type"]])
        
        mg_pred = el_dot(mg_x, mg_data_train["total_edge_index"], model.mg_relw[mg_data_train["total_edge_type"]])
        
        with torch.no_grad():
            if metrics is not None: # compute metrics on meta-graph per batch
                for m in ['roc_meta', 'ap_meta', 'acc_meta', 'f1_meta']:
                    _ = metrics[m](mg_pred, mg_data_train['y'].to(device=device, dtype=torch.int32))

        # Compute predictions for PPI layers
        print('--- compute predictions for ppi graphs')
        ppi_preds = dict()
        
        for celltype, x in ppi_x.items():
            # 1. Make prediction (assumes el_dot is GPU-compatible)
            ppi_preds[celltype] = el_dot(x, ppi_data_batch[celltype]['total_edge_index'], [])
            
            with torch.no_grad():
                if store_global_pred_train:
                    # 2. Store predictions on CPU
                    pred_cpu = ppi_preds[celltype].detach().cpu()
                    if celltype not in ppi_preds_all:
                        ppi_preds_all[celltype] = torch.empty((0, *pred_cpu.shape[1:])) if pred_cpu.ndim > 1 else torch.tensor([])
                    ppi_preds_all[celltype] = torch.cat([ppi_preds_all[celltype], pred_cpu])
                    # 3. Store labels on CPU
                    y_cpu = ppi_data_batch[celltype]['y'].detach().cpu()
                    ppi_data_y[celltype]['y'] = torch.cat([ppi_data_y[celltype]['y'], y_cpu])
                    # 4. Store edge types on CPU
                    edge_type_cpu = ppi_data_batch[celltype]['total_edge_type'].detach().cpu()
                    ppi_data_y[celltype]['total_edge_type'] = torch.cat([
                        ppi_data_y[celltype]['total_edge_type'], edge_type_cpu
                    ])
                    # 5. Store node embeddings on CPU (ensure index is also on CPU)
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

        # Get embeddings
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
        
        # Update
        for param in center_loss.parameters():
            param.grad.data *= (hparams["lr_cent"] / (hparams["lambda"] * hparams["lr"]))
        if hparams['gradclip'] != -1: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['gradclip'])
        optimizer.step()
        
        # Calculate loss
        total_samples += batch_size
        total_loss += float(combined_loss) * batch_size
        # Note that here for simplicity the total loss rather than only the link prediction BCEloss is weighted by edge batch size. 

    print('count - number of different packed_batches')
    total_loss = total_loss/total_samples  # Weighted total train loss
    
    if metrics is not None:
        # aggregate metrics across batches for the epoch
        computed_metrics = {}
        for m in metrics.keys():
            computed_metrics[m] = metrics[m].compute().item()

    return ppi_x_out, mg_x, mg_pred, ppi_preds_all, ppi_data_y, total_loss, computed_metrics



def iterate_predict_batch(ppi_loader_dict: dict, ppi_x_ori: dict, ppi_metapaths_eval: dict, mg_x_ori: dict,  mg_metapaths: list, mg_data: dict, tissue_neighbors: dict, model: torch.nn.Module, hparams: dict, device: str) -> tuple:
    """
    Iterate batches for prediction (val/test). To ensure consistent results, the full :code:`ppi_x` is being updated each round with train (for validation), or train & val metapaths (for test), respectively. Minibatching is only performed for edges used for link prediction here to reduce memory cost. Setting val/test batch num to 1 is recommended wherever probable.
    
    :return: :code:`ppi_x`, :code:`mg_x`, :code:`mg_pred`, :code:`ppi_preds_all`, and :code:`ppi_data_y`.
    """
    ppi_preds_all = {}
    ppi_data_y = {key:{'y':torch.tensor([]), 'total_edge_type':torch.tensor([])} for key in ppi_x_ori.keys()}
    count = 0
    
    for packed_batch in zip(*ppi_loader_dict.values()):
        count += 1
        
        # Unpack batches and reinitialize mg_x
        ppi_data_batch, ppi_x, mg_x = pred_batch2dict(
            packed_batch, mg_x_ori, ppi_x_ori, list(ppi_loader_dict.keys()), device,
            evaluation_mode='standard')

        # Generate PPI and metagraph embeddings & Compute predictions for metagraph
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
    """
    Iterate batches for prediction (val/test). To ensure consistent results, the full :code:`ppi_x` is being updated each round with train (for validation), or train & val metapaths (for test), respectively. Minibatching is only performed for edges used for link prediction here to reduce memory cost. Setting val/test batch num to 1 is recommended wherever probable.
    
    :return: :code:`ppi_x`, :code:`mg_x`, :code:`mg_pred`, :code:`ppi_preds_all`, and :code:`ppi_data_y`.
    """
    ppi_preds_all = {}
    ppi_data_y = {key:{
        'y':None, 'total_edge_type':None
        } for key in ppi_x_ori.keys()}
    model.eval()
    
    with torch.no_grad():
        
        # Unpack batches and reinitialize mg_x
        ppi_data_batch, ppi_x, mg_x = pred_batch2dict(
            ppi_loader_dict, mg_x_ori, ppi_x_ori, list(ppi_loader_dict.keys()), device,
            evaluation_mode='corrected')

        # Generate PPI and metagraph embeddings & Compute predictions for metagraph
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
    """
    Construct the center loss mask for a batch of nodes during training. We first concatenate batched indices across cell types. Note that we must add each index a certain amount correponding to its cell type to represent its position in the concatenated list. We then find the intersection between the indices and the original train mask. The POSITIONS of the matching concatenated batched indices are the mask we want.
    
    :param original_mask: The original train mask for nodes for calculating center_loss.
    :param ppi_node_ind_batch: A dictionary of ppi node indices that are sampled in this minibatch.
    :param ppi_x_ori: A dictionary of original :code:`ppi_x`. We need it in order to know the original numbers of PPI nodes of each cell type, and the order of cell types in the original train mask.
    
    :return: A list of new train mask for center loss that can be directly applied on the batch center loss labels and the batch labels and the batch embeddings.
        
    """
    ppi_size = [x.shape[0] for x in ppi_x_ori.values()]
    ppi_cum_size = np.append(0, np.cumsum(ppi_size))
    ppi_node_ind_batch_concat = torch.cat([value + ppi_cum_size[i] for i, value in enumerate(ppi_node_ind_batch.values())]).detach().cpu().numpy()
    original_mask_set = set(original_mask)
    train_mask_batch = [i for i, ind in enumerate(ppi_node_ind_batch_concat) if ind in original_mask_set]
    
    return train_mask_batch
    

def train_batch2dict(packed_batch: object, mg_x_ori: dict, ppi_metapaths: dict, cell_type_order: list, device: str) -> dict:
    """
    
    
    Re-initialize :code:`metagraph`, and transform packed batches of all graphs to a dictionary of batches.
    Note that different from :code:`pred_batch2dict`, we are not re-initializing :code:`ppi_x` here because we are only feeding sampled nodes in the batch to the model during training.
    
    :param packed_batch: An iterable (tuple if directly following unpacking of the output of :code:`generatePPIBatch`) storing batches of :class:`Data` from all graphs in one round.
    :param mg_x_ori: metagraph original node embeddings.
    :param ppi_metapaths: All metapaths.
    :param cell_type_order: Cell type order.
    :param device: A string indicating the device. Default is "cuda".
    
    :return: A dictionary of edge data from all graphs in one round, :code:`ppi_x_batch`, :code:`ppi_node_ind_batch` and :code:`ppi_metapaths_batch` extracted from batches, and the re-initialized node embeddings :code:`mg_x_init`.
    """
    # Unpack batches
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
        
        # Metapath adjs
        ppi_metapaths_batch = construct_metapath(ppi_metapaths, batch.edge_index[:, batch.y.type(torch.bool)], batch.edge_attr[batch.y.type(torch.bool)], batch.x.shape[0])
        
        ppi_metapaths_out[i] = [ppi_metapaths_batch[0].to(device)]
    
    return ppi_data_batch, ppi_x_batch, ppi_node_ind_batch, ppi_metapaths_out, []#, mg_x_init
