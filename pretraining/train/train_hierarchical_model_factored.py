# General
import os
import numpy as np
import pandas as pd
import random

# Pytorch
import torch

# W&B
import wandb

# Own code
from .. import utils
from . import minibatch_factored_utils as mb_utils

from time import time


def _capture_rng_state():
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def _build_checkpoint_payload(epoch, model, optimizer, score=None, score_metric=None):
    payload = {
        "epoch": epoch,
        "model": model,
        "optimizer": optimizer,
        "rng_state": _capture_rng_state(),
    }
    if score is not None:
        payload["score"] = float(score)
    if score_metric is not None:
        payload["score_metric"] = score_metric
    return payload


def _save_checkpoint(payload, path):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
    os.replace(tmp_path, path)


def get_training_data(
    cfg,
    ppi_data,
    edge_attr_dict,
    device='cuda:0',
    n_jobs=None,
    evaluation_mode='global',
    root_gsnorm_path=None):
    # Generate PPI batches for train and validation
    
    # Also notice that batch_size is only used for the PPI data, not for the metagraph
    # and relates to the number of subgraphs designed by loaders like GraphSAINT or NeighborSampler
    # to handle large scale graphs
    
    # for train - evaluation mode is always 'local' to operate on all contexts simultaneously
    ppi_train_loader_dict = mb_utils.generate_batch(
        ppi_data, edge_attr_dict, "train", cfg.batch_size, device, ppi=True,
        loader_type=cfg.loader, n_jobs=n_jobs, evaluation_mode='local',
        graph_saint_norm= cfg.graph_saint_norm, root_gsnorm_path=root_gsnorm_path,
        weighted_ppi_loss=cfg.weighted_ppi_loss)
    
    ppi_val_loader_dict = mb_utils.generate_batch(
        ppi_data, edge_attr_dict, "val", cfg.batch_size, device, ppi=True,
        loader_type=cfg.loader, n_jobs=n_jobs, evaluation_mode=evaluation_mode,
        graph_saint_norm=False, weighted_ppi_loss=cfg.weighted_ppi_loss)
    
    # Generate metagraph batches for train and validation when requested
    if getattr(cfg, "use_metagraph", False):
        # Metagraph batches are built on-the-fly for CCI-only edges in minibatch_factored_utils.
        pass
       
    return ppi_train_loader_dict, ppi_val_loader_dict


def train(
        cfg,
        epoch,
        model,
        optimizer,
        train_mask,
        dict_save_models,
        save_latest_model,
        ppi_data,
        mg_data,
        edge_attr_dict,
        celltype_map,
        tissue_neighbors,
        CT_map,
        device='cuda:0',
        evaluation_mode='global',
        root_gsnorm_path=None,
        n_jobs=None):
    """
    Train the model for one epoch.
    Args:
        cfg (dict): Configuration parameters.
        epoch (int): Current epoch number.
        model (nn.Module): The hierarchical model.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        center_loss (torch.nn.Module): Center loss module.
        train_mask (torch.Tensor): Mask for training data.
        log_f (file): File to log training metrics.
        dict_save_models (dict): Dictionary of paths to save models.
        save_latest_model (str): Path to save the latest model.
        ppi_data (dict): PPI data for training.
        mg_data (torch_geometric.data.Data): Metagraph data for training.
        edge_attr_dict (dict): Dictionary mapping edge types to indices.
        celltype_map (dict): Mapping of cell types to indices.
        tissue_neighbors (dict): Neighbors of tissues in the metagraph.
        CT_map (dict): : cell to tissues assignments as a one-hot encoded label, see generate_input::compute_CT_map
        device (torch.device): Device to run the model on.
        n_jobs (int): number of CPUs to parallelize the dataloading on CPUs or not. Default is None = no parallelization.
    Returns:
        tuple: Updated PPI and metagraph metapaths for training and validation.
    """
    
    start_dataloading = time()
    # get formated training data for train and validation splits
    ppi_train_loader_dict, ppi_val_loader_dict = get_training_data(
        cfg, ppi_data, edge_attr_dict, device, n_jobs,
        evaluation_mode=evaluation_mode,
        root_gsnorm_path=root_gsnorm_path)

    print('(time) end dataloading:', time() - start_dataloading)
    
    model.train()
    
    # Run batch training
    start_train = time()
    
    loss, train_metrics = mb_utils.iterate_train_batch_hierarchical_model(
        cfg=cfg,
        ppi_train_loader_dict=ppi_train_loader_dict,
        CT_map=CT_map,
        mg_data=mg_data,
        model=model,
        device=device,
        optimizer=optimizer,
        edge_attr_dict=edge_attr_dict,
        current_epoch=epoch)
    

    print('end train:', time() - start_train)
    
    # Training metrics
    with torch.no_grad():
        
        roc_score, ap_score, train_acc, train_f1 = train_metrics['roc'], train_metrics['ap'], train_metrics['acc'], train_metrics['f1'] 
        roc_meta = train_metrics.get('roc_meta', None)
        ap_meta = train_metrics.get('ap_meta', None)
        acc_meta = train_metrics.get('acc_meta', None)
        f1_meta = train_metrics.get('f1_meta', None)
    
        df_train_total_metrics = {
            f'train_total_roc': [roc_score],
            f'train_total_ap': [ap_score],
            f'train_total_acc': [train_acc],
            f'train_total_f1': [train_f1]
        }
        if roc_meta is not None:
            df_train_total_metrics.update({
                'train_total_roc_meta': [roc_meta],
                'train_total_ap_meta': [ap_meta],
                'train_total_acc_meta': [acc_meta],
                'train_total_f1_meta': [f1_meta],
            })
        df_train_total_metrics = pd.DataFrame(df_train_total_metrics)
        
        print("Training Metrics:", "ROC", roc_score, "AP", ap_score, "ACC", train_acc, "F1", train_f1)
        if roc_meta is not None:
            print("Training Metagraph Metrics:", "ROC", roc_meta, "AP", ap_meta, "ACC", acc_meta, "F1", f1_meta)
        
        wandb_log_dict = {
                "train_roc": roc_score,
                "train_ap": ap_score,
                "train_acc": train_acc,
                "train_f1": train_f1
                }
        if roc_meta is not None:
            wandb_log_dict.update({
                "train_roc_meta": roc_meta,
                "train_ap_meta": ap_meta,
                "train_acc_meta": acc_meta,
                "train_f1_meta": f1_meta
            })
        wandb.log(wandb_log_dict)

        df_train_metrics = df_train_total_metrics

    del ppi_train_loader_dict
    
    # Validation set predictions
    model.eval()
    eval_on_gpu = True
    mg_pred_val, mg_data_val_y = None, None
    with torch.no_grad():
        if evaluation_mode == 'local':
            # evaluation per subgraphs sampled with {train, val} edges
            mg_pred_val, mg_data_val_y, ppi_preds_all, ppi_labels_all = mb_utils.iterate_predict_batch_hierarchical_model(
                cfg=cfg,
                ppi_loader_dict=ppi_val_loader_dict,
                CT_map=CT_map,
                mg_data=mg_data,
                model=model,
                device=device,
                edge_attr_dict=edge_attr_dict,
                eval_on_gpu=eval_on_gpu,
                split="val")  # Using train metapaths.
        elif evaluation_mode == 'global':
            # evaluation on the full val graph
            mg_pred_val, mg_data_val_y, ppi_preds_all, ppi_labels_all = mb_utils.iterate_predict_hierarchical_model(
                cfg=cfg,
                ppi_loader_dict=ppi_val_loader_dict,
                CT_map=CT_map,
                mg_data=mg_data,
                model=model,
                device=device,
                edge_attr_dict=edge_attr_dict,
                eval_on_gpu=eval_on_gpu,
                split="val")  # Using train metapaths.
        # Validation metrics
        val_metrics = utils.calc_metrics_factored(
            mg_pred_val if getattr(cfg, "use_metagraph", False) else None,
            mg_data_val_y if getattr(cfg, "use_metagraph", False) else None,
            ppi_preds_all,
            ppi_labels_all)

        print('val_metrics:', val_metrics)
        print("Validation Metrics:", "ROC", val_metrics['roc_ppi'],
              "AP", val_metrics['ap_ppi'],
              "ACC", val_metrics['acc_ppi'],
              "F1", val_metrics['f1_ppi'])
        if val_metrics.get('roc_meta') is not None:
            print("Validation Metagraph Metrics:", "ROC", val_metrics['roc_meta'],
                  "AP", val_metrics['ap_meta'],
                  "ACC", val_metrics['acc_meta'],
                  "F1", val_metrics['f1_meta'])
        
        df_celltype_val_metrics = utils.metrics_per_rel_factored(
            mg_pred_val if getattr(cfg, "use_metagraph", False) else None,
            mg_data_val_y if getattr(cfg, "use_metagraph", False) else None,
            ppi_preds_all, ppi_labels_all,
            edge_attr_dict, celltype_map, split="val")
        print('df_celltype_val_metrics:', df_celltype_val_metrics)  
        #calinski_harabasz, davies_bouldin = utils.calc_cluster_metrics(ppi_x)
        
        # Save metrics
        wandb_dict = {
            "total_loss": loss,
        }
        for m in ['roc', 'ap', 'acc', 'f1']:
            wandb_dict[f'total_val_{m}_ppi'] = val_metrics[f'{m}_ppi']
            if val_metrics.get(f'{m}_meta') is not None:
                wandb_dict[f'total_val_{m}_meta'] = val_metrics[f'{m}_meta']
        wandb.log(wandb_dict
            )

        # Save best model and parameters
        for metric in dict_save_models.keys():
            val_score = val_metrics[f'{metric}_ppi']
            score_metric = f'{metric}_ppi'
            if metric == 'ap' and val_metrics.get('ap_meta') is not None:
                val_score = val_metrics['ap_ppi'] + val_metrics['ap_meta']
                score_metric = 'ap_ppi_plus_ap_meta'
            
            if dict_save_models[metric]['value'] <= val_score:
                dict_save_models[metric]['value'] = val_score
                payload = _build_checkpoint_payload(
                    epoch, model, optimizer, score=val_score, score_metric=score_metric)
                _save_checkpoint(payload, dict_save_models[metric]['model_path'])
                primary_model_path = dict_save_models[metric].get('primary_model_path')
                if primary_model_path is not None:
                    _save_checkpoint(payload, primary_model_path)
                
                # save metrics for the validation set 
                df_val_total_metrics = {}
                for m in ['roc', 'ap', 'acc', 'f1']:
                    df_val_total_metrics[f'best_val_{m}_ppi'] = [val_metrics[f'{m}_ppi']]
                    if val_metrics.get(f'{m}_meta') is not None:
                        df_val_total_metrics[f'best_val_{m}_meta'] = [val_metrics[f'{m}_meta']]
                df_val_total_metrics = pd.DataFrame(df_val_total_metrics)
                df_val_metrics = pd.concat([df_val_total_metrics, df_celltype_val_metrics], axis=1)
                df_val_metrics.to_csv(dict_save_models[metric]['val_metrics_path'], index=False)
                
                # save corresponding metrics for the training set
                df_train_metrics.to_csv(dict_save_models[metric]['train_metrics_path'], index=False)

        # save after validation
        _save_checkpoint(_build_checkpoint_payload(epoch, model, optimizer), save_latest_model)
        
        """  
        if mg_metapaths_train is not None:
            for i, val in enumerate(mg_metapaths_train):
                mg_metapaths_train[i] = val.detach().cpu()
        """

    return val_metrics
    
    
@torch.no_grad()
def test(
        cfg,
        model,
        save_metrics_path,
        ppi_data,
        mg_data,
        edge_attr_dict,
        celltype_map,
        tissue_neighbors,
        CT_map,
        device='cuda:0',
        n_jobs=None,
        split='test',
        eval_on_gpu=False,
        evaluation_mode='global',
        ):
    """
    Test the model on the test set.
    Args:
        config (dict): Configuration parameters.
        model (nn.Module): The Pinnacle model.
        ppi_data (dict): PPI data for testing.
        mg_data (torch_geometric.data.Data): Metagraph data for training.
        edge_attr_dict (dict): Dictionary mapping edge types to indices.
        celltype_map (dict): Mapping of cell types to indices.
        tissue_neighbors (dict): Neighbors of tissues in the metagraph.
        CT_map (dict): : cell to tissues assignments as a one-hot encoded label, see generate_input::compute_CT_map
        device (torch.device): Device to run the model on.
        n_jobs (int): number of CPUs to parallelize the dataloading on CPUs or not. Default is None = no parallelization.
    Returns:
        None
    """
    model.to(device)
    model.eval()
    # Generate PPI batches
    ppi_test_loader_dict = mb_utils.generate_batch(
        ppi_data, edge_attr_dict, split,
        cfg.batch_size, device, ppi=True,
        loader_type=cfg.loader, n_jobs=n_jobs,
        evaluation_mode=evaluation_mode)
    
    """
    # Generate metagraph batches
    mg_data_test = None
    if getattr(cfg, "use_metagraph", False):
    
        _, mg_data_test, _, mg_x = mb_utils.generate_batch(
            {0: mg_data}, mg_metapaths, edge_attr_dict, split,
            cfg.batch_size, device, ppi=False, loader_type=cfg.loader, n_jobs=n_jobs)
        mg_data_test = mg_data_test[0]
        mg_x = mg_x[0]
    """
    mg_pred_test, mg_data_test_y = None, None
    with torch.no_grad():
        if evaluation_mode == 'local':
            mg_pred_test, mg_data_test_y, ppi_preds_all, ppi_labels_all = mb_utils.iterate_predict_batch_hierarchical_model(
            cfg=cfg,
            ppi_loader_dict=ppi_test_loader_dict,
            CT_map=CT_map,
            mg_data=mg_data,
            model=model,
            device=device,
            edge_attr_dict=edge_attr_dict,
            eval_on_gpu=eval_on_gpu,
            split=split,
            ) 
        elif evaluation_mode == 'global':
            mg_pred_test, mg_data_test_y, ppi_preds_all, ppi_labels_all = mb_utils.iterate_predict_hierarchical_model(
            cfg=cfg,
            ppi_loader_dict=ppi_test_loader_dict,
            CT_map=CT_map,
            mg_data=mg_data,
            model=model,
            device=device,
            edge_attr_dict=edge_attr_dict,
            eval_on_gpu=eval_on_gpu,
            split=split,
            )
        # Test metrics on proteins
        test_metrics = utils.calc_metrics_factored(
            mg_pred_test if getattr(cfg, "use_metagraph", False) else None,
            mg_data_test_y if getattr(cfg, "use_metagraph", False) else None,
            ppi_preds_all, 
            ppi_labels_all)
        
        test_roc = test_metrics['roc_ppi']
        test_ap = test_metrics['ap_ppi']
        test_acc = test_metrics['acc_ppi']
        test_f1 = test_metrics['f1_ppi']
    
        # Log test metrics
        df_test_total_metrics = {}
        for m in ['roc', 'ap', 'acc', 'f1']:
            df_test_total_metrics[f'test_{m}_ppi'] = [test_metrics[f'{m}_ppi']]
            meta_val = test_metrics.get(f'{m}_meta')
            if meta_val is not None:
                df_test_total_metrics[f'test_{m}_meta'] = [meta_val]
                # total = mean over available ppi/meta metrics
                df_test_total_metrics[f'test_{m}_total'] = [np.mean([test_metrics[f'{m}_ppi'], meta_val])]

        df_test_total_metrics = pd.DataFrame(df_test_total_metrics)
        df_celltype_test_metrics = utils.metrics_per_rel_factored(
            mg_pred_test if getattr(cfg, "use_metagraph", False) else None,
            mg_data_test_y if getattr(cfg, "use_metagraph", False) else None,
            ppi_preds_all, ppi_labels_all,
            edge_attr_dict, celltype_map, split=split,
            )
        df_test_metrics = pd.concat([df_test_total_metrics, df_celltype_test_metrics], axis=1)
        print('df_test_metrics:', df_test_metrics)
        df_test_metrics.to_csv(save_metrics_path, index=False)
        # log aggregated test metrics to wandb
        wandb_log = {}
        for col in df_test_total_metrics.columns:
            wandb_log[col] = df_test_total_metrics.iloc[0][col]
        if wandb_log:
            wandb.log(wandb_log)
    
