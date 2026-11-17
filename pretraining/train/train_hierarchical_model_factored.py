import numpy as np
import pandas as pd
import torch
import wandb
from ..checkpoints import build_training_checkpoint, save_training_checkpoint
from .. import utils
from . import factored_batching as batch_utils
from . import factored_training as training_utils

from time import time


def get_training_data(
    cfg,
    ppi_data,
    edge_attr_dict,
    device='cuda:0',
    n_jobs=None,
    evaluation_mode='global',
    root_gsnorm_path=None):
    """Build sampled training graphs and local or full-graph validation inputs."""
    ppi_train_loader_dict = batch_utils.generate_batch(
        ppi_data, edge_attr_dict, "train", cfg.batch_size, device, ppi=True,
        loader_type=cfg.loader, n_jobs=n_jobs, evaluation_mode='local',
        graph_saint_norm= cfg.graph_saint_norm, root_gsnorm_path=root_gsnorm_path,
        weighted_ppi_loss=cfg.weighted_ppi_loss)

    ppi_val_loader_dict = batch_utils.generate_batch(
        ppi_data, edge_attr_dict, "val", cfg.batch_size, device, ppi=True,
        loader_type=cfg.loader, n_jobs=n_jobs, evaluation_mode=evaluation_mode,
        graph_saint_norm=False, weighted_ppi_loss=cfg.weighted_ppi_loss)


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
    """Train one ProtScape epoch, evaluate validation edges and save improved models."""

    start_dataloading = time()
    ppi_train_loader_dict, ppi_val_loader_dict = get_training_data(
        cfg, ppi_data, edge_attr_dict, device, n_jobs,
        evaluation_mode=evaluation_mode,
        root_gsnorm_path=root_gsnorm_path)

    print('(time) end dataloading:', time() - start_dataloading)

    model.train()

    # Run batch training
    start_train = time()

    loss, train_metrics = training_utils.iterate_train_batch_hierarchical_model(
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
            # Evaluate sampled subgraphs containing training and validation edges.
            mg_pred_val, mg_data_val_y, ppi_preds_all, ppi_labels_all = training_utils.iterate_predict_batch_hierarchical_model(
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
            # Evaluate complete cell graphs.
            mg_pred_val, mg_data_val_y, ppi_preds_all, ppi_labels_all = training_utils.iterate_predict_hierarchical_model(
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
                payload = build_training_checkpoint(
                    epoch, model, optimizer, score=float(val_score), score_metric=score_metric)
                save_training_checkpoint(payload, dict_save_models[metric]['model_path'])
                primary_model_path = dict_save_models[metric].get('primary_model_path')
                if primary_model_path is not None:
                    save_training_checkpoint(payload, primary_model_path)

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
        save_training_checkpoint(build_training_checkpoint(epoch, model, optimizer), save_latest_model)


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
    """Evaluate a selected model and write protein, metagraph and tissue metrics."""
    model.to(device)
    model.eval()
    # Generate PPI batches
    ppi_test_loader_dict = batch_utils.generate_batch(
        ppi_data, edge_attr_dict, split,
        cfg.batch_size, device, ppi=True,
        loader_type=cfg.loader, n_jobs=n_jobs,
        evaluation_mode=evaluation_mode)

    mg_pred_test, mg_data_test_y = None, None
    with torch.no_grad():
        if evaluation_mode == 'local':
            mg_pred_test, mg_data_test_y, ppi_preds_all, ppi_labels_all = training_utils.iterate_predict_batch_hierarchical_model(
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
            mg_pred_test, mg_data_test_y, ppi_preds_all, ppi_labels_all = training_utils.iterate_predict_hierarchical_model(
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
