import numpy as np
import pandas as pd

import torch
import wandb
from ..checkpoints import build_training_checkpoint, save_training_checkpoint
from .. import utils
from . import minibatch_utils as mb_utils


def train(
        cfg,
        epoch,
        model,
        best_val_ap,
        optimizer,
        center_loss,
        train_mask,
        save_train_metrics_path,
        save_val_metrics_path,
        save_best_model,
        save_latest_model,
        ppi_data,
        ppi_metapaths,
        mg_data,
        mg_metapaths,
        edge_attr_dict,
        celltype_map,
        tissue_neighbors,
        store_global_pred_train=False,
        device='cuda:0'):
    """Train one PINNACLE epoch, evaluate validation edges and save improved models."""

    use_corrected_eval = cfg.get("split_mode") == "global"

    # PPI training and validation batches
    ppi_train_loader_dict, _, ppi_metapaths_train, ppi_x_ori = mb_utils.generate_batch(
        ppi_data, ppi_metapaths, edge_attr_dict, "train", cfg.batch_size, device, ppi=True, loader_type=cfg.loader)

    if use_corrected_eval:
        ppi_val_loader_dict, _, ppi_metapaths_val, _ = mb_utils.generate_batch_corrected(
            ppi_data, ppi_metapaths, edge_attr_dict, "val", cfg.batch_size, device, ppi=True, loader_type=cfg.loader)
    else:
        ppi_val_loader_dict, _, ppi_metapaths_val, _ = mb_utils.generate_batch(
            ppi_data, ppi_metapaths, edge_attr_dict, "val", cfg.batch_size, device, ppi=True, loader_type=cfg.loader)

    # Metagraph training and validation batches
    _, mg_data_train, mg_metapaths_train, mg_x_ori = mb_utils.generate_batch(
        {0: mg_data}, mg_metapaths, edge_attr_dict, "train", cfg.batch_size, device, ppi=False, loader_type=cfg.loader)
    _, mg_data_val, mg_metapaths_val, _ = mb_utils.generate_batch(
        {0: mg_data}, mg_metapaths, edge_attr_dict, "val", cfg.batch_size, device, ppi=False, loader_type=cfg.loader)

    # Unpack single-item batches
    mg_x_ori = mg_x_ori[0]
    mg_data_train = mg_data_train[0]
    mg_data_val = mg_data_val[0]
    mg_metapaths_train = mg_metapaths_train[0]
    mg_metapaths_val = mg_metapaths_val[0]

    for i, val in enumerate(mg_metapaths_train):
        mg_metapaths_train[i] = val.to(device)


    for key, val in ppi_metapaths_train.items():
        ppi_metapaths_train[key] = [val[0].to(device)]

    for i, val in enumerate(mg_metapaths_val):
        mg_metapaths_val[i] = val.to(device)

    for key, val in ppi_metapaths_val.items():
        ppi_metapaths_val[key] = [val[0].to(device)]

    model.train()


    _, _, mg_pred, ppi_preds_all, ppi_data_train_y, loss, train_metrics = mb_utils.iterate_train_batch(
        ppi_train_loader_dict, ppi_x_ori, ppi_metapaths, mg_x_ori, mg_metapaths_train, mg_data_train,
        tissue_neighbors, model, cfg, device, wandb, center_loss, optimizer, train_mask,
        store_global_pred_train)


    # Summarize training edge metrics.
    with torch.no_grad():
        if train_metrics is None:
            assert store_global_pred_train # enforce that all predictions across batches were saved to compute metrics

            roc_score_ppi, ap_score_ppi, train_acc_ppi, train_f1_ppi = utils.calc_metrics(
                None, None, ppi_preds_all, ppi_data_train_y, 'torch')
            roc_score_meta, ap_score_meta, train_acc_meta, train_f1_meta = utils.calc_metrics(
                mg_pred, mg_data_train, None, None, 'torch')


        else:
            roc_score_ppi, ap_score_ppi, train_acc_ppi, train_f1_ppi = train_metrics['roc_ppi'], train_metrics['ap_ppi'], train_metrics['acc_ppi'], train_metrics['f1_ppi']
            roc_score_meta, ap_score_meta, train_acc_meta, train_f1_meta = train_metrics['roc_meta'], train_metrics['ap_meta'], train_metrics['acc_meta'], train_metrics['f1_meta']


        df_train_total_metrics = {
            f'train_total_roc_ppi': [roc_score_ppi],
            f'train_total_ap_ppi': [ap_score_ppi],
            f'train_total_acc_ppi': [train_acc_ppi],
            f'train_total_f1_ppi': [train_f1_ppi],
            f'train_total_roc_meta': [roc_score_meta],
            f'train_total_ap_meta': [ap_score_meta],
            f'train_total_acc_meta': [train_acc_meta],
            f'train_total_f1_meta': [train_f1_meta],
        }
        df_train_total_metrics = pd.DataFrame(df_train_total_metrics)
        print(df_train_total_metrics.to_string(index=False))

        wandb.log(
            {
                "train_roc_ppi": roc_score_ppi,
                "train_ap_ppi": ap_score_ppi,
                "train_acc_ppi": train_acc_ppi,
                "train_f1_ppi": train_f1_ppi,
                "train_roc_meta": roc_score_meta,
                "train_ap_meta": ap_score_meta,
                "train_acc_meta": train_acc_meta,
                "train_f1_meta": train_f1_meta
            }
            )

        if store_global_pred_train:

            df_celltype_train_metrics = utils.metrics_per_rel(
                mg_pred, mg_data_train, ppi_preds_all, ppi_data_train_y,
                edge_attr_dict, celltype_map, split="train",
                version="torch")
            df_train_metrics = pd.concat([df_train_total_metrics, df_celltype_train_metrics], axis=1)
        else:
            df_train_metrics = df_train_total_metrics

    del ppi_preds_all, ppi_data_train_y, ppi_train_loader_dict, mg_pred, mg_data_train


    # Validation step
    model.eval()
    with torch.no_grad():
        if use_corrected_eval:
            ppi_x, _, mg_pred, ppi_preds_all, ppi_data_val_y = mb_utils.iterate_predict(
                ppi_val_loader_dict, ppi_x_ori, ppi_metapaths_val, mg_x_ori, mg_metapaths_train, mg_data_val,
                tissue_neighbors, model, cfg, device)
        else:
            ppi_x, _, mg_pred, ppi_preds_all, ppi_data_val_y = mb_utils.iterate_predict_batch(
                ppi_val_loader_dict, ppi_x_ori, ppi_metapaths_train, mg_x_ori, mg_metapaths_train, mg_data_val,
                tissue_neighbors, model, cfg, device)

        # Validation metrics
        val_metrics = utils.calc_metrics(mg_pred, mg_data_val, ppi_preds_all, ppi_data_val_y, version='torch')
        for local_type in ['ppi', 'meta']:
            print(f"[Validation Metrics ({local_type}):", "ROC", val_metrics[f'roc_{local_type}'],
                "AP", val_metrics[f'ap_{local_type}'],
                "ACC", val_metrics[f'acc_{local_type}'],
                "F1", val_metrics[f'f1_{local_type}'])

        df_celltype_val_metrics = utils.metrics_per_rel(
            None, None, ppi_preds_all, ppi_data_val_y,
            edge_attr_dict, celltype_map, "val", version="torch")

        # Log validation results (metrics)

        wandb_dict = {
            "total_loss": loss,
        }
        for m in ['roc', 'ap', 'acc', 'f1']:
            wandb_dict[f'total_val_{m}_ppi'] = val_metrics[f'{m}_ppi']
            wandb_dict[f'total_val_{m}_meta'] = val_metrics[f'{m}_meta']
        wandb.log(wandb_dict
            )

        # save latest model
        save_training_checkpoint(build_training_checkpoint(epoch, model, optimizer), save_latest_model)

        # Select the checkpoint by summed PPI and metagraph average precision.
        val_score = val_metrics['ap_ppi']
        score_metric = "ap_ppi"
        if val_metrics.get('ap_meta') is not None:
            val_score = val_metrics['ap_ppi'] + val_metrics['ap_meta']
            score_metric = "ap_ppi_plus_ap_meta"
        if best_val_ap <= val_score:
            best_val_ap = val_score
            save_training_checkpoint(
                build_training_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    score_metric=score_metric,
                    score=float(val_score),
                ),
                save_best_model,
            )

            df_val_best_metrics = {}
            for key in val_metrics.keys():
                df_val_best_metrics[f'best_val_{key}'] = [val_metrics[key]]

            # compute total best val metrics dataframe
            for m in ['roc', 'ap', 'acc', 'f1']:
                df_val_best_metrics[f'best_val_{m}_total'] = [np.mean([val_metrics[f'{m}_ppi'], val_metrics[f'{m}_meta']])]
            df_val_best_metrics = pd.DataFrame(df_val_best_metrics)
            df_val_metrics = pd.concat([df_val_best_metrics, df_celltype_val_metrics], axis=1)

            df_val_metrics.to_csv(save_val_metrics_path, index=False)

            # save corresponding metrics for the training set
            df_train_metrics.to_csv(save_train_metrics_path, index=False)

        # Move cached metapaths back to CPU.
        for i, val in enumerate(mg_metapaths_train):
            mg_metapaths_train[i] = val.detach().cpu()
        for key, val in ppi_metapaths_train.items():
            ppi_metapaths_train[key] = [val[0].detach().cpu()]
        for i, val in enumerate(mg_metapaths_val):
            mg_metapaths_val[i] = val.detach().cpu()
        for key, val in ppi_metapaths_val.items():
            ppi_metapaths_val[key] = [val[0].detach().cpu()]

        del ppi_x, mg_pred, ppi_preds_all, ppi_data_val_y, mg_data_val

    return ppi_metapaths_train, mg_metapaths_train, ppi_metapaths_val, mg_metapaths_val, best_val_ap


@torch.no_grad()
def test(
        config,
        model,
        save_metrics_path,
        ppi_data,
        ppi_metapaths,
        mg_data,
        mg_metapaths,
        ppi_metapaths_test,
        mg_metapaths_test,
        edge_attr_dict,
        celltype_map,
        tissue_neighbors,
        device,
        split=None,
        evaluation_mode='standard'):

    """Evaluate a saved PINNACLE model and write overall and per-cell edge metrics."""
    model.to(device)
    model.eval()

    # Generate test batches for PPI and metagraph
    if evaluation_mode == 'standard':
        eval_split = "test" if split is None else split
        assert eval_split in {"val", "test"}, "standard PINNACLE evaluation only supports val/test splits"
        assert ppi_metapaths_test is not None
        assert mg_metapaths_test is not None

        ppi_test_loader_dict, _, _, ppi_x = mb_utils.generate_batch(
            ppi_data, ppi_metapaths, edge_attr_dict, eval_split,
            config.batch_size, device, ppi=True, loader_type=config.loader)

        _, mg_data_test, _, mg_x = mb_utils.generate_batch(
            {0: mg_data}, mg_metapaths, edge_attr_dict, eval_split, config.batch_size,
            device, ppi=False, loader_type=config.loader)

    elif evaluation_mode == 'corrected':
        eval_split = "test" if split is None else split
        assert eval_split in {"val", "test"}, "corrected PINNACLE evaluation only supports val/test splits"
        assert ppi_metapaths_test is None
        assert mg_metapaths_test is None

        ppi_test_loader_dict, _, ppi_metapaths_test, ppi_x = mb_utils.generate_batch_corrected(
            ppi_data, ppi_metapaths, edge_attr_dict, eval_split,
            config.batch_size, device, ppi=True, loader_type=config.loader)

        # Keep the original metagraph protocol: all metagraph masks are true in generate_input.py.
        # The corrected path only changes PPI val/test edge selection.
        _, mg_data_test, mg_metapaths_test, mg_x = mb_utils.generate_batch(
            {0: mg_data}, mg_metapaths, edge_attr_dict, eval_split,
            config.batch_size, device, ppi=False, loader_type=config.loader)

        # Unpack single-item batches
        mg_metapaths_test = mg_metapaths_test[0]
        for i, val in enumerate(mg_metapaths_test):
            mg_metapaths_test[i] = val.to(device)


        for key, val in ppi_metapaths_test.items():
            ppi_metapaths_test[key] = [val[0].to(device)]

    # Unpack single-item batches
    mg_data_test = mg_data_test[0]
    mg_x = mg_x[0]

    # Predict on test set
    if evaluation_mode == 'standard':
        _, _, mg_pred, ppi_preds_all, ppi_data_test_y = mb_utils.iterate_predict_batch(
            ppi_test_loader_dict, ppi_x, ppi_metapaths_test, mg_x, mg_metapaths_test, mg_data_test,
            tissue_neighbors, model, config, device)
    elif evaluation_mode == 'corrected':
        _, _, mg_pred, ppi_preds_all, ppi_data_test_y = mb_utils.iterate_predict(
            ppi_test_loader_dict, ppi_x, ppi_metapaths_test, mg_x, mg_metapaths_test, mg_data_test,
            tissue_neighbors, model, config, device)

    # Compute performance metrics on the selected evaluation split.
    test_metrics = utils.calc_metrics(
        mg_pred, mg_data_test, ppi_preds_all, ppi_data_test_y, version='torch')

    # Use the same metric columns as ProtScape.
    df_test_total_metrics = {}
    for m in ['roc', 'ap', 'acc', 'f1']:
        df_test_total_metrics[f'test_{m}_ppi'] = [test_metrics[f'{m}_ppi']]
        meta_val = test_metrics.get(f'{m}_meta')
        if meta_val is not None:
            df_test_total_metrics[f'test_{m}_meta'] = [meta_val]
            df_test_total_metrics[f'test_{m}_total'] = [np.mean([test_metrics[f'{m}_ppi'], meta_val])]

    df_test_total_metrics = pd.DataFrame(df_test_total_metrics)
    # Log detailed performance per relation type
    df_celltype_test_metrics = utils.metrics_per_rel(
        mg_pred, mg_data_test, ppi_preds_all, ppi_data_test_y,
        edge_attr_dict, celltype_map, "test",
        version='torch')

    df_test_metrics = pd.concat([df_test_total_metrics, df_celltype_test_metrics], axis=1)
    print('df_test_metrics:', df_test_metrics)
    df_test_metrics.to_csv(save_metrics_path, index=False)
    wandb_log = {col: df_test_total_metrics.iloc[0][col] for col in df_test_total_metrics.columns}
    if wandb_log:
        wandb.log(wandb_log)

    del mg_pred, ppi_preds_all, ppi_data_test_y, ppi_test_loader_dict, mg_data_test
