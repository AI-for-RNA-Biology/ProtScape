"""Train PINNACLE variants and export the best validation checkpoint."""

import os
import argparse
import traceback
import sys
import csv
import torch
import wandb
from tqdm import tqdm
import yaml

from .data_handler.generate_input import read_data, get_metapaths, get_centerloss_labels
from .models.pinnacle_model import Pinnacle
from .checkpoints import restore_rng_state, save_portable_checkpoint
from .losses.center_loss import CenterLoss
from .train.train_pinnacle_model import train, test
from . import utils
from omegaconf import DictConfig
import pickle
from pathlib import Path

from itertools import product

def parse_training_options():
    """Read feature settings and the small grid of PINNACLE training parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn-method", choices=["GATv2", "GIN", "ACM_RandomWalk"], default="GATv2")
    parser.add_argument("--gnn-lambda", default="0.01", help="Comma-separated center-loss lambda values, e.g. 0.1")
    parser.add_argument("--gnn-dropout", default="0.2", help="Comma-separated dropout values, e.g. 0 or 0.4")
    parser.add_argument("--gnn-output", default="32", help="Comma-separated output dims, e.g. 16 or 32")
    parser.add_argument("--features-mode", choices=["random", "ESM2", "ProstT5"], default="ESM2")
    parser.add_argument("--esm2-pooling", choices=["bos", "mean"], default="bos")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--dataset-mode", type=str, default="bulk", choices=["bulk", "legacy"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpointing",
        action="store_true",
        default=os.environ.get("PINNACLE_CHECKPOINTING", "false").lower() in ["1", "true", "yes"],
    )
    return parser.parse_args()

cli_args = parse_training_options()
seed = int(cli_args.seed)
utils.set_seed(seed)

gnn_method = str(cli_args.gnn_method)
list_gnn_lambda = [float(x) for x in cli_args.gnn_lambda.split(',')] # weight parameter for the center loss [0.1,0.01]

list_gnn_dropout = [float(x) for x in cli_args.gnn_dropout.split(',')]  # dropout rate [0,0.2,0.4,0.6]

list_gnn_output = [int(x) for x in cli_args.gnn_output.split(',')]     # output size
features_mode = cli_args.features_mode

lr = 0.01
use_scheduler = True
batch_size = 64
epochs = int(cli_args.epochs)
checkpointing = bool(cli_args.checkpointing)
loader = "graphsaint"
split_mode = 'global'
# A protein pair keeps the same split in every cell type.

lr_str = '' if lr==0.01 else 'lr' + str(lr).replace('.', '')
batch_size_str = '' if batch_size == 64 else '_b' + str(batch_size)
epochs_str = f'_ep{epochs}'
optim_str = lr_str + batch_size_str + epochs_str
split_mode_str = '' if split_mode == 'context' else f'{split_mode}split_'

dataset_mode = str(cli_args.dataset_mode)
print('dataset_mode:', dataset_mode)

symmetric_ppi = True

# Input and output roots are kept in one editable project config.
paths_file = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
with paths_file.open("r", encoding="utf-8") as handle:
    project_paths = yaml.safe_load(handle)

network_dir = str(Path(project_paths["networks_bulk"]).expanduser())
global_ppi_edges = os.path.join(network_dir, "global_ppi_edgelist.txt")
ppi_dir = os.path.join(network_dir, "ppi_edgelists")
metagraph_edges = os.path.join(network_dir, "mg_edgelist.txt")
exp_dir = os.path.join(
    str(Path(project_paths["output_root"]).expanduser()),
    "pretraining",
    "pinnacle",
)
print('exp_dir:', exp_dir)
os.makedirs(exp_dir, exist_ok=True)
if split_mode == 'global':
    count_edge_path = os.path.join(network_dir, "count_edge_dict.pkl")
else:
    count_edge_path = None

device = (
    "cuda" if cli_args.device == "auto" and torch.cuda.is_available()
    else "cpu" if cli_args.device == "auto"
    else cli_args.device
)

# shared by all GNN methods
dict_cfg = {
        'gnn_method': gnn_method,
        'dataset_mode': dataset_mode,
        'feat_mat': 1024,
        'hidden': 64,
        'wd': 1e-05,
        'lr': lr,
        'theta': 0.3,
        'lr_cent': 0.1,
        'batch_size': batch_size,
        'norm': None,
        'pc_att_channels': 16,

        'symmetric_ppi': symmetric_ppi,
        'features_mode': features_mode,
        'ppi_feat_dir' : None,
        'protein_feat_dim': None,
        "split_mode": split_mode,

        'seed' : seed,
        'loader': loader,
        'epochs': epochs,
        'resume_run': '',
        'use_scheduler': use_scheduler,
        'loss_type': "BCE",
        'gradclip': 1.0,

        'save_prefix': None,
        "plot": False,
        "wandb_mode": cli_args.wandb_mode,
    }

dict_cfg['n_heads'] = 8 if gnn_method == 'GATv2' else None
features_mode_dict = {
        "random": 1024,
        "ESM2": 2560,
        "ProstT5": 1024,
}

def train_configuration(
    cfg,
    ppi_data,
    ppi_metapaths,
    mg_data,
    mg_metapaths,
    ppi_layers,
    metagraph,
    edge_attr_dict,
    celltype_map,
    tissue_neighbors,
    device,
    store_global_pred_train=False,
    run_train=True,
    run_test=True,
    checkpointing=False,
    ):
    """Train or resume one configuration, export its best checkpoint, and evaluate.

    The validation score selects the checkpoint. Test edges are scored only
    after training, using the training/validation message-passing edges.
    """

    # Checkpoints and metric tables for this configuration.
    save_best_model = cfg.save_prefix + "/best_model_save.pth"
    save_latest_model = cfg.save_prefix + "/latest_model_save.pth"

    if cfg.get("split_mode") == "global":
        save_bestap_train_metrics = cfg.save_prefix + "/df_bestap_global_train_metrics.csv"
        save_bestap_val_metrics = cfg.save_prefix + "/df_bestap_global_val_metrics.csv"
        save_bestap_test_metrics = cfg.save_prefix + "/df_bestap_global_test_metrics.csv"
    else:
        save_bestap_train_metrics = cfg.save_prefix + "/df_bestap_train_metrics.csv"
        save_bestap_val_metrics = cfg.save_prefix + "/df_bestap_val_metrics.csv"
        save_bestap_test_metrics = cfg.save_prefix + "/df_bestap_test_metrics.csv"


    device = torch.device(device)
    print('Using device:', device)
    if device.type == 'cuda': print(torch.cuda.get_device_name(device))
    best_val_ap = -1
    best_model = None


    if run_train or run_test:
        center_loss_labels, train_mask, val_mask, test_mask = get_centerloss_labels(
        celltype_map, ppi_layers)

        model = Pinnacle(
            cfg['gnn_method'],
            mg_data.x.shape[1],
            cfg['hidden'],
            cfg['output'],
            len(ppi_metapaths),
            len(mg_metapaths),
            ppi_data,
            cfg['n_heads'],
            cfg['pc_att_channels'],
            cfg['dropout'],
            device=device).to(device)
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr = cfg['lr'], weight_decay = cfg['wd'])
        center_loss = CenterLoss(
            num_classes=len(set(center_loss_labels)),
            feat_dim=model.output
            )

        wandb.watch(model)
        print(model)

    # Train model
    if run_train:
        start_epoch = 0
        if checkpointing and os.path.exists(save_latest_model):
            print(f"Resuming from checkpoint: {save_latest_model}")
            ckpt = torch.load(save_latest_model, weights_only=False)
            model.load_state_dict(ckpt['model'].state_dict())
            optimizer.load_state_dict(ckpt['optimizer'].state_dict())
            restore_rng_state(ckpt.get('rng_state'))
            start_epoch = ckpt['epoch'] + 1
            if os.path.exists(save_best_model):
                best_ckpt = torch.load(save_best_model, map_location="cpu", weights_only=False)
                best_score = None
                if best_ckpt.get("score_metric") == "ap_ppi_plus_ap_meta" and best_ckpt.get("score") is not None:
                    best_score = float(best_ckpt["score"])
                elif os.path.exists(save_bestap_val_metrics):
                    with open(save_bestap_val_metrics, newline="") as csv_file:
                        row = next(csv.DictReader(csv_file), None)
                    if row is not None:
                        ppi_ap = row.get("best_val_ap_ppi")
                        meta_ap = row.get("best_val_ap_meta")
                        if ppi_ap not in [None, ""] and meta_ap not in [None, ""]:
                            best_score = float(ppi_ap) + float(meta_ap)
                if best_score is None and best_ckpt.get("score") is not None:
                    best_score = float(best_ckpt["score"])
                if best_score is not None:
                    best_val_ap = best_score
                    print(f"Restored best validation AP selection score: {best_val_ap:.6f}")
            print(f"Resuming training from epoch {start_epoch}/{cfg.epochs}")

        for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Training Epochs"):
            print(f"\n===== Epoch {epoch + 1}/{cfg.epochs} =====", flush=True)
            ppi_metapaths_train, mg_metapaths_train, ppi_metapaths_val, mg_metapaths_val, best_val_ap = train(cfg,
                epoch,
                model,
                best_val_ap,
                optimizer,
                center_loss,
                train_mask,
                save_bestap_train_metrics,
                save_bestap_val_metrics,
                save_best_model,
                save_latest_model,
                ppi_data,
                ppi_metapaths,
                mg_data,
                mg_metapaths,
                edge_attr_dict,
                celltype_map,
                tissue_neighbors,
                store_global_pred_train,
                device)

        print("Optimization finished!")

        # Save the train/validation metapaths used by test-only evaluation.
        torch.save(ppi_metapaths_train, os.path.join(cfg.save_prefix, "ppi_metapaths_train.pth"))
        torch.save(mg_metapaths_train, os.path.join(cfg.save_prefix, "mg_metapaths_train.pth"))
        torch.save(ppi_metapaths_val, os.path.join(cfg.save_prefix, "ppi_metapaths_val.pth"))
        torch.save(mg_metapaths_val, os.path.join(cfg.save_prefix, "mg_metapaths_val.pth"))

    else:
        print('Training disabled.')

    if os.path.exists(save_best_model):
        id_to_name = {cell_id: name for name, cell_id in celltype_map.items()}
        cell_ids = list(ppi_data)
        portable_config = {
            "dataset_mode": cfg.dataset_mode,
            "split_mode": cfg.split_mode,
            "features_mode": cfg.features_mode,
            "esm2_pooling": cfg.esm2_pooling,
            "symmetric_ppi": cfg.symmetric_ppi,
            "seed": cfg.seed,
            "epochs": cfg.epochs,
            "gnn_method": cfg.gnn_method,
            "input_dim": int(mg_data.x.shape[1]),
            "hidden": cfg.hidden,
            "output": cfg.output,
            "num_ppi_relations": len(ppi_metapaths),
            "num_mg_relations": len(mg_metapaths),
            "n_heads": cfg.n_heads,
            "pc_att_channels": cfg.pc_att_channels,
            "dropout": cfg.dropout,
            "shared_ppi_gnn": False,
        }
        save_portable_checkpoint(
            Path(cfg.save_prefix) / "best_model_state_dict.pt",
            save_best_model,
            model_type="pinnacle",
            config=DictConfig(portable_config),
            cell_ids=cell_ids,
            cell_names=[id_to_name[cell_id] for cell_id in cell_ids],
        )

    if run_test:
        if not run_train:
            ppi_metapaths_train = torch.load(os.path.join(cfg.save_prefix, "ppi_metapaths_train.pth"))
            mg_metapaths_train = torch.load(os.path.join(cfg.save_prefix, "mg_metapaths_train.pth"))
            ppi_metapaths_val = torch.load(os.path.join(cfg.save_prefix, "ppi_metapaths_val.pth"))
            mg_metapaths_val = torch.load(os.path.join(cfg.save_prefix, "mg_metapaths_val.pth"))

        if best_model is None:
            print('loading model weights')
            best_model_dict = torch.load(save_best_model, map_location="cpu", weights_only=False)
            print('Saved best epoch:', best_model_dict['epoch'])
            best_model = model
            best_model.load_state_dict(best_model_dict['model'].state_dict())
        best_model.to(device)
        best_model.eval()

        if cfg.get("split_mode") == "global":
            ppi_metapaths_test = None
            mg_metapaths_test = None
            evaluation_mode = "corrected"
            print('Evaluating the best validation AP model with global PPI batches')
        else:
            ppi_metapaths_test = {}
            mg_metapaths_test = []
            for key in ppi_metapaths_train.keys():
                ppi_metapaths_test[key] = [torch.cat(ppi_metapaths_val[key] + ppi_metapaths_train[key], dim=1).to(device)]
            for mg_mt_t, mg_mt_v in zip(mg_metapaths_train, mg_metapaths_val):
                mg_metapaths_test.append(torch.cat([mg_mt_t, mg_mt_v], dim=1).to(device))
            evaluation_mode = "standard"
            print('Evaluating the best validation AP model with context-specific PPI batches')

        test(
            cfg,
            best_model,
            save_bestap_test_metrics,
            ppi_data,
            ppi_metapaths,
            mg_data,
            mg_metapaths,
            ppi_metapaths_test,
            mg_metapaths_test,
            edge_attr_dict,
            celltype_map,
            tissue_neighbors,
            device=device,
            evaluation_mode=evaluation_mode,
            split='test')

if __name__ == "__main__":
    # Adapted PINNACLE configurations.
    dict_cfg['features_mode'] = features_mode
    dict_cfg['esm2_pooling'] = cli_args.esm2_pooling
    dict_cfg['protein_feat_dim'] = features_mode_dict[features_mode]
    if features_mode == "ESM2":
        feature_key = "esm2_embeddings" if cli_args.esm2_pooling == "bos" else "esm2_mean_embeddings"
        dict_cfg['ppi_feat_dir'] = str(Path(project_paths[feature_key]).expanduser())
    elif features_mode == "ProstT5":
        dict_cfg['ppi_feat_dir'] = str(Path(project_paths['prostt5_embeddings']).expanduser())
    else:
        dict_cfg['ppi_feat_dir'] = None

    had_failure = False
    for gnn_lambda, gnn_dropout, gnn_output in product(
        list_gnn_lambda, list_gnn_dropout, list_gnn_output):

        dict_cfg['lambda'] = gnn_lambda
        dict_cfg['dropout'] = gnn_dropout
        dict_cfg['output'] = gnn_output
        if dict_cfg['n_heads'] is None:
            dict_cfg['hidden'] = dict_cfg['output']
        cfg = DictConfig(dict_cfg)

        ppi_data, mg_data, edge_attr_dict, celltype_map, tissue_neighbors, ppi_layers, metagraph = read_data(
            global_ppi_edges, ppi_dir, metagraph_edges,
            feat_mat_dim = cfg['protein_feat_dim'],
            get_CT_map=False,
            ppi_feat_dir= cfg['ppi_feat_dir'],
            symmetric_ppi= cfg['symmetric_ppi'],
            dataset_mode=dataset_mode,
            split_mode=split_mode,
            count_edge_path=count_edge_path
        )
        ppi_metapaths, mg_metapaths = get_metapaths()

        gnn_lambda_str = '%s'%(str(cfg['lambda']).replace('.', ''))
        gnn_dropout_str = '%s'%(str(cfg['dropout']).replace('.', ''))

        gnn_str = f"{gnn_method}_H{cfg['hidden']}_lambda{gnn_lambda_str}_drop{gnn_dropout_str}_out{cfg['output']}"

        feature_name = features_mode + ('_mean' if features_mode == 'ESM2' and cli_args.esm2_pooling == 'mean' else '')
        experiment_name = f'PINNACLE_model_{split_mode_str}{feature_name}_symmetric_PPI-{symmetric_ppi}_{gnn_str}'
        if optim_str != '':
            experiment_name += f'_{optim_str}'
        experiment_name += f'_{dataset_mode}'
        print(experiment_name)

        save_prefix = os.path.join(exp_dir, experiment_name)
        print("save prefix", save_prefix)
        run_train = True
        run_test = True
        existing_repo = os.path.exists(save_prefix)
        if not existing_repo:
            os.makedirs(save_prefix, exist_ok=True)
        if existing_repo:
            print('Checking existing training run')
            save_latest_model = save_prefix + "/latest_model_save.pth"
            save_best_model = save_prefix + "/best_model_save.pth"
            if not os.path.exists(save_latest_model):
                if os.path.exists(save_best_model):
                    print('Evaluating the saved best checkpoint')
                    run_train = False
                    run_test = True
                else:
                    print('Starting training')
                    run_train = True
                    run_test = True
            else:
                latest_model_dict = torch.load(save_latest_model, weights_only=False)

                if latest_model_dict['epoch'] == (epochs - 1):
                    run_train = False
                    run_test = True
                elif checkpointing:
                    print(f'-- resuming training from epoch {latest_model_dict["epoch"] + 1}')
                    run_train = True
                    run_test = True
                else:
                    raise RuntimeError('Incomplete run: use --checkpointing to resume.')

        try:
            if run_train or run_test:
                print('run_train:', run_train, ' | run_test:', run_test)
                print('save_prefix:', save_prefix)
                cfg['save_prefix'] = save_prefix

                if not existing_repo:
                    config_file = save_prefix + '/config_dict_seed%s.pkl'%cfg.seed
                    with open(config_file, 'wb') as f:
                        pickle.dump(dict_cfg, f)

                utils.set_seed(seed)
                utils.setup_wandb(cfg, experiment_name)

                train_configuration(cfg,
                    ppi_data,
                    ppi_metapaths,
                    mg_data,
                    mg_metapaths,
                    ppi_layers,
                    metagraph,
                    edge_attr_dict,
                    celltype_map,
                    tissue_neighbors,
                    store_global_pred_train=False,
                    run_train=run_train,
                    run_test=run_test,
                    device=device,
                    checkpointing=checkpointing)
            else:
                print('-- experiment already run successfully -> skipping')
        except Exception:
            print('failed running experiment;', experiment_name)
            traceback.print_exc()
            had_failure = True
            continue
    if had_failure:
        sys.exit(1)
