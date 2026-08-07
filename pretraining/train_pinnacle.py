import os
import argparse
import traceback
import sys
import csv
import random
import torch
import wandb
import numpy as np
from tqdm import tqdm
import yaml

from .data_handler.generate_input import read_data, get_metapaths, get_centerloss_labels
from .models.pinnacle_model import Pinnacle
from .checkpoints import save_portable_checkpoint
from .losses.center_loss import CenterLoss
from .train.train_pinnacle_model import train, test
from . import utils
from omegaconf import DictConfig
import pickle 
from pathlib import Path


from itertools import product


def _restore_rng_state(rng_state):
    if not rng_state:
        return

    python_state = rng_state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = rng_state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = rng_state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)

    torch_cuda_state = rng_state.get("torch_cuda")
    if torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_state)


def _parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("gnn_method", type=str, choices=["GATv2", "GIN", "ACM_RandomWalk"])
    parser.add_argument("gnn_lambda", type=str, help="Comma-separated center-loss lambda values, e.g. 0.1")
    parser.add_argument("gnn_dropout", type=str, help="Comma-separated dropout values, e.g. 0 or 0.4")
    parser.add_argument("gnn_output", type=str, help="Comma-separated output dims, e.g. 16 or 32")
    parser.add_argument("features_mode", type=str, help="random|ESM2|ProstT5")
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


def _normalize_features_mode(raw_value):
    feature_mode_aliases = {
        'random': 'random',
        'esm2': 'ESM2',
        'prostt5': 'ProstT5',
        'prosst5': 'ProstT5',
    }
    normalized = feature_mode_aliases.get(str(raw_value).strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported features_mode: {raw_value}")
    return normalized


def _get_ppi_feat_dir(paths, features_mode):
    if features_mode == "ESM2":
        return str(Path(paths["esm2_embeddings"]).expanduser())
    elif features_mode == "ProstT5":
        return str(Path(paths["prostt5_embeddings"]).expanduser())
    else:
        return None




cli_args = _parse_cli_args()
seed = int(cli_args.seed)
utils.set_seed(seed)

gnn_method = str(cli_args.gnn_method)
list_gnn_lambda = [float(x) for x in cli_args.gnn_lambda.split(',')] # weight parameter for the center loss [0.1,0.01]
assert np.all([x in [0.1,0.01] for x in list_gnn_lambda])

list_gnn_dropout = [float(x) for x in cli_args.gnn_dropout.split(',')]  # dropout rate [0,0.2,0.4,0.6]
assert np.all([x in [0,0.2,0.4,0.6] for x in list_gnn_dropout])

list_gnn_output = [int(x) for x in cli_args.gnn_output.split(',')]     # output size
if gnn_method in ['GATv2']:
    assert np.all([x in [16,32] for x in list_gnn_output]) # output size to be multiplied by n_heads (8) to get protein embedding dim
else:
    assert np.all([x in [64, 128, 256, 512] for x in list_gnn_output])

features_mode = _normalize_features_mode(cli_args.features_mode)
# global variables

lr = 0.01 #0.01
use_scheduler = True
batch_size = 64
epochs = int(cli_args.epochs) #500 #300
checkpointing = bool(cli_args.checkpointing)
loader = "graphsaint"
split_mode = 'global'
assert split_mode in ['context', 'global']
# context: corresponds to initial pinnacle split with allowing edge leakage across contexts
# global: defines splits over the global ppi, stratified by edge frequencies across contextes, preventing edge leakage across contexts.


lr_str = '' if lr==0.01 else 'lr' + str(lr).replace('.', '')
batch_size_str = '' if batch_size == 64 else '_b' + str(batch_size)
epochs_str = f'_ep{epochs}'
optim_str = lr_str + batch_size_str + epochs_str
split_mode_str = '' if split_mode == 'context' else f'{split_mode}split_'

dataset_mode = str(cli_args.dataset_mode)
print('dataset_mode:', dataset_mode)

symmetric_ppi = True #False
if symmetric_ppi == False:
    print('Warning: Using asymmetric PPI network to replicate PINNACLE experiments - this is not recommended!')
    
    
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


plot = False
device = 'cuda:0'


# shared by all GNN methods
dict_cfg = {
        'gnn_method': gnn_method, 
        'feat_mat': 1024,          # Random Gaussian vectors of shape (1 x 2048)                 2048  
        'hidden': 64,              # GAT hidden                                                  64
        'wd': 1e-05,               # Weight decay                                                5e-4
        'lr': lr,               # Learning rate                                               0.001
        'theta': 0.3,              # Theta (for PPI loss)                                        0.1
        'lr_cent': 0.1,            # Learning rate for center loss                               0.01
        'batch_size': batch_size,           # Batch size                                                  8
        'norm': None,              # Type of normalization layer to use in up-pooling            None           
        'pc_att_channels': 16,     # Protein-Context attention channels                          8

        'symmetric_ppi': symmetric_ppi,     # PPI network should be undirected 
        'features_mode': features_mode,   # Mode for generating features {random, ESM2, MaSIF}          None
        'ppi_feat_dir' : None,     # protein node features directory (Used for ESM2 and MaSIF)   None
        'protein_feat_dim': None,  # Protein_feat_dim 
        "split_mode": split_mode,     

        # Parameters
        'seed' : seed,
        'loader': loader,    # {"neighbor", "graphsaint"} 
        'epochs': epochs,             # Number of epochs to train 300
        'resume_run': '',    
        'use_scheduler': use_scheduler,     # Whether to use a learning rate scheduler
        'loss_type': "BCE", 
        'gradclip': 1.0,
        
        # Save    
        'save_prefix': None,       # Prefix of all saved files
        "plot": False,             # Bool to fit and plot a UMAP
        "wandb_mode": cli_args.wandb_mode,
    }

if gnn_method == 'GATv2':
    specific_dict_cfg = {
        'n_heads' : 8
    }

else:
    specific_dict_cfg = {
    'n_heads': None,              # Number of heads                                             8
               }

dict_cfg.update(specific_dict_cfg)
features_mode_dict = {
        # key : method 
        # value list containing n
        "random": 1024,
        "ESM2": 2560,
        "ProstT5": 1024,
}

def main(
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
    # ++ center_loss_labels ,
    device,
    store_global_pred_train=False,
    run_train=True,
    run_test=True,
    checkpointing=False,
    ):

     # saving folders 
    save_graph = cfg.save_prefix + "/graph.pkl"
    save_best_model = cfg.save_prefix + "/best_model_save.pth"
    save_latest_model = cfg.save_prefix + "/latest_model_save.pth"
    save_ppi_embed = cfg.save_prefix + "/protein_embed.pth"
    save_mg_embed = cfg.save_prefix + "/mg_embed.pth"
    save_labels_dict = cfg.save_prefix + "/labels_dict.txt"
    
    if cfg.get("split_mode") == "global":
        save_bestap_train_metrics = cfg.save_prefix + "/df_bestap_global_train_metrics.csv"
        save_bestap_val_metrics = cfg.save_prefix + "/df_bestap_global_val_metrics.csv"
        save_bestap_test_metrics = cfg.save_prefix + "/df_bestap_global_test_metrics.csv"
    else:
        save_bestap_train_metrics = cfg.save_prefix + "/df_bestap_train_metrics.csv"
        save_bestap_val_metrics = cfg.save_prefix + "/df_bestap_val_metrics.csv"
        save_bestap_test_metrics = cfg.save_prefix + "/df_bestap_test_metrics.csv"

    if plot:
        save_plots = cfg.save_prefix + "/train_embed_plots.pdf"
        
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda': print(torch.cuda.get_device_name(0))
    best_val_ap = -1
    best_model = None
    eps = 10e-4


    # Read data  
 
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
            feat_dim=cfg['hidden'] if cfg['n_heads'] is None else cfg['output'] * cfg['n_heads'],
            use_gpu=torch.cuda.is_available()
            )
        params += list(center_loss.parameters())

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
            _restore_rng_state(ckpt.get('rng_state'))
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
            # ppi_metapaths_train, mg_metapaths_train, ppi_metapaths_val, mg_metapaths_val = train(epoch, model, optimizer, center_loss)
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
        print('Skipping training as per user request.')

    if os.path.exists(save_best_model):
        id_to_name = {cell_id: name for name, cell_id in celltype_map.items()}
        cell_ids = list(ppi_data)
        portable_config = {
            "dataset_mode": cfg.dataset_mode,
            "split_mode": cfg.split_mode,
            "features_mode": cfg.features_mode,
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
            best_model_dict = torch.load(save_best_model, map_location="cpu", weights_only=False) # containing keys (epoch, model, optimizer)
            print('Saved best epoch:', best_model_dict['epoch'])
            best_model = model
            best_model.load_state_dict(best_model_dict['model'].state_dict()) # conversion ensuring that the weight loading process remains valid across older-newer model versions
        best_model.to(device)
        best_model.eval()

        if cfg.get("split_mode") == "global":
            ppi_metapaths_test = None
            mg_metapaths_test = None
            evaluation_mode = "corrected"
            print('Refreshing test metrics for best val AP model using corrected global PPI batching')
        else:
            ppi_metapaths_test = {}
            mg_metapaths_test = []
            for key in ppi_metapaths_train.keys():
                ppi_metapaths_test[key] = [torch.cat(ppi_metapaths_val[key] + ppi_metapaths_train[key], dim=1).to(device)]
            for mg_mt_t, mg_mt_v in zip(mg_metapaths_train, mg_metapaths_val):
                mg_metapaths_test.append(torch.cat([mg_mt_t, mg_mt_v], dim=1).to(device))
            evaluation_mode = "standard"
            print('Refreshing test metrics for best val AP model using original PINNACLE batching')

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
    dict_cfg['protein_feat_dim'] = features_mode_dict[features_mode]
    dict_cfg['ppi_feat_dir'] = _get_ppi_feat_dir(project_paths, features_mode)

    had_failure = False
    for gnn_lambda, gnn_dropout, gnn_output in product(
        list_gnn_lambda, list_gnn_dropout, list_gnn_output):
        
        dict_cfg['lambda'] = gnn_lambda
        dict_cfg['dropout'] = gnn_dropout
        dict_cfg['output'] = gnn_output
        if dict_cfg['n_heads'] is None:
            dict_cfg['hidden'] = dict_cfg['output']
        cfg = DictConfig(dict_cfg)

        # setup data 
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
        # get_centerloss_labels inside main method 

        # Define experiment_name
        gnn_lambda_str = '%s'%(str(cfg['lambda']).replace('.', ''))
        gnn_dropout_str = '%s'%(str(cfg['dropout']).replace('.', ''))
        
        gnn_str = f"{gnn_method}_H{cfg['hidden']}_lambda{gnn_lambda_str}_drop{gnn_dropout_str}_out{cfg['output']}"
        
        experiment_name = f'PINNACLE_model_{split_mode_str}{features_mode}_symmetric_PPI-{symmetric_ppi}_{gnn_str}'
        if optim_str != '':
            experiment_name += f'_{optim_str}'
        experiment_name += f'_{dataset_mode}'
        print(experiment_name)

        # Define full path experiment folder 
        save_prefix = os.path.join(exp_dir, experiment_name)
        print("save prefix", save_prefix)
        run_train = True
        run_test = True
        existing_repo = os.path.exists(save_prefix)
        if not existing_repo:
            os.makedirs(save_prefix, exist_ok=True)
        if existing_repo:
            # sanity check to see if the experiment was already done successfully
            print('-- already done experiment -> checking if experiment was run successfully')
            save_latest_model = save_prefix + "/latest_model_save.pth"
            save_best_model = save_prefix + "/best_model_save.pth"
            if not os.path.exists(save_latest_model):
                if os.path.exists(save_best_model):
                    print('-- latest model missing but best model exists, skipping training and refreshing eval metrics')
                    run_train = False
                    run_test = True
                else:
                    print('-- latest model missing, treating as fresh run with test refresh at the end')
                    run_train = True
                    run_test = True
            else:
                latest_model_dict = torch.load(save_latest_model, weights_only=False) # containing keys (epoch, model, optimizer)

                if latest_model_dict['epoch'] == (epochs - 1):
                    run_train = False
                    run_test = True
                elif checkpointing:
                    print(f'-- resuming training from epoch {latest_model_dict["epoch"] + 1}')
                    run_train = True
                    run_test = True
                else:
                    raise Exception('Model exists but did not finish to train ... set PINNACLE_CHECKPOINTING=true to resume')
        
        try:
            if run_train or run_test:  
                print('run_train:', run_train, ' | run_test:', run_test)    
                print('save_prefix:', save_prefix)
                # complete config dict and save it
                cfg['save_prefix'] = save_prefix 
            
                # save config 
                if not existing_repo:
                    config_file = save_prefix + '/config_dict_seed%s.pkl'%cfg.seed # cfg['save_prefix'] + '/config_dict_seed%s.pkl'%cfg.seed
                    with open(config_file, 'wb') as f:
                        pickle.dump(dict_cfg, f)

                # Use the best parameters from the PINNACLE implementation.
                utils.set_seed(seed)
                # setup wandb tracker
                utils.setup_wandb(cfg, experiment_name)
                
            
                main(cfg,
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
