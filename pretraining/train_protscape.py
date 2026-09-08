import os
import shutil
import csv
import random
import argparse
import torch
import wandb
import numpy as np

from .data_handler.generate_input import read_data
from .models.hierarchical_model import hierarchical_model
from .checkpoints import save_portable_checkpoint

from . import utils
from .train import train_hierarchical_model_factored
from .train import factored_batching as batch_utils

from omegaconf import DictConfig
import pickle 
from itertools import product
import yaml
from pathlib import Path


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


def _load_best_metric_value(metric, metric_cfg):
    metrics_score = None
    metrics_path = metric_cfg["val_metrics_path"]
    if os.path.exists(metrics_path):
        with open(metrics_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            row = next(reader, None)
        if row is not None:
            ppi_key = f"best_val_{metric}_ppi"
            meta_key = f"best_val_{metric}_meta"
            ppi_value = row.get(ppi_key)
            meta_value = row.get(meta_key)
            if metric == "ap" and ppi_value not in [None, ""] and meta_value not in [None, ""]:
                metrics_score = float(ppi_value) + float(meta_value)
            elif ppi_value not in [None, ""]:
                metrics_score = float(ppi_value)

    model_path = metric_cfg["model_path"]
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        score = ckpt.get("score")
        score_metric = ckpt.get("score_metric")
        if score is not None and (metric != "ap" or score_metric == "ap_ppi_plus_ap_meta" or metrics_score is None):
            return float(score)

    return metrics_score


def _restore_best_trackers(dict_save_models):
    for metric, metric_cfg in dict_save_models.items():
        restored_value = _load_best_metric_value(metric, metric_cfg)
        if restored_value is not None:
            metric_cfg["value"] = restored_value
            print(f"Restored best {metric} tracker to {restored_value:.6f}")


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


def _str2bool(raw):
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}")


def _parse_csv(raw_value, cast):
    return [cast(x) for x in str(raw_value).split(",")]


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Run factored hierarchical PINNACLE pretraining.")

    # positional model-hyperparameters
    parser.add_argument("gnn_method", type=str, choices=["GATv2", "GIN", "ACM_RandomWalk"])
    parser.add_argument("jumping_knowledge", type=str, help="Comma-separated JK values, e.g. concat or concat_full")
    parser.add_argument("hidden_dim", type=str, help="Comma-separated hidden dims, e.g. 512")
    parser.add_argument("gnn_dropout", type=str, help="Comma-separated dropouts, e.g. 0.2")
    parser.add_argument("gnn_n_layers", type=str, help="Comma-separated layer counts, e.g. 3")
    parser.add_argument("reg_CTassignment", type=str, help="Comma-separated CTassignment regs, e.g. 1")
    parser.add_argument("cell_pooling", type=str, choices=["attention", "mean", "vn", "learnedvn"])
    parser.add_argument("features_mode", type=str, help="random|ESM2|ProstT5")
    parser.add_argument("add_virtual_node", type=int, choices=[0, 1])
    parser.add_argument("s2gae_enabled", nargs="?", type=int, default=0, choices=[0, 1])
    parser.add_argument("s2gae_decode_channels", nargs="?", type=int, default=512)
    parser.add_argument("s2gae_decoder_layers", nargs="?", type=int, default=2)
    parser.add_argument("graph_saint_norm", nargs="?", type=int, default=0, choices=[0, 1])

    # runtime/training controls
    parser.add_argument("--dataset-mode", type=str, default="bulk", choices=["bulk", "legacy"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--checkpointing", type=_str2bool, default=False)
    parser.add_argument("--eval-only", type=_str2bool, default=False)
    parser.add_argument("--eval-save-prefix", type=str, default=None)
    parser.add_argument("--eval-output-prefix", type=str, default=None)
    parser.add_argument("--k-negatives", type=int, default=1)
    parser.add_argument("--use-metagraph", type=_str2bool, default=True)
    parser.add_argument("--loader", type=str, default="graphsaint")
    parser.add_argument("--split-mode", type=str, default="global", choices=["context", "global"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weighted-ppi-loss", type=_str2bool, default=False)
    parser.add_argument("--ppi-loss", type=str, default="bce", choices=["bce", "l1", "phuber"])
    parser.add_argument("--ppi-phuber-tau", type=float, default=10.0)

    # S2GAE controls
    parser.add_argument("--s2gae-mask-ratio", type=float, default=0.5)
    parser.add_argument("--s2gae-mask-type", type=str, default="dm", choices=["um", "dm"])
    parser.add_argument("--s2gae-decoder-type", type=str, default="cross_layer", choices=["cross_layer", "simple"])
    parser.add_argument("--s2gae-decoder-dropout", type=float, default=0.0)
    parser.add_argument("--s2gae-loss-weight", type=float, default=1.0)
    parser.add_argument("--metagraph-loss-weight", type=float, default=1.0)

    # uniformity loss controls
    parser.add_argument("--uniformity-enabled", type=_str2bool, default=False)
    parser.add_argument("--uniformity-lambda-reg", type=float, default=0.0005)
    parser.add_argument("--uniformity-t", type=float, default=2.0)
    parser.add_argument("--uniformity-dim", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:1")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


# Configuration

cli_args = _parse_cli_args()

# Fixed cell-memory setting.
cell_memory = 'average'
cell_memory_str = '' if cell_memory == 'average' else f'_{cell_memory}cellmemory'

k_negatives = int(cli_args.k_negatives)
k_negatives_str = '' if k_negatives == 1 else f'_kneg{str(k_negatives)}'

# Model hyperparameters
gnn_method = str(cli_args.gnn_method)
list_jumping_knowledge = _parse_csv(cli_args.jumping_knowledge, str)
assert all(x in ['concat', 'concat_full', 'max', 'lstm', 'sum', 'none'] for x in list_jumping_knowledge)

list_hidden_dim = _parse_csv(cli_args.hidden_dim, int)
list_gnn_dropout = _parse_csv(cli_args.gnn_dropout, float)
list_gnn_n_layers = _parse_csv(cli_args.gnn_n_layers, int)
list_reg_CTassignment = [float(x) if float(x)!=1. else int(x) for x in _parse_csv(cli_args.reg_CTassignment, float)]
cell_pooling = str(cli_args.cell_pooling)
features_mode = _normalize_features_mode(cli_args.features_mode)
add_virtual_node = bool(int(cli_args.add_virtual_node))
print('--add_virtual_node:', add_virtual_node)

if add_virtual_node:
    assert cell_pooling in ['attention']

# Virtual nodes are used for both message passing and pooling with ``vn`` or
# ``learnedvn``; with attention pooling they are used only for message passing.

input_dim = None
ppi_feat_dir = None

# S2GAE positional controls
S2GAE_ENABLED = bool(int(cli_args.s2gae_enabled))
print('-- S2GAE enabled:', S2GAE_ENABLED)
s2gae_decode_channels = int(cli_args.s2gae_decode_channels)
print('s2gae_decode_channels:', s2gae_decode_channels)
s2gae_decoder_layers = int(cli_args.s2gae_decoder_layers)
print('s2gae_decoder_layers:', s2gae_decoder_layers)


graph_saint_norm = bool(int(cli_args.graph_saint_norm))
print('-- graph_saint_norm enabled:', graph_saint_norm)
weighted_ppi_loss = bool(cli_args.weighted_ppi_loss)
ppi_loss = str(cli_args.ppi_loss).lower()
ppi_phuber_tau = float(cli_args.ppi_phuber_tau)
if ppi_loss == "phuber" and ppi_phuber_tau <= 1.0:
    raise ValueError("--ppi-phuber-tau must be > 1 for partially Huberized BCE.")

# S2GAE and loss hyperparameters
s2gae_mask_ratio = float(cli_args.s2gae_mask_ratio)
s2gae_mask_type = str(cli_args.s2gae_mask_type)
if s2gae_mask_type not in {"um", "dm"}:
    raise ValueError(f"Invalid s2gae_mask_type={s2gae_mask_type!r}; expected 'um' or 'dm'.")
s2gae_decoder_type = str(cli_args.s2gae_decoder_type)
if s2gae_decoder_type not in {"cross_layer", "simple"}:
    raise ValueError(
        f"Invalid s2gae_decoder_type={s2gae_decoder_type!r}; expected 'cross_layer' or 'simple'."
    )
s2gae_decoder_dropout = float(cli_args.s2gae_decoder_dropout)
s2gae_loss_weight = float(cli_args.s2gae_loss_weight)
metagraph_loss_weight = float(cli_args.metagraph_loss_weight)

uniformity_enabled = bool(cli_args.uniformity_enabled)
uniformity_lambda_reg = float(cli_args.uniformity_lambda_reg)
uniformity_t = float(cli_args.uniformity_t)
uniformity_dim = int(cli_args.uniformity_dim) if uniformity_enabled else 0
if uniformity_enabled and uniformity_lambda_reg <= 0.0:
    raise ValueError("--uniformity-enabled=true requires --uniformity-lambda-reg > 0.")
if uniformity_enabled and uniformity_t <= 0.0:
    raise ValueError("--uniformity-t must be > 0.")
if uniformity_enabled and uniformity_dim < 0:
    raise ValueError("--uniformity-dim must be >= 0.")
s2gae_str = (
    f"s2gae_{s2gae_mask_type}"
    f"_mr{utils._fmt_float(s2gae_mask_ratio)}"
    f"_dc{s2gae_decode_channels}"
    f"_dl{s2gae_decoder_layers}"
    f"_do{utils._fmt_float(s2gae_decoder_dropout)}"
) if S2GAE_ENABLED else ''

print('gnn_method: %s / list_jumping_knowledge %s / list_hidden_dim %s / list_gnn_dropout %s / list_gnn_n_layers %s / list_reg_CTassignment %s / cell pooling : %s '%(
    gnn_method, list_jumping_knowledge, list_hidden_dim, list_gnn_dropout, list_gnn_n_layers, list_reg_CTassignment, cell_pooling
))

seed = int(cli_args.seed)
utils.set_seed(seed)

# Dataset mode (controls data-specific tweaks, e.g. bulk CL IDs vs legacy naming)
dataset_mode = str(cli_args.dataset_mode).lower()
assert dataset_mode in ["bulk", "legacy"]

print("Dataset mode:", dataset_mode)
# Use CCI metagraph or PPI only
use_metagraph = bool(cli_args.use_metagraph)


# Optimization hyperparameters

lr = float(cli_args.lr)
use_scheduler = False
batch_size = int(cli_args.batch_size)
epochs = int(cli_args.epochs)
checkpointing = bool(cli_args.checkpointing)
loader = str(cli_args.loader)
split_mode = str(cli_args.split_mode)
assert split_mode in ['context', 'global']
# context: corresponds to initial pinnacle split with allowing edge leakage across contexts
# global: defines splits over the global ppi, stratified by edge frequencies across contexts, preventing edge leakage across contexts.


lr_str = '' if lr == 0.001 else '_lr' + str(lr).replace('.', '')
batch_size_str = '' if batch_size == 64 else f'_b{batch_size}'
epochs_str = f'_ep{epochs}'
loader_str = '' if loader == 'graphsaint' else f'_{loader}'
graph_saint_norm_str = '_gsnorm' if graph_saint_norm else ''
weighted_ppi_loss_str = '_weightedppiloss' if weighted_ppi_loss else ''
if ppi_loss == 'bce':
    ppi_loss_str = ''
elif ppi_loss == 'phuber':
    ppi_loss_str = f'_phubertau{utils._fmt_float(ppi_phuber_tau)}'
else:
    ppi_loss_str = f'_{ppi_loss}'
if graph_saint_norm:
    assert not weighted_ppi_loss
    
optim_str = lr_str + batch_size_str + epochs_str + loader_str + weighted_ppi_loss_str + graph_saint_norm_str + cell_memory_str + k_negatives_str + ppi_loss_str

split_mode_str = '' if split_mode == 'context' else f'{split_mode}split_'

# Data paths

paths_file = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
with paths_file.open("r", encoding="utf-8") as handle:
    project_paths = yaml.safe_load(handle)

output_root = Path(project_paths["output_root"]).expanduser()
network_dir = str(Path(project_paths["networks_bulk"]).expanduser())
global_ppi_edges = os.path.join(network_dir, "global_ppi_edgelist.txt")
ppi_dir = os.path.join(network_dir, "ppi_edgelists")
metagraph_edges = os.path.join(network_dir, "mg_edgelist.txt")
if graph_saint_norm:
    graphsaint_dir = os.path.join(str(output_root), "pretraining", "graphsaint_cache")
    os.makedirs(graphsaint_dir, exist_ok=True)
else:
    graphsaint_dir = None

exp_dir = os.path.join(str(output_root), "pretraining", "protscape")
if split_mode == 'global':
    count_edge_path = os.path.join(network_dir, "count_edge_dict.pkl")
else:
    count_edge_path = None

if features_mode in ["ESM2", "ProstT5"]:
    ppi_feat_dir = _get_ppi_feat_dir(project_paths, features_mode)

hierarchical_mode = 'CTassignment'
assert hierarchical_mode in ['CTassignment']
# will see later if we need a more informative metagraph knowledge

if features_mode == "random":
    input_dim = 1024 # default values in pinnacle for testing


if cell_pooling == 'attention':
    cell_att_embedding_dim = 128 #256
    cell_str = f'_{cell_pooling}{cell_att_embedding_dim}'
elif cell_pooling == 'mean':
    cell_att_embedding_dim = None
    cell_str = '' # default for the method
elif cell_pooling in ['vn', 'learnedvn']:
    cell_att_embedding_dim = None
    cell_str = f'_{cell_pooling}'
    
device = (
    "cuda" if cli_args.device == "auto" and torch.cuda.is_available()
    else "cpu" if cli_args.device == "auto"
    else cli_args.device
)
plot = False
symmetric_ppi = True #False only for original Pinnacle
n_jobs = None # number of cpus used for parallel dataloading
# Setup


dict_cfg = { # mostly default settings for general learning
    'epochs': epochs,
    'loader': loader,  # minibatch loader type
    'graph_saint_norm':graph_saint_norm, # whether or not to use graph saint batch normalization on train set
    'batch_size': batch_size,  # Batch size for training
    'lr': lr,  # Learning rate for the optimizer
    'use_scheduler': use_scheduler,  # Whether to use a learning rate scheduler
    'features_mode': features_mode,  # Mode for generating features (random, ESM2 or ProstT5)
    'hierarchical_mode': hierarchical_mode, # Mode to handle the metagraph
    'reg_centerloss' : 0., # regularization coefficient for the center loss (default is 0.)
    'reg_CTassignment' : None, # regularization coefficient for the CT multi-label assingments
    'use_metagraph': use_metagraph,
    'dataset_mode': dataset_mode,
    'split_mode': split_mode,
    'metagraph_loss_weight': metagraph_loss_weight,
    'weighted_ppi_loss': weighted_ppi_loss,
    'ppi_loss': ppi_loss,
    'ppi_phuber_tau': ppi_phuber_tau,
    'add_virtual_node': add_virtual_node,
    'k_negatives': k_negatives, # number of structured negatives sampled per positive edge (default 1)
    'wandb_mode': cli_args.wandb_mode,
    # S2GAE (Self-Supervised Graph Autoencoders) configuration
    # Set 'enabled' to True to use S2GAE training with edge masking
    's2gae_config': {
        'enabled': S2GAE_ENABLED,
        'mask_ratio': s2gae_mask_ratio,
        'mask_type': s2gae_mask_type,
        'decoder_type': s2gae_decoder_type,  # 'cross_layer' or 'simple'
        'decode_channels': s2gae_decode_channels,
        'decoder_layers': s2gae_decoder_layers,
        'decoder_dropout': s2gae_decoder_dropout,
        'loss_weight': s2gae_loss_weight,
    },
    'uniformity_config': {
        'enabled': uniformity_enabled,
        'lambda_reg': uniformity_lambda_reg,
        't': uniformity_t,
        'dim': uniformity_dim,
    },
    }

use_virtual_node = cell_pooling in ['vn', 'learnedvn']

protein_config = {
    'input_dim': input_dim,  # Input dimension for protein features
    'gnn_method': None,  # GNN method to use (GATv2, GIN, ACM)
    'hidden_dim': None,  # Hidden dimension for GNN layers
    'gnn_dropout': None,  # Dropout rate for GNN layers
    'gnn_batchnorm': True,  # Whether to use batch normalization
    'gnn_activation': 'leaky_relu',  # Activation function for GNN layers
    'jumping_knowledge': 'concat',  # Jumping knowledge method
    'n_layers': None,  # Number of GNN layers
    'gat_heads': 8,  # Number of heads for GATv2
    'use_virtual_node': use_virtual_node,  # Whether to use virtual node for cell embeddings
    'n_cells': None,
    'add_virtual_node': add_virtual_node,  # Whether to add virtual node for attention-based pooling
}

cell_config = {
    'pooling': cell_pooling,
    'att_embedding_dim': cell_att_embedding_dim,
    'memory': cell_memory,
    'window_size': 50, # in this case the window size is automatically defined by the number of batches used during one epoch
    'n_cells': None,
}


def main(
    cfg,
    ppi_data,
    mg_data,
    edge_attr_dict,
    celltype_map,
    tissue_neighbors,
    CT_map,
    device,
    n_jobs,
    run_train=True,
    run_test=True,
    evaluation_mode='global',
    checkpointing=False,
    ):
    assert evaluation_mode in ['global', 'local']
    
    print('--- start main / device:', device)
    # saving folders 
    save_latest_model = cfg.save_prefix + "/latest_model_save.pth"
    metrics_save_prefix = str(getattr(cfg, "eval_output_prefix", None) or cfg.save_prefix)
    os.makedirs(metrics_save_prefix, exist_ok=True)
    primary_metric = 'ap'
    primary_model_path = cfg.save_prefix + "/best_model_save.pth"
    list_save_metrics = ['acc', 'roc', 'f1', 'ap']
    dict_save_models = {
    }
    for metric in list_save_metrics:
        dict_save_models[metric] = {
            'value': -1, # keep track of the best value for this metric
            'model_path': cfg.save_prefix + f"/best_{metric}_model_save.pth"
        }
        if metric == primary_metric:
            dict_save_models[metric]['primary_model_path'] = primary_model_path
        for set_ in ['train', 'val', 'test']:
            if evaluation_mode == 'local':
                dict_save_models[metric][f'{set_}_metrics_path'] = metrics_save_prefix + f"/df_best{metric}_{set_}_metrics.csv"
            elif evaluation_mode == 'global':
                dict_save_models[metric][f'{set_}_metrics_path'] = metrics_save_prefix + f"/df_best{metric}_global_{set_}_metrics.csv"

    if checkpointing:
        _restore_best_trackers(dict_save_models)

    ppi_metapaths_train = None
    best_model = None

    
    if run_train or run_test:
        if cfg.hierarchical_mode == 'CTassignment':
            s2gae_config = getattr(cfg, 's2gae_config', None)
            uni_cfg = getattr(cfg, 'uniformity_config', None) or {}
            model = hierarchical_model(
                cfg.hierarchical_mode,
                cfg.protein_config,
                cfg.cell_config,
                cfg.tissue_config,
                ppi_data,
                device,
                s2gae_config=s2gae_config,
                use_metagraph=cfg.use_metagraph,
                graph_saint_norm=cfg.graph_saint_norm,
                uniformity_dim=int(uni_cfg.get('dim', 0)),
            ).to(device)
            if cfg.use_metagraph:
                cell_ids = list(ppi_data.keys())
                cci_edge_index = None
                cci_splits = getattr(mg_data, "cci_edge_splits", None)
                if cci_splits is not None:
                    cci_edge_index = cci_splits.get("train", None)
                if cci_edge_index is None:
                    cci_edge_index = batch_utils.build_cci_edge_index(
                        mg_data, edge_attr_dict, cell_ids
                    )
                if cci_edge_index is not None:
                    model.set_cci_graph(cci_edge_index.to(device), mg_data.num_nodes)
        else:
            raise f'not implemented hierarchical mode : {cfg.hierarchical_mode}'    
        
    # values instantiated before updates at train or test time
    # ppi_metapaths_train, mg_metapaths_train, ppi_metapaths_val, mg_metapaths_val = None, None, None, None
    
    if run_train:
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr = cfg.lr, weight_decay = 0.)
        
        assert cfg.reg_centerloss == 0.
        center_loss_labels, train_mask, val_mask, test_mask = None, None, None, None

        wandb.watch(model)
        print(model)

        start_epoch = 0
        if checkpointing and os.path.exists(save_latest_model):
            print(f"Resuming from checkpoint: {save_latest_model}")
            ckpt = torch.load(save_latest_model, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt['model'].state_dict())
            optimizer.load_state_dict(ckpt['optimizer'].state_dict())
            _restore_rng_state(ckpt.get('rng_state'))
            start_epoch = ckpt['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}/{cfg.epochs}")

        for epoch in range(start_epoch, cfg.epochs):
            print(f"\n===== Epoch {epoch + 1}/{cfg.epochs} =====", flush=True)
            train_hierarchical_model_factored.train(
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
                CT_map=CT_map,
                device=device,
                evaluation_mode=evaluation_mode,
                root_gsnorm_path=graphsaint_dir,
                n_jobs=n_jobs)

        print("Optimization finished!")

    else:
        print('Training disabled.')

    portable_source = primary_model_path
    if not os.path.exists(portable_source):
        portable_source = dict_save_models[primary_metric]["model_path"]
    if os.path.exists(portable_source):
        id_to_name = {cell_id: name for name, cell_id in celltype_map.items()}
        cell_ids = list(ppi_data)
        portable_config_keys = (
            "epochs", "loader", "graph_saint_norm", "batch_size", "lr",
            "use_scheduler", "features_mode", "hierarchical_mode",
            "reg_centerloss", "reg_CTassignment", "use_metagraph",
            "dataset_mode", "split_mode", "metagraph_loss_weight",
            "weighted_ppi_loss", "ppi_loss", "ppi_phuber_tau",
            "add_virtual_node", "k_negatives", "s2gae_config",
            "uniformity_config", "protein_config", "cell_config",
            "tissue_config", "seed",
        )
        portable_config = {key: cfg[key] for key in portable_config_keys}
        portable_config["symmetric_ppi"] = symmetric_ppi
        save_portable_checkpoint(
            Path(cfg.save_prefix) / "best_model_state_dict.pt",
            portable_source,
            model_type="protscape",
            config=DictConfig(portable_config),
            cell_ids=cell_ids,
            cell_names=[id_to_name[cell_id] for cell_id in cell_ids],
        )

    if run_test:
            
        # Load best model for different metric checkpoints
        
        for metric in dict_save_models.keys():
            save_best_model = dict_save_models[metric]['model_path']
            if not os.path.exists(save_best_model):
                legacy_best_model = cfg.save_prefix + "/best_model_save.pth"
                if metric == 'f1' and os.path.exists(legacy_best_model):
                    print(f"Using legacy best_model_save.pth for metric 'f1': {legacy_best_model}")
                    save_best_model = legacy_best_model
                else:
                    print(f"Skipping test for metric '{metric}': missing checkpoint {save_best_model}")
                    continue
            print('loading model weights')
            best_model_dict = torch.load(save_best_model, map_location="cpu", weights_only=False) # containing keys (epoch, model, optimizer)
            print('Saved best epoch:', best_model_dict['epoch'])
            best_model = model
            best_model.load_state_dict(best_model_dict['model'].state_dict()) # conversion ensuring that the weight loading process remains valid across older-newer model versions

            # Recompute test metrics from the selected checkpoint.
            train_hierarchical_model_factored.test(
                cfg,
                best_model,
                dict_save_models[metric]['test_metrics_path'],
                ppi_data,
                mg_data,
                edge_attr_dict,
                celltype_map,
                tissue_neighbors,
                CT_map,
                device=device,
                n_jobs=n_jobs,
                split='test')
            
        
        
if __name__ == "__main__":
    
    # setup data
    ppi_data, mg_data, edge_attr_dict, celltype_map, tissue_neighbors, ppi_layers, metagraph, CT_map = read_data(
        global_ppi_edges,
        ppi_dir,
        metagraph_edges,
        feat_mat_dim=input_dim,
        get_CT_map=True,
        ppi_feat_dir=ppi_feat_dir,
        symmetric_ppi=symmetric_ppi,
        dataset_mode=dataset_mode,
        split_mode=split_mode,
        count_edge_path=count_edge_path,
        weighted_ppi_loss=weighted_ppi_loss
        )
    if input_dim is None:
        input_dim = ppi_data[0].x.shape[-1]
        protein_config['input_dim'] = input_dim
    protein_config['n_cells'] = len(ppi_data)
    cell_config['n_cells'] = len(ppi_data)

    if use_metagraph:
        mg_cell_nodes = torch.nonzero(mg_data.node_type == 1, as_tuple=False).view(-1).cpu().tolist()
        ppi_cells = list(ppi_data.keys())
        print("ppi #cells", len(ppi_cells), "mg #cell nodes", len(mg_cell_nodes))
        print("overlap", len(set(ppi_cells) & set(mg_cell_nodes)))
        assert set(ppi_cells) == set(mg_cell_nodes), "PPI keys are not metagraph cell node ids"

    if use_metagraph:
        cell_ids = list(ppi_data.keys())
        batch_utils.get_cci_edge_splits(
            mg_data,
            edge_attr_dict,
            cell_ids=cell_ids,
            split_ratios=(0.8, 0.1, 0.1),
            seed=seed,
        )
    
    tissue_config = {
        'n_tissues': CT_map[0]['onehot'].shape[0]
    }
    
    tested_hyperparameters = {
        'gnn_method':[gnn_method],
        'jumping_knowledge': list_jumping_knowledge,
        'hidden_dim': list_hidden_dim,
        'gnn_dropout': list_gnn_dropout,
        'gnn_n_layers': list_gnn_n_layers,
        'reg_CTassignment': list_reg_CTassignment
    }
    
    # iterate over all hyperparameters
    for gnn_method, jumping_knowledge, hidden_dim, gnn_dropout, gnn_n_layers, reg_CTassignment in product(
        *(tested_hyperparameters[key] for key in tested_hyperparameters.keys())):
        
        print(gnn_method, jumping_knowledge, hidden_dim, gnn_dropout, gnn_n_layers, reg_CTassignment)
        
        utils.set_seed(seed)

        protein_config['gnn_method'] = gnn_method
        protein_config['hidden_dim'] = hidden_dim
        protein_config['gnn_dropout'] = gnn_dropout
        protein_config['n_layers'] = gnn_n_layers
        protein_config['jumping_knowledge'] = jumping_knowledge
        
        dict_cfg['reg_CTassignment'] = reg_CTassignment
        
        gnn_dropout_str = '' if gnn_dropout == 0 else '_dropout%s'%(str(gnn_dropout).replace('.', ''))
        reg_CTassignment_str = '_regCTassignment%s'%(str(reg_CTassignment).replace('.', ''))
        symmetric_ppi_str = 'symmetricPPI' if symmetric_ppi else ''
        mg_str = 'MG' if use_metagraph else 'noMG'
        mg_weight_str = ''
        if use_metagraph and metagraph_loss_weight != 1.0:
            mg_weight_str = f'_mgw{utils._fmt_float(metagraph_loss_weight)}'
        uni_str = ''
        if uniformity_enabled and uniformity_lambda_reg > 0.0:
            uni_str = (
                f'_unil{utils._fmt_float(uniformity_lambda_reg)}'
                f'_unit{utils._fmt_float(uniformity_t)}'
                f'_unid{uniformity_dim}'
            )
        
        add_virtual_node_str = '_addVN' if add_virtual_node else ''
        gnn_str = f'{gnn_method}{jumping_knowledge}{add_virtual_node_str}_H{hidden_dim}L{gnn_n_layers}{gnn_dropout_str}{reg_CTassignment_str}{mg_weight_str}{uni_str}'
        s2gae_name_str = f'_{s2gae_str}' if s2gae_str else ''
        experiment_name = f'/HC{hierarchical_mode}{split_mode_str}{features_mode}{symmetric_ppi_str}{cell_str}{s2gae_name_str}_{mg_str}_{dataset_mode}'
        if optim_str == '':
            experiment_name += f'_{gnn_str}'
        else:
            experiment_name += f'_{optim_str}_{gnn_str}'
        save_prefix = exp_dir + experiment_name
        if cli_args.eval_save_prefix is not None:
            save_prefix = str(cli_args.eval_save_prefix)
        eval_output_prefix = str(cli_args.eval_output_prefix) if cli_args.eval_output_prefix is not None else None
        config_file = save_prefix + '/config_dict_seed%s.pkl'%seed
        print('experiment_name:', experiment_name)
        
        run_train = True
        run_test = True
        if bool(cli_args.eval_only):
            if not os.path.exists(save_prefix):
                raise FileNotFoundError(f"--eval-only requested but experiment directory is missing: {save_prefix}")
            print(f'-- eval-only requested for {save_prefix}; skipping training and recomputing test metrics.')
            run_train = False
        elif os.path.exists(save_prefix):
            if checkpointing:
                save_latest_model = save_prefix + "/latest_model_save.pth"
                if os.path.exists(save_latest_model):
                    ckpt = torch.load(save_latest_model, map_location="cpu", weights_only=False)
                    if ckpt['epoch'] >= (epochs - 1):
                        print(f'-- training already completed (epoch {ckpt["epoch"]}), skipping to test.')
                        run_train = False
                    else:
                        print(f'-- resuming training from epoch {ckpt["epoch"] + 1}')
                else:
                    print('-- checkpoint dir exists but no latest model, training from scratch.')
            elif cli_args.overwrite:
                print(f'-- existing experiment directory found at {save_prefix}, removing it.')
                shutil.rmtree(save_prefix)
            else:
                raise FileExistsError(
                    f"Experiment already exists: {save_prefix}. Use --checkpointing, "
                    "--eval-only, or --overwrite."
                )
            
        print('save_prefix:', save_prefix)
        os.makedirs(save_prefix, exist_ok=True)
        if eval_output_prefix is not None:
            print('eval_output_prefix:', eval_output_prefix)
            os.makedirs(eval_output_prefix, exist_ok=True)
        # Set up configuration
        dict_cfg['protein_config'] = protein_config.copy()
        dict_cfg['cell_config'] = cell_config.copy()
        dict_cfg['tissue_config'] = tissue_config.copy()
        dict_cfg['save_prefix'] = save_prefix 
        dict_cfg['eval_output_prefix'] = eval_output_prefix
        dict_cfg['seed'] = seed
        
        cfg = DictConfig(dict_cfg)
        
        # save config
        if not bool(cli_args.eval_only):
            with open(config_file, 'wb') as f:
                pickle.dump(dict_cfg, f)
        
        # setup wandb tracker
        utils.setup_wandb(cfg, experiment_name)
            
        main(
            cfg,
            ppi_data,
            mg_data,
            edge_attr_dict,
            celltype_map,
            tissue_neighbors,
            CT_map,
            device,
            n_jobs,
            run_train=run_train,
            run_test=run_test,
            checkpointing=checkpointing)
