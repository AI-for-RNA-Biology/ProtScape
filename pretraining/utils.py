import os
import random
from decimal import Decimal
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch_sparse
import torch.nn.functional as F
from torch.nn import Sigmoid
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, roc_curve, precision_recall_curve, silhouette_score, calinski_harabasz_score, davies_bouldin_score
import omegaconf
import wandb

from tqdm import tqdm

from torchmetrics import AUROC, AveragePrecision, Accuracy, F1Score
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    # Seed
    print("SEED:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_wandb(cfg, experiment_name):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'project': 'ProtScape',
              'config': config_dict,
              'reinit': True,
              'mode': str(getattr(cfg, 'wandb_mode', 'disabled')),
              'group': str(cfg.seed),
              'settings': wandb.Settings(_disable_stats=True),
              'name': experiment_name}
    wandb.init(**kwargs)
    return cfg

def calc_individual_metrics(pred, y):
    try: 
        roc_score = roc_auc_score(y, pred)
    except ValueError: 
        roc_score = 0.5 
    ap_score = average_precision_score(y, pred)
    acc = accuracy_score(y, pred > 0.5)
    f1 = f1_score(y, pred > 0.5, average='macro', labels=[0, 1], zero_division=0)
    return roc_score, ap_score, acc, f1


def calc_individual_metrics_torch(pred, y):
    y=y.to(dtype=torch.int32, device=pred.device)
    roc_auc_score_ = AUROC(task='binary').to(pred.device)
    average_precision_score_ = AveragePrecision(task='binary').to(pred.device)
    accuracy_score_ = Accuracy(task='binary').to(pred.device)
    try: 
        roc_score = roc_auc_score_(pred, y).item()
    except ValueError: 
        roc_score = 0.5 
    ap_score = average_precision_score_(pred, y).item()
    acc = accuracy_score_(pred, y).item()
    # TorchMetrics ignores ``average='macro'`` for ``task='binary'`` and
    # returns positive-class F1.  Compute both class-wise F1 scores explicitly
    # so this matches ``calc_individual_metrics`` above.
    pred_label = pred > 0.5
    y_bool = y.bool()
    tp = torch.sum(pred_label & y_bool).float()
    tn = torch.sum(~pred_label & ~y_bool).float()
    fp = torch.sum(pred_label & ~y_bool).float()
    fn = torch.sum(~pred_label & y_bool).float()

    def class_f1(true_positive, false_positive, false_negative):
        denominator = 2 * true_positive + false_positive + false_negative
        return torch.where(
            denominator > 0,
            2 * true_positive / denominator,
            torch.zeros_like(denominator),
        )

    f1 = torch.mean(torch.stack([class_f1(tp, fp, fn), class_f1(tn, fn, fp)])).item()
    return roc_score, ap_score, acc, f1


def calc_metrics_factored(mg_pred, mg_data, ppi_preds, ppi_labels):
    
    levels = ['ppi']
    if mg_pred is not None:
        levels.append('meta')
    
    list_metrics = ['roc', 'ap', 'acc', 'f1']
    metrics = {}
    for level in levels:
        for metric in list_metrics:
            metrics[f'{metric}_{level}'] = []

    # Compute metrics for metagraph
    if mg_pred is not None:
        if len(mg_pred) > 0:
            roc_score, ap_score, acc, f1 = calc_individual_metrics_torch(
                mg_pred.detach(), mg_data["y"].detach())

            metrics['roc_meta'].append(roc_score)
            metrics['ap_meta'].append(ap_score)
            metrics['acc_meta'].append(acc)
            metrics['f1_meta'].append(f1)

    # Compute metrics for PPI layers
    for celltype in ppi_preds.keys():
        curr_roc_score, curr_ap_score, curr_acc, curr_f1 = calc_individual_metrics_torch(
            ppi_preds[celltype], ppi_labels[celltype])

        metrics['roc_ppi'].append(curr_roc_score)
        metrics['ap_ppi'].append(curr_ap_score)
        metrics['acc_ppi'].append(curr_acc)
        metrics['f1_ppi'].append(curr_f1)

    for m in metrics.keys():
        if len(metrics[m]) > 0:
            metrics[m] = np.mean(metrics[m])
        else:
            metrics[m] = None
    return metrics


def calc_metrics(mg_pred, mg_data, ppi_preds, ppi_data, version='numpy'):
    assert version in ['numpy', 'torch']
    metrics = {
        'roc_ppi': [],
        'ap_ppi': [],
        'acc_ppi': [],
        'f1_ppi': [],
        'roc_meta': [],
        'ap_meta': [],
        'acc_meta': [],
        'f1_meta': [],
    }
    # Compute metrics for metagraph
    if mg_pred is not None:
        if len(mg_pred) > 0:
            if version == 'numpy':
                roc_score, ap_score, acc, f1 = calc_individual_metrics(
                    mg_pred.cpu().detach().numpy(), mg_data["y"].cpu().detach().numpy())
            else:
                roc_score, ap_score, acc, f1 = calc_individual_metrics_torch(
                    mg_pred.detach(), mg_data["y"].detach())

            metrics['roc_meta'].append(roc_score)
            metrics['ap_meta'].append(ap_score)
            metrics['acc_meta'].append(acc)
            metrics['f1_meta'].append(f1)

    # Compute metrics for PPI layers
    for celltype, ppi in ppi_preds.items():
        if version == 'numpy':
            curr_roc_score, curr_ap_score, curr_acc, curr_f1 = calc_individual_metrics(
                ppi.numpy(), ppi_data[celltype]["y"].numpy())
        else:
            
            curr_roc_score, curr_ap_score, curr_acc, curr_f1 = calc_individual_metrics_torch(
                ppi, ppi_data[celltype]["y"])
            ### for sanity check
            #print('torchmetrics: ', curr_roc_score, curr_ap_score, curr_acc, curr_f1)
            #curr_roc_score_, curr_ap_score_, curr_acc_, curr_f1_ = calc_individual_metrics(
            #    ppi.cpu().detach().numpy(), ppi_data[celltype]["y"].cpu().numpy())
            #print('numpy: ', curr_roc_score_, curr_ap_score_, curr_acc_, curr_f1_)

        metrics['roc_ppi'].append(curr_roc_score)
        metrics['ap_ppi'].append(curr_ap_score)
        metrics['acc_ppi'].append(curr_acc)
        metrics['f1_ppi'].append(curr_f1)

    for m in metrics.keys():
        if len(metrics[m]) > 0:
            metrics[m] = np.mean(metrics[m])
        else:
            metrics[m] = None
    return metrics


def metrics_per_rel_factored(
    mg_pred, mg_data,
    ppi_preds, ppi_labels,
    edge_attr_dict, celltype_map,
    split):
    
    celltype_map = {v: k for k, v in celltype_map.items()}

    df_metrics = {}
    # Compute metrics per rel for metagraph
    if mg_pred is not None:
        if len(mg_pred) > 0:
            roc_meta, ap_meta, acc_meta, f1_meta = calc_individual_metrics_torch(
                mg_pred.detach(), mg_data["y"].detach()
            )
            wandb.log({
                f"metagraph_{split}_roc": roc_meta,
                f"metagraph_{split}_ap": ap_meta,
                f"metagraph_{split}_acc": acc_meta,
                f"metagraph_{split}_f1": f1_meta,
            })
            local_res = [('roc', roc_meta), ('ap', ap_meta), ('acc', acc_meta), ('f1', f1_meta)]
            for name_metric, metric_value in local_res:
                df_metrics[f'metagraph_cci_{split}_{name_metric}'] = [metric_value]
                    
    # Compute metrics per rel per PPI layer
    for celltype in ppi_preds.keys():
        ppi = ppi_preds[celltype]
        y_per_rel = ppi_labels[celltype]
        attr = 'ppi'
        roc_per_rel, ap_per_rel, acc_per_rel, f1_per_rel = calc_individual_metrics_torch(
            ppi, y_per_rel)
    
        
        wandb.log({"%s_%s_%s_roc" % (celltype_map[celltype], attr, split): roc_per_rel, "%s_%s_%s_ap" % (celltype_map[celltype], attr, split): ap_per_rel, "%s_%s_%s_acc" % (celltype_map[celltype], attr, split): acc_per_rel, "%s_%s_%s_f1" % (celltype_map[celltype], attr, split): f1_per_rel})
            
        local_res = [('roc', roc_per_rel), ('ap', ap_per_rel), ('acc', acc_per_rel), ('f1', f1_per_rel)]
        for name_metric, metric_value in local_res:
            df_metrics[f'cell{celltype}_ppi_{split}_{name_metric}'] = [metric_value]

    df_metrics = pd.DataFrame(df_metrics)
    return df_metrics


def metrics_per_rel(
    mg_pred, mg_data,
    ppi_preds, ppi_data,
    edge_attr_dict, celltype_map,
    split, version='numpy'):
    
    assert version in ['numpy', 'torch']

    celltype_map = {v: k for k, v in celltype_map.items()}

    df_metrics = {}
    # Compute metrics per rel for metagraph
    if mg_pred is not None:
        if len(mg_pred) > 0:
            for attr, idx in edge_attr_dict.items():
                if version == 'numpy':
                    mask = (mg_data["total_edge_type"].cpu().detach().numpy() == idx)
                    if mask.sum() == 0:
                        continue
                    pred_per_rel = mg_pred.cpu().detach().numpy()[mask]
                    y_per_rel = mg_data["y"].cpu().detach().numpy()[mask]
                    roc_per_rel, ap_per_rel, acc_per_rel, f1_per_rel = calc_individual_metrics(pred_per_rel, y_per_rel)
                else:
                    mask = (mg_data["total_edge_type"].detach() == idx)
                    if mask.sum() == 0:
                        continue
                    pred_per_rel = mg_pred.detach()[mask]
                    y_per_rel = mg_data["y"].detach()[mask]
                    roc_per_rel, ap_per_rel, acc_per_rel, f1_per_rel = calc_individual_metrics_torch(pred_per_rel, y_per_rel)
                
                wandb.log({"%s_%s_roc" % (attr, split): roc_per_rel, "%s_%s_ap" % (attr, split): ap_per_rel, "%s_%s_acc" % (attr, split): acc_per_rel, "%s_%s_f1" % (attr, split): f1_per_rel})
                
                # build df with metrics to save
                local_res = [('roc', roc_per_rel), ('ap', ap_per_rel), ('acc', acc_per_rel), ('f1', f1_per_rel)]
                for name_metric, metric_value in local_res:
                    df_metrics[f'metagraph_rel_{attr}_{split}_{name_metric}'] = [metric_value]
                    
    # Compute metrics per rel per PPI layer
    for celltype, ppi in ppi_preds.items():
        for attr, idx in edge_attr_dict.items():
            if version == 'numpy':
                mask = (ppi_data[celltype]["total_edge_type"].numpy() == idx)
                if mask.sum() == 0:
                    continue
                pred_per_rel = ppi.cpu().detach().numpy()[mask]
                y_per_rel = ppi_data[celltype]["y"].numpy()[mask]
                roc_per_rel, ap_per_rel, acc_per_rel, f1_per_rel = calc_individual_metrics(
                    pred_per_rel, y_per_rel)
            else:
                mask = (ppi_data[celltype]["total_edge_type"] == idx)
                if mask.sum() == 0:
                    continue
                pred_per_rel = ppi.detach()[mask]
                y_per_rel = ppi_data[celltype]["y"][mask]
                roc_per_rel, ap_per_rel, acc_per_rel, f1_per_rel = calc_individual_metrics_torch(
                    pred_per_rel, y_per_rel)
            
            
            wandb.log({"%s_%s_%s_roc" % (celltype_map[celltype], attr, split): roc_per_rel, "%s_%s_%s_ap" % (celltype_map[celltype], attr, split): ap_per_rel, "%s_%s_%s_acc" % (celltype_map[celltype], attr, split): acc_per_rel, "%s_%s_%s_f1" % (celltype_map[celltype], attr, split): f1_per_rel})
                
            local_res = [('roc', roc_per_rel), ('ap', ap_per_rel), ('acc', acc_per_rel), ('f1', f1_per_rel)]
            for name_metric, metric_value in local_res:
                df_metrics[f'cell{celltype}_ppi_{split}_{name_metric}'] = [metric_value]

    df_metrics = pd.DataFrame(df_metrics)
    return df_metrics



def construct_metapath(metapaths, edge_index, edge_type, num_nodes, verbose=False):
    """ 
    Construct metapath adjacency matrices from edge_index and edge_type.
    Args:
        metapaths (list): List of metapaths, where each metapath is a list of edge types.
            - Example: [[0, 1], [1, 2]] for two metapaths with edge types 0->1 and 1->2.
            This applies for metapaths between tissues and cell types.
            - For proteins, metapaths=ppi_metapaths, with only [4] as edge type
        edge_index (torch.Tensor): Edge index tensor of shape [2, num_edges].
        edge_type (torch.Tensor): Edge type tensor of shape [num_edges].
        num_nodes (int): Number of nodes in the graph.
    Returns:
        mp_adjs (list): List of metapath adjacency matrices, where each matrix is a SparseTensor."""
    unique_edge_types = edge_type.unique()
    if verbose:
        print('unique_edge_types:', unique_edge_types)
        print('metapaths:', metapaths)
    adjs = {}
    for et in unique_edge_types:
        row, col = edge_index[:, edge_type == et]
        adj = torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        adjs[int(et)] = adj

    mp_adjs = []
    for metapath in metapaths:
        if verbose:
            print('metapath:', metapath)
        mp_adj = None
        for idx in metapath:
            if verbose:
                print('idx in metapath:', idx)
            
            if idx not in adjs:
                continue
            if mp_adj:
                if verbose:
                    print('update mp_adj with matrix multiplication')
                mp_adj @= adjs[idx]
            else:
                if verbose:
                    print('replace mp_adj with adjs[idx]')
                mp_adj = adjs[idx]
        if mp_adj:
            mp_adjs.append(mp_adj)

    mp_adjs = [torch.stack([mp_adj.storage.row(), mp_adj.storage.col()]) for mp_adj in mp_adjs]
    return mp_adjs


@torch.no_grad()
def get_embeddings(model, ppi_x, mg_x, ppi_metapaths, mg_metapaths, ppi_edge_index, mg_edge_index, tissue_neighbors):
    model.eval()
    ppi_x, mg_x = model(ppi_x, mg_x, ppi_metapaths, mg_metapaths, ppi_edge_index, mg_edge_index, tissue_neighbors)
    return ppi_x, mg_x 


def plot_emb(best_ppi_x, best_mg_x, celltype_map, ppi_layers, metagraph, wandb, finetune_labels, plot=False):
    celltype_map = {v: k for k, v in celltype_map.items()}
    embed, labels_df, mg_labels = combine_embed(best_ppi_x, best_mg_x, celltype_map, ppi_layers, metagraph, finetune_labels)

    if plot:
        mapping, embedding = fit_umap(embed, min_dist=0.5)
        labels_df["x"] = embedding[:, 0]
        labels_df["y"] = embedding[:, 1]
        plot_umap(labels_df, wandb, "umap.all")
        if len(best_mg_x) > 0:
            mg_labels["x"] = embedding[0:len(celltype_map), 0]
            mg_labels["y"] = embedding[0:len(celltype_map), 1]
            plot_umap(mg_labels, wandb, "umap.ccibto")
        labels_df.pop("x")
        labels_df.pop("y")
    return labels_df


def combine_embed(ppi_embed, mg_embed, key, ppi_layers, metagraph, finetune_labels):
    labels_df = dict()
    mg_labels = dict()

    if len(mg_embed) > 0:

        # Set metagraph labels
        labels_df["Cell Type"] = ["CCI_" + v if "BTO" not in v else v for k, v in key.items()]
        mg_labels["Cell Type"] = [v if "BTO" not in v else v for k, v in key.items()]
        labels_df["Name"] = ["CCI_" + v if "BTO" not in v else v for k, v in key.items()]

        labels_df["Degree"] = [100] * len(metagraph.nodes) # Artificially increase size
        labels_df["Relative Degree"] = [1] * len(metagraph.nodes)
        mg_labels["Degree"] = [metagraph.degree[n] for n in metagraph.nodes]
        
        pcount_labels = [0] * len(key)
        combined = [mg_embed]

    else: 
        labels_df["Cell Type"] = []
        labels_df["Name"] = []
        labels_df["Degree"] = []
        labels_df["Relative Degree"] = []
        pcount_labels = []
        combined = []

    # Get per protein counts & node degrees
    protein_counts = []
    for cluster, ppi in ppi_layers.items():
        if cluster in key.values(): protein_counts += list(ppi)
    protein_counts = Counter(protein_counts)
    
    # Get labels for PPI
    sanity = dict()
    for celltype, x in ppi_embed.items():
        labels_df["Cell Type"] += [key[celltype]] * x.size(0)
        degrees = [ppi_layers[key[celltype]].degree[n] for n in ppi_layers[key[celltype]].nodes]
        labels_df["Degree"] += degrees
        labels_df["Relative Degree"] += [round(d / max(degrees), 5) for d in degrees]
        labels_df["Name"] += list(ppi_layers[key[celltype]].nodes)
        max_rank = len(ppi_layers[key[celltype]])
        combined.append(x)
        for protein in ppi_layers[key[celltype]]:
            pcount_labels.append(protein_counts[protein])
        sanity[key[celltype]] = torch.mean(x, 0)

    labels_df["Overlap"] = pcount_labels

    # Concatenate
    combined = torch.cat(combined)

    # Sanity check
    combined = torch.cat((combined, torch.stack(list(sanity.values()))))
    labels_df["Cell Type"] += ["Sanity Check %s" % k for k in sanity]
    labels_df["Degree"] += [100] * len(sanity)
    labels_df["Relative Degree"] += [1] * len(sanity)
    labels_df["Name"] += ["Sanity Check %s" % k for k in sanity]
    labels_df["Overlap"] += [0] * len(sanity)
    
    return combined, labels_df, mg_labels


def fit_umap(embed, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=3):
    import umap

    mapping = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_state=random_state).fit(embed)
    embedding = mapping.transform(embed)
    print("UMAP reduced:", embedding.shape)
    return mapping, embedding


def plot_umap(labels, wandb, wb_title, color_category="default", finetune_labels=[]):
    import matplotlib
    import plotly.express as px
    from matplotlib import pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    hover_keys = list(labels.keys())
    df = pd.DataFrame(labels)
    fig = px.scatter(df, x="x", y="y", color="Cell Type", size="Degree", hover_data=hover_keys)
    wandb.log({wb_title: fig})
    plt.close()

    
def calc_cluster_metrics(ppi_x: dict) -> tuple:
    """
    Calculate Calinski-Harabasz score and Davies-Bouldin score of PPI embeddings.
    
    :param ppi_x: PPI node embeddings output from the model.
    
    :return: calinski_harabasz, and davies_bouldin scores.
    """
    X = torch.cat(list(ppi_x.values())).detach().cpu().numpy()
    labels = np.concatenate([[key] * x.shape[0] for key, x in ppi_x.items()])
    
    if len(np.unique(labels))==1:
        return 0, 0

    return calinski_harabasz_score(X, labels), davies_bouldin_score(X, labels)


def check_train_log_file(log_path, max_epoch, marg_epoch=5, checked_last_lines=10000):
    with open(log_path) as f:
        f = f.readlines()
    n_lines = len(f)
    epoch_lines = []
    for i, line in tqdm(enumerate(f[n_lines - checked_last_lines : n_lines]), desc='process train log lines'):
        if 'Epoch' in line:
            epoch_lines.append(line)
    ep_str = epoch_lines[-1].split("\t")[0]
    print('ep_str:', ep_str)
    ep_int = int(ep_str.split(' ')[-1])
    print('ep_int:', ep_int)
    if ep_int >= max_epoch - marg_epoch:
        return True
    else:
        return False


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name, "")
    return default if val == "" else val


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name, "")
    return default if val == "" else int(val)


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name, "")
    return default if val == "" else float(val)


def _fmt_float(val: float) -> str:
    normalized = format(Decimal(str(val)), "f")
    if "." not in normalized:
        return normalized

    whole, frac = normalized.split(".", 1)
    frac = frac.rstrip("0")
    if frac == "":
        frac = "0"
    return f"{whole}{frac}"
