import random
from decimal import Decimal
import numpy as np
import pandas as pd
import torch
import torch_sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    roc_auc_score,
)
import omegaconf
import wandb

from torchmetrics import AUROC, AveragePrecision, Accuracy


def set_seed(seed):
    """Seed Python, NumPy and torch before splitting data or initializing a model."""
    print("SEED:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_wandb(cfg, experiment_name):
    """Initialize the optional run log from the training configuration."""
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
    """Return AUROC, average precision, accuracy and macro F1 for binary predictions."""
    try:
        roc_score = roc_auc_score(y, pred)
    except ValueError:
        roc_score = 0.5
    ap_score = average_precision_score(y, pred)
    acc = accuracy_score(y, pred > 0.5)
    f1 = f1_score(y, pred > 0.5, average='macro', labels=[0, 1], zero_division=0)
    return roc_score, ap_score, acc, f1


def calc_individual_metrics_torch(pred, y):
    """Compute the same four binary metrics directly from tensors."""
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
        """Return the class F1 score, using zero when the class has no predicted or true positives."""
        denominator = 2 * true_positive + false_positive + false_negative
        return torch.where(
            denominator > 0,
            2 * true_positive / denominator,
            torch.zeros_like(denominator),
        )

    f1 = torch.mean(torch.stack([class_f1(tp, fp, fn), class_f1(tn, fn, fp)])).item()
    return roc_score, ap_score, acc, f1


def calc_metrics_factored(mg_pred, mg_data, ppi_preds, ppi_labels):

    """Average PPI metrics across cells and score CCI predictions when provided."""
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
    """Average per-cell PPI metrics and score metagraph edges using their stored labels."""
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

    """Log per-cell PPI and CCI metrics and return a one-row results table."""
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

    """Log metrics for each cell/relation and return a one-row results table."""
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
    """Compose relation adjacency matrices and return one edge-index tensor per metapath."""
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
    """Run PINNACLE in evaluation mode and return protein and metagraph embeddings."""
    model.eval()
    ppi_x, mg_x = model(ppi_x, mg_x, ppi_metapaths, mg_metapaths, ppi_edge_index, mg_edge_index, tissue_neighbors)
    return ppi_x, mg_x


def calc_cluster_metrics(ppi_x: dict) -> tuple:
    """Measure separation of cell-specific protein embeddings with CH and DB scores."""
    X = torch.cat(list(ppi_x.values())).detach().cpu().numpy()
    labels = np.concatenate([[key] * x.shape[0] for key, x in ppi_x.items()])

    if len(np.unique(labels))==1:
        return 0, 0

    return calinski_harabasz_score(X, labels), davies_bouldin_score(X, labels)


def format_float(val: float) -> str:
    """Format a hyperparameter for existing run names, e.g. 0.5 becomes 05."""
    normalized = format(Decimal(str(val)), "f")
    if "." not in normalized:
        return normalized

    whole, frac = normalized.split(".", 1)
    frac = frac.rstrip("0")
    if frac == "":
        frac = "0"
    return f"{whole}{frac}"
