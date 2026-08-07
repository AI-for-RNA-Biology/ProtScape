#!/usr/bin/env python3
"""Evaluate the pretraining checkpoints and generate their plots.

This analysis starts from the processed networks and portable checkpoints. It
scores the held-out PPI edges at the five class-imbalance ratios used in the
paper from one context-ID-seeded negative bank and evaluates CCI link prediction.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from pretraining.checkpoints import load_pinnacle_model, load_protscape_model
from pretraining.data_handler.generate_input import get_metapaths, read_data
from pretraining.inference import _LazyPPIFeatures
from pretraining.models.hierarchical_model import add_virtual_node
from pretraining.train import minibatch_factored_utils as factored_mb
from pretraining.train import minibatch_utils as pinnacle_mb


REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS_FILE = REPO_ROOT / "configs" / "paths.yaml"
K_NEGATIVES = [1, 10, 50, 100, 500]
NEGATIVE_BANK_SIZE = max(K_NEGATIVES)

CORE_MODEL_ORDER = [
    "pinnacle_random",
    "pinnacle_esm2_acm",
    "pinnacle_esm2",
    "gae_att",
    "s2gae_att_k1_uni",
]
LOSS_MODEL_ORDER = [
    "s2gae_att_k1_uni",
    "s2gae_att_k1_l1_do00",
    "s2gae_att_k1_phuber",
]
POOLING_MODEL_ORDER = ["gae_att", "gae_vn", "gae_learnedvn"]
TABLE2_MODEL_ORDER = [
    "pinnacle_random",
    "pinnacle_esm2",
    "pinnacle_esm2_acm",
    "gae_att",
    "gae_att_uni",
    "gae_vn",
    "gae_vn_uni",
    "gae_learnedvn",
    "gae_learnedvn_uni",
    "s2gae_att_k1_no_uni",
    "s2gae_att_k1_uni",
]

MODELS = {
    "pinnacle_random": {
        "checkpoint": "pinnacle_random_checkpoint",
        "name": "PINNACLE random fixed drop02",
        "short_name": "PINNACLE random",
        "kind": "pinnacle",
        "k_values": K_NEGATIVES,
        "encoder_pooling": "GATv2",
        "uniformity": None,
    },
    "pinnacle_esm2_acm": {
        "checkpoint": "pinnacle_esm2_acm_checkpoint",
        "name": "PINNACLE ESM2 ACM H64 drop02",
        "short_name": "PINNACLE ESM ACM",
        "kind": "pinnacle",
        "k_values": K_NEGATIVES,
        "encoder_pooling": "ACM",
        "uniformity": None,
    },
    "pinnacle_esm2": {
        "checkpoint": "pinnacle_esm2_gat_checkpoint",
        "name": "PINNACLE ESM2 fixed drop02",
        "short_name": "PINNACLE ESM2",
        "kind": "pinnacle",
        "k_values": K_NEGATIVES,
        "encoder_pooling": "GATv2",
        "uniformity": None,
    },
    "gae_att": {
        "checkpoint": "protscape_gae_checkpoint",
        "name": "GAE att fixed do06",
        "short_name": "GAE att",
        "kind": "protscape",
        "k_values": K_NEGATIVES,
        "metagraph": True,
        "encoder_pooling": "Attention",
        "uniformity": False,
    },
    "gae_att_uni": {
        "checkpoint": "protscape_gae_uniformity_checkpoint",
        "name": "GAE att fixed do06 uni5e-5",
        "short_name": "GAE att + uniformity",
        "kind": "protscape",
        "k_values": [1],
        "metagraph": True,
        "encoder_pooling": "Attention",
        "uniformity": True,
    },
    "gae_vn": {
        "checkpoint": "protscape_gae_virtual_node_checkpoint",
        "name": "GAE VN fixed do04",
        "short_name": "GAE VN",
        "kind": "protscape",
        "k_values": [1],
        "metagraph": True,
        "encoder_pooling": "VN",
        "uniformity": False,
    },
    "gae_vn_uni": {
        "checkpoint": "protscape_gae_virtual_node_uniformity_checkpoint",
        "name": "GAE VN fixed do04 uni5e-5",
        "short_name": "GAE VN + uniformity",
        "kind": "protscape",
        "k_values": [1],
        "metagraph": True,
        "encoder_pooling": "VN",
        "uniformity": True,
    },
    "gae_learnedvn": {
        "checkpoint": "protscape_gae_learned_virtual_node_checkpoint",
        "name": "GAE learnedVN fixed do06",
        "short_name": "GAE LVN",
        "kind": "protscape",
        "k_values": [1],
        "metagraph": True,
        "encoder_pooling": "LVN",
        "uniformity": False,
    },
    "gae_learnedvn_uni": {
        "checkpoint": "protscape_gae_learned_virtual_node_uniformity_checkpoint",
        "name": "GAE learnedVN fixed do06 uni5e-5",
        "short_name": "GAE LVN + uniformity",
        "kind": "protscape",
        "k_values": [1],
        "metagraph": True,
        "encoder_pooling": "LVN",
        "uniformity": True,
    },
    "s2gae_att_k1_no_uni": {
        "checkpoint": "protscape_no_uniformity_checkpoint",
        "name": "S2GAE att k=1 fixed do04",
        "short_name": "S2GAE k=1",
        "kind": "protscape",
        "k_values": [1],
        "metagraph": True,
        "encoder_pooling": "Attention",
        "uniformity": False,
    },
    "s2gae_att_k1_uni": {
        "checkpoint": "protscape_bce_checkpoint",
        "name": "S2GAE att k=1 fixed do04 uni5e-5",
        "short_name": "S2GAE k=1 uni",
        "kind": "protscape",
        "k_values": K_NEGATIVES,
        "metagraph": True,
        "encoder_pooling": "Attention",
        "uniformity": True,
    },
    "s2gae_att_k1_phuber": {
        "checkpoint": "protscape_phuber_checkpoint",
        "name": "S2GAE att k=1 fixed do04 pHuber uni",
        "short_name": "S2GAE pHuber",
        "kind": "protscape",
        "k_values": K_NEGATIVES,
    },
    "s2gae_att_k1_l1_do00": {
        "checkpoint": "protscape_l1_checkpoint",
        "name": "S2GAE att k=1 fixed do00 L1 uni",
        "short_name": "S2GAE L1",
        "kind": "protscape",
        "k_values": K_NEGATIVES,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_paths() -> dict:
    with PATHS_FILE.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing released checkpoint: {path}")
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def load_dataset(checkpoint: dict, paths: dict, *, defer_features: bool):
    config = checkpoint["config"]
    seed = int(config.get("seed", 0))
    set_seed(seed)

    networks = Path(paths["networks_bulk"]).expanduser()
    features_mode = config["features_mode"]
    if features_mode == "ESM2":
        feature_path = Path(paths["esm2_embeddings"]).expanduser()
        feature_dim = None
    elif features_mode == "random":
        feature_path = None
        feature_dim = int(config["input_dim"])
    else:
        raise ValueError(f"Unsupported protein features: {features_mode}")

    data = read_data(
        networks / "global_ppi_edgelist.txt",
        networks / "ppi_edgelists",
        networks / "mg_edgelist.txt",
        feat_mat_dim=feature_dim,
        get_CT_map=True,
        ppi_feat_dir=feature_path,
        symmetric_ppi=config.get("symmetric_ppi", True),
        dataset_mode=config["dataset_mode"],
        split_mode=config["split_mode"],
        count_edge_path=networks / "count_edge_dict.pkl",
        weighted_ppi_loss=config.get("weighted_ppi_loss", False),
        defer_ppi_features=defer_features,
        verbose=False,
    )

    ppi_data, _, _, celltype_map, _, _, _, _ = data
    cell_ids = [int(cell_id) for cell_id in checkpoint["cell_ids"]]
    if set(cell_ids) != set(ppi_data):
        raise ValueError("Checkpoint cell IDs do not match the released PPI contexts.")
    ppi_data = {cell_id: ppi_data[cell_id] for cell_id in cell_ids}
    data = (ppi_data, *data[1:])

    id_to_name = {cell_id: name for name, cell_id in celltype_map.items()}
    cell_names = [id_to_name[cell_id] for cell_id in cell_ids]
    if checkpoint["cell_names"] != cell_names:
        raise ValueError("Checkpoint cell names do not match the released PPI contexts.")
    return data, id_to_name


def binary_metrics(scores, labels) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    roc = 0.5 if np.unique(labels).size < 2 else roc_auc_score(labels, scores)
    return {
        "roc": float(roc),
        "ap": float(average_precision_score(labels, scores)),
        "acc": float(accuracy_score(labels, scores >= 0.5)),
        "f1": float(
            f1_score(
                labels,
                scores >= 0.5,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        ),
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
    }


def metrics_from_pos_neg(pos_scores, neg_scores) -> dict[str, float]:
    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate(
        [np.ones(pos_scores.size, dtype=np.int8), np.zeros(neg_scores.size, dtype=np.int8)]
    )
    return binary_metrics(scores, labels)


def sample_structured_negatives(pos_edges, all_positive_edges, num_nodes, max_k):
    """Sample ``max_k`` non-edges for each positive edge, as in paper evaluation."""
    device = pos_edges.device
    sources = pos_edges[0].detach().cpu().numpy().astype(np.int64, copy=False)
    positives = all_positive_edges.detach().cpu().numpy().astype(np.int64, copy=False)

    known_targets = {}
    for left, right in zip(positives[0], positives[1]):
        known_targets.setdefault(int(left), []).append(int(right))
        known_targets.setdefault(int(right), []).append(int(left))

    source_rows = {}
    for row, source in enumerate(sources):
        source_rows.setdefault(int(source), []).append(row)

    negative_targets = np.empty((sources.size, max_k), dtype=np.int64)
    for source, rows in source_rows.items():
        rows = np.asarray(rows, dtype=np.int64)
        forbidden = np.asarray(known_targets.get(source, ()), dtype=np.int64)
        needed = rows.size * max_k
        draws = []
        collected = 0
        while collected < needed:
            candidates = np.random.randint(0, num_nodes, size=max(4096, 2 * (needed - collected)))
            keep = candidates != source
            if forbidden.size:
                keep &= ~np.isin(candidates, forbidden)
            candidates = candidates[keep]
            take = min(candidates.size, needed - collected)
            draws.append(candidates[:take])
            collected += take
        negative_targets[rows] = np.concatenate(draws).reshape(rows.size, max_k)

    negative_edges = np.stack(
        [np.repeat(sources, max_k), negative_targets.reshape(-1)], axis=0
    )
    return torch.from_numpy(negative_edges).to(device=device, dtype=torch.long)


def canonical_undirected_edges(edge_index):
    edge_index = edge_index.detach().cpu().long()
    pairs = torch.stack(
        [torch.minimum(edge_index[0], edge_index[1]),
         torch.maximum(edge_index[0], edge_index[1])],
        dim=1,
    )
    return torch.unique(pairs, dim=0).t().contiguous()


def shared_cci_test_edges(mg_data, edge_types, cell_ids):
    """Return the train graph and balanced test targets used in Methods/Table 2."""
    all_positive = canonical_undirected_edges(
        factored_mb.build_cci_edge_index(
            mg_data, edge_types, cell_ids=cell_ids, device="cpu"
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(0)
    order = torch.randperm(all_positive.size(1), generator=generator)
    n_train = int(0.8 * all_positive.size(1))
    n_val = int(0.1 * all_positive.size(1))
    train_positive = all_positive[:, order[:n_train]]
    test_positive = all_positive[:, order[n_train + n_val :]]

    cell_ids = sorted(int(cell_id) for cell_id in cell_ids)
    known = {tuple(pair) for pair in all_positive.t().tolist()}
    candidates = {
        source: [
            target
            for target in cell_ids
            if target != source
            and (min(source, target), max(source, target)) not in known
        ]
        for source in cell_ids
    }
    generator.manual_seed(3)
    candidate_order = {
        source: [
            values[index]
            for index in torch.randperm(len(values), generator=generator).tolist()
        ]
        for source, values in candidates.items()
    }
    cursor = {source: 0 for source in cell_ids}
    used = set()

    def draw(source):
        values = candidate_order[source]
        while cursor[source] < len(values):
            target = values[cursor[source]]
            cursor[source] += 1
            pair = (min(source, target), max(source, target))
            if pair not in used:
                used.add(pair)
                return source, target
        return None

    test_negative = []
    for source, target in test_positive.t().tolist():
        negative = draw(int(source)) or draw(int(target))
        if negative is None:
            raise ValueError(f"No structured CCI negative for ({source}, {target}).")
        test_negative.append(negative)
    test_negative = torch.tensor(test_negative, dtype=torch.long).t().contiguous()

    train_edges = torch.unique(
        torch.cat([train_positive, train_positive.flip(0)], dim=1), dim=1
    )
    test_edges = torch.cat([test_positive, test_negative], dim=1)
    test_labels = torch.cat(
        [torch.ones(test_positive.size(1)), torch.zeros(test_negative.size(1))]
    )
    return train_edges, test_edges, test_labels


def dot_scores(embeddings, edges, relation=None, batch_size=100_000):
    scores = []
    for start in range(0, edges.size(1), batch_size):
        edge_batch = edges[:, start : start + batch_size].to(embeddings.device)
        source = embeddings[edge_batch[0]].float()
        target = embeddings[edge_batch[1]].float()
        if relation is not None:
            rel = relation[start : start + batch_size].to(embeddings.device).float()
            logits = (source * rel * target).sum(dim=1)
        else:
            logits = (source * target).sum(dim=1)
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def s2gae_scores(decoder, decoder_input, edges, batch_size=100_000):
    scores = []
    for start in range(0, edges.size(1), batch_size):
        edge_batch = edges[:, start : start + batch_size]
        logits = factored_mb.undirected_decoder_logits(
            decoder, decoder_input, edge_batch
        )
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def context_records(model_key, cell_id, cell_name, metrics) -> list[dict]:
    spec = MODELS[model_key]
    rows = []
    for metric in ["ap", "f1"]:
        rows.append(
            {
                "model_key": model_key,
                "model": spec["name"],
                "model_short": spec["short_name"],
                "k_negatives": 1,
                "pos_neg_ratio": "1:1",
                "cell_index": int(cell_id),
                "edgelist": cell_name,
                "metric": metric,
                "score": 100.0 * metrics[metric],
                "n_pos": metrics["n_pos"],
                "n_neg": metrics["n_neg"],
                "message_passing_edges": "train+val",
                "negative_sampling": "structured_target_corruption",
                "negative_seed": int(cell_id),
                "negative_bank_size": NEGATIVE_BANK_SIZE,
            }
        )
    return rows


def aggregate_context_metrics(metrics_by_k: dict[int, list[dict]]) -> dict[int, dict]:
    output = {}
    for k, rows in metrics_by_k.items():
        output[k] = {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ["roc", "ap", "acc", "f1"]
        }
        output[k]["n_pos"] = int(sum(row["n_pos"] for row in rows))
        output[k]["n_neg"] = int(sum(row["n_neg"] for row in rows))
        output[k]["n_cells"] = len(rows)
    return output


def protein_graph(model, graph, cell_id, device):
    """Build one train+validation message-passing graph for ProtScape."""
    local = graph.clone()
    # Keep the train-then-validation ordering used by the submitted evaluator.
    local.edge_index = torch.cat(
        [graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.val_mask]],
        dim=1,
    )
    local.edge_attr = torch.cat(
        [graph.edge_attr[graph.train_mask], graph.edge_attr[graph.val_mask]],
        dim=0,
    )
    local = local.to(device)

    pooling = model.cell_config["pooling"]
    if pooling == "vn" or model.protein_config.get("add_virtual_node"):
        local = add_virtual_node(local, model.virtual_node_features, clone=False)
    elif pooling == "learnedvn":
        vn_id = model.celltype_to_vn_id[cell_id]
        local = add_virtual_node(local, model.virtual_node_features[vn_id], clone=False)
    return local


@torch.no_grad()
def evaluate_protscape_ppi(model_key, model, data, id_to_name, device):
    spec = MODELS[model_key]
    ppi_data, _, _, _, _, _, _, _ = data
    k_values = spec["k_values"]
    max_k = max(k_values)
    metrics_by_k = {k: [] for k in k_values}
    contextwise = []
    pooled_cells = []

    set_seed(0)
    model.eval()
    for index, (cell_id, graph) in enumerate(ppi_data.items(), start=1):
        local = protein_graph(model, graph, cell_id, device)
        if model.s2gae_enabled:
            embeddings, layers = model.prot_encoder(
                {cell_id: local}, batching=False, return_layer_outputs=True
            )
            embeddings = embeddings[cell_id]
            layers = layers[cell_id]
        else:
            embeddings = model.prot_encoder(
                {cell_id: local}, batching=False, return_layer_outputs=False
            )[cell_id]
            layers = None

        pooling = model.cell_config["pooling"]
        uses_virtual_node = (
            pooling in {"vn", "learnedvn"}
            or model.protein_config.get("add_virtual_node")
        )
        pooling_embeddings = (
            embeddings[:-1]
            if uses_virtual_node and pooling not in {"vn", "learnedvn"}
            else embeddings
        )
        pooled_cells.append(
            model.cell_encoder.get_cell_embedding_per_celltype(pooling_embeddings).cpu()
        )
        if uses_virtual_node:
            embeddings = embeddings[:-1]
            if layers is not None:
                layers = [layer[:-1] for layer in layers]

        pos_edges_cpu = graph.edge_index[:, graph.test_mask]
        np.random.seed(int(cell_id))
        negative_bank = sample_structured_negatives(
            pos_edges_cpu,
            graph.edge_index,
            int(graph.num_nodes),
            NEGATIVE_BANK_SIZE,
        ).reshape(2, pos_edges_cpu.size(1), NEGATIVE_BANK_SIZE)
        pos_edges = pos_edges_cpu.to(device)
        neg_edges = negative_bank[:, :, :max_k].reshape(2, -1).to(device)
        eval_edges = torch.cat([pos_edges, neg_edges], dim=1)

        if model.s2gae_enabled:
            decoder_input = factored_mb._prepare_s2gae_decoder_input(
                model.s2gae_decoder,
                layers,
                raw_embeddings=local.x[: graph.num_nodes],
            )
            scores = s2gae_scores(model.s2gae_decoder, decoder_input, eval_edges)
            del decoder_input
        else:
            scores = dot_scores(embeddings, eval_edges)

        n_pos = pos_edges.size(1)
        pos_scores = scores[:n_pos]
        neg_scores = scores[n_pos:].reshape(n_pos, max_k)
        for k in k_values:
            metrics = metrics_from_pos_neg(pos_scores, neg_scores[:, :k].reshape(-1))
            metrics_by_k[k].append(metrics)
            if k == 1 and model_key in CORE_MODEL_ORDER:
                contextwise.extend(
                    context_records(model_key, cell_id, id_to_name[cell_id], metrics)
                )

        if index % 25 == 0 or index == len(ppi_data):
            print(f"  {model_key}: scored {index}/{len(ppi_data)} PPI contexts", flush=True)
        del local, embeddings, pooling_embeddings, layers, pos_edges_cpu, pos_edges, negative_bank, neg_edges, eval_edges

    return aggregate_context_metrics(metrics_by_k), contextwise, torch.stack(pooled_cells)


@torch.no_grad()
def evaluate_protscape_metagraph(model, data, pooled_cells, device):
    """Evaluate held-out CCI edges from the cell vectors pooled during PPI scoring."""
    ppi_data, mg_data, edge_types, _, _, _, _, _ = data
    cell_ids = list(ppi_data)
    train_edges, test_edges, test_labels = shared_cci_test_edges(
        mg_data, edge_types, cell_ids
    )
    model.set_cci_graph(train_edges.to(device), mg_data.num_nodes)

    cells = pooled_cells.to(device)
    memory = model.cell_encoder.cell_memory_layers
    if memory is not None:
        cells = memory(cells)
    cells = model.apply_cci(cells, cell_ids)
    logits = model.predict_cci_edges(
        cells,
        cell_ids,
        test_edges.to(device),
        assume_cci_encoded=True,
    )
    return binary_metrics(torch.sigmoid(logits).cpu().numpy(), test_labels.numpy())


def pinnacle_eval_dtype(device):
    return torch.float16 if device.type == "cuda" else torch.float32


@torch.no_grad()
def evaluate_pinnacle(model_key, model, data, id_to_name, device):
    spec = MODELS[model_key]
    ppi_data, mg_data, edge_types, _, tissue_neighbors, _, _, _ = data
    k_values = spec["k_values"]
    max_k = max(k_values)
    dtype = pinnacle_eval_dtype(device)
    model.to(device=device, dtype=dtype).eval()
    set_seed(0)

    ppi_metapaths, mg_metapaths = get_metapaths()
    _, metagraph_test, mg_adjacencies, mg_features = pinnacle_mb.generate_batch(
        {0: mg_data},
        mg_metapaths,
        edge_types,
        "test",
        int(model.output),
        device,
        ppi=False,
        loader_type="graphsaint",
    )
    metagraph_test = metagraph_test[0]
    mg_x = mg_features[0].to(dtype=dtype)
    mg_adjacencies = [adj.to(device) for adj in mg_adjacencies[0]]

    # PINNACLE is evaluated in half precision on GPU to fit the ESM2 model.
    # Cast the lazily loaded protein features as well as the model and metagraph.
    global_features = mg_data.global_protein_features.to(dtype=dtype)
    ppi_x = _LazyPPIFeatures(ppi_data, global_features, device)
    ppi_adjacencies = {}
    for cell_id, graph in ppi_data.items():
        train_val = graph.train_mask | graph.val_mask
        ppi_adjacencies[cell_id] = [
            adjacency.to(device)
            for adjacency in pinnacle_mb.construct_metapath(
                ppi_metapaths,
                graph.edge_index[:, train_val],
                graph.edge_attr[train_val],
                int(graph.num_nodes),
            )
        ]

    protein_embeddings, metagraph_embeddings = model(
        ppi_x,
        mg_x,
        ppi_adjacencies,
        mg_adjacencies,
        {},
        metagraph_test["total_edge_index"],
        tissue_neighbors,
    )

    relations = model.mg_relw[metagraph_test["total_edge_type"]]
    metagraph_scores = dot_scores(
        metagraph_embeddings,
        metagraph_test["total_edge_index"],
        relation=relations,
    )
    metagraph_labels = metagraph_test["y"].cpu().numpy()
    metagraph_metrics = binary_metrics(metagraph_scores, metagraph_labels)
    cci_mask = (
        metagraph_test["total_edge_type"].detach().cpu().numpy()
        == edge_types["cell_cell"]
    )
    cci_metrics = binary_metrics(
        metagraph_scores[cci_mask], metagraph_labels[cci_mask]
    )

    metrics_by_k = {k: [] for k in k_values}
    contextwise = []
    for index, (cell_id, embeddings) in enumerate(protein_embeddings.items(), start=1):
        graph = ppi_data[cell_id]
        pos_edges_cpu = graph.edge_index[:, graph.test_mask]
        np.random.seed(int(cell_id))
        negative_bank = sample_structured_negatives(
            pos_edges_cpu,
            graph.edge_index,
            int(graph.num_nodes),
            NEGATIVE_BANK_SIZE,
        ).reshape(2, pos_edges_cpu.size(1), NEGATIVE_BANK_SIZE)
        pos_edges = pos_edges_cpu.to(device)
        neg_edges = negative_bank[:, :, :max_k].reshape(2, -1).to(device)
        pos_scores = dot_scores(embeddings, pos_edges)
        neg_scores = dot_scores(embeddings, neg_edges).reshape(pos_edges.size(1), max_k)
        for k in k_values:
            metrics = metrics_from_pos_neg(pos_scores, neg_scores[:, :k].reshape(-1))
            metrics_by_k[k].append(metrics)
            if k == 1:
                contextwise.extend(
                    context_records(model_key, cell_id, id_to_name[cell_id], metrics)
                )
        if index % 25 == 0 or index == len(ppi_data):
            print(f"  {model_key}: scored {index}/{len(ppi_data)} PPI contexts", flush=True)
        del pos_edges_cpu, pos_edges, negative_bank, neg_edges

    return (
        aggregate_context_metrics(metrics_by_k),
        metagraph_metrics,
        cci_metrics,
        contextwise,
    )


def add_context_metadata(table: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path)
    frequent = mapping["cell_type_class"].value_counts()
    frequent = set(frequent[frequent > 10].index)
    mapping = mapping.copy()
    mapping["plot_cell_type_class"] = mapping["cell_type_class"].where(
        mapping["cell_type_class"].isin(frequent), "Other"
    )
    columns = [
        "edgelist",
        "cl_id",
        "canonical_name",
        "cell_type_class",
        "plot_cell_type_class",
        "primary_dataset",
    ]
    return table.merge(mapping[columns], on="edgelist", how="left")


def robust_rows(model_key, metrics_by_k) -> list[dict]:
    rows = []
    for k in sorted(metrics_by_k):
        metrics = metrics_by_k[k]
        rows.append(
            {
                "model_key": model_key,
                "k_negatives": k,
                "pos_neg_ratio": f"1:{k}",
                "test_roc_ppi": metrics["roc"],
                "test_ap_ppi": metrics["ap"],
                "test_acc_ppi": metrics["acc"],
                "test_f1_ppi": metrics["f1"],
                "n_cells": metrics["n_cells"],
                "n_pos": metrics["n_pos"],
                "n_neg": metrics["n_neg"],
            }
        )
    return rows


def curve_table(robust: pd.DataFrame, order, metric, *, chance=False) -> pd.DataFrame:
    table = robust[robust["model_key"].isin(order)][
        ["model_key", "k_negatives", "pos_neg_ratio", metric]
    ].copy()
    table["model_key"] = pd.Categorical(table["model_key"], order, ordered=True)
    table = table.sort_values(["model_key", "k_negatives"]).reset_index(drop=True)
    table["model_key"] = table["model_key"].astype(object)
    table["score_percent"] = 100.0 * table[metric]
    if chance:
        baseline = pd.DataFrame(
            {
                "model_key": "chance_auprc",
                "k_negatives": K_NEGATIVES,
                "pos_neg_ratio": [f"1:{k}" for k in K_NEGATIVES],
                metric: np.nan,
                "score_percent": [100.0 / (1.0 + k) for k in K_NEGATIVES],
            }
        )
        table = pd.concat([table, baseline], ignore_index=True)
    return table


def full_metrics_table(robust: pd.DataFrame, cci: dict) -> pd.DataFrame:
    """Build the complete balanced PPI/CCI table reported in the manuscript."""
    by_model = robust.set_index(["model_key", "k_negatives"])
    rows = []
    for key in TABLE2_MODEL_ORDER:
        ppi = by_model.loc[(key, 1)]
        cci_metrics = cci[key]
        is_pinnacle = MODELS[key]["kind"] == "pinnacle"
        rows.append(
            {
                "model_key": key,
                "model": MODELS[key]["short_name"],
                "encoder_pooling": MODELS[key]["encoder_pooling"],
                "uniformity_enabled": MODELS[key]["uniformity"],
                "ppi_auprc": ppi["test_ap_ppi"],
                "ppi_macro_f1": ppi["test_f1_ppi"],
                "ppi_accuracy": ppi["test_acc_ppi"],
                "ppi_auroc": ppi["test_roc_ppi"],
                "cci_auprc": cci_metrics["ap"],
                "cci_macro_f1": cci_metrics["f1"],
                "cci_accuracy": cci_metrics["acc"],
                "cci_auroc": cci_metrics["roc"],
                "ppi_n_contexts": ppi["n_cells"],
                "ppi_n_pos": ppi["n_pos"],
                "ppi_n_neg": ppi["n_neg"],
                "cci_n_pos": cci_metrics["n_pos"],
                "cci_n_neg": cci_metrics["n_neg"],
                "encoder_dtype": "float16" if is_pinnacle else "float32",
                "evaluation_device": "CUDA",
                "ppi_protocol": "held_out_per_context_seeded_500_bank_macro_across_contexts",
                "ppi_negative_sampling": "structured_target_corruption",
                "ppi_negative_seed": "stable_cell_id",
                "ppi_negative_bank_size": NEGATIVE_BANK_SIZE,
                "cci_protocol": (
                    "in_sample_reconstruction"
                    if is_pinnacle
                    else "held_out_1to1_train_only_message_passing"
                ),
                "cci_scope": "cell_cell",
            }
        )
    return pd.DataFrame(rows)


def build_output_tables(robust, metagraph, cci, parameters, contextwise):
    robust = pd.DataFrame(robust)
    by_model = robust.set_index(["model_key", "k_negatives"])

    metagraph_rows = []
    for key in CORE_MODEL_ORDER:
        ppi_score = by_model.loc[(key, 1), "test_ap_ppi"]
        meta_score = metagraph[key]["ap"]
        is_pinnacle = MODELS[key]["kind"] == "pinnacle"
        metagraph_rows.append(
            {
                "model_key": key,
                "metric": "ap",
                "protein_score": ppi_score,
                "metagraph_score": meta_score,
                "protein_percent": 100.0 * ppi_score,
                "metagraph_percent": 100.0 * meta_score,
                "higher_level_scope": "full_metagraph" if is_pinnacle else "cell_cell",
                "higher_level_protocol": (
                    "in_sample_reconstruction"
                    if is_pinnacle
                    else "held_out_1to1_train_only_message_passing"
                ),
            }
        )

    pooling_rows = []
    for key in POOLING_MODEL_ORDER:
        for metric, label in [("ap", "AUPRC"), ("f1", "F1")]:
            pooling_rows.extend(
                [
                    {
                        "model_key": key,
                        "metric": f"PPI - {label}",
                        "score": 100.0 * by_model.loc[(key, 1), f"test_{metric}_ppi"],
                        "higher_level_scope": "not_applicable",
                        "higher_level_protocol": "not_applicable",
                    },
                    {
                        "model_key": key,
                        "metric": f"Metagraph - {label}",
                        "score": 100.0 * metagraph[key][metric],
                        "higher_level_scope": "cell_cell",
                        "higher_level_protocol": "held_out_1to1_train_only_message_passing",
                    },
                ]
            )

    contextwise = pd.DataFrame(contextwise)
    outputs = {
        "robust_ppi_auprc.csv": curve_table(
            robust, CORE_MODEL_ORDER, "test_ap_ppi", chance=True
        ),
        "robust_ppi_f1.csv": curve_table(
            robust, CORE_MODEL_ORDER, "test_f1_ppi"
        ),
        "loss_sensitivity_auprc.csv": curve_table(
            robust, LOSS_MODEL_ORDER, "test_ap_ppi", chance=True
        ),
        "loss_sensitivity_f1.csv": curve_table(
            robust, LOSS_MODEL_ORDER, "test_f1_ppi"
        ),
        "metagraph_auprc.csv": pd.DataFrame(metagraph_rows),
        "parameter_counts.csv": pd.DataFrame(
            [
                {
                    "model_key": key,
                    "model": MODELS[key]["name"],
                    "parameter_count": parameters[key],
                }
                for key in CORE_MODEL_ORDER
            ]
        ),
        "contextwise_ppi_auprc.csv": contextwise[contextwise["metric"] == "ap"],
        "contextwise_ppi_f1.csv": contextwise[contextwise["metric"] == "f1"],
        "pooling_sensitivity.csv": pd.DataFrame(pooling_rows),
        "pretraining_full_metrics.csv": full_metrics_table(robust, cci),
    }
    return outputs


def main() -> None:
    paths = load_paths()
    output_dir = Path(paths["output_root"]) / "analysis" / "pretraining_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Exact pretraining evaluation requires CUDA; PINNACLE is evaluated "
            "in the submitted FP16 GPU protocol."
        )
    device = torch.device("cuda")

    robust = []
    metagraph = {}
    cci = {}
    parameters = {}
    contextwise = []

    # All ProtScape variants use the same ESM2 input data.
    first_checkpoint = load_checkpoint(Path(paths[MODELS["gae_att"]["checkpoint"]]))
    factored_data, id_to_name = load_dataset(first_checkpoint, paths, defer_features=False)
    for model_key in [
        "gae_att",
        "gae_att_uni",
        "gae_vn",
        "gae_vn_uni",
        "gae_learnedvn",
        "gae_learnedvn_uni",
        "s2gae_att_k1_no_uni",
        "s2gae_att_k1_uni",
        "s2gae_att_k1_phuber",
        "s2gae_att_k1_l1_do00",
    ]:
        print(f"\nEvaluating {MODELS[model_key]['name']}", flush=True)
        checkpoint = (
            first_checkpoint
            if model_key == "gae_att"
            else load_checkpoint(Path(paths[MODELS[model_key]["checkpoint"]]))
        )
        cell_ids = [int(cell_id) for cell_id in checkpoint["cell_ids"]]
        model_data = (
            {cell_id: factored_data[0][cell_id] for cell_id in cell_ids},
            *factored_data[1:],
        )
        expected_names = [id_to_name[cell_id] for cell_id in cell_ids]
        if checkpoint["cell_names"] != expected_names:
            raise ValueError(f"Cell names do not match for {model_key}.")
        model = load_protscape_model(checkpoint, model_data[0], device=device)
        parameters[model_key] = sum(parameter.numel() for parameter in model.parameters())
        ppi_metrics, local_context, pooled_cells = evaluate_protscape_ppi(
            model_key, model, model_data, id_to_name, device
        )
        robust.extend(robust_rows(model_key, ppi_metrics))
        contextwise.extend(local_context)
        if MODELS[model_key].get("metagraph"):
            cci[model_key] = evaluate_protscape_metagraph(
                model, model_data, pooled_cells, device
            )
            metagraph[model_key] = cci[model_key]
            print(f"  {model_key} CCI metrics: {cci[model_key]}", flush=True)
        del checkpoint, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del first_checkpoint, factored_data

    # Adapted PINNACLE random inputs are generated with the checkpoint seed.
    for model_key in ["pinnacle_random", "pinnacle_esm2_acm", "pinnacle_esm2"]:
        print(f"\nEvaluating {MODELS[model_key]['name']}", flush=True)
        checkpoint = load_checkpoint(Path(paths[MODELS[model_key]["checkpoint"]]))
        data, id_to_name = load_dataset(checkpoint, paths, defer_features=True)
        model = load_pinnacle_model(checkpoint, data[0], device="cpu")
        parameters[model_key] = sum(parameter.numel() for parameter in model.parameters())
        ppi_metrics, metagraph_metrics, cci_metrics, local_context = evaluate_pinnacle(
            model_key, model, data, id_to_name, device
        )
        robust.extend(robust_rows(model_key, ppi_metrics))
        metagraph[model_key] = metagraph_metrics
        cci[model_key] = cci_metrics
        print(f"  {model_key} metagraph metrics: {metagraph_metrics}", flush=True)
        print(f"  {model_key} cell-cell metrics: {cci_metrics}", flush=True)
        contextwise.extend(local_context)
        del checkpoint, data, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    contextwise = add_context_metadata(
        pd.DataFrame(contextwise), Path(paths["celltype_class_mapping"])
    )
    outputs = build_output_tables(robust, metagraph, cci, parameters, contextwise)
    for filename, table in outputs.items():
        path = output_dir / filename
        table.to_csv(path, index=False)
        print(f"Saved {path}", flush=True)

    from exploration.plotting.model_evaluation_plots import (
        plot_pretraining as plot_model_evaluation,
    )
    from exploration.plotting.model_diagnostic_plots import (
        plot_pretraining as plot_model_diagnostics,
    )

    plot_model_evaluation(output_dir, output_dir)
    plot_model_diagnostics(output_dir, output_dir)


if __name__ == "__main__":
    main()
