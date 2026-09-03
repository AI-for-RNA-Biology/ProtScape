"""Evaluate PPI, metagraph and cell-cell link prediction checkpoints."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from pretraining.contextwise_ppi import (
    aggregate_context_metrics,
    binary_metrics,
    metrics_from_pos_neg,
    sample_structured_negatives,
)
from pretraining.data_handler.generate_input import get_metapaths, read_data
from pretraining.inference import _LazyPPIFeatures
from pretraining.models.hierarchical_model import add_virtual_node
from pretraining.s2gae_utils import undirected_decoder_logits
from pretraining.train.factored_batching import build_cci_edge_index
from pretraining.train.factored_training import _prepare_s2gae_decoder_input
from pretraining.train import minibatch_utils as pinnacle_mb
from exploration.analysis.pretraining_model_specs import (
    CORE_MODEL_ORDER,
    MODELS,
    NEGATIVE_BANK_SIZE,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def canonical_undirected_edges(edge_index):
    edge_index = edge_index.detach().cpu().long()
    pairs = torch.stack(
        [torch.minimum(edge_index[0], edge_index[1]),
         torch.maximum(edge_index[0], edge_index[1])],
        dim=1,
    )
    return torch.unique(pairs, dim=0).t().contiguous()


def shared_cci_test_edges(mg_data, edge_types, cell_ids):
    """Return the train graph and balanced CCI test targets."""
    all_positive = canonical_undirected_edges(
        build_cci_edge_index(
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
        logits = undirected_decoder_logits(
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


def protein_graph(model, graph, cell_id, device):
    """Build one train+validation message-passing graph for ProtScape."""
    local = graph.clone()
    # Keep the train-then-validation ordering used by the evaluator.
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
            decoder_input = _prepare_s2gae_decoder_input(
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
