"""Load ProtScape models and score context-specific protein pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from downstream_tasks.config import PATHS
from pretraining.checkpoints import load_protscape_model
from pretraining.data_handler.generate_input import read_data
from pretraining.s2gae_utils import undirected_decoder_logits
from pretraining.train.factored_training import _prepare_s2gae_decoder_input


LOSSES = ("BCE", "pHuber", "L1")
CHECKPOINT_KEYS = {
    "BCE": "protscape_bce_checkpoint",
    "pHuber": "protscape_phuber_checkpoint",
    "L1": "protscape_l1_checkpoint",
}
LABEL_NAMES = {1: "Labelled positive", 0: "Labelled negative"}
CLASS_NAMES = (
    "Consensus negative",
    "Weak disagreement negative",
    "Strong disagreement negative",
    "Strong disagreement positive",
    "Weak disagreement positive",
    "Consensus positive",
)
SCORE_CUTOFF = 0.5
EDGE_CHUNK = 100_000


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing exhaustive-inference input: {path}")
    return path


def clean_gene(value: object) -> str:
    return str(value).strip().upper()


def pair_ids(src: np.ndarray, dst: np.ndarray, n_genes: int) -> np.ndarray:
    lo = np.minimum(src, dst).astype(np.int64, copy=False)
    hi = np.maximum(src, dst).astype(np.int64, copy=False)
    return lo * n_genes - (lo * (lo + 1)) // 2 + (hi - lo - 1)


def load_models_and_data(device: torch.device):
    paths = PATHS
    checkpoints = {
        loss: torch.load(
            require_file(Path(paths[key]).expanduser()),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        for loss, key in CHECKPOINT_KEYS.items()
    }
    reference = checkpoints["BCE"]
    config = reference["config"]
    networks = Path(paths["networks_bulk"]).expanduser()
    features = Path(paths["esm2_embeddings"]).expanduser()
    ppi_data, _, _, celltype_map, _, ppi_layers, _, _ = read_data(
        networks / "global_ppi_edgelist.txt",
        networks / "ppi_edgelists",
        networks / "mg_edgelist.txt",
        feat_mat_dim=None,
        get_CT_map=True,
        ppi_feat_dir=features,
        symmetric_ppi=config.get("symmetric_ppi", True),
        dataset_mode=config["dataset_mode"],
        split_mode=config["split_mode"],
        count_edge_path=networks / "count_edge_dict.pkl",
        weighted_ppi_loss=config.get("weighted_ppi_loss", False),
    )

    cell_ids = [int(value) for value in reference["cell_ids"]]
    if set(cell_ids) != set(ppi_data):
        raise ValueError("Released checkpoint cells do not match the processed networks")
    ppi_data = {cell_id: ppi_data[cell_id] for cell_id in cell_ids}
    id_to_name = {cell_id: name for name, cell_id in celltype_map.items()}
    expected_names = [id_to_name[cell_id] for cell_id in cell_ids]

    models = {}
    checkpoint_rows = []
    for loss in LOSSES:
        checkpoint = checkpoints[loss]
        if checkpoint["cell_ids"] != cell_ids or checkpoint["cell_names"] != expected_names:
            raise ValueError(f"{loss} checkpoint uses a different context order")
        model = load_protscape_model(checkpoint, ppi_data, device=device)
        model.use_metagraph = False
        models[loss] = model
        checkpoint_path = Path(paths[CHECKPOINT_KEYS[loss]]).expanduser()
        checkpoint_rows.append(
            {
                "loss": loss,
                "checkpoint": checkpoint_path.name,
                "checkpoint_epoch": int(checkpoint["epoch"]),
            }
        )
        print(f"Loaded {loss} checkpoint", flush=True)

    genes = sorted(
        {
            clean_gene(gene)
            for graph in ppi_layers.values()
            for gene in graph.nodes()
        }
    )
    gene_to_id = {gene: index for index, gene in enumerate(genes)}
    data = (ppi_data, celltype_map, ppi_layers)
    return data, models, genes, gene_to_id, pd.DataFrame(checkpoint_rows)


def unique_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    src = edge_index[0].detach().cpu()
    dst = edge_index[1].detach().cpu()
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    keep = lo != hi
    return torch.unique(torch.stack([lo[keep], hi[keep]], dim=1), dim=0).t().contiguous()


def positive_chunks(graph):
    edges = unique_undirected_edges(graph.edge_index)
    for start in range(0, edges.size(1), EDGE_CHUNK):
        yield edges[:, start : start + EDGE_CHUNK]


def nonedge_chunks(num_nodes: int, positive_edge_index: torch.Tensor):
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    src = positive_edge_index[0].detach().cpu()
    dst = positive_edge_index[1].detach().cpu()
    adjacency[src, dst] = True
    adjacency[dst, src] = True

    src_parts, dst_parts, total = [], [], 0
    for row in range(num_nodes - 1):
        destinations = torch.arange(row + 1, num_nodes, dtype=torch.long)
        keep = ~adjacency[row, row + 1 :]
        if keep.any():
            selected = destinations[keep]
            src_parts.append(torch.full((selected.numel(),), row, dtype=torch.long))
            dst_parts.append(selected)
            total += int(selected.numel())
        if total >= EDGE_CHUNK:
            yield torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])
            src_parts, dst_parts, total = [], [], 0
    if total:
        yield torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])


@torch.no_grad()
def decoder_inputs(models: dict[str, torch.nn.Module], cell_id: int, graph, device):
    graph_on_device = graph.clone().to(device)
    output = {}
    for loss, model in models.items():
        _, _, _, _, layer_outputs = model(
            {cell_id: graph_on_device}, batching=False, return_layer_outputs=True
        )
        decoder_input = _prepare_s2gae_decoder_input(
            model.s2gae_decoder,
            layer_outputs[cell_id],
            raw_embeddings=graph_on_device.x,
        )
        output[loss] = (model.s2gae_decoder, decoder_input)
    return output


@torch.no_grad()
def score_edges(decoders, edges: torch.Tensor, device: torch.device) -> torch.Tensor:
    edges = edges.to(device)
    values = []
    for loss in LOSSES:
        decoder, decoder_input = decoders[loss]
        logits = undirected_decoder_logits(decoder, decoder_input, edges)
        values.append(torch.sigmoid(logits))
    return torch.stack(values)


def class_ids(scores: torch.Tensor, sigma_threshold: float) -> np.ndarray:
    means = scores.mean(dim=0)
    sigma = scores.std(dim=0, unbiased=False)
    votes = (scores >= SCORE_CUTOFF).sum(dim=0)
    classes = torch.empty(scores.size(1), dtype=torch.long, device=scores.device)
    classes[votes == 0] = 0
    classes[votes == len(LOSSES)] = 5
    disagreement = (votes > 0) & (votes < len(LOSSES))
    negative = disagreement & (means < SCORE_CUTOFF)
    positive = disagreement & (means >= SCORE_CUTOFF)
    classes[negative & (sigma <= sigma_threshold)] = 1
    classes[negative & (sigma > sigma_threshold)] = 2
    classes[positive & (sigma > sigma_threshold)] = 3
    classes[positive & (sigma <= sigma_threshold)] = 4
    return classes.detach().cpu().numpy().astype(np.int8, copy=False)
