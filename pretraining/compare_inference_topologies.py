"""Paired PPI evaluation for global, local-topology, and contextual inference."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data

from pretraining.checkpoints import load_protscape_model
from pretraining.contextwise_ppi import (
    CONTEXTWISE_K_VALUES,
    sample_structured_negatives,
)
from pretraining.data_handler.generate_input import read_data
from pretraining.evaluate_global_s2gae import build_model as build_global_model
from pretraining.global_s2gae import load_global_ppi_data
from pretraining.models.hierarchical_model import add_virtual_node
from pretraining.s2gae_utils import undirected_decoder_logits


INFERENCE_ORDER = ("global_context_free", "cell_context_free", "protscape")
INFERENCE_LABELS = {
    "global_context_free": "Context-free / global topology",
    "cell_context_free": "Context-free / Cell-PPI topology",
    "protscape": "ProtScape / Cell-PPI topology",
}
INFERENCE_COLORS = {
    "global_context_free": "#d31529",
    "cell_context_free": "#1f77b4",
    "protscape": "#e6550d",
}
SPLIT_ORDER = ("train", "validation", "test")
SPLIT_SEED_OFFSET = {"train": 1_000_000, "validation": 2_000_000, "test": 0}
CM = 1 / 2.54
PLOT_RC = {
    "font.family": "Liberation Sans",
    "axes.labelsize": 8.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "font.size": 7.0,
    "legend.fontsize": 6.0,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--protscape-checkpoint", type=Path, required=True)
    parser.add_argument("--networks-dir", type=Path, required=True)
    parser.add_argument("--esm2-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decoder-chunk-size", type=int, default=100_000)
    parser.add_argument("--plot-samples-per-class-cell", type=int, default=250)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_edges(edge_index: torch.Tensor) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long()
    return torch.stack(
        [
            torch.minimum(edge_index[0], edge_index[1]),
            torch.maximum(edge_index[0], edge_index[1]),
        ],
        dim=0,
    )


def prepare_decoder_input(decoder, layers, raw_embeddings: torch.Tensor):
    if getattr(decoder, "raw_input_dim", 0) > 0:
        return {"layers": layers, "raw": raw_embeddings}
    return layers


def split_masks(graph: Data, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        return graph.train_mask, graph.train_mask
    if split == "validation":
        return graph.train_mask, graph.val_mask
    if split == "test":
        return graph.train_mask | graph.val_mask, graph.test_mask
    raise ValueError(f"Unknown split: {split}")


def score_decoder(
    decoder,
    decoder_input,
    edge_index: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """Score symmetric edges once per unique pair and restore occurrence order."""
    canonical = canonical_edges(edge_index).t().contiguous()
    unique_edges, inverse = torch.unique(
        canonical, dim=0, sorted=True, return_inverse=True
    )
    outputs = []
    for start in range(0, unique_edges.size(0), chunk_size):
        batch = unique_edges[start : start + chunk_size].t().to(device)
        logits = undirected_decoder_logits(decoder, decoder_input, batch)
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    if not outputs:
        return np.empty(0, dtype=np.float32)
    unique_scores = np.concatenate(outputs)
    return unique_scores[inverse.numpy()]


def load_context_data(
    checkpoint: dict,
    networks_dir: Path,
    esm2_embeddings: Path,
) -> tuple[dict[int, Data], torch.Tensor, list[str], dict[int, str]]:
    config = checkpoint["config"]
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        loaded = read_data(
            networks_dir / "global_ppi_edgelist.txt",
            networks_dir / "ppi_edgelists",
            networks_dir / "mg_edgelist.txt",
            get_CT_map=True,
            ppi_feat_dir=esm2_embeddings,
            symmetric_ppi=config.get("symmetric_ppi", True),
            dataset_mode=config["dataset_mode"],
            split_mode=config["split_mode"],
            count_edge_path=networks_dir / "count_edge_dict.pkl",
            weighted_ppi_loss=config.get("weighted_ppi_loss", False),
            defer_ppi_features=True,
            seed=int(config.get("seed", 0)),
            verbose=False,
        )
    contexts, metagraph, _, celltype_map, _, _, _, _ = loaded
    cell_ids = [int(cell_id) for cell_id in checkpoint["cell_ids"]]
    if set(cell_ids) != set(contexts):
        raise ValueError("ProtScape checkpoint and released Cell-PPIs differ.")
    contexts = {cell_id: contexts[cell_id] for cell_id in cell_ids}
    id_to_name = {int(cell_id): str(name) for name, cell_id in celltype_map.items()}
    if checkpoint["cell_names"] != [id_to_name[cell_id] for cell_id in cell_ids]:
        raise ValueError("ProtScape checkpoint and released cell names differ.")
    return (
        contexts,
        metagraph.global_protein_features.float().contiguous(),
        [str(name) for name in metagraph.global_protein_names],
        id_to_name,
    )


@torch.no_grad()
def encode_context_free_local(
    model,
    features: torch.Tensor,
    graph: Data,
    message_mask: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    local_features = features[graph.feature_index].to(device)
    message_edges = graph.edge_index[:, message_mask].to(device)
    _, layers = model.encode(local_features, message_edges)
    return layers


@torch.no_grad()
def encode_protscape_local(
    model,
    features: torch.Tensor,
    graph: Data,
    cell_id: int,
    message_mask: torch.Tensor,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    local = Data(
        x=features[graph.feature_index],
        edge_index=graph.edge_index[:, message_mask],
        edge_attr=graph.edge_attr[message_mask],
    ).to(device)
    pooling = model.cell_config["pooling"]
    uses_virtual_node = (
        pooling in {"vn", "learnedvn"}
        or model.protein_config.get("add_virtual_node")
    )
    if pooling == "vn" or model.protein_config.get("add_virtual_node"):
        local = add_virtual_node(local, model.virtual_node_features, clone=False)
    elif pooling == "learnedvn":
        local = add_virtual_node(
            local,
            model.virtual_node_features[model.celltype_to_vn_id[cell_id]],
            clone=False,
        )
    _, layers = model.prot_encoder(
        {cell_id: local}, batching=False, return_layer_outputs=True
    )
    layers = layers[cell_id]
    if uses_virtual_node:
        layers = [layer[:-1] for layer in layers]
    raw = local.x[: graph.num_nodes]
    return layers, raw


def correlation(left: np.ndarray, right: np.ndarray, method: str) -> float:
    if left.size < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    result = pearsonr(left, right) if method == "pearson" else spearmanr(left, right)
    return float(result.statistic)


def top_fraction_jaccard(left: np.ndarray, right: np.ndarray, fraction: float = 0.1) -> float:
    count = max(1, int(np.ceil(fraction * left.size)))
    left_top = set(np.argpartition(left, -count)[-count:].tolist())
    right_top = set(np.argpartition(right, -count)[-count:].tolist())
    return len(left_top & right_top) / len(left_top | right_top)


def agreement_row(
    cell_id: int,
    cell_name: str,
    split: str,
    left_key: str,
    right_key: str,
    left: np.ndarray,
    right: np.ndarray,
) -> dict:
    return {
        "cell_id": cell_id,
        "cell": cell_name,
        "split": split,
        "left_inference": left_key,
        "right_inference": right_key,
        "pearson": correlation(left, right, "pearson"),
        "spearman": correlation(left, right, "spearman"),
        "mae": float(np.mean(np.abs(left - right))),
        "rmse": float(np.sqrt(np.mean((left - right) ** 2))),
        "top_10pct_jaccard": top_fraction_jaccard(left, right),
        "n_scored": int(left.size),
    }


def add_plot_sample(
    rows: list[dict],
    scores: dict[str, np.ndarray],
    n_pos: int,
    cell_id: int,
    cell_name: str,
    split: str,
    samples_per_class: int,
) -> None:
    generator = np.random.default_rng(SPLIT_SEED_OFFSET[split] + cell_id)
    selections = []
    for label, indices in (
        (1, np.arange(n_pos)),
        (0, np.arange(n_pos, 2 * n_pos)),
    ):
        count = min(samples_per_class, indices.size)
        chosen = generator.choice(indices, size=count, replace=False)
        selections.extend((label, int(index)) for index in chosen)
    for label, index in selections:
        rows.append(
            {
                "cell_id": cell_id,
                "cell": cell_name,
                "split": split,
                "label": label,
                **{f"score_{key}": float(scores[key][index]) for key in INFERENCE_ORDER},
            }
        )


def auprc_from_pos_neg(positive: np.ndarray, negative: np.ndarray) -> dict:
    scores = np.concatenate([positive, negative])
    labels = np.concatenate(
        [
            np.ones(positive.size, dtype=np.int8),
            np.zeros(negative.size, dtype=np.int8),
        ]
    )
    return {
        "ap": float(average_precision_score(labels, scores)),
        "n_pos": int(positive.size),
        "n_neg": int(negative.size),
    }


def evaluate_score_banks(
    scores: dict[str, np.ndarray],
    n_pos: int,
    max_k: int,
    k_values: tuple[int, ...],
) -> tuple[dict[tuple[str, int], dict], dict[str, np.ndarray]]:
    """Evaluate each model on nested prefixes of its shared negative bank."""
    metrics = {}
    paired_scores = {}
    for inference_key in INFERENCE_ORDER:
        positive = scores[inference_key][:n_pos]
        negative = scores[inference_key][n_pos:].reshape(n_pos, max_k)
        paired_scores[inference_key] = np.concatenate([positive, negative[:, 0]])
        for k in k_values:
            metrics[(inference_key, k)] = auprc_from_pos_neg(
                positive,
                negative[:, :k].reshape(-1),
            )
    return metrics, paired_scores


def metric_rows(per_cell: list[dict]) -> list[dict]:
    output = []
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in per_cell:
        grouped[(row["inference_key"], row["split"], row["k_negatives"])].append(row)
    for inference_key in INFERENCE_ORDER:
        for split in SPLIT_ORDER:
            k_values = CONTEXTWISE_K_VALUES if split == "test" else (1,)
            for k in k_values:
                rows = grouped[(inference_key, split, k)]
                output.append(
                    {
                        "inference_key": inference_key,
                        "inference": INFERENCE_LABELS[inference_key],
                        "split": split,
                        "k_negatives": k,
                        "pos_neg_ratio": f"1:{k}",
                        "auprc_percent": 100.0
                        * float(np.mean([row["ap"] for row in rows])),
                        "n_pos": sum(row["n_pos"] for row in rows),
                        "n_neg": sum(row["n_neg"] for row in rows),
                        "n_cells": len(rows),
                        "targets_in_message_graph": split == "train",
                    }
                )
    return output


def aggregate_agreement(per_cell: list[dict]) -> list[dict]:
    output = []
    keys = ("pearson", "spearman", "mae", "rmse", "top_10pct_jaccard")
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in per_cell:
        grouped[(row["split"], row["left_inference"], row["right_inference"])].append(row)
    for (split, left, right), rows in grouped.items():
        output.append(
            {
                "split": split,
                "left_inference": left,
                "right_inference": right,
                **{
                    f"macro_{key}": float(
                        np.nanmean([float(row[key]) for row in rows])
                    )
                    for key in keys
                },
                "n_cells": len(rows),
            }
        )
    return output


def plot_results(
    metrics_rows: list[dict],
    samples: list[dict],
    variability: list[dict],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def clean_axis(ax) -> None:
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", width=0.6, length=2.5)

    def save_figure(fig, stem: str) -> None:
        options = {"dpi": 300, "bbox_inches": "tight", "transparent": True}
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output_dir / f"{stem}.{suffix}", **options)
        plt.close(fig)

    with matplotlib.rc_context(PLOT_RC):
        test = [row for row in metrics_rows if row["split"] == "test"]
        fig, ax = plt.subplots(figsize=(7.2 * CM, 6.0 * CM))
        for key in INFERENCE_ORDER:
            rows = sorted(
                (row for row in test if row["inference_key"] == key),
                key=lambda row: row["k_negatives"],
            )
            ax.plot(
                [row["k_negatives"] for row in rows],
                [row["auprc_percent"] for row in rows],
                marker="o",
                markersize=3.5,
                linewidth=2,
                color=INFERENCE_COLORS[key],
                label=INFERENCE_LABELS[key],
            )
        ax.set_xscale("log")
        ax.set_xticks(CONTEXTWISE_K_VALUES)
        ax.set_xticklabels([str(k) for k in CONTEXTWISE_K_VALUES])
        ax.set_xlabel("Negatives per positive edge")
        ax.set_ylabel("AUPRC")
        ax.set_ylim(0, 100)
        clean_axis(ax)
        ax.legend(frameon=False, fontsize=5.5)
        fig.tight_layout()
        save_figure(fig, "topology_inference_test_auprc")

        fig, ax = plt.subplots(figsize=(8.0 * CM, 6.0 * CM))
        x = np.arange(len(SPLIT_ORDER))
        width = 0.24
        for index, key in enumerate(INFERENCE_ORDER):
            values = [
                next(
                    row["auprc_percent"]
                    for row in metrics_rows
                    if row["inference_key"] == key
                    and row["split"] == split
                    and row["k_negatives"] == 1
                )
                for split in SPLIT_ORDER
            ]
            ax.bar(
                x + (index - 1) * width,
                values,
                width,
                color=INFERENCE_COLORS[key],
                label=INFERENCE_LABELS[key],
            )
            for xpos, value in zip(x + (index - 1) * width, values):
                ax.text(xpos, value + 1.0, f"{value:.1f}", ha="center", fontsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Train*", "Validation", "Test"])
        ax.set_ylabel("AUPRC at 1:1")
        ax.set_ylim(0, 105)
        clean_axis(ax)
        ax.legend(frameon=False, fontsize=5.2, loc="lower left")
        fig.tight_layout()
        save_figure(fig, "topology_inference_split_auprc")

        sample_rows = [row for row in samples if row["split"] == "test"]
        pairs = (
            ("global_context_free", "cell_context_free"),
            ("cell_context_free", "protscape"),
            ("global_context_free", "protscape"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(17.5 * CM, 5.5 * CM))
        labels = np.asarray([row["label"] for row in sample_rows])
        for ax, (left, right) in zip(axes, pairs):
            x_values = np.asarray([row[f"score_{left}"] for row in sample_rows])
            y_values = np.asarray([row[f"score_{right}"] for row in sample_rows])
            for label, color_map in ((0, "Greys"), (1, "Reds")):
                mask = labels == label
                ax.hexbin(
                    x_values[mask],
                    y_values[mask],
                    gridsize=35,
                    mincnt=1,
                    bins="log",
                    cmap=color_map,
                    alpha=0.7,
                )
            ax.plot([0, 1], [0, 1], linestyle="--", color="#222222", linewidth=0.7)
            ax.set_xlabel(INFERENCE_LABELS[left])
            ax.set_ylabel(INFERENCE_LABELS[right])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            clean_axis(ax)
        fig.tight_layout()
        save_figure(fig, "topology_inference_test_score_agreement")

        repeated = [row for row in variability if row["n_contexts"] > 1]
        fig, ax = plt.subplots(figsize=(7.2 * CM, 6.0 * CM))
        for key in INFERENCE_ORDER:
            values = np.sort(
                np.asarray([row[f"std_{key}"] for row in repeated], dtype=float)
            )
            cumulative = np.arange(1, values.size + 1) / values.size
            ax.plot(
                values,
                cumulative,
                linewidth=2,
                color=INFERENCE_COLORS[key],
                label=INFERENCE_LABELS[key],
            )
        ax.set_xlabel("Within-pair score SD across Cell-PPIs")
        ax.set_ylabel("Cumulative fraction of repeated pairs")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        clean_axis(ax)
        ax.legend(frameon=False, fontsize=5.5, loc="lower right")
        fig.tight_layout()
        save_figure(fig, "topology_inference_context_variability")


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    paths = (
        args.global_checkpoint,
        args.protscape_checkpoint,
        args.networks_dir,
        args.esm2_embeddings,
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing evaluation input(s): " + ", ".join(missing))
    if (
        args.decoder_chunk_size < 1
        or args.plot_samples_per_class_cell < 1
    ):
        raise ValueError("Chunk and plot-sample sizes must be positive.")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type != "cuda":
        raise RuntimeError("The paired evaluation requires a CUDA device.")

    global_checkpoint = torch.load(
        args.global_checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    protscape_checkpoint = torch.load(
        args.protscape_checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    if global_checkpoint.get("model_type") != "global_s2gae":
        raise ValueError("Expected a context-free global-S2GAE checkpoint.")
    if protscape_checkpoint.get("model_type") != "protscape":
        raise ValueError("Expected a portable ProtScape checkpoint.")
    if not protscape_checkpoint["config"]["s2gae_config"]["enabled"]:
        raise ValueError("The ProtScape checkpoint does not contain an S2GAE decoder.")

    split_seed = int(global_checkpoint["training_config"]["split_seed"])
    global_data = load_global_ppi_data(
        args.networks_dir,
        args.esm2_embeddings,
        seed=split_seed,
        verbose=False,
    )
    contexts, features, protein_names, id_to_name = load_context_data(
        protscape_checkpoint,
        args.networks_dir,
        args.esm2_embeddings,
    )
    if len(contexts) != global_data.source_context_count:
        raise ValueError(
            "Global and Cell-PPI loaders produced different context counts."
        )
    for checkpoint_key, observed in (
        ("graph_fingerprint", global_data.graph_fingerprint),
        ("feature_fingerprint", global_data.feature_fingerprint),
        ("split_fingerprint", global_data.split_fingerprint),
    ):
        if global_checkpoint[checkpoint_key] != observed:
            raise ValueError(f"Context-free checkpoint {checkpoint_key} differs from the release.")
    if protein_names != global_data.protein_names:
        raise ValueError("Global and Cell-PPI loaders produced different protein orders.")
    if not torch.equal(features, global_data.features):
        raise ValueError("Global and Cell-PPI loaders produced different normalized features.")
    if global_checkpoint["protein_names"] != protein_names:
        raise ValueError("Context-free checkpoint protein order differs from the release.")

    global_model = build_global_model(global_checkpoint, device).eval()
    protscape_model = load_protscape_model(
        protscape_checkpoint, contexts, device=device
    ).eval()
    features = features.cpu()
    global_features = global_data.features.to(device)
    _, global_train_layers = global_model.encode(
        global_features, global_data.train_edge_index.to(device)
    )
    _, global_test_layers = global_model.encode(
        global_features, global_data.train_val_edge_index.to(device)
    )

    per_cell_metrics: list[dict] = []
    per_cell_agreement: list[dict] = []
    plot_samples: list[dict] = []
    positive_scores: list[dict] = []
    inference_pairs = tuple(
        (INFERENCE_ORDER[left], INFERENCE_ORDER[right])
        for left in range(len(INFERENCE_ORDER))
        for right in range(left + 1, len(INFERENCE_ORDER))
    )
    num_global_nodes = len(protein_names)
    global_test_canonical = canonical_edges(global_data.test_edge_index)
    global_test_keys = set(
        (
            global_test_canonical[0] * num_global_nodes
            + global_test_canonical[1]
        ).tolist()
    )
    for context_index, (cell_id, graph) in enumerate(contexts.items(), start=1):
        cell_name = id_to_name[cell_id]
        train_message, _ = split_masks(graph, "validation")
        test_message, _ = split_masks(graph, "test")
        context_free_layers = {
            "train": encode_context_free_local(
                global_model, features, graph, train_message, device
            ),
            "test": encode_context_free_local(
                global_model, features, graph, test_message, device
            ),
        }
        protscape_train_layers, protscape_train_raw = encode_protscape_local(
            protscape_model, features, graph, cell_id, train_message, device
        )
        protscape_test_layers, protscape_test_raw = encode_protscape_local(
            protscape_model, features, graph, cell_id, test_message, device
        )
        protscape_inputs = {
            "train": prepare_decoder_input(
                protscape_model.s2gae_decoder,
                protscape_train_layers,
                raw_embeddings=protscape_train_raw,
            ),
            "test": prepare_decoder_input(
                protscape_model.s2gae_decoder,
                protscape_test_layers,
                raw_embeddings=protscape_test_raw,
            ),
        }
        local_to_global = graph.feature_index.detach().cpu().long()

        for split in SPLIT_ORDER:
            _, target_mask = split_masks(graph, split)
            positive_local = graph.edge_index[:, target_mask].detach().cpu().long()
            if positive_local.size(1) == 0:
                raise ValueError(f"Cell-PPI {cell_id} has no {split} targets.")
            max_k = max(CONTEXTWISE_K_VALUES) if split == "test" else 1
            np.random.seed(SPLIT_SEED_OFFSET[split] + cell_id)
            negative_local = sample_structured_negatives(
                positive_local,
                graph.edge_index.detach().cpu(),
                int(graph.num_nodes),
                max_k,
            ).reshape(2, positive_local.size(1), max_k)
            evaluation_local = torch.cat(
                [positive_local, negative_local.reshape(2, -1)], dim=1
            )
            evaluation_global = local_to_global[evaluation_local]
            topology_key = "test" if split == "test" else "train"
            global_layers = (
                global_test_layers if split == "test" else global_train_layers
            )
            scores = {
                "global_context_free": score_decoder(
                    global_model.decoder,
                    global_layers,
                    evaluation_global,
                    device,
                    args.decoder_chunk_size,
                ),
                "cell_context_free": score_decoder(
                    global_model.decoder,
                    context_free_layers[topology_key],
                    evaluation_local,
                    device,
                    args.decoder_chunk_size,
                ),
                "protscape": score_decoder(
                    protscape_model.s2gae_decoder,
                    protscape_inputs[topology_key],
                    evaluation_local,
                    device,
                    args.decoder_chunk_size,
                ),
            }
            n_pos = positive_local.size(1)
            k_values = CONTEXTWISE_K_VALUES if split == "test" else (1,)
            results, paired_scores = evaluate_score_banks(
                scores, n_pos, max_k, k_values
            )
            for inference_key in INFERENCE_ORDER:
                for k in k_values:
                    per_cell_metrics.append(
                        {
                            "inference_key": inference_key,
                            "inference": INFERENCE_LABELS[inference_key],
                            "cell_id": cell_id,
                            "cell": cell_name,
                            "split": split,
                            "k_negatives": k,
                            "pos_neg_ratio": f"1:{k}",
                            **results[(inference_key, k)],
                        }
                    )

            for left, right in inference_pairs:
                per_cell_agreement.append(
                    agreement_row(
                        cell_id,
                        cell_name,
                        split,
                        left,
                        right,
                        paired_scores[left],
                        paired_scores[right],
                    )
                )
            add_plot_sample(
                plot_samples,
                paired_scores,
                n_pos,
                cell_id,
                cell_name,
                split,
                args.plot_samples_per_class_cell,
            )

            if split == "test":
                positive_global = local_to_global[positive_local]
                canonical = canonical_edges(positive_global)
                pair_keys = (
                    canonical[0] * num_global_nodes + canonical[1]
                ).numpy()
                if not set(pair_keys.tolist()).issubset(global_test_keys):
                    raise ValueError(
                        f"Cell-PPI {cell_id} has a test pair outside the global test split."
                    )
                _, first_indices = np.unique(pair_keys, return_index=True)
                for occurrence in first_indices:
                    source = int(canonical[0, occurrence])
                    target = int(canonical[1, occurrence])
                    positive_scores.append(
                        {
                            "pair_key": int(pair_keys[occurrence]),
                            "source_protein": protein_names[source],
                            "target_protein": protein_names[target],
                            "cell_id": cell_id,
                            "cell": cell_name,
                            **{
                                f"score_{key}": float(scores[key][occurrence])
                                for key in INFERENCE_ORDER
                            },
                        }
                    )

        if context_index % 10 == 0 or context_index == len(contexts):
            print(
                f"paired topology evaluation: {context_index}/{len(contexts)} Cell-PPIs",
                flush=True,
            )
        del context_free_layers, protscape_inputs
        del protscape_train_layers, protscape_test_layers
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_metrics = metric_rows(per_cell_metrics)
    aggregate_agreements = aggregate_agreement(per_cell_agreement)

    grouped_positive: dict[int, list[dict]] = defaultdict(list)
    for row in positive_scores:
        grouped_positive[int(row["pair_key"])].append(row)
    variability = []
    for pair_key, rows in grouped_positive.items():
        variability.append(
            {
                "pair_key": pair_key,
                "source_protein": rows[0]["source_protein"],
                "target_protein": rows[0]["target_protein"],
                "n_contexts": len(rows),
                **{
                    f"std_{key}": float(
                        np.std([row[f"score_{key}"] for row in rows])
                    )
                    for key in INFERENCE_ORDER
                },
            }
        )
    repeated_variability = [row for row in variability if row["n_contexts"] > 1]
    variability_summary = []
    for inference_key in INFERENCE_ORDER:
        values = np.asarray(
            [row[f"std_{inference_key}"] for row in repeated_variability],
            dtype=float,
        )
        variability_summary.append(
            {
                "inference_key": inference_key,
                "inference": INFERENCE_LABELS[inference_key],
                "mean_within_pair_sd": float(values.mean()) if values.size else "",
                "median_within_pair_sd": (
                    float(np.median(values)) if values.size else ""
                ),
                "p95_within_pair_sd": (
                    float(np.quantile(values, 0.95)) if values.size else ""
                ),
                "max_within_pair_sd": float(values.max()) if values.size else "",
                "n_repeated_pairs": int(values.size),
            }
        )

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir.with_name(
        f".{output_dir.name}.incomplete-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    )
    work_dir.mkdir(parents=False, exist_ok=False)
    write_csv(work_dir / "aggregate_metrics.csv", aggregate_metrics)
    write_csv(work_dir / "per_cell_metrics.csv", per_cell_metrics)
    write_csv(work_dir / "aggregate_score_agreement.csv", aggregate_agreements)
    write_csv(work_dir / "per_cell_score_agreement.csv", per_cell_agreement)
    write_csv(work_dir / "plot_score_samples.csv", plot_samples)
    write_gzip_csv(
        work_dir / "test_positive_pair_context_scores.csv.gz", positive_scores
    )
    write_csv(work_dir / "test_positive_context_variability.csv", variability)
    write_csv(work_dir / "aggregate_context_variability.csv", variability_summary)
    plot_results(aggregate_metrics, plot_samples, variability, work_dir)

    summary = {
        "protocol_version": "paired_topology_inference_v1",
        "inference_labels": INFERENCE_LABELS,
        "test_k_values": list(CONTEXTWISE_K_VALUES),
        "diagnostic_splits": ["train", "validation"],
        "diagnostic_k_values": [1],
        "split_policy": {
            "train": "train targets scored on train topology; descriptive and seen",
            "validation": "validation targets scored on train topology",
            "test": "test targets scored on train+validation topology",
        },
        "negative_sampling": {
            "method": "within-Cell-PPI structured target corruption",
            "test_seed": "cell_id (identical to paper evaluator)",
            "train_seed": "1000000 + cell_id",
            "validation_seed": "2000000 + cell_id",
        },
        "aggregation": "unweighted macro-average over Cell-PPIs",
        "n_cells": len(contexts),
        "n_global_proteins": len(protein_names),
        "global_checkpoint": str(args.global_checkpoint.resolve()),
        "global_checkpoint_sha256": sha256_file(args.global_checkpoint),
        "protscape_checkpoint": str(args.protscape_checkpoint.resolve()),
        "protscape_checkpoint_sha256": sha256_file(args.protscape_checkpoint),
        "git_commit": os.environ.get("PROTSCAPE_GIT_COMMIT", "uncommitted"),
        "aggregate_metrics": aggregate_metrics,
        "aggregate_score_agreement": aggregate_agreements,
        "aggregate_context_variability": variability_summary,
    }
    with (work_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (work_dir / "completed.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol_version": summary["protocol_version"],
                "global_checkpoint_sha256": summary["global_checkpoint_sha256"],
                "protscape_checkpoint_sha256": summary[
                    "protscape_checkpoint_sha256"
                ],
                "git_commit": summary["git_commit"],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    os.replace(work_dir, output_dir)
    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
