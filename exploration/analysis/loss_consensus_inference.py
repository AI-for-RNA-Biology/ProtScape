#!/usr/bin/env python3
"""Score every context PPI edge and non-edge for the loss analyses."""

from __future__ import annotations

import gzip
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from downstream_tasks.config import PATHS
from pretraining.checkpoints import load_protscape_model
from pretraining.data_handler.generate_input import read_data
from pretraining.train.minibatch_factored_utils import (
    _prepare_s2gae_decoder_input,
    undirected_decoder_logits,
)

LOSSES = ("BCE", "pHuber", "L1")
LOSS_PAIRS = ((0, 1), (0, 2))
CHECKPOINT_KEYS = {
    "BCE": "protscape_bce_checkpoint",
    "pHuber": "protscape_phuber_checkpoint",
    "L1": "protscape_l1_checkpoint",
}
LABEL_NAMES = {1: "Labelled positive", 0: "Labelled negative"}
SIMILARITY_GROUPS = {1: "Labelled positives", 0: "Labelled negatives"}
CLASS_NAMES = (
    "Consensus negative",
    "Weak disagreement negative",
    "Strong disagreement negative",
    "Strong disagreement positive",
    "Weak disagreement positive",
    "Consensus positive",
)
DISAGREEMENT_CLASSES = CLASS_NAMES[1:5]

SCORE_CUTOFF = 0.5
EDGE_CHUNK = 100_000
SAMPLE_SEED = 13
SAMPLE_PER_STRATUM = 50_000

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


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def exact_similarity_rows(
    context: str, group: str, scores: np.ndarray
) -> list[dict]:
    """Compute exact score and thresholded-call similarities."""
    n = scores.shape[1]
    if n == 0:
        return []

    ranks = np.empty((len(LOSSES), n), dtype=np.float64)
    for loss_index in range(len(LOSSES)):
        ranks[loss_index] = rankdata(scores[loss_index], method="average")

    calls = scores >= SCORE_CUTOFF
    rows = []
    for first, second in LOSS_PAIRS:
        first_call = calls[first]
        second_call = calls[second]
        intersection = int((first_call & second_call).sum())
        union = int((first_call | second_call).sum())
        agreement = int((first_call == second_call).sum())
        first_positive = int(first_call.sum())
        second_positive = int(second_call.sum())
        rows.append(
            {
                "context": context,
                "group": group,
                "comparison": f"{LOSSES[first]} vs {LOSSES[second]}",
                "edge_contexts": n,
                "pearson_r": safe_corrcoef(scores[first], scores[second]),
                "spearman_rho": safe_corrcoef(ranks[first], ranks[second]),
                "intersection_positive": intersection,
                "union_positive": union,
                "binary_agreement_count": agreement,
                "first_positive_count": first_positive,
                "second_positive_count": second_positive,
            }
        )
    return rows


def pooled_binary_phi(
    n: int, first_positive: int, second_positive: int, intersection: int
) -> float:
    first_fraction = first_positive / n
    second_fraction = second_positive / n
    variance = (
        first_fraction
        * (1 - first_fraction)
        * second_fraction
        * (1 - second_fraction)
    )
    if variance == 0:
        return np.nan
    return (
        intersection / n - first_fraction * second_fraction
    ) / np.sqrt(variance)


def summarize_exact_similarity(
    by_context: pd.DataFrame, pooled_positive: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    grouped = by_context.groupby(["group", "comparison"], sort=False)
    for (group, comparison), table in grouped:
        n = int(table["edge_contexts"].sum())
        first_positive = int(table["first_positive_count"].sum())
        second_positive = int(table["second_positive_count"].sum())
        intersection = int(table["intersection_positive"].sum())
        union = int(table["union_positive"].sum())
        agreement = int(table["binary_agreement_count"].sum())
        rows.append(
            {
                "group": group,
                "comparison": comparison,
                "contexts": int(table["context"].nunique()),
                "edge_contexts": n,
                "pearson_context_mean": table["pearson_r"].mean(),
                "pearson_context_q25": table["pearson_r"].quantile(0.25),
                "pearson_context_median": table["pearson_r"].median(),
                "pearson_context_q75": table["pearson_r"].quantile(0.75),
                "spearman_context_mean": table["spearman_rho"].mean(),
                "spearman_context_q25": table["spearman_rho"].quantile(0.25),
                "spearman_context_median": table["spearman_rho"].median(),
                "spearman_context_q75": table["spearman_rho"].quantile(0.75),
                "pooled_binary_phi": pooled_binary_phi(
                    n, first_positive, second_positive, intersection
                ),
                "pooled_positive_call_jaccard": (
                    intersection / union if union else np.nan
                ),
                "pooled_binary_agreement": agreement / n,
                "pooled_first_positive_fraction": first_positive / n,
                "pooled_second_positive_fraction": second_positive / n,
                "pooled_exact_pearson": np.nan,
                "pooled_exact_spearman": np.nan,
                "score_summary_estimand": (
                    "macro distribution of exact within-context correlations"
                ),
            }
        )

    summary = pd.DataFrame(rows)
    pooled_lookup = pooled_positive.set_index("comparison")
    positive_rows = summary["group"] == SIMILARITY_GROUPS[1]
    summary.loc[positive_rows, "pooled_exact_pearson"] = summary.loc[
        positive_rows, "comparison"
    ].map(pooled_lookup["pearson_r"])
    summary.loc[positive_rows, "pooled_exact_spearman"] = summary.loc[
        positive_rows, "comparison"
    ].map(pooled_lookup["spearman_rho"])
    return summary


def first_pass(data, models, device: torch.device):
    ppi_data, celltype_map, _ = data
    id_to_name = {cell_id: name for name, cell_id in celltype_map.items()}
    count = {label: 0 for label in LABEL_NAMES}
    total = {label: 0.0 for label in LABEL_NAMES}
    total_sq = {label: 0.0 for label in LABEL_NAMES}
    correlation_rows = []
    pooled_positive_blocks = []

    for context_index, (cell_id, graph) in enumerate(ppi_data.items(), start=1):
        context = id_to_name[cell_id]
        decoders = decoder_inputs(models, cell_id, graph, device)
        for label, chunks in (
            (1, positive_chunks(graph)),
            (0, nonedge_chunks(int(graph.num_nodes), graph.edge_index)),
        ):
            score_blocks = []
            for edges in chunks:
                scores = score_edges(decoders, edges, device)
                votes = (scores >= SCORE_CUTOFF).sum(dim=0)
                mixed = (votes > 0) & (votes < len(LOSSES))
                sigma = scores.std(dim=0, unbiased=False)[mixed].cpu().numpy()
                count[label] += int(sigma.size)
                total[label] += float(sigma.sum(dtype=np.float64))
                total_sq[label] += float(np.square(sigma, dtype=np.float64).sum())
                score_blocks.append(scores.cpu().numpy())

            context_scores = np.concatenate(score_blocks, axis=1)
            del score_blocks
            correlation_rows.extend(
                exact_similarity_rows(
                    context, SIMILARITY_GROUPS[label], context_scores
                )
            )
            if label == 1:
                pooled_positive_blocks.append(context_scores)
            del context_scores
        del decoders
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"Statistics pass: {context} ({context_index}/{len(ppi_data)})",
            flush=True,
        )

    n_mixed = sum(count.values())
    if n_mixed == 0:
        raise RuntimeError("No mixed-vote edge-contexts were found")
    threshold = sum(total.values()) / n_mixed
    rows = []
    for scope, n, value, value_sq in [
        ("All edge-contexts", n_mixed, sum(total.values()), sum(total_sq.values())),
        *[
            (LABEL_NAMES[label], count[label], total[label], total_sq[label])
            for label in (1, 0)
        ],
    ]:
        mean = value / n if n else np.nan
        variance = max(value_sq / n - mean * mean, 0.0) if n else np.nan
        rows.append(
            {
                "scope": scope,
                "sigma_threshold": threshold,
                "mixed_edge_contexts": n,
                "mixed_sigma_mean": mean,
                "mixed_sigma_sd": np.sqrt(variance),
                "threshold_method": "exact mean population s.d. over all mixed-vote edge-contexts",
                "score_cutoff": SCORE_CUTOFF,
                "score_sd_ddof": 0,
                "contexts": len(ppi_data),
            }
        )

    pooled_positive_scores = np.concatenate(pooled_positive_blocks, axis=1)
    pooled_positive = pd.DataFrame(
        exact_similarity_rows(
            "All contexts pooled", SIMILARITY_GROUPS[1], pooled_positive_scores
        )
    )
    by_context = pd.DataFrame(correlation_rows)
    summary = summarize_exact_similarity(by_context, pooled_positive)
    return threshold, pd.DataFrame(rows), by_context, summary


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("r")


def build_string_scores(genes: list[str], paths: dict) -> np.ndarray:
    info_path = require_file(Path(paths["string_protein_info"]).expanduser())
    links_path = require_file(Path(paths["string_links_detailed"]).expanduser())
    protein_to_gene = {}
    with open_text(info_path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        protein_col = header.index("#string_protein_id")
        gene_col = header.index("preferred_name")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            protein_to_gene[fields[protein_col]] = clean_gene(fields[gene_col])

    gene_to_id = {gene: index for index, gene in enumerate(genes)}
    n_genes = len(genes)
    scores = np.zeros(n_genes * (n_genes - 1) // 2, dtype=np.float32)
    with open_text(links_path) as handle:
        header = handle.readline().split()
        p1_col = header.index("protein1")
        p2_col = header.index("protein2")
        score_col = header.index("combined_score")
        for line in handle:
            fields = line.split()
            gene_a = gene_to_id.get(protein_to_gene.get(fields[p1_col], ""))
            gene_b = gene_to_id.get(protein_to_gene.get(fields[p2_col], ""))
            if gene_a is None or gene_b is None or gene_a == gene_b:
                continue
            lo, hi = sorted((gene_a, gene_b))
            pair = lo * n_genes - (lo * (lo + 1)) // 2 + (hi - lo - 1)
            score = int(fields[score_col]) / 1000.0
            if score > scores[pair]:
                scores[pair] = score
    print(f"Aligned STRING scores for {(scores > 0).sum():,} protein pairs", flush=True)
    return scores


def splitmix64(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=False) + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


class PrioritySample:
    """Deterministic uniform hash sample for each label-by-class stratum."""

    def __init__(self):
        strata = [(label, cls) for label in (1, 0) for cls in range(len(CLASS_NAMES))]
        self.hashes = {key: np.empty(0, dtype=np.uint64) for key in strata}
        self.scores = {
            key: np.empty((len(LOSSES), 0), dtype=np.float32) for key in strata
        }

    def update(self, label, classes, scores, pairs, context_index):
        scores = scores.astype(np.float32, copy=False)
        for cls in range(len(CLASS_NAMES)):
            selected = classes == cls
            if not selected.any():
                continue
            salt = np.uint64(
                SAMPLE_SEED
                + 1_000_003 * context_index
                + 10_007 * label
                + 101 * cls
            )
            hashes = splitmix64(pairs[selected].astype(np.uint64) ^ salt)
            values = scores[:, selected]
            key = (label, cls)
            hashes = np.concatenate([self.hashes[key], hashes])
            values = np.concatenate([self.scores[key], values], axis=1)
            if hashes.size > SAMPLE_PER_STRATUM:
                keep = np.argpartition(hashes, SAMPLE_PER_STRATUM - 1)[:SAMPLE_PER_STRATUM]
                hashes = hashes[keep]
                values = values[:, keep]
            self.hashes[key] = hashes
            self.scores[key] = values

    def save(self, output_dir: Path, population_counts: np.ndarray):
        labels, classes, blocks = [], [], {loss: [] for loss in LOSSES}
        rows = []
        for label in (1, 0):
            for cls, class_name in enumerate(CLASS_NAMES):
                order = np.argsort(self.hashes[(label, cls)])
                values = self.scores[(label, cls)][:, order]
                n = values.shape[1]
                labels.append(np.full(n, label, dtype=np.int8))
                classes.append(np.full(n, cls, dtype=np.int8))
                for loss_index, loss in enumerate(LOSSES):
                    blocks[loss].append(values[loss_index])
                rows.append(
                    {
                        "label": LABEL_NAMES[label],
                        "loss_class": class_name,
                        "edge_contexts_seen": int(population_counts[label, cls]),
                        "edge_contexts_sampled": n,
                    }
                )
        np.savez_compressed(
            output_dir / "whole_ppi_score_sample.npz",
            labels=np.concatenate(labels),
            class_ids=np.concatenate(classes),
            **{loss: np.concatenate(values) for loss, values in blocks.items()},
        )
        pd.DataFrame(rows).to_csv(
            output_dir / "whole_ppi_score_sample_summary.csv", index=False
        )


def similarity_table(exact_summary: pd.DataFrame) -> pd.DataFrame:
    """Build the compact source table consumed by Supplementary Figure 2e."""
    rows = []
    for record in exact_summary.to_dict("records"):
        other_loss = record["comparison"].replace("BCE vs ", "", 1)
        # The first label is retained because the plotting/source-table code
        # already uses it; the value is the exact within-context macro mean.
        metrics = {
            "Binned Spearman score correlation": record[
                "spearman_context_mean"
            ],
            "Positive-call Jaccard": record["pooled_positive_call_jaccard"],
            "Binary agreement": record["pooled_binary_agreement"],
            "BCE positive-call fraction": record[
                "pooled_first_positive_fraction"
            ],
            f"{other_loss} positive-call fraction": record[
                "pooled_second_positive_fraction"
            ],
            "Thresholded Spearman call correlation": record[
                "pooled_binary_phi"
            ],
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "group": record["group"],
                    "comparison": record["comparison"],
                    "metric": metric,
                    "value": value,
                    "edge_contexts": int(record["edge_contexts"]),
                }
            )
    return pd.DataFrame(rows)


def population_summary(population_counts, positive_calls, totals) -> pd.DataFrame:
    rows = []
    for label in (1, 0):
        for cls, class_name in enumerate(CLASS_NAMES):
            count = int(population_counts[label, cls])
            rows.append(
                {
                    "edge_label": LABEL_NAMES[label],
                    "summary": class_name,
                    "edge_contexts": count,
                    "fraction": count / totals[label],
                }
            )
        for loss_index, loss in enumerate(LOSSES):
            count = int(positive_calls[label, loss_index])
            rows.append(
                {
                    "edge_label": LABEL_NAMES[label],
                    "summary": f"{loss} score >= {SCORE_CUTOFF}",
                    "edge_contexts": count,
                    "fraction": count / totals[label],
                }
            )
    return pd.DataFrame(rows)


def second_pass(data, models, genes, gene_to_id, string_scores, threshold, device):
    ppi_data, celltype_map, ppi_layers = data
    id_to_name = {cell_id: name for name, cell_id in celltype_map.items()}
    n_genes = len(genes)
    n_pairs = n_genes * (n_genes - 1) // 2
    class_counts = np.zeros((2, len(CLASS_NAMES), n_pairs), dtype=np.uint16)
    population_counts = np.zeros((2, len(CLASS_NAMES)), dtype=np.int64)
    positive_calls = np.zeros((2, len(LOSSES)), dtype=np.int64)
    totals = np.zeros(2, dtype=np.int64)
    sample = PrioritySample()

    string_pairs = np.flatnonzero(string_scores > 0).astype(np.int64)
    string_lookup = np.full(n_pairs, -1, dtype=np.int32)
    string_lookup[string_pairs] = np.arange(string_pairs.size, dtype=np.int32)
    string_sum = np.zeros((len(LOSSES), string_pairs.size), dtype=np.float64)
    string_sum_sq = np.zeros_like(string_sum)
    string_count = np.zeros(string_pairs.size, dtype=np.uint16)
    string_observed = np.zeros(string_pairs.size, dtype=np.uint16)

    for context_index, (cell_id, graph) in enumerate(ppi_data.items(), start=1):
        context = id_to_name[cell_id]
        local_names = [clean_gene(gene) for gene in ppi_layers[context].nodes()]
        local_to_global = np.asarray([gene_to_id[gene] for gene in local_names], dtype=np.int64)
        decoders = decoder_inputs(models, cell_id, graph, device)
        for label, chunks in (
            (1, positive_chunks(graph)),
            (0, nonedge_chunks(int(graph.num_nodes), graph.edge_index)),
        ):
            for edges in chunks:
                scores_tensor = score_edges(decoders, edges, device)
                classes = class_ids(scores_tensor, threshold)
                scores = scores_tensor.cpu().numpy()
                edge_array = edges.numpy()
                pairs = pair_ids(
                    local_to_global[edge_array[0]],
                    local_to_global[edge_array[1]],
                    n_genes,
                )
                for cls in range(len(CLASS_NAMES)):
                    selected = classes == cls
                    if selected.any():
                        np.add.at(class_counts[label, cls], pairs[selected], 1)
                        population_counts[label, cls] += int(selected.sum())
                totals[label] += scores.shape[1]
                positive_calls[label] += (scores >= SCORE_CUTOFF).sum(axis=1)
                sample.update(label, classes, scores, pairs, context_index)

                compact = string_lookup[pairs]
                supported = compact >= 0
                if supported.any():
                    ids = compact[supported]
                    values = scores[:, supported].astype(np.float64, copy=False)
                    string_sum[:, ids] += values
                    string_sum_sq[:, ids] += values * values
                    string_count[ids] += 1
                    if label == 1:
                        string_observed[ids] += 1
        del decoders
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"Classification pass: {context} ({context_index}/{len(ppi_data)})",
            flush=True,
        )

    return {
        "class_counts": class_counts,
        "population_counts": population_counts,
        "positive_calls": positive_calls,
        "totals": totals,
        "sample": sample,
        "string_pairs": string_pairs,
        "string_sum": string_sum,
        "string_sum_sq": string_sum_sq,
        "string_count": string_count,
        "string_observed": string_observed,
    }


def save_string_moments(results: dict, string_scores: np.ndarray, output_dir: Path):
    keep = results["string_count"] >= 2
    counts = results["string_count"][keep]
    sums = results["string_sum"][:, keep]
    sum_squares = results["string_sum_sq"][:, keep]
    means = sums / counts
    variances = np.maximum(sum_squares / counts - means * means, 0.0)
    pair_ids_selected = results["string_pairs"][keep]
    np.savez_compressed(
        output_dir / "string_positive_loss_score_moments.npz",
        pair_scope=np.asarray("string_positive_copresent_at_least_2"),
        losses=np.asarray(LOSSES),
        pair_ids=pair_ids_selected,
        string_combined=string_scores[pair_ids_selected],
        mean_scores=means.astype(np.float32),
        std_scores=np.sqrt(variances).astype(np.float32),
        n_copresent_contexts=counts,
        n_observed_contexts=results["string_observed"][keep],
    )


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("No CUDA GPU found; exhaustive inference will be very slow.", flush=True)

    data, models, genes, gene_to_id, checkpoints = load_models_and_data(device)
    threshold, threshold_table, similarity_by_context, similarity_summary = first_pass(
        data, models, device
    )
    print(f"Exact mixed-vote sigma threshold: {threshold:.8f}", flush=True)
    string_scores = build_string_scores(genes, PATHS)
    results = second_pass(
        data, models, genes, gene_to_id, string_scores, threshold, device
    )

    np.save(output_dir / "pair_class_counts_uint16.npy", results["class_counts"])
    np.save(output_dir / "string_combined_scores.npy", string_scores)
    (output_dir / "global_gene_order.txt").write_text("\n".join(genes) + "\n")
    checkpoints.to_csv(output_dir / "checkpoints_used.csv", index=False)
    population = population_summary(
        results["population_counts"], results["positive_calls"], results["totals"]
    )
    population.to_csv(output_dir / "whole_ppi_score_population_summary.csv", index=False)
    mixed_by_label = (
        population[population["summary"].isin(DISAGREEMENT_CLASSES)]
        .groupby("edge_label")["edge_contexts"]
        .sum()
    )
    classification_counts = {
        LABEL_NAMES[label]: int(mixed_by_label.get(LABEL_NAMES[label], 0))
        for label in (1, 0)
    }
    classification_counts["All edge-contexts"] = sum(classification_counts.values())
    threshold_table["classification_pass_mixed_edge_contexts"] = threshold_table[
        "scope"
    ].map(classification_counts)
    threshold_table["classification_minus_statistics_pass"] = (
        threshold_table["classification_pass_mixed_edge_contexts"]
        - threshold_table["mixed_edge_contexts"]
    )
    threshold_table.to_csv(
        output_dir / "weak_strong_disagreement_sigma_threshold.csv", index=False
    )
    results["sample"].save(output_dir, results["population_counts"])
    similarity_by_context.to_csv(
        output_dir / "loss_similarity_exact_by_context.csv", index=False
    )
    similarity_summary.to_csv(
        output_dir / "loss_similarity_exact_summary.csv", index=False
    )
    similarity_table(similarity_summary).to_csv(
        output_dir / "loss_similarity_to_bce_metrics.csv", index=False
    )
    save_string_moments(results, string_scores, output_dir)
    print(f"Saved exhaustive loss-consensus inference to {output_dir}", flush=True)
