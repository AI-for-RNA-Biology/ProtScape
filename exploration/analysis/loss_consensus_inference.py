#!/usr/bin/env python3
"""Score every context PPI edge and non-edge for the loss analyses."""

from __future__ import annotations

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

from downstream_tasks.config import PATHS
from exploration.analysis.loss_consensus_scoring import (
    CLASS_NAMES,
    LABEL_NAMES,
    LOSSES,
    SCORE_CUTOFF,
    class_ids,
    clean_gene,
    decoder_inputs,
    load_models_and_data,
    nonedge_chunks,
    pair_ids,
    positive_chunks,
    score_edges,
)
from exploration.analysis.loss_consensus_statistics import (
    SIMILARITY_GROUPS,
    exact_similarity_rows,
    population_summary,
    similarity_table,
    summarize_exact_similarity,
)
from exploration.analysis.loss_consensus_string import (
    build_string_scores,
    save_string_moments,
)


DISAGREEMENT_CLASSES = CLASS_NAMES[1:5]
SAMPLE_SEED = 13
SAMPLE_PER_STRATUM = 50_000


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
