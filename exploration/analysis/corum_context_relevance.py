"""Compute CORUM context-relevance and coverage analyses."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from downstream_tasks.data.preprocessing import zscore_normalize_bags
from exploration.analysis.corum_complex_properties import (
    CELL_PPI_DIR,
    GLOBAL_PPI,
    complex_members,
    read_ppi,
    sorted_edge,
)
from exploration.analysis.corum_model_evaluation import (
    MODEL_LABELS,
    build_abmil,
    load_folds,
    train_indices,
)
from exploration.analysis.therapeutic_target.xmil_lrp import explain_late_fusion_bag


def build_lrp_model(
    data: dict[str, object],
    checkpoint_dir: Path,
    fold: int,
) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_abmil("abmil8_pdl_id2_dropout", data, device)
    model.load_state_dict(
        torch.load(
            checkpoint_dir / f"fold_{fold}.pt",
            map_location=device,
            weights_only=True,
        )
    )
    return model.eval()


def xmil_lrp_values(
    model_key: str,
    data: dict[str, object],
    checkpoint_dir: Path,
    split_file: Path,
) -> pd.DataFrame:
    """Recompute normalized positive xMIL relevance from fold checkpoints."""
    ctx_bags = data["ctx_bags"]
    esm = data["esm"]
    cell_ids = data["cell_ids"]
    folds = load_folds(split_file, data["genes"])
    test_idx = folds[0]
    sums: dict[tuple[str, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for fold in range(5):
        train_idx = train_indices(folds, fold)
        _, bag_mean, bag_std = zscore_normalize_bags([ctx_bags[i] for i in train_idx])
        esm_mean = esm[train_idx].mean(axis=0)
        esm_std = np.maximum(esm[train_idx].std(axis=0), 1e-8)
        model = build_lrp_model(data, checkpoint_dir, fold)
        device = next(model.parameters()).device

        for gene_idx in test_idx:
            positive_classes = np.flatnonzero(data["labels"][gene_idx] > 0)
            if len(positive_classes) == 0:
                continue
            bag = ((ctx_bags[gene_idx] - bag_mean) / bag_std).astype(
                np.float32, copy=False
            )
            esm_vector = ((esm[gene_idx] - esm_mean) / esm_std).astype(
                np.float32, copy=False
            )
            scores = explain_late_fusion_bag(
                model,
                torch.as_tensor(bag, dtype=torch.float32, device=device),
                torch.as_tensor(esm_vector, dtype=torch.float32, device=device),
                positive_classes,
            )
            gene = data["genes"][gene_idx]
            for class_idx, score in scores.items():
                signed = score["context_evidence"]
                esm_relevance = score["esm_evidence"]
                positive = np.clip(signed, a_min=0.0, a_max=None)
                denominator = positive.sum() + max(esm_relevance, 0.0)
                normalized = (
                    positive / denominator
                    if denominator > 1e-12
                    else np.zeros_like(positive)
                )
                complex_id = data["class_names"][class_idx]
                for context_idx, cell_id in enumerate(cell_ids[gene_idx]):
                    key = (complex_id, gene, str(cell_id))
                    sums[key] += float(normalized[context_idx])
                    counts[key] += 1
        del model

    rows = [
        {
            "inference_key": model_key,
            "complex_id": complex_id,
            "gene": gene,
            "cell_type": cell_type,
            "complete_positive_lrp": sums[key] / counts[key],
            "n_folds": counts[key],
        }
        for key in sorted(sums)
        for complex_id, gene, cell_type in [key]
    ]
    return pd.DataFrame(rows)


def add_observed_context_coverage(lrp: pd.DataFrame) -> pd.DataFrame:
    _, members_by_complex = complex_members()
    _, global_edges = read_ppi(GLOBAL_PPI)
    member_pairs = {
        complex_id: {
            sorted_edge(a, b)
            for a, b in combinations(sorted(members), 2)
        }
        for complex_id, members in members_by_complex.items()
    }
    reference_edges = {
        complex_id: pairs & global_edges
        for complex_id, pairs in member_pairs.items()
    }
    requested = lrp[["complex_id", "cell_type"]].drop_duplicates()
    metrics = []
    for cell_type, group in requested.groupby("cell_type"):
        ppi_path = CELL_PPI_DIR / f"{cell_type}.txt"
        if not ppi_path.is_file():
            raise FileNotFoundError(f"Missing cell PPI for LRP context: {ppi_path}")
        nodes, edges = read_ppi(ppi_path)
        for complex_id in group["complex_id"]:
            members = members_by_complex[complex_id]
            reference = reference_edges[complex_id]
            metrics.append(
                {
                    "complex_id": complex_id,
                    "cell_type": cell_type,
                    "context_member_coverage": len(members & nodes) / len(members),
                    "context_edge_coverage": len(reference & edges) / len(reference),
                }
            )
        del nodes, edges
    metrics = pd.DataFrame(metrics)
    lrp = lrp.merge(metrics, on=["complex_id", "cell_type"], how="inner")
    return (
        lrp.groupby(
            ["inference_key", "complex_id", "cell_type"], as_index=False
        )
        .agg(
            complete_positive_lrp=("complete_positive_lrp", "mean"),
            context_member_coverage=("context_member_coverage", "first"),
            context_edge_coverage=("context_edge_coverage", "first"),
            n_observed_member_contexts=("gene", "nunique"),
        )
    )


def xmil_tables(
    values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binned_parts = []
    definition_parts = []
    per_complex_parts = []
    correlation_parts = []
    for coverage_column in ("context_member_coverage", "context_edge_coverage"):
        reference = (
            values[values["inference_key"] == "s2gae_bce_uni"]
            [["complex_id", "cell_type", coverage_column]]
            .dropna()
            .drop_duplicates(["complex_id", "cell_type"])
            .sort_values([coverage_column, "complex_id", "cell_type"], kind="mergesort")
        )
        labels = [f"bin_{index}" for index in range(10)]
        bin_numbers = np.minimum(
            np.arange(len(reference), dtype=int) * len(labels) // len(reference),
            len(labels) - 1,
        )
        reference["coverage_bin"] = pd.Categorical(
            [labels[index] for index in bin_numbers], categories=labels, ordered=True
        )
        definitions = (
            reference.groupby("coverage_bin", observed=True)
            .agg(
                lower=(coverage_column, "min"),
                upper=(coverage_column, "max"),
                n_complex_context_pairs=("complex_id", "size"),
            )
            .reindex(labels)
            .reset_index()
        )
        definitions["percentile_midpoint"] = 100.0 * (
            np.arange(len(labels), dtype=float) + 0.5
        ) / len(labels)
        definitions["coverage_metric"] = coverage_column
        definition_parts.append(definitions)

        rows = values.dropna(
            subset=[coverage_column, "complete_positive_lrp"]
        ).copy()
        lookup = reference[["complex_id", "cell_type", "coverage_bin"]].copy()
        lookup["coverage_bin"] = lookup["coverage_bin"].astype(str)
        rows = rows.merge(lookup, on=["complex_id", "cell_type"], how="inner")
        rows["coverage_bin"] = pd.Categorical(
            rows["coverage_bin"], categories=labels, ordered=True
        )
        per_complex_bin = (
            rows.groupby(
                ["inference_key", "complex_id", "coverage_bin"], observed=True
            )["complete_positive_lrp"]
            .mean()
            .rename("aggregated_lrp")
            .reset_index()
            .merge(
                definitions[["coverage_bin", "percentile_midpoint"]],
                on="coverage_bin",
                how="left",
            )
        )
        per_complex_bin["coverage_metric"] = coverage_column
        per_complex_bin["evidence_score"] = "complete_positive_lrp"
        per_complex_parts.append(per_complex_bin)

        summary = (
            per_complex_bin.groupby(
                ["inference_key", "coverage_bin"], observed=True
            )
            .agg(
                mean_lrp=("aggregated_lrp", "mean"),
                n_complexes=("complex_id", "nunique"),
            )
            .reset_index()
            .merge(
                definitions[
                    [
                        "coverage_bin",
                        "lower",
                        "upper",
                        "n_complex_context_pairs",
                        "percentile_midpoint",
                    ]
                ],
                on="coverage_bin",
                how="left",
            )
        )
        summary["coverage_metric"] = coverage_column
        summary["evidence_score"] = "complete_positive_lrp"
        summary["within_complex_aggregation"] = "mean"
        binned_parts.append(summary)

        corr_rows = []
        for (model_key, complex_id), group in values.groupby(
            ["inference_key", "complex_id"]
        ):
            sub = group[[coverage_column, "complete_positive_lrp"]].dropna()
            if len(sub) < 3 or sub[coverage_column].nunique() < 2:
                continue
            constant = sub["complete_positive_lrp"].nunique() < 2
            rho = (
                0.0
                if constant
                else float(
                    spearmanr(
                        sub[coverage_column], sub["complete_positive_lrp"]
                    ).statistic
                )
            )
            corr_rows.append(
                {
                    "inference_key": model_key,
                    "complex_id": complex_id,
                    "coverage_metric": coverage_column,
                    "evidence_score": "complete_positive_lrp",
                    "spearman": rho,
                    "constant_evidence": constant,
                }
            )
        correlations = pd.DataFrame(corr_rows)
        for model_key, group in correlations.groupby("inference_key"):
            rhos = group["spearman"].to_numpy(dtype=float)
            correlation_parts.append(
                {
                    "inference_key": model_key,
                    "inference_label": MODEL_LABELS[model_key],
                    "coverage_metric": coverage_column,
                    "evidence_score": "complete_positive_lrp",
                    "n_complexes": len(group),
                    "n_variable_evidence": int((~group["constant_evidence"]).sum()),
                    "n_constant_evidence": int(group["constant_evidence"].sum()),
                    "median_spearman": float(np.median(rhos)),
                    "fraction_positive_spearman": float(np.mean(rhos > 0)),
                }
            )
    all_per_complex_bins = pd.concat(per_complex_parts, ignore_index=True)
    return (
        pd.concat(binned_parts, ignore_index=True),
        pd.concat(
            [part for part in definition_parts if not part.empty], ignore_index=True
        ),
        all_per_complex_bins,
        pd.DataFrame(correlation_parts),
    )
