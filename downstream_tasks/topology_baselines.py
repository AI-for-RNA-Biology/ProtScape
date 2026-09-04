#!/usr/bin/env python
"""Run direct-neighbour voting and finite-depth label propagation.

The held-out proteins remain vertices in every graph, but only the current
training folds are label seeds.  In the ``annotated`` universe, the graph is
induced by the proteins in the frozen downstream split.  In the ``full``
universe, every reference-PPI protein is retained and proteins outside the
downstream task are unlabelled bridge vertices.

``unknown`` is deliberately represented by an all-zero seed row.  It is not
an output class and is never used as a negative label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from .data.global_split import split_fingerprint
from .data.task_loaders import MultiLabelMembershipLoader


FORMAT_VERSION = 3
DEFAULT_DEPTHS = tuple(range(1, 11))
DEFAULT_METHODS = ("direct_neighbor", "label_propagation")
DEFAULT_PROPAGATION_NORMALIZATIONS = ("random_walk", "symmetric")
DEFAULT_UNIVERSES = ("annotated", "full")


@dataclass(frozen=True)
class TaskData:
    genes: Tuple[str, ...]
    labels: np.ndarray
    class_ids: Tuple[str, ...]
    class_labels: Tuple[str, ...]
    sources: Tuple[str, ...]


@dataclass(frozen=True)
class SplitData:
    genes: Tuple[str, ...]
    labels: np.ndarray
    folds: Tuple[np.ndarray, ...]
    class_ids: Tuple[str, ...]
    class_labels: Tuple[str, ...]
    sources: Tuple[str, ...]
    fingerprint: str
    stratification: str


@dataclass(frozen=True)
class GraphData:
    name: str
    universe: str
    genes: Tuple[str, ...]
    adjacency: sparse.csr_matrix
    random_walk_operator: sparse.csr_matrix
    symmetric_operator: sparse.csr_matrix
    components: np.ndarray
    task_to_graph: np.ndarray
    n_retained_edges: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_csv_arg(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _int_csv_arg(value: str) -> Tuple[int, ...]:
    values = tuple(int(item) for item in _split_csv_arg(value))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return values


def load_task_data(path: Path) -> TaskData:
    """Load the strict frozen membership table and its display metadata."""
    genes, labels, class_ids = MultiLabelMembershipLoader(path).load()
    table = pd.read_csv(path, dtype=str)
    for column in table.columns:
        table[column] = table[column].str.strip()
    class_column = "label_id" if "label_id" in table else "label"
    metadata = (
        table[[class_column, "label"] + (["source"] if "source" in table else [])]
        .drop_duplicates()
        .sort_values(class_column)
    )
    by_class: Dict[str, Tuple[str, str]] = {}
    for row in metadata.itertuples(index=False, name=None):
        class_id = str(row[0])
        label = str(row[1])
        source = str(row[2]) if len(row) == 3 else "unspecified"
        previous = by_class.get(class_id)
        if previous is not None and previous != (label, source):
            raise ValueError(
                f"Class metadata is not unique for {class_id!r} in {path}"
            )
        by_class[class_id] = (label, source)
    return TaskData(
        genes=tuple(genes),
        labels=np.asarray(labels, dtype=np.float32),
        class_ids=tuple(class_ids),
        class_labels=tuple(by_class[item][0] for item in class_ids),
        sources=tuple(by_class[item][1] for item in class_ids),
    )


def load_split_data(path: Path, task: TaskData) -> SplitData:
    """Align labels to an existing ProtScape six-fold split artifact."""
    with np.load(path, allow_pickle=False) as archive:
        if "genes" not in archive.files:
            raise ValueError(f"Split archive has no 'genes' array: {path}")
        genes = tuple(str(item).strip().upper() for item in archive["genes"].tolist())
        fold_keys = sorted(
            (key for key in archive.files if key.startswith("fold_")),
            key=lambda item: int(item.split("_", 1)[1]),
        )
        if len(fold_keys) != 6 or fold_keys != [f"fold_{i}" for i in range(6)]:
            raise ValueError(f"Expected exactly fold_0 through fold_5 in {path}")
        folds = tuple(np.asarray(archive[key], dtype=np.int64) for key in fold_keys)
        stratification = (
            str(np.asarray(archive["split_stratification"]).item())
            if "split_stratification" in archive.files
            else "unspecified"
        )
        embedded_fingerprint = (
            str(np.asarray(archive["split_fingerprint"]).item())
            if "split_fingerprint" in archive.files
            else None
        )

    if not genes or len(set(genes)) != len(genes):
        raise ValueError(f"Split genes are empty or duplicated: {path}")
    concatenated = np.concatenate(folds)
    if (
        len(concatenated) != len(genes)
        or np.any(concatenated < 0)
        or np.any(concatenated >= len(genes))
        or not np.array_equal(np.sort(concatenated), np.arange(len(genes)))
    ):
        raise ValueError(f"Split folds do not partition the gene array exactly: {path}")

    task_index = {gene: index for index, gene in enumerate(task.genes)}
    missing = sorted(set(genes).difference(task_index))
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"{len(missing)} split genes are absent from the task table, e.g. {preview}"
        )
    labels = task.labels[[task_index[gene] for gene in genes]]
    fingerprint = split_fingerprint(genes, folds)
    if embedded_fingerprint is not None and embedded_fingerprint != fingerprint:
        raise ValueError(
            f"Embedded split fingerprint does not match genes/folds in {path}"
        )
    return SplitData(
        genes=genes,
        labels=np.asarray(labels, dtype=np.float32),
        folds=folds,
        class_ids=task.class_ids,
        class_labels=task.class_labels,
        sources=task.sources,
        fingerprint=fingerprint,
        stratification=stratification,
    )


def read_undirected_edges(path: Path) -> Tuple[Tuple[str, str], ...]:
    """Read a strict whitespace-delimited two-column undirected edge list."""
    edges = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 2:
                raise ValueError(
                    f"Expected two columns at {path}:{line_number}, got {len(fields)}"
                )
            left, right = (field.strip().upper() for field in fields)
            if not left or not right:
                raise ValueError(f"Empty protein identifier at {path}:{line_number}")
            if left != right:
                edges.add((left, right) if left < right else (right, left))
    if not edges:
        raise ValueError(f"No non-self edges found in {path}")
    return tuple(sorted(edges))


def build_graph(
    *,
    name: str,
    universe: str,
    reference_edges: Sequence[Tuple[str, str]],
    task_genes: Sequence[str],
) -> GraphData:
    """Build a binary sparse graph without converting unknowns to labels."""
    if universe not in {"annotated", "full"}:
        raise ValueError(f"Unknown graph universe: {universe}")

    task_gene_set = set(task_genes)
    if universe == "annotated":
        genes = tuple(task_genes)
    else:
        reference_genes = {gene for edge in reference_edges for gene in edge}
        genes = tuple(sorted(reference_genes | task_gene_set))
    gene_index = {gene: index for index, gene in enumerate(genes)}
    task_to_graph = np.asarray(
        [gene_index[gene] for gene in task_genes], dtype=np.int64
    )

    rows: List[int] = []
    columns: List[int] = []
    values: List[float] = []
    retained = 0
    for left, right in reference_edges:
        if left not in gene_index or right not in gene_index:
            continue
        left_index = gene_index[left]
        right_index = gene_index[right]
        rows.extend((left_index, right_index))
        columns.extend((right_index, left_index))
        values.extend((1.0, 1.0))
        retained += 1
    adjacency = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(len(genes), len(genes)),
        dtype=np.float32,
    )
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    _, components = connected_components(adjacency, directed=False)
    return GraphData(
        name=name,
        universe=universe,
        genes=genes,
        adjacency=adjacency,
        random_walk_operator=normalize_adjacency(adjacency, "random_walk"),
        symmetric_operator=normalize_adjacency(adjacency, "symmetric"),
        components=np.asarray(components, dtype=np.int64),
        task_to_graph=task_to_graph,
        n_retained_edges=retained,
    )


def normalize_adjacency(
    adjacency: sparse.csr_matrix, normalization: str
) -> sparse.csr_matrix:
    """Return D^-1 A or D^-1/2 A D^-1/2 for a binary adjacency."""
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    scale = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    if normalization == "random_walk":
        scale[nonzero] = 1.0 / degree[nonzero]
        return sparse.diags(scale, format="csr") @ adjacency
    if normalization == "symmetric":
        scale[nonzero] = 1.0 / np.sqrt(degree[nonzero])
        diagonal = sparse.diags(scale, format="csr")
        return diagonal @ adjacency @ diagonal
    raise ValueError(f"Unknown propagation normalization: {normalization}")


def make_seed_matrix(
    graph: GraphData,
    labels: np.ndarray,
    train_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create graph-aligned seeds; every non-training vertex stays exactly zero."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    seed = np.zeros((len(graph.genes), labels.shape[1]), dtype=np.float32)
    train_mask = np.zeros(len(graph.genes), dtype=np.float32)
    graph_train = graph.task_to_graph[train_indices]
    train_mask[graph_train] = 1.0
    seed[graph_train] = np.asarray(labels[train_indices], dtype=np.float32)
    return seed, train_mask


def structural_coverage(
    graph: GraphData, train_mask: np.ndarray, evaluation_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return direct-neighbour and connected-to-seed masks for task indices."""
    training_neighbor_count = np.asarray(graph.adjacency @ train_mask).ravel()
    direct = training_neighbor_count[graph.task_to_graph[evaluation_indices]] > 0
    seeded_components = np.unique(graph.components[train_mask > 0])
    reachable = np.isin(
        graph.components[graph.task_to_graph[evaluation_indices]], seeded_components
    )
    return direct, reachable


def propagate_label_depths(
    *,
    graph: GraphData,
    labels: np.ndarray,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    normalization: str,
    depths: Sequence[int],
) -> Dict[int, Tuple[np.ndarray, Dict[str, float]]]:
    """Return F^k Y scores for every requested message-passing depth."""
    requested_depths = tuple(sorted(int(depth) for depth in depths))
    if (
        not requested_depths
        or len(set(requested_depths)) != len(requested_depths)
        or any(depth < 1 for depth in requested_depths)
    ):
        raise ValueError("Label-propagation depths must be unique positive integers")
    if normalization == "random_walk":
        operator = graph.random_walk_operator
    elif normalization == "symmetric":
        operator = graph.symmetric_operator
    else:
        raise ValueError(f"Unknown propagation normalization: {normalization}")

    evaluation_indices = np.asarray(evaluation_indices, dtype=np.int64)
    seed, train_mask = make_seed_matrix(graph, labels, train_indices)
    graph_eval = graph.task_to_graph[evaluation_indices]
    direct, reachable = structural_coverage(graph, train_mask, evaluation_indices)
    coverage = {
        "direct_coverage": float(direct.mean()) if len(direct) else math.nan,
        "reachable_coverage": float(reachable.mean()) if len(reachable) else math.nan,
    }

    scores = seed
    propagated_train_mass = train_mask
    requested = set(requested_depths)
    results: Dict[int, Tuple[np.ndarray, Dict[str, float]]] = {}
    for depth in range(1, requested_depths[-1] + 1):
        scores = np.asarray(operator @ scores, dtype=np.float32)
        propagated_train_mass = np.asarray(
            operator @ propagated_train_mass, dtype=np.float32
        ).ravel()
        if depth in requested:
            depth_covered = propagated_train_mass[graph_eval] > 0
            metadata = {
                **coverage,
                "depth_coverage": (
                    float(depth_covered.mean()) if len(depth_covered) else math.nan
                ),
            }
            results[depth] = (scores[graph_eval].copy(), metadata)
    return results


def propagate_scores(
    *,
    graph: GraphData,
    labels: np.ndarray,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    method: str,
    normalization: str,
    depth: Optional[int],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Score held-out task proteins using training-fold labels only."""
    if method == "label_propagation":
        if depth is None or int(depth) != depth or depth < 1:
            raise ValueError("Label-propagation depth must be a positive integer")
        return propagate_label_depths(
            graph=graph,
            labels=labels,
            train_indices=train_indices,
            evaluation_indices=evaluation_indices,
            normalization=normalization,
            depths=(int(depth),),
        )[int(depth)]

    if method != "direct_neighbor":
        raise ValueError(f"Unknown topology method: {method}")
    if normalization != "none":
        raise ValueError("Direct-neighbour voting does not use a normalization")
    if depth is not None:
        raise ValueError("Direct-neighbour voting does not use a depth")

    evaluation_indices = np.asarray(evaluation_indices, dtype=np.int64)
    seed, train_mask = make_seed_matrix(graph, labels, train_indices)
    graph_eval = graph.task_to_graph[evaluation_indices]
    direct, reachable = structural_coverage(graph, train_mask, evaluation_indices)
    metadata = {
        "direct_coverage": float(direct.mean()) if len(direct) else math.nan,
        "reachable_coverage": float(reachable.mean()) if len(reachable) else math.nan,
    }
    fallback_prevalence = np.asarray(
        labels[np.asarray(train_indices)].mean(axis=0), dtype=np.float32
    )
    numerator = np.asarray(graph.adjacency @ seed, dtype=np.float32)
    denominator = np.asarray(graph.adjacency @ train_mask).ravel()
    scores = np.tile(fallback_prevalence, (len(graph.genes), 1))
    covered = denominator > 0
    scores[covered] = numerator[covered] / denominator[covered, None]
    metadata["depth_coverage"] = metadata["direct_coverage"]
    return scores[graph_eval], metadata


def _macro_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    valid = np.asarray(y_true).sum(axis=0) > 0
    if not valid.any():
        return math.nan
    return float(
        np.mean(
            [
                average_precision_score(y_true[:, index], y_score[:, index])
                for index in np.flatnonzero(valid)
            ]
        )
    )


def _macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    values = []
    for index in range(y_true.shape[1]):
        if np.unique(y_true[:, index]).size == 2:
            values.append(roc_auc_score(y_true[:, index], y_score[:, index]))
    return float(np.mean(values)) if values else math.nan


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "auprc_macro": _macro_average_precision(y_true, y_score),
        "auprc_micro": float(
            average_precision_score(y_true.ravel(), y_score.ravel())
        ),
        "auroc_macro": _macro_auroc(y_true, y_score),
    }


def _depth_grid(method: str, depths: Sequence[int]) -> Tuple[Optional[int], ...]:
    if method == "direct_neighbor":
        return (None,)
    if method == "label_propagation":
        return tuple(int(value) for value in depths)
    raise ValueError(f"Unknown topology method: {method}")


def _baseline_configurations(
    propagation_normalizations: Sequence[str],
) -> Tuple[Tuple[str, str], ...]:
    return (("direct_neighbor", "none"),) + tuple(
        ("label_propagation", normalization)
        for normalization in propagation_normalizations
    )


def _depth_text(depth: Optional[int]) -> str:
    return "none" if depth is None else str(depth)


def _cv_indices(folds: Sequence[np.ndarray], validation_rotation: int):
    validation_fold = validation_rotation + 1
    validation = np.asarray(folds[validation_fold], dtype=np.int64)
    training = np.concatenate(
        [
            np.asarray(folds[index], dtype=np.int64)
            for index in range(1, 6)
            if index != validation_fold
        ]
    )
    test = np.asarray(folds[0], dtype=np.int64)
    return training, validation, test, validation_fold


def _per_label_rows(
    *,
    split: SplitData,
    truth: np.ndarray,
    scores: np.ndarray,
    base: Mapping[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, class_id in enumerate(split.class_ids):
        label_truth = truth[:, index]
        prevalence = float(label_truth.mean())
        ap = (
            float(average_precision_score(label_truth, scores[:, index]))
            if label_truth.sum() > 0
            else math.nan
        )
        row = dict(base)
        row.update(
            {
                "class_id": class_id,
                "class_label": split.class_labels[index],
                "source": split.sources[index],
                "n_positive": int(label_truth.sum()),
                "prevalence": prevalence,
                "average_precision": ap,
                "ap_lift_over_prevalence": (
                    ap / prevalence if prevalence > 0 and not math.isnan(ap) else math.nan
                ),
            }
        )
        rows.append(row)
    return rows


def run_baselines(
    *,
    task_csv: Path,
    split_indices: Path,
    global_ppi: Path,
    output_dir: Path,
    universes: Sequence[str] = DEFAULT_UNIVERSES,
    propagation_normalizations: Sequence[str] = DEFAULT_PROPAGATION_NORMALIZATIONS,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    save_predictions: bool = False,
) -> Dict[str, Path]:
    """Run validation selection and fixed-test evaluation for all baselines."""
    invalid_universes = sorted(set(universes).difference(DEFAULT_UNIVERSES))
    if invalid_universes:
        raise ValueError(f"Invalid graph universes: {invalid_universes}")
    invalid_normalizations = sorted(
        set(propagation_normalizations).difference(
            DEFAULT_PROPAGATION_NORMALIZATIONS
        )
    )
    if invalid_normalizations or not propagation_normalizations:
        raise ValueError(
            "Invalid propagation normalizations: "
            f"{invalid_normalizations or propagation_normalizations}"
        )
    if not depths or any(int(depth) != depth or depth < 1 for depth in depths):
        raise ValueError("Label-propagation depths must be positive integers")
    if len(set(depths)) != len(depths):
        raise ValueError("Label-propagation depths must be unique")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )

    task = load_task_data(task_csv)
    split = load_split_data(split_indices, task)
    reference_edges = read_undirected_edges(global_ppi)
    graphs = [
        build_graph(
            name="global_binary",
            universe=universe,
            reference_edges=reference_edges,
            task_genes=split.genes,
        )
        for universe in universes
    ]

    validation_rows: List[Dict[str, object]] = []
    propagation_cache: Dict[
        Tuple[str, str, str, int], Dict[int, Tuple[np.ndarray, Dict[str, float]]]
    ] = {}
    for graph in graphs:
        for method, normalization in _baseline_configurations(
            propagation_normalizations
        ):
            for depth in _depth_grid(method, depths):
                for rotation in range(5):
                    train, validation, _, validation_fold = _cv_indices(
                        split.folds, rotation
                    )
                    if method == "label_propagation":
                        cache_key = (
                            graph.name,
                            graph.universe,
                            normalization,
                            rotation,
                        )
                        if cache_key not in propagation_cache:
                            propagation_cache[cache_key] = propagate_label_depths(
                                graph=graph,
                                labels=split.labels,
                                train_indices=train,
                                evaluation_indices=validation,
                                normalization=normalization,
                                depths=depths,
                            )
                        scores, propagation = propagation_cache[cache_key][int(depth)]
                    else:
                        scores, propagation = propagate_scores(
                            graph=graph,
                            labels=split.labels,
                            train_indices=train,
                            evaluation_indices=validation,
                            method=method,
                            normalization=normalization,
                            depth=depth,
                        )
                    metrics = compute_metrics(split.labels[validation], scores)
                    validation_rows.append(
                        {
                            "graph": graph.name,
                            "universe": graph.universe,
                            "method": method,
                            "normalization": normalization,
                            "depth": depth,
                            "validation_rotation": rotation,
                            "validation_fold": validation_fold,
                            "n_train": len(train),
                            "n_validation": len(validation),
                            **metrics,
                            **propagation,
                        }
                    )

    validation_table = pd.DataFrame(validation_rows)
    validation_table["depth"] = validation_table["depth"].astype("Int64")
    selection_rows: List[Dict[str, object]] = []
    selected: Dict[Tuple[str, str, str], Tuple[str, Optional[int]]] = {}
    group_columns = ["graph", "universe", "method", "normalization", "depth"]
    summaries = (
        validation_table.groupby(group_columns, sort=True, dropna=False)
        .agg(
            validation_auprc_macro_mean=("auprc_macro", "mean"),
            validation_auprc_macro_std=("auprc_macro", "std"),
            validation_auprc_micro_mean=("auprc_micro", "mean"),
            validation_auroc_macro_mean=("auroc_macro", "mean"),
        )
        .reset_index()
    )
    for keys, group in summaries.groupby(
        ["graph", "universe", "method"], sort=True
    ):
        best = group.sort_values(
            ["validation_auprc_macro_mean", "depth", "normalization"],
            ascending=[False, True, True],
            kind="mergesort",
        ).iloc[0]
        selected_depth = None if pd.isna(best["depth"]) else int(best["depth"])
        selected[tuple(str(item) for item in keys)] = (
            str(best["normalization"]),
            selected_depth,
        )
        selection_rows.append(best.to_dict())

    test_rows: List[Dict[str, object]] = []
    per_label_rows: List[Dict[str, object]] = []
    ensemble_rows: List[Dict[str, object]] = []
    ensemble_per_label_rows: List[Dict[str, object]] = []
    prediction_payload: Dict[str, np.ndarray] = {}
    graph_lookup = {(graph.name, graph.universe): graph for graph in graphs}
    for (graph_name, universe, method), (normalization, depth) in selected.items():
        graph = graph_lookup[(graph_name, universe)]
        rotation_test_scores: List[np.ndarray] = []
        for rotation in range(5):
            train, validation, test, validation_fold = _cv_indices(
                split.folds, rotation
            )
            depth_text = _depth_text(depth)
            scores, propagation = propagate_scores(
                graph=graph,
                labels=split.labels,
                train_indices=train,
                evaluation_indices=test,
                method=method,
                normalization=normalization,
                depth=depth,
            )
            rotation_test_scores.append(np.asarray(scores, dtype=np.float32))
            metrics = compute_metrics(split.labels[test], scores)
            base: Dict[str, object] = {
                "graph": graph_name,
                "universe": universe,
                "method": method,
                "normalization": normalization,
                "selected_depth": depth,
                "validation_rotation": rotation,
                "validation_fold": validation_fold,
                "test_fold": 0,
                "n_train": len(train),
                "n_validation": len(validation),
                "n_test": len(test),
            }
            test_rows.append({**base, **metrics, **propagation})
            per_label_rows.extend(
                _per_label_rows(
                    split=split,
                    truth=split.labels[test],
                    scores=scores,
                    base=base,
                )
            )
            if save_predictions:
                prefix = "__".join(
                    (
                        graph_name,
                        universe,
                        method,
                        normalization,
                        depth_text,
                        str(rotation),
                    )
                )
                prediction_payload[f"{prefix}__scores"] = scores.astype(np.float32)

        if len(rotation_test_scores) != 5:
            raise RuntimeError(
                "Expected exactly five fixed-test score matrices for the ensemble"
            )
        ensemble_scores = np.mean(
            np.stack(rotation_test_scores, axis=0), axis=0, dtype=np.float64
        ).astype(np.float32)
        ensemble_metrics = compute_metrics(split.labels[test], ensemble_scores)
        ensemble_base: Dict[str, object] = {
            "graph": graph_name,
            "universe": universe,
            "method": method,
            "normalization": normalization,
            "selected_depth": depth,
            "test_fold": 0,
            "n_test": len(test),
            "n_ensemble_members": len(rotation_test_scores),
            "ensemble_aggregation": "arithmetic_mean_of_rotation_scores",
        }
        ensemble_rows.append({**ensemble_base, **ensemble_metrics})
        ensemble_per_label_rows.extend(
            _per_label_rows(
                split=split,
                truth=split.labels[test],
                scores=ensemble_scores,
                base=ensemble_base,
            )
        )
        if save_predictions:
            ensemble_prefix = "__".join(
                (
                    graph_name,
                    universe,
                    method,
                    normalization,
                    _depth_text(depth),
                    "ensemble",
                )
            )
            prediction_payload[f"{ensemble_prefix}__scores"] = ensemble_scores

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "validation": output_dir / "validation_results.csv",
        "validation_summary": output_dir / "validation_summary.csv",
        "selection": output_dir / "selected_hyperparameters.csv",
        "test": output_dir / "test_fold_results.csv",
        "per_label": output_dir / "test_per_label.csv",
        "ensemble": output_dir / "test_ensemble_results.csv",
        "ensemble_per_label": output_dir / "test_ensemble_per_label.csv",
        "manifest": output_dir / "manifest.json",
    }
    validation_table.sort_values(group_columns + ["validation_rotation"]).to_csv(
        paths["validation"], index=False
    )
    summaries.sort_values(group_columns).to_csv(
        paths["validation_summary"], index=False
    )
    pd.DataFrame(selection_rows).sort_values(
        ["graph", "universe", "method", "normalization"]
    ).to_csv(paths["selection"], index=False)
    pd.DataFrame(test_rows).sort_values(
        ["graph", "universe", "method", "normalization", "validation_rotation"]
    ).to_csv(paths["test"], index=False)
    pd.DataFrame(per_label_rows).sort_values(
        [
            "graph",
            "universe",
            "method",
            "normalization",
            "validation_rotation",
            "class_id",
        ]
    ).to_csv(paths["per_label"], index=False)
    pd.DataFrame(ensemble_rows).sort_values(
        ["graph", "universe", "method", "normalization"]
    ).to_csv(paths["ensemble"], index=False)
    pd.DataFrame(ensemble_per_label_rows).sort_values(
        ["graph", "universe", "method", "normalization", "class_id"]
    ).to_csv(paths["ensemble_per_label"], index=False)
    if save_predictions:
        prediction_path = output_dir / "test_predictions.npz"
        np.savez_compressed(
            prediction_path,
            genes=np.asarray(split.genes),
            test_indices=np.asarray(split.folds[0], dtype=np.int64),
            test_genes=np.asarray([split.genes[index] for index in split.folds[0]]),
            y_true=np.asarray(split.labels[split.folds[0]], dtype=np.float32),
            class_ids=np.asarray(split.class_ids),
            **prediction_payload,
        )
        paths["predictions"] = prediction_path

    manifest = {
        "format_version": FORMAT_VERSION,
        "analysis": "topology_only_multilabel_baselines",
        "unknown_semantics": (
            "non-training and external proteins have an all-zero graph state; "
            "unknown is never an output class"
        ),
        "split_protocol": "fold_0_test_folds_1_to_5_rotate_validation",
        "selection_metric": "mean_validation_macro_average_precision",
        "label_propagation_selection": (
            "normalization and depth selected jointly within each graph universe"
        ),
        "test_selection_rule": "test evaluated only for validation-selected parameters",
        "test_rotation_warning": (
            "the five scores reuse fixed test proteins and are correlated"
        ),
        "test_ensemble_policy": (
            "arithmetic mean of the five fixed-test score matrices produced by the "
            "validation-selected configuration"
        ),
        "task_csv": str(task_csv.resolve()),
        "task_csv_sha256": _sha256_file(task_csv),
        "split_indices": str(split_indices.resolve()),
        "split_indices_sha256": _sha256_file(split_indices),
        "split_fingerprint": split.fingerprint,
        "split_stratification": split.stratification,
        "global_ppi": str(global_ppi.resolve()),
        "global_ppi_sha256": _sha256_file(global_ppi),
        "self_loop_policy": "drop",
        "unreachable_policy": (
            "label propagation keeps zero scores; direct-neighbour voting uses "
            "training prevalence when no training neighbour is present"
        ),
        "n_task_csv_genes": len(task.genes),
        "n_split_genes": len(split.genes),
        "n_task_csv_genes_excluded_from_split": len(task.genes) - len(split.genes),
        "n_labels": len(split.class_ids),
        "n_reference_edges": len(reference_edges),
        "graphs": [
            {
                "graph": graph.name,
                "universe": graph.universe,
                "n_nodes": len(graph.genes),
                "n_edges": graph.n_retained_edges,
            }
            for graph in graphs
        ],
        "methods": list(DEFAULT_METHODS),
        "propagation_normalizations": list(propagation_normalizations),
        "depths": list(depths),
        "label_propagation_filter": "H_k = F^k Y_train",
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "code": {
            "git_commit": os.environ.get("PROTSCAPE_GIT_COMMIT", "uncommitted"),
            "fingerprint": os.environ.get(
                "PROTSCAPE_CODE_FINGERPRINT", "unspecified"
            ),
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--global-ppi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--universes", type=_split_csv_arg, default=DEFAULT_UNIVERSES)
    parser.add_argument(
        "--propagation-normalizations",
        type=_split_csv_arg,
        default=DEFAULT_PROPAGATION_NORMALIZATIONS,
    )
    parser.add_argument("--depths", type=_int_csv_arg, default=DEFAULT_DEPTHS)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = run_baselines(
        task_csv=args.task_csv,
        split_indices=args.split_indices,
        global_ppi=args.global_ppi,
        output_dir=args.output_dir,
        universes=args.universes,
        propagation_normalizations=args.propagation_normalizations,
        depths=args.depths,
        save_predictions=args.save_predictions,
    )
    for name, path in sorted(paths.items()):
        print(f"[OK] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
