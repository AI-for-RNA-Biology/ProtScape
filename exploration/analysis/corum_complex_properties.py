"""Compute CORUM dataset, topology and coverage statistics."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from downstream_tasks.config import PATHS
from exploration.analysis.corum_model_evaluation import (
    MAIN_MODEL_ORDER,
    MODEL_LABELS,
)


CORUM_DIR = Path(PATHS["corum_dataset_dir"])
CORUM_COMPLEXES = CORUM_DIR / "corum_complexes_filtered.csv"
CORUM_MEMBERSHIPS = CORUM_DIR / "corum_memberships_filtered.csv"
GLOBAL_PPI = Path(PATHS["networks_bulk"]) / "global_ppi_edgelist.txt"
CELL_PPI_DIR = GLOBAL_PPI.parent / "ppi_edgelists"


DRIVER_SPECS = [
    ("complex_size", "n_members_in_ppi_universe", "Complex size", False),
    (
        "cell_ppi_member_coverage",
        "mean_cell_ppi_member_coverage",
        "Cell-PPI node coverage",
        True,
    ),
    (
        "cell_ppi_edge_coverage",
        "mean_cell_ppi_edge_coverage",
        "Cell-PPI edge coverage",
        True,
    ),
    (
        "global_ppi_edge_density",
        "induced_ppi_edge_density",
        "Global PPI edge density",
        True,
    ),
]


def dataset_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complexes = pd.read_csv(CORUM_COMPLEXES, dtype={"complex_id": str})
    memberships = pd.read_csv(CORUM_MEMBERSHIPS, dtype={"complex_id": str})
    valid_ids = set(complexes["complex_id"])
    memberships["protein"] = memberships["protein"].astype(str).str.upper()
    memberships = memberships[memberships["complex_id"].isin(valid_ids)].drop_duplicates(
        ["protein", "complex_id"]
    )
    per_protein = memberships.groupby("protein")["complex_id"].nunique()
    positives = int(len(memberships))
    n_proteins = int(per_protein.size)
    n_complexes = int(complexes["complex_id"].nunique())
    stats = pd.DataFrame(
        [
            {
                "scope": "filtered_corum_dataset",
                "n_proteins": n_proteins,
                "n_complexes": n_complexes,
                "n_complexes_with_positives": int(memberships["complex_id"].nunique()),
                "n_positive_memberships": positives,
                "n_negative_memberships": n_proteins * n_complexes - positives,
                "positive_label_fraction": positives / (n_proteins * n_complexes),
                "mean_complexes_per_protein": float(per_protein.mean()),
                "median_complexes_per_protein": float(per_protein.median()),
                "max_complexes_per_protein": int(per_protein.max()),
                "label_density_percent": 100.0 * positives / (n_proteins * n_complexes),
            }
        ]
    )
    size_distribution = (
        complexes["n_members_in_ppi_universe"]
        .value_counts(sort=False)
        .sort_index()
        .rename_axis("n_members")
        .reset_index(name="n_complexes")
    )
    per_protein_distribution = (
        per_protein.value_counts(sort=False)
        .sort_index()
        .rename_axis("complexes_per_protein")
        .reset_index(name="n_proteins")
    )
    return stats, size_distribution, per_protein_distribution


def sorted_edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def read_ppi(path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0].upper(), parts[1].upper()
            nodes.update((a, b))
            if a != b:
                edges.add(sorted_edge(a, b))
    return nodes, edges


def complex_members() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    complexes = pd.read_csv(CORUM_COMPLEXES, dtype={"complex_id": str})
    members = {
        row.complex_id: {
            gene.strip().upper()
            for gene in str(row.member_hgnc).split(";")
            if gene.strip()
        }
        for row in complexes.itertuples(index=False)
    }
    return complexes, members


def build_topology_and_cell_coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    complexes, members_by_complex = complex_members()
    _, global_edges = read_ppi(GLOBAL_PPI)
    pairs = {
        complex_id: {
            sorted_edge(a, b)
            for a, b in combinations(sorted(members), 2)
        }
        for complex_id, members in members_by_complex.items()
    }
    reference_edges = {
        complex_id: member_pairs & global_edges
        for complex_id, member_pairs in pairs.items()
    }

    topology_rows = []
    for row in complexes.itertuples(index=False):
        complex_id = str(row.complex_id)
        n_members = len(members_by_complex[complex_id])
        n_possible = len(pairs[complex_id])
        n_edges = len(reference_edges[complex_id])
        topology_rows.append(
            {
                "complex_id": complex_id,
                "complex_name": row.complex_name,
                "n_members_in_ppi_universe": n_members,
                "n_possible_ppi_edges": n_possible,
                "n_induced_ppi_edges": n_edges,
                "induced_ppi_edge_density": n_edges / n_possible,
            }
        )

    member_coverage_sum = defaultdict(float)
    edge_coverage_sum = defaultdict(float)
    n_cell_ppis = 0
    for ppi_path in sorted(CELL_PPI_DIR.glob("*.txt")):
        nodes, edges = read_ppi(ppi_path)
        n_cell_ppis += 1
        for complex_id, members in members_by_complex.items():
            member_coverage_sum[complex_id] += len(members & nodes) / len(members)
            reference = reference_edges[complex_id]
            edge_coverage_sum[complex_id] += len(reference & edges) / len(reference)
        del nodes, edges

    coverage_rows = [
        {
            "complex_id": complex_id,
            "mean_cell_ppi_member_coverage": member_coverage_sum[complex_id]
            / n_cell_ppis,
            "mean_cell_ppi_edge_coverage": edge_coverage_sum[complex_id]
            / n_cell_ppis,
            "n_cell_ppis": n_cell_ppis,
        }
        for complex_id in members_by_complex
    ]
    return pd.DataFrame(topology_rows), pd.DataFrame(coverage_rows)


def equal_count_bins(
    rows: pd.DataFrame,
    value_column: str,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_column]).copy()
    rows = rows.sort_values([value_column, "complex_id"], kind="mergesort")
    labels = [f"bin_{index}" for index in range(n_bins)]
    bin_numbers = np.minimum(
        np.arange(len(rows), dtype=int) * n_bins // len(rows), n_bins - 1
    )
    rows["driver_bin"] = pd.Categorical(
        [labels[index] for index in bin_numbers], categories=labels, ordered=True
    )
    return rows, labels


def driver_tables(
    per_complex: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluable = set(
        per_complex.loc[
            per_complex["inference_key"].eq("s2gae_bce_uni")
            & per_complex["n_test_pos"].gt(0)
            & per_complex["mean_test_auprc"].notna(),
            "complex_id",
        ]
    )
    per_complex = per_complex[
        per_complex["inference_key"].isin(MAIN_MODEL_ORDER)
        & per_complex["complex_id"].isin(evaluable)
    ].copy()
    summaries = []
    bin_tables = []
    correlations = []
    for driver_key, column, _label, is_percent in DRIVER_SPECS:
        unique = per_complex[["complex_id", column]].drop_duplicates("complex_id")
        unique, labels = equal_count_bins(unique, column)
        bin_lookup = dict(zip(unique["complex_id"], unique["driver_bin"].astype(str)))
        rows = per_complex.copy()
        rows["driver_bin"] = pd.Categorical(
            rows["complex_id"].map(bin_lookup), categories=labels, ordered=True
        )
        observed = (
            unique.groupby("driver_bin", observed=True)
            .agg(
                lower=(column, "min"),
                upper=(column, "max"),
                x=(column, "mean"),
                n_complexes=("complex_id", "nunique"),
            )
            .reindex(labels)
        )
        if is_percent:
            display = [
                f"{int(round(100 * row.lower))}-{int(round(100 * row.upper))}%"
                for row in observed.itertuples()
            ]
        else:
            display = [
                f"{int(round(row.lower))}-{int(round(row.upper))}"
                for row in observed.itertuples()
            ]
        bins = pd.DataFrame(
            {
                "driver": driver_key,
                "driver_bin": labels,
                "bin_label": display,
                "lower": observed["lower"].to_numpy(dtype=float),
                "upper": observed["upper"].to_numpy(dtype=float),
                "x": observed["x"].to_numpy(dtype=float),
                "n_complexes": observed["n_complexes"].to_numpy(dtype=int),
                "percentile_midpoint": 100.0
                * (np.arange(len(labels), dtype=float) + 0.5)
                / len(labels),
            }
        )
        bin_tables.append(bins)

        available = rows.dropna(subset=["driver_bin", "mean_test_auprc"])
        summary = (
            available.groupby(["inference_key", "driver_bin"], observed=True)
            .agg(
                mean_score=("mean_test_auprc", "mean"),
                n_complexes=("complex_id", "nunique"),
            )
            .reset_index()
        )
        summary["driver"] = driver_key
        summaries.append(summary)

        for model_key, group in available.groupby("inference_key"):
            sub = group[[column, "mean_test_auprc"]].dropna()
            statistic = spearmanr(sub[column], sub["mean_test_auprc"])
            correlations.append(
                {
                    "driver": driver_key,
                    "inference_key": model_key,
                    "inference_label": MODEL_LABELS[model_key],
                    "n_complexes": len(sub),
                    "spearman": float(statistic.statistic),
                    "spearman_p": float(statistic.pvalue),
                }
            )
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(bin_tables, ignore_index=True),
        pd.DataFrame(correlations),
    )
