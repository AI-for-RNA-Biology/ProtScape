"""Compare Parkinson STRING modularity with degree-preserving random networks."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from downstream_tasks.config import PATHS
from exploration.analysis.parkinson_modules import leiden_communities


ANALYSIS_DIR = Path(PATHS["output_root"]) / "analysis/parkinson_target_analysis"
N_NULL = 1000
NULL_RESTARTS = 5
SWAPS_PER_EDGE = 10
RANDOM_SEED = 20260729
MODULARITY_RESOLUTION = 1.0


def best_modularity(graph, leiden_resolution, seed_offset):
    best_q, best_n_modules = -np.inf, 0
    for restart in range(NULL_RESTARTS):
        communities = leiden_communities(
            graph, resolution=leiden_resolution, seed=seed_offset + restart
        )
        modularity = nx.community.modularity(
            graph, communities, weight="experimental_score",
            resolution=MODULARITY_RESOLUTION,
        )
        if modularity > best_q:
            best_q, best_n_modules = modularity, len(communities)
    return float(best_q), int(best_n_modules)


def modularity_null(graph, n_null, leiden_resolution):
    """Preserve connectivity, node degrees and the global edge-weight distribution."""
    if (
        graph.is_directed() or graph.is_multigraph()
        or len(graph) < 4 or nx.number_of_selfloops(graph)
        or not nx.is_connected(graph)
    ):
        raise ValueError(
            "The connected-swap null requires a connected, simple undirected graph "
            "with at least four nodes. Analyse disconnected networks separately."
        )
    if n_null < 2:
        raise ValueError("At least two null networks are needed to estimate the SD.")

    observed_q, observed_modules = best_modularity(graph, leiden_resolution, 0)
    weights = np.asarray([
        data["experimental_score"] for _, _, data in graph.edges(data=True)
    ])
    rng = np.random.default_rng(RANDOM_SEED)
    n_swaps = SWAPS_PER_EDGE * graph.number_of_edges()
    rows = []
    for null_index in range(n_null):
        null_graph = nx.Graph()
        null_graph.add_nodes_from(graph)
        null_graph.add_edges_from(graph.edges())
        successful_swaps = nx.connected_double_edge_swap(
            null_graph, nswap=n_swaps,
            seed=int(rng.integers(0, 2**32 - 1)),
        )
        for (first, second), weight in zip(null_graph.edges(), rng.permutation(weights)):
            null_graph[first][second]["experimental_score"] = float(weight)
        modularity, n_modules = best_modularity(
            null_graph, leiden_resolution, 10_000 + null_index * NULL_RESTARTS
        )
        rows.append({
            "null_index": null_index,
            "requested_degree_preserving_swap_attempts": n_swaps,
            "successful_degree_preserving_swaps": successful_swaps,
            "modularity": modularity,
            "n_modules": n_modules,
        })

    null = pd.DataFrame(rows)
    null_mean = float(null["modularity"].mean())
    null_std = float(null["modularity"].std(ddof=1))
    summary = {
        "null_comparison_observed_modularity": observed_q,
        "null_comparison_observed_n_modules": observed_modules,
        "null_mean_modularity": null_mean,
        "null_std_modularity": null_std,
        "null_z_score": (observed_q - null_mean) / null_std if null_std > 0 else None,
        "null_empirical_p_value": (1 + int(null["modularity"].ge(observed_q).sum())) / (n_null + 1),
        "n_null_networks": n_null,
        "leiden_restarts_per_observed_and_null_network": NULL_RESTARTS,
        "leiden_resolution_for_null_comparison": leiden_resolution,
        "modularity_resolution_for_cross_resolution_and_null_comparison": MODULARITY_RESOLUTION,
        "null_model": (
            "connected double-edge swaps preserving unweighted degree; "
            "observed experimental scores permuted across rewired edges"
        ),
        "requested_swap_attempts_per_null_network": n_swaps,
        "null_preserves_global_edge_weight_distribution": True,
        "null_preserves_node_strength": False,
    }
    return null, summary


def main():
    edges = pd.read_csv(ANALYSIS_DIR / "parkinson_string_network_edges.csv")
    edges = edges.dropna(subset=["protein_a", "protein_b", "experimental_score"])
    graph = nx.from_pandas_edgelist(
        edges, "protein_a", "protein_b", edge_attr="experimental_score"
    )
    selection = pd.read_csv(ANALYSIS_DIR / "parkinson_leiden_selection.csv")
    resolution = float(selection.loc[selection["selected_resolution"], "resolution"].item())
    null, summary = modularity_null(graph, N_NULL, resolution)
    null.to_csv(ANALYSIS_DIR / "parkinson_modularity_null.csv", index=False)
    (ANALYSIS_DIR / "parkinson_modularity_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
