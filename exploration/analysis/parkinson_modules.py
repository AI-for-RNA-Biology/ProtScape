"""Select Leiden communities and summarize their Reactome enrichment."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from exploration.analysis.parkinson_string import benjamini_hochberg


MODULE_ANCHORS = ["DRD2", "GABRB3", "GRIN1", "GRM5", "SCN2A", "POLE"]
MODULE_LABELS = {
    1: "Class-A GPCR and monoaminergic receptors",
    2: "GABA/cholinergic ligand-gated receptors",
    3: "Ionotropic glutamate/NMDA receptor signalling",
    4: "Metabotropic glutamate and Class-C GPCR signalling",
    5: "Voltage-gated ion channels and excitability",
    6: "DNA replication/repair",
}
LEIDEN_RESOLUTIONS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
LEIDEN_SEEDS = tuple(range(10))


def leiden_communities(
    graph: nx.Graph, resolution: float, seed: int
) -> list[set[str]]:
    import igraph as ig
    import leidenalg

    nodes = sorted(graph)
    node_index = {node: index for index, node in enumerate(nodes)}
    edges = list(graph.edges())
    igraph_graph = ig.Graph(
        n=len(nodes),
        edges=[
            (node_index[first], node_index[second]) for first, second in edges
        ],
        directed=False,
    )
    igraph_graph.es["weight"] = [
        float(graph[first][second]["experimental_score"])
        for first, second in edges
    ]
    partition = leidenalg.find_partition(
        igraph_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=seed,
    )
    return [{nodes[index] for index in community} for community in partition]


def min_max_normalize(values: pd.Series) -> pd.Series:
    value_range = values.max() - values.min()
    if np.isclose(value_range, 0.0):
        return pd.Series(1.0, index=values.index)
    return (values - values.min()) / value_range


def select_leiden_partition(
    graph: nx.Graph,
) -> tuple[list[set[str]], pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import adjusted_rand_score

    nodes = sorted(graph)
    runs_by_resolution = {}
    resolution_rows = []
    stability_rows = []
    for resolution in LEIDEN_RESOLUTIONS:
        runs = []
        for seed in LEIDEN_SEEDS:
            communities = leiden_communities(graph, resolution, seed)
            labels = {
                node: community_index
                for community_index, community in enumerate(communities)
                for node in community
            }
            runs.append(
                {
                    "resolution": resolution,
                    "seed": seed,
                    "communities": communities,
                    "labels": np.asarray([labels[node] for node in nodes]),
                    "modularity": nx.community.modularity(
                        graph,
                        communities,
                        weight="experimental_score",
                        resolution=1.0,
                    ),
                    "n_modules": len(communities),
                }
            )
        agreement = np.eye(len(runs))
        pairwise = []
        for first in range(len(runs)):
            for second in range(first + 1, len(runs)):
                ari = adjusted_rand_score(
                    runs[first]["labels"], runs[second]["labels"]
                )
                agreement[first, second] = agreement[second, first] = ari
                pairwise.append(ari)
        mean_ari = (agreement.sum(axis=1) - 1.0) / (len(runs) - 1)
        for run, value in zip(runs, mean_ari):
            run["mean_ari"] = float(value)
        runs_by_resolution[resolution] = runs
        module_counts = [run["n_modules"] for run in runs]
        resolution_rows.append(
            {
                "resolution": resolution,
                "mean_modularity": np.mean(
                    [run["modularity"] for run in runs]
                ),
                "mean_pairwise_ari": np.mean(pairwise),
                "mean_n_modules": np.mean(module_counts),
                "minimum_n_modules": min(module_counts),
                "maximum_n_modules": max(module_counts),
                "eligible_more_than_one_module_in_every_run": (
                    min(module_counts) > 1
                ),
            }
        )
    selection = pd.DataFrame(resolution_rows)
    eligible = selection[
        selection["eligible_more_than_one_module_in_every_run"]
    ].copy()
    eligible["mean_modularity_normalized"] = min_max_normalize(
        eligible["mean_modularity"]
    )
    eligible["mean_pairwise_ari_normalized"] = min_max_normalize(
        eligible["mean_pairwise_ari"]
    )
    eligible["combined_score"] = 0.5 * (
        eligible["mean_modularity_normalized"]
        + eligible["mean_pairwise_ari_normalized"]
    )
    selected_resolution = float(
        eligible.sort_values(
            [
                "combined_score",
                "mean_pairwise_ari",
                "mean_modularity",
                "resolution",
            ],
            ascending=[False, False, False, True],
        ).iloc[0]["resolution"]
    )
    for column in [
        "mean_modularity_normalized",
        "mean_pairwise_ari_normalized",
        "combined_score",
    ]:
        selection[column] = selection["resolution"].map(
            eligible.set_index("resolution")[column]
        )
    selection["selected_resolution"] = selection["resolution"].eq(
        selected_resolution
    )

    selected_runs = runs_by_resolution[selected_resolution]
    best_ari = max(run["mean_ari"] for run in selected_runs)
    medoid_candidates = [
        index
        for index, run in enumerate(selected_runs)
        if np.isclose(run["mean_ari"], best_ari, rtol=0.0, atol=1e-12)
    ]
    medoid_index = max(
        medoid_candidates,
        key=lambda index: (
            selected_runs[index]["modularity"],
            -selected_runs[index]["seed"],
        ),
    )
    for resolution, runs in runs_by_resolution.items():
        for index, run in enumerate(runs):
            stability_rows.append(
                {
                    "resolution": resolution,
                    "seed": run["seed"],
                    "modularity": run["modularity"],
                    "n_modules": run["n_modules"],
                    "mean_ari_to_other_runs_at_same_resolution": run[
                        "mean_ari"
                    ],
                    "selected_resolution": resolution == selected_resolution,
                    "selected_medoid_partition": (
                        resolution == selected_resolution
                        and index == medoid_index
                    ),
                }
            )
    return (
        selected_runs[medoid_index]["communities"],
        selection,
        pd.DataFrame(stability_rows),
    )


def order_modules(communities: list[set[str]]) -> list[set[str]]:
    ordered = []
    for anchor in MODULE_ANCHORS:
        match = next(
            (community for community in communities if anchor in community),
            None,
        )
        if match is not None and match not in ordered:
            ordered.append(match)
    ordered.extend(
        sorted(
            (
                community
                for community in communities
                if community not in ordered
            ),
            key=lambda community: (-len(community), min(community)),
        )
    )
    return ordered


def build_leiden_tables(
    network_nodes: pd.DataFrame,
    edges: pd.DataFrame,
    membership: pd.DataFrame,
    candidates: pd.DataFrame,
    support: pd.DataFrame,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    graph = nx.from_pandas_edgelist(
        edges,
        "protein_a",
        "protein_b",
        edge_attr=["experimental_score", "combined_score"],
    )
    connected = network_nodes[network_nodes["connected"]].copy()
    if set(graph) != set(connected["protein"]):
        raise ValueError("Connected STRING nodes and edge endpoints disagree")
    medoid, selection, stability = select_leiden_partition(graph)
    modules = order_modules(medoid)
    if len(modules) != len(MODULE_LABELS):
        raise RuntimeError(
            f"Expected six Leiden communities, found {len(modules)}"
        )
    cluster_by_protein = {
        protein: module_id
        for module_id, module in enumerate(modules, start=1)
        for protein in module
    }
    connected["leiden_cluster"] = connected["protein"].map(
        cluster_by_protein
    )
    connected["leiden_cluster_label"] = connected["leiden_cluster"].map(
        MODULE_LABELS
    )
    connected = connected.merge(
        membership[["protein", "benchmark_label", "benchmark_split"]],
        on="protein",
        how="left",
        validate="one_to_one",
    )
    prot_candidates = candidates[candidates["model"].eq("protscape")][
        ["protein", "candidate_rank", "mean_probability"]
    ].rename(
        columns={
            "candidate_rank": "protscape_candidate_rank",
            "mean_probability": "protscape_mean_probability",
        }
    )
    connected = connected.merge(
        prot_candidates,
        on="protein",
        how="left",
        validate="one_to_one",
    )
    support_columns = [
        "protein",
        "current_opentargets_parkinson_association_non_literature_only",
        "approved_human_drugbank_target_any_indication",
        "other_opentargets_disease_association",
    ]
    prot_support = support[support["model"].eq("protscape")][support_columns]
    connected = connected.merge(
        prot_support, on="protein", how="left", validate="one_to_one"
    )
    for column in support_columns[1:]:
        connected[column] = connected[column].fillna(False).astype(bool)
    return graph, connected, selection, stability


def build_module_reactome_enrichment(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    string_by_protein = nodes.set_index("protein")["string_id"].to_dict()
    protein_by_string = {
        string_id: protein
        for protein, string_id in string_by_protein.items()
    }
    network_ids = set(protein_by_string)
    reactome = annotations[
        annotations["source"].eq("Reactome")
        & annotations["string_id"].isin(network_ids)
    ].copy()
    reactome["protein"] = reactome["string_id"].map(protein_by_string)
    background = set(graph)
    background_size = len(background)
    rows = []
    for (term, description), term_rows in reactome.groupby(
        ["term", "description"]
    ):
        members = set(term_rows["protein"]) & background
        if not (3 <= len(members) <= 0.8 * background_size):
            continue
        for module_id in sorted(MODULE_LABELS):
            module = set(
                nodes.loc[
                    nodes["leiden_cluster"].eq(module_id), "protein"
                ]
            )
            hits = sorted(module & members)
            fold = (len(hits) / len(module)) / (
                len(members) / background_size
            )
            rows.append(
                {
                    "leiden_cluster": module_id,
                    "leiden_cluster_label": MODULE_LABELS[module_id],
                    "module_size": len(module),
                    "term": term,
                    "description": description,
                    "term_network_size": len(members),
                    "module_hits": len(hits),
                    "module_hit_genes": ";".join(hits),
                    "fold_enrichment": fold,
                    "p_value": hypergeom.sf(
                        len(hits) - 1,
                        background_size,
                        len(members),
                        len(module),
                    ),
                }
            )
    enrichment = pd.DataFrame(rows)
    enrichment["fdr"] = benjamini_hochberg(enrichment["p_value"])
    enrichment["minus_log10_fdr"] = -np.log10(
        enrichment["fdr"].clip(lower=1e-300)
    )
    enrichment["significant"] = (
        enrichment["fdr"].lt(0.05)
        & enrichment["module_hits"].ge(3)
        & enrichment["fold_enrichment"].gt(1)
    )
    displayed = (
        enrichment[enrichment["significant"]]
        .sort_values(
            [
                "leiden_cluster",
                "fdr",
                "fold_enrichment",
                "module_hits",
                "term",
            ],
            ascending=[True, True, False, False, True],
        )
        .groupby("leiden_cluster", as_index=False)
        .head(5)
        .sort_values("leiden_cluster")
        .reset_index(drop=True)
    )
    if set(displayed["leiden_cluster"]) != set(MODULE_LABELS):
        raise RuntimeError(
            "One or more Leiden clusters lack five displayable Reactome terms"
        )
    return enrichment, displayed


def build_module_summary(nodes: pd.DataFrame) -> pd.DataFrame:
    summary = (
        nodes.assign(
            protscape_protein=nodes["node_role"].eq("candidate"),
            known_target_protein=nodes["node_role"].eq(
                "benchmark_positive"
            ),
        )
        .groupby("leiden_cluster", as_index=False)
        .agg(
            total_proteins=("protein", "size"),
            protscape_proteins=("protscape_protein", "sum"),
            known_target_proteins=("known_target_protein", "sum"),
        )
    )
    summary["leiden_cluster_label"] = summary["leiden_cluster"].map(
        MODULE_LABELS
    )
    summary["protscape_percent"] = (
        100 * summary["protscape_proteins"] / summary["total_proteins"]
    )
    return summary
