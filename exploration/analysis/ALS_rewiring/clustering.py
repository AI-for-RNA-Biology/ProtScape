"""Select Leiden resolution using modularity and partition stability.

For each resolution, average modularity and pairwise adjusted Rand index
across seeds. Min-max normalise both metrics across resolutions and combine
them using `TRADEOFF_MODULARITY_WEIGHT`.
"""
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def build_igraph(giant: nx.Graph):
    """Return an igraph graph and protein names in fixed vertex order.

    Sorting nodes and edges makes seeded Leiden runs independent of Python
    hash order. Membership arrays align with the returned protein names.
    """
    nodes = sorted(giant.nodes())
    index_of = {n: i for i, n in enumerate(nodes)}
    edges = [(index_of[u], index_of[v]) for u, v in sorted(giant.edges())]
    return ig.Graph(n=len(nodes), edges=edges), nodes


def run_leiden(ig_graph: ig.Graph, resolution: float, seed: int, n_iterations: int):
    partition = leidenalg.find_partition(
        ig_graph, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed, n_iterations=n_iterations,
    )
    membership = np.array(partition.membership)
    modularity = ig_graph.modularity(membership)
    return membership, int(membership.max()) + 1, modularity


def _minmax_norm(s: pd.Series) -> pd.Series:
    span = s.max() - s.min()
    return pd.Series(1.0, index=s.index) if span == 0 else (s - s.min()) / span


def benchmark_resolution(giant: nx.Graph, resolutions, n_runs: int, plot: bool = True,
                          n_iterations: int = -1, n_jobs: int = -1,
                          modularity_weight: float = 0.5):
    ig_graph, nodes = build_igraph(giant)

    # Every (resolution, seed) run is independent, so flatten the grid and run it in
    # parallel instead of two nested sequential loops - this is the main cost driver.
    jobs = [(res, seed) for res in resolutions for seed in range(n_runs)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_leiden)(ig_graph, res, seed, n_iterations) for res, seed in jobs
    )

    records = []
    for idx, res in enumerate(resolutions):
        chunk = results[idx * n_runs:(idx + 1) * n_runs]
        partitions = [p for p, _, _ in chunk]
        n_modules = [m for _, m, _ in chunk]
        modularities = [q for _, _, q in chunk]

        ari_vals, nmi_vals = [], []
        for i in range(n_runs - 1):
            for j in range(i + 1, n_runs):
                ari_vals.append(adjusted_rand_score(partitions[i], partitions[j]))
                nmi_vals.append(normalized_mutual_info_score(partitions[i], partitions[j]))

        records.append({
            "resolution": res,
            "mean_n_modules": np.mean(n_modules),
            "sd_n_modules": np.std(n_modules, ddof=1),
            "mean_modularity": np.mean(modularities),
            "sd_modularity": np.std(modularities, ddof=1),
            "mean_ari_stability": np.mean(ari_vals),
            "sd_ari_stability": np.std(ari_vals, ddof=1),
            "mean_nmi_stability": np.mean(nmi_vals),
            "sd_nmi_stability": np.std(nmi_vals, ddof=1),
        })

    sweep_df = pd.DataFrame(records)

    candidates = sweep_df[sweep_df["mean_n_modules"] > 1].copy()
    candidates["modularity_norm"] = _minmax_norm(candidates["mean_modularity"])
    candidates["ari_norm"] = _minmax_norm(candidates["mean_ari_stability"])
    candidates["combined_score"] = (
        modularity_weight * candidates["modularity_norm"]
        + (1 - modularity_weight) * candidates["ari_norm"]
    )
    sweep_df = sweep_df.merge(
        candidates[["resolution", "modularity_norm", "ari_norm", "combined_score"]],
        on="resolution", how="left",
    )
    best_row = candidates.loc[candidates["combined_score"].idxmax()]

    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(6, 13))

        axes[0].errorbar(sweep_df.resolution, sweep_df.mean_n_modules,
                          yerr=sweep_df.sd_n_modules, fmt="-o", color="steelblue")
        axes[0].set(xlabel="Resolution", ylabel="Number of modules",
                    title=f"Modules detected vs resolution (mean over {n_runs} runs)")

        axes[1].errorbar(sweep_df.resolution, sweep_df.mean_modularity,
                          yerr=sweep_df.sd_modularity, fmt="-o", color="darkorange")
        axes[1].set(xlabel="Resolution", ylabel="Modularity Q", title="Modularity vs resolution")

        axes[2].errorbar(sweep_df.resolution, sweep_df.mean_ari_stability,
                          yerr=sweep_df.sd_ari_stability, fmt="-o", color="indianred", label="ARI stability")
        axes[2].errorbar(sweep_df.resolution, sweep_df.mean_nmi_stability,
                          yerr=sweep_df.sd_nmi_stability, fmt="-s", color="seagreen",
                          label="NMI stability (reference only)")
        axes[2].set(xlabel="Resolution", ylabel="Partition stability", title="Stability vs resolution")
        axes[2].legend(loc="lower right", frameon=False)

        axes[3].plot(sweep_df.resolution, sweep_df.combined_score, "-o", color="purple")
        axes[3].axvline(best_row.resolution, linestyle="--", color="black", label="suggested")
        axes[3].set(xlabel="Resolution", ylabel="Combined score",
                    title=f"Modularity/ARI trade-off (weight={modularity_weight})")
        axes[3].legend(loc="lower right", frameon=False)
        fig.tight_layout()

    return best_row.resolution, sweep_df


def build_module_palette(mod_ids) -> dict:
    """One HSV color per module id, in the given order - the same scheme
    `assign_modules` uses, factored out so any other partition (e.g. one
    assembled by hand, or by the hierarchical-recluster experiment) can get a
    palette without duplicating this."""
    mod_ids = sorted(mod_ids)
    cmap = plt.get_cmap("hsv", len(mod_ids))
    return {m: cmap(i) for i, m in enumerate(mod_ids)}


def assign_modules(giant: nx.Graph, resolution: float, seed: int = 42, n_iterations: int = -1):
    ig_graph, nodes = build_igraph(giant)
    membership, n_modules, _ = run_leiden(ig_graph, resolution, seed, n_iterations)
    module_of = {nodes[i]: int(membership[i]) + 1 for i in range(len(nodes))}
    nx.set_node_attributes(giant, module_of, "module")

    mod_ids = sorted(set(module_of.values()))
    mod_palette = build_module_palette(mod_ids)

    print(f"Number of modules detected: {len(mod_ids)}")
    return module_of, mod_ids, mod_palette
