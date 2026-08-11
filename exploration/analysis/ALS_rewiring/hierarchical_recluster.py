"""Core pipeline step (stage 05b): split the least intra-connected
module(s) from stage 05's Leiden partition, re-benchmarking the resolution
inside each one, and track how global modularity / intra-connectivity evolve
as the partition gets refined.

Motivation: `clustering.benchmark_resolution` picks a single, global
resolution for the whole giant graph. Modules found at that resolution don't
all have the same internal connectivity - some are dense/well-separated,
others are loose agglomerations that a single global resolution may
under-split. This asks: if we rank modules by internal edge density and
peel off the least-connected one at a time, re-clustering only inside it with
the same modularity/ARI-stability procedure, does the *global* partition
modularity keep improving, and where does it plateau?

`recursive_refine_by_density` is purely additive (reads a `(giant, module_of,
mod_ids)` triple, returns a new triple plus a history DataFrame - never
touches `giant.pickle`/`module_of.json`/etc. in place): it ranks the
*original* modules once, splits each in ascending density order. Newly
created sub-modules are never re-split, even if one ends up looser than the
remaining originals.
"""
import networkx as nx
import numpy as np
import pandas as pd

from .clustering import assign_modules, benchmark_resolution, build_igraph
from .plotting import module_density_graph


def intra_module_density(giant: nx.Graph, module_of: dict, mod_ids) -> dict:
    """Within-module edge density per module (`actual_edges / possible_edges`,
    the same self-loop weight `plotting.module_density_graph` computes) - the
    ranking criterion for "least intra-connected"."""
    mod_g = module_density_graph(giant, module_of, mod_ids)
    return {m: mod_g.edges[m, m]["weight"] for m in mod_ids}


def global_modularity(giant: nx.Graph, module_of: dict) -> float:
    """RBConfiguration modularity (resolution=1, igraph's default `modularity`)
    of the whole graph under `module_of` - comparable across iterations since
    it's always evaluated on the same fixed `giant`. Modularity trades off
    internal density against a null-model *expectation* (based on node
    degrees and total edge count), so a high value can partly reflect that
    the rest of the graph is sparse rather than that any given module is
    tightly-knit in absolute terms - see `mean_intra_density` for a metric
    that isolates intra-module connectivity without that null-model term."""
    ig_graph, nodes = build_igraph(giant)
    labels = [module_of[n] for n in nodes]
    codes, _ = pd.factorize(labels)
    return ig_graph.modularity(codes.tolist())


def mean_intra_density(giant: nx.Graph, module_of: dict, mod_ids, weighted: bool = True) -> float:
    """Average of each module's own `intra_module_density` (actual/possible
    edges within that module alone - no reference to the rest of the graph at
    all). `weighted=True` (default) weights each module's density by its
    node count, so the summary reflects how tightly-knit the modules are for
    a typical protein rather than being dominated by many small modules."""
    densities = intra_module_density(giant, module_of, mod_ids)
    if not weighted:
        return float(np.mean(list(densities.values())))
    sizes = {m: sum(1 for mm in module_of.values() if mm == m) for m in mod_ids}
    total = sum(sizes.values())
    if total == 0:
        return 0.0
    return sum(densities[m] * sizes[m] for m in mod_ids) / total


def recursive_refine_by_density(giant: nx.Graph, module_of: dict, mod_ids,
                                 resolutions, n_runs: int, seed: int = 42,
                                 n_iterations: int = -1, modularity_weight: float = 0.5,
                                 min_module_size: int = 10, max_splits: "int | None" = None,
                                 on_step=None):
    """Starting from an existing partition, repeatedly: rank current *original*
    top-level modules by ascending intra-module density, take the least
    intra-connected one not yet processed, re-run the full
    `benchmark_resolution` procedure on its induced subgraph alone, and (if
    that finds >1 sub-module) replace it in the partition with its
    sub-modules under new ids. Global modularity is recomputed after every
    split. Only the original modules from `mod_ids` are ever candidates for
    splitting (sub-modules are not recursively re-split) - this matches "take
    the next less intra-connected cluster" as a single pass over the starting
    partition, not unbounded recursion.

    Modules with fewer than `min_module_size` nodes, or for which the
    sub-benchmark doesn't find more than one sub-module, are left untouched
    and skipped (recorded in the history with `n_subclusters=1`).

    Returns `(new_module_of, new_mod_ids, history_df)`. `history_df` has one
    row per processed original module, in the order it was split, with its
    intra-module density, the resolution the sub-benchmark chose, how many
    sub-modules it produced, that sub-benchmark's own ARI/NMI stability
    (mean pairwise ARI/NMI across its `n_runs` seeds at the chosen
    resolution - how reproducible this particular split was, not a
    comparison against any other partition), the resulting *global*
    modularity, `mean_intra_density` - a size-weighted average of each
    module's own edge density, isolating intra-module connectivity without
    modularity's null-model correction - and `n_modules_total`, the total
    module count after this step.

    `on_step`, if given, is called after every step (whether it split the
    module or left it as-is) as `on_step(step_index, new_module_of,
    new_mod_ids, history_row)` - `step_index` is 1-indexed over `split_order`,
    `history_row` is the dict just appended to `history`. This is the hook
    the pipeline scripts use to snapshot e.g. a pathway embedding of the
    partition after each split, without this function needing to know
    anything about pathway embeddings itself.
    """
    densities = intra_module_density(giant, module_of, mod_ids)
    split_order = sorted(mod_ids, key=lambda m: densities[m])
    if max_splits is not None:
        split_order = split_order[:max_splits]

    new_module_of = dict(module_of)
    new_mod_ids = list(mod_ids)
    next_id = max(mod_ids) + 1
    history = []

    def _connectivity_snapshot():
        return {
            "global_modularity": global_modularity(giant, new_module_of),
            "mean_intra_density": mean_intra_density(giant, new_module_of, new_mod_ids),
            "n_modules_total": len(new_mod_ids),
        }

    for step_index, m in enumerate(split_order, start=1):
        nodes_m = [n for n, mm in new_module_of.items() if mm == m]
        sub_g = giant.subgraph(nodes_m).copy()

        if sub_g.number_of_nodes() < min_module_size:
            print(f"Module {m}: {sub_g.number_of_nodes()} nodes < min_module_size={min_module_size}, skipping")
            history.append({
                "split_module": m, "n_nodes": sub_g.number_of_nodes(), "intra_density": densities[m],
                "resolution_used": np.nan, "n_subclusters": 1,
                "sub_ari_stability": np.nan, "sub_nmi_stability": np.nan,
                **_connectivity_snapshot(),
            })
            if on_step is not None:
                on_step(step_index, new_module_of, new_mod_ids, history[-1])
            continue

        print(f"Module {m}: {sub_g.number_of_nodes()} nodes, intra-density={densities[m]:.4f} - re-benchmarking...")
        best_resolution, sweep_df = benchmark_resolution(
            sub_g, resolutions=resolutions, n_runs=n_runs, plot=False,
            n_iterations=n_iterations, modularity_weight=modularity_weight,
        )
        # `benchmark_resolution`'s own stability metrics (mean pairwise ARI/NMI
        # across its `n_runs` seeds) for the resolution it picked - how
        # reproducible *this particular split* was, not a comparison against
        # any other partition.
        chosen_row = sweep_df.loc[sweep_df["resolution"] == best_resolution].iloc[0]
        sub_ari = chosen_row["mean_ari_stability"]
        sub_nmi = chosen_row["mean_nmi_stability"]
        sub_module_of, sub_mod_ids, _ = assign_modules(sub_g, resolution=best_resolution, seed=seed)

        if len(sub_mod_ids) <= 1:
            print(f"  -> sub-benchmark found only 1 sub-module, leaving module {m} as-is")
            history.append({
                "split_module": m, "n_nodes": sub_g.number_of_nodes(), "intra_density": densities[m],
                "resolution_used": best_resolution, "n_subclusters": 1,
                "sub_ari_stability": sub_ari, "sub_nmi_stability": sub_nmi,
                **_connectivity_snapshot(),
            })
            if on_step is not None:
                on_step(step_index, new_module_of, new_mod_ids, history[-1])
            continue

        relabel = {sid: next_id + i for i, sid in enumerate(sub_mod_ids)}
        next_id += len(sub_mod_ids)
        for n, sid in sub_module_of.items():
            new_module_of[n] = relabel[sid]
        new_mod_ids.remove(m)
        new_mod_ids.extend(relabel.values())

        snapshot = _connectivity_snapshot()
        print(f"  -> split into {len(sub_mod_ids)} sub-modules (res={best_resolution}), "
              f"ARI={sub_ari:.3f} NMI={sub_nmi:.3f}, global modularity now {snapshot['global_modularity']:.4f}, "
              f"mean intra-density now {snapshot['mean_intra_density']:.4f}")
        history.append({
            "split_module": m, "n_nodes": sub_g.number_of_nodes(), "intra_density": densities[m],
            "resolution_used": best_resolution, "n_subclusters": len(sub_mod_ids),
            "sub_ari_stability": sub_ari, "sub_nmi_stability": sub_nmi,
            **snapshot,
        })
        if on_step is not None:
            on_step(step_index, new_module_of, new_mod_ids, history[-1])

    return new_module_of, new_mod_ids, pd.DataFrame(history)
