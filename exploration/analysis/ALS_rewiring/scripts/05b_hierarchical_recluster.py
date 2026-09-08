"""Core pipeline stage 05b. Ranks the modules from stage 05 by intra-module
edge density, then repeatedly peels off the least-connected one and
re-benchmarks + re-clusters just its induced subgraph, tracking global
partition modularity after each split. Always runs as part of
ALS_rewiring_analysis.py, right after stage 05.

Always processes every original stage-05 module (a full pass, least- to
most-connected) and always writes both the fully-refined partition and -
when at least 3 modules were processed - the elbow partition, as diagnostic
snapshots under OUT_DIR/hierarchical_recluster/ (see below).

Then, whichever of the two - elbow or fully-refined - config.HIERARCHICAL_
RECLUSTER_STOP_AT_ELBOW selects becomes the pipeline's CANONICAL partition:
this OVERWRITES INTERMEDIATE_DIR/giant.pickle's "module" node attribute and
module_of.json/mod_ids.json/mod_palette.json/pathway_embedding.parquet/
pathway_module_distances.parquet with it, after backing up stage 05's
original values as module_of_pre_recluster.json/mod_ids_pre_recluster.json.
Stages 06-10 read INTERMEDIATE_DIR unmodified, so they automatically run on
the hierarchically-refined partition, not stage 05's original one.

Reads:
    INTERMEDIATE_DIR/giant.pickle, module_of.json, mod_ids.json (stage 05 output)

Writes (own, isolated diagnostics dir - never overlaps stage 06-10's own outputs):
    OUT_DIR/hierarchical_recluster/initial_density_cumulative.pdf
    OUT_DIR/hierarchical_recluster/history.csv
    OUT_DIR/hierarchical_recluster/connectivity_vs_split.pdf
    OUT_DIR/hierarchical_recluster/sub_benchmark_stability_vs_split.pdf
    OUT_DIR/hierarchical_recluster/module_of_refined.json (final, after ALL splits)
    OUT_DIR/hierarchical_recluster/mod_ids_refined.json    (ditto)
    OUT_DIR/hierarchical_recluster/pathway_embedding_refined.parquet (ditto)
    OUT_DIR/hierarchical_recluster/pathway_module_distances_refined.parquet (ditto)
    OUT_DIR/hierarchical_recluster/module_of_elbow.json (partition AT the elbow
    OUT_DIR/hierarchical_recluster/mod_ids_elbow.json    split step)
    OUT_DIR/hierarchical_recluster/pathway_embeddings/step_00_initial.parquet
        (GO pathway x module embedding of the ORIGINAL, pre-split partition -
        "step 0" in pathway_distance_vs_split.pdf below)
    OUT_DIR/hierarchical_recluster/pathway_embeddings/step_00_initial_distances.parquet
    OUT_DIR/hierarchical_recluster/pathway_embeddings/step_{NN}_mod{split_module}.parquet
        (GO pathway x module embedding of the partition as it stood right
        after that split step - one per row of history.csv)
    OUT_DIR/hierarchical_recluster/pathway_embeddings/step_{NN}_mod{split_module}_distances.parquet
        (module x module distance between that step's pathway-embedding
        columns - see pathway_embedding.compute_module_distance_fast, cheap
        enough to run once per split step)
    OUT_DIR/hierarchical_recluster/pathway_distance_vs_split.pdf
        (evolution of the pairwise pathway-distance distribution across split
        steps, starting from the pre-split partition (step 0) - one bar per
        step, mean pairwise distance with std errorbar, y-axis zoomed to the
        data's own range rather than starting at 0; dashed vertical line marks
        the same elbow-of-mean-intra-density split step used to cut
        module_of_elbow.json, i.e. connectivity_vs_split.pdf's elbow line)
    OUT_DIR/hierarchical_recluster/pathway_embedding_elbow.parquet (only
        when the elbow partition is written)
    OUT_DIR/hierarchical_recluster/pathway_module_distances_elbow.parquet (ditto)

Also OVERWRITES, with the chosen (elbow or fully-refined, per
config.HIERARCHICAL_RECLUSTER_STOP_AT_ELBOW) partition:
    INTERMEDIATE_DIR/giant.pickle ("module" node attribute updated)
    INTERMEDIATE_DIR/module_of.json, mod_ids.json, mod_palette.json
    INTERMEDIATE_DIR/pathway_embedding.parquet, pathway_module_distances.parquet
And backs up stage 05's original partition first, so nothing is lost:
    INTERMEDIATE_DIR/module_of_pre_recluster.json, mod_ids_pre_recluster.json

Usage:
    python 05b_hierarchical_recluster.py
    python 05b_hierarchical_recluster.py --max-splits 3   # only peel off the 3 least-connected modules
"""
import argparse
import hashlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import config, io_utils, pathway_embedding
from exploration.analysis.ALS_rewiring.clustering import build_module_palette
from exploration.analysis.ALS_rewiring.hierarchical_recluster import (
    global_modularity, intra_module_density, mean_intra_density, recursive_refine_by_density,
)


def _find_elbow(y: np.ndarray) -> int:
    """Index of the "elbow" of `y` (assumed roughly monotonic, e.g. rising with
    diminishing returns): the point of maximum perpendicular distance from the
    straight line connecting its first and last points. Standard robust
    knee/elbow detector - unlike a literal second-derivative sign change, it
    doesn't require a true inflection (mean_intra_density here is concave and
    noisy split-to-split, not S-shaped, so a strict sign change would fire
    multiple times) and always returns exactly one point."""
    x = np.arange(len(y), dtype=float)
    y = y.astype(float)
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0
    y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y * 0

    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len if line_len > 0 else line_vec

    points = np.column_stack([x_norm, y_norm])
    vec_from_start = points - start
    proj_len = vec_from_start @ line_unit
    proj = np.outer(proj_len, line_unit)
    distances = np.linalg.norm(vec_from_start - proj, axis=1)
    return int(np.argmax(distances))


def _input_fingerprint(giant, module_of, mod_ids) -> str:
    """Hash graph and partition inputs to identify the data used for a run."""
    h = hashlib.sha256()
    h.update(repr(sorted(giant.nodes())).encode())
    h.update(repr(sorted(giant.edges())).encode())
    h.update(repr(sorted(module_of.items())).encode())
    h.update(repr(sorted(mod_ids)).encode())
    return h.hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-splits", type=int, default=None,
                         help="Only process this many least-connected modules (default: all).")
    parser.add_argument("--min-module-size", type=int, default=10,
                         help="Skip (leave untouched) modules smaller than this.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp_out_dir = config.OUT_DIR / "hierarchical_recluster"
    exp_out_dir.mkdir(parents=True, exist_ok=True)

    giant = io_utils.load_graph(config.INTERMEDIATE_DIR / "giant.pickle")
    module_of = io_utils.load_json(config.INTERMEDIATE_DIR / "module_of.json")
    mod_ids = io_utils.load_json(config.INTERMEDIATE_DIR / "mod_ids.json")
    print(f"Input fingerprint: {_input_fingerprint(giant, module_of, mod_ids)}")

    # Step-0 (pre-split) connectivity snapshot - plotted as the "initial"
    # point in connectivity_vs_split.pdf below, same convention
    # pathway_distance_vs_split.pdf already uses for its own step 0.
    initial_global_modularity = global_modularity(giant, module_of)
    initial_mean_intra_density = mean_intra_density(giant, module_of, mod_ids)
    print(f"Initial (pre-split) global_modularity={initial_global_modularity:.4f}, "
          f"mean_intra_density={initial_mean_intra_density:.4f}")

    # Diagnostic, computed BEFORE any splitting: rank the starting modules by
    # intra-density, highest first, and plot the cumulative SHARE of total
    # intra-density "mass" captured as more modules are included (a Pareto /
    # cumulative-variance-explained-style curve) - monotonically rises from
    # the densest module up to 1.0 once every module is included, showing how
    # concentrated intra-connectivity is among the top few modules.
    initial_densities = intra_module_density(giant, module_of, mod_ids)
    ranked_ids = sorted(mod_ids, key=lambda m: initial_densities[m], reverse=True)
    ranked_vals = np.array([initial_densities[m] for m in ranked_ids])
    cumulative_share = np.cumsum(ranked_vals) / ranked_vals.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(1, len(ranked_ids) + 1)
    ax.plot(x, cumulative_share, "-o", color="teal")
    for xi, m in zip(x, ranked_ids):
        ax.annotate(f"mod {m}", (xi, cumulative_share[xi - 1]), fontsize=6, ha="center", va="bottom")
    ax.set(xlabel="module rank (highest to lowest intra-density)",
           ylabel="cumulative share of total intra-density",
           title="Starting-partition intra-connectivity, ranked before any splitting")
    fig.tight_layout()
    cumulative_fig_path = exp_out_dir / "initial_density_cumulative.pdf"
    fig.savefig(cumulative_fig_path)
    plt.close(fig)
    print(f"Saved {cumulative_fig_path}")

    go_terms = pathway_embedding.load_go_terms(config.GO_TERMS_CSV)
    embeddings_dir = exp_out_dir / "pathway_embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Pre-split snapshot ("step 0") - lets pathway_distance_vs_split.pdf show
    # how far re-clustering actually moved modules apart in pathway-embedding
    # space, not just each step relative to the previous one.
    initial_embedding = pathway_embedding.compute_pathway_embedding(module_of, mod_ids, go_terms)
    initial_step_stem = "step_00_initial"
    initial_embedding_path = embeddings_dir / f"{initial_step_stem}.parquet"
    pathway_embedding.save_pathway_embedding(initial_embedding, initial_embedding_path)
    print(f"Saved {initial_embedding_path}")

    initial_distances = pathway_embedding.compute_module_distance_fast(initial_embedding, metric='euclidean')
    initial_distances_path = embeddings_dir / f"{initial_step_stem}_distances.parquet"
    pathway_embedding.save_module_distances(initial_distances, initial_distances_path)
    print(f"Saved {initial_distances_path}")

    initial_mean_dist, initial_std_dist, initial_n_pairs = pathway_embedding.pairwise_distance_stats(
        initial_distances
    )
    print(f"Initial pairwise distance: mean={initial_mean_dist:.4f}, std={initial_std_dist:.4f} "
          f"(n={initial_n_pairs} module pairs)")

    # One row per split step: the mean/std/n of that step's partition's
    # pairwise pathway distances (upper triangle of the module x module
    # matrix) - the data behind pathway_distance_vs_split.pdf below. Uses
    # compute_module_distance_fast (cosine, vectorized), cheap enough to call
    # once per split step of a full re-clustering run.
    distance_history = []

    def _save_step_embedding(step_index, step_module_of, step_mod_ids, history_row):
        embedding = pathway_embedding.compute_pathway_embedding(step_module_of, step_mod_ids, go_terms)
        step_stem = f"step_{step_index:02d}_mod{int(history_row['split_module'])}"
        step_path = embeddings_dir / f"{step_stem}.parquet"
        pathway_embedding.save_pathway_embedding(embedding, step_path)
        print(f"  -> Saved {step_path}")

        distances = pathway_embedding.compute_module_distance_fast(embedding, metric='euclidean')
        distances_path = embeddings_dir / f"{step_stem}_distances.parquet"
        pathway_embedding.save_module_distances(distances, distances_path)
        print(f"  -> Saved {distances_path}")

        mean_dist, std_dist, n_pairs = pathway_embedding.pairwise_distance_stats(distances)
        distance_history.append({
            "step_index": step_index, "split_module": int(history_row["split_module"]),
            "n_modules_total": int(history_row["n_modules_total"]),
            "mean_pairwise_distance": mean_dist, "std_pairwise_distance": std_dist,
            "n_pairs": n_pairs,
        })

    new_module_of, new_mod_ids, history_df = recursive_refine_by_density(
        giant, module_of, mod_ids,
        resolutions=config.RESOLUTION_SWEEP, n_runs=config.N_STABILITY_RUNS,
        seed=args.seed, n_iterations=config.LEIDEN_N_ITERATIONS,
        modularity_weight=config.TRADEOFF_MODULARITY_WEIGHT,
        min_module_size=args.min_module_size, max_splits=args.max_splits,
        on_step=_save_step_embedding,
    )

    # distance_history is appended by _save_step_embedding in the same step
    # order as history - one entry per row, so a positional zip is safe.
    history_df["mean_pairwise_distance"] = [r["mean_pairwise_distance"] for r in distance_history]
    history_df["std_pairwise_distance"] = [r["std_pairwise_distance"] for r in distance_history]
    history_df["n_pairs_distance"] = [r["n_pairs"] for r in distance_history]

    history_path = exp_out_dir / "history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved {history_path}")

    # Computed once, up front, so both pathway_distance_vs_split.pdf and
    # connectivity_vs_split.pdf mark the same cut point - `elbow_x` is a
    # split-step index (1 = right after the first split), which lines up
    # directly with connectivity_vs_split.pdf's x-axis and with
    # pathway_distance_vs_split.pdf's (0 = initial, pre-split partition).
    elbow_idx = elbow_x = elbow_y = elbow_module = None
    if len(history_df) >= 3:
        elbow_idx = _find_elbow(history_df["mean_intra_density"].to_numpy())
        elbow_x = elbow_idx + 1
        elbow_y = history_df["mean_intra_density"].iloc[elbow_idx]
        elbow_module = int(history_df["split_module"].iloc[elbow_idx])
        print(f"Elbow of mean intra-density curve: split step {elbow_x} (module {elbow_module}, "
              f"mean_intra_density={elbow_y:.4f})")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(history_df) + 1)  # 0 = initial, pre-split partition
    means = [initial_mean_dist] + history_df["mean_pairwise_distance"].tolist()
    stds = [initial_std_dist] + history_df["std_pairwise_distance"].tolist()
    labels = [f"initial\n[{len(mod_ids)} total]"] + [
        f"mod {int(m)}\n[{int(n)} total]"
        for m, n in zip(history_df["split_module"], history_df["n_modules_total"])
    ]
    ax.bar(x, means, yerr=stds, capsize=3, color="steelblue")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)

    # Zoom to where the bars actually differ instead of the default 0-based
    # range - cosine distance between module pairs tends to cluster tightly
    # near 1 (most modules share little to no pathway mass), so a 0-1 axis
    # makes step-to-step movement invisible.
    finite = [(m, s) for m, s in zip(means, stds) if not np.isnan(m)]
    if finite:
        lows = [m - s for m, s in finite]
        highs = [m + s for m, s in finite]
        pad = 0.05 * ((max(highs) - min(lows)) or 1.0)
        ax.set_ylim(min(lows) - pad, max(highs) + pad)

    if elbow_x is not None:
        ax.axvline(elbow_x, color="black", linestyle="--", linewidth=1, alpha=0.6, label="elbow cut")
        ax.annotate(f"elbow\n(mod {elbow_module})", (elbow_x, ax.get_ylim()[1]), fontsize=7, color="black",
                    ha="center", va="top", xytext=(0, -3), textcoords="offset points")
        ax.legend(loc="upper right", fontsize=7, frameon=False)

    ax.set(xlabel="split step (0 = initial partition, then least- to most-connected original module)",
           ylabel="pairwise distance",
           title="Distribution of pairwise distances between modules'\n"
                 "pathway embeddings, as least-connected modules are re-split\n"
                 "(bar = mean over module pairs, errorbar = std)")
    fig.tight_layout()
    distance_fig_path = exp_out_dir / "pathway_distance_vs_split.pdf"
    fig.savefig(distance_fig_path)
    plt.close(fig)
    print(f"Saved {distance_fig_path}")

    fig, (ax_mod, ax_dens) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    x = range(len(history_df) + 1)  # 0 = initial, pre-split partition
    mod_vals = [initial_global_modularity] + history_df["global_modularity"].tolist()
    dens_vals = [initial_mean_intra_density] + history_df["mean_intra_density"].tolist()

    ax_mod.plot(x, mod_vals, "-o", color="purple")
    ax_mod.annotate(f"initial\n[{len(mod_ids)} total]", (0, initial_global_modularity),
                     fontsize=6, ha="center", va="bottom")
    for i, row in history_df.reset_index(drop=True).iterrows():
        ax_mod.annotate(f"mod {int(row.split_module)}\n(n={int(row.n_subclusters)})\n"
                         f"[{int(row.n_modules_total)} total]",
                         (i + 1, row.global_modularity), fontsize=6, ha="center", va="bottom")
    ax_mod.set(ylabel="global modularity", title="Connectivity metrics as least-connected modules are re-split")

    ax_dens.plot(x, dens_vals, "-^", color="darkorange")
    ax_dens.set(xlabel="split step (0 = initial partition, then least- to most-connected original module)",
                ylabel="mean intra-connectivity")

    if elbow_x is not None:
        for ax in (ax_mod, ax_dens):
            ax.axvline(elbow_x, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax_dens.annotate(f"elbow\n(mod {elbow_module})", (elbow_x, elbow_y), fontsize=7, color="black",
                          ha="left", va="bottom", xytext=(5, 5), textcoords="offset points")

        # Recompute the partition at the selected elbow with the same seed.
        module_of_elbow, mod_ids_elbow, _ = recursive_refine_by_density(
            giant, module_of, mod_ids,
            resolutions=config.RESOLUTION_SWEEP, n_runs=config.N_STABILITY_RUNS,
            seed=args.seed, n_iterations=config.LEIDEN_N_ITERATIONS,
            modularity_weight=config.TRADEOFF_MODULARITY_WEIGHT,
            min_module_size=args.min_module_size, max_splits=elbow_x,
        )
        io_utils.save_json(module_of_elbow, exp_out_dir / "module_of_elbow.json")
        io_utils.save_json(mod_ids_elbow, exp_out_dir / "mod_ids_elbow.json")
        print(f"Saved module_of_elbow.json, mod_ids_elbow.json ({len(mod_ids_elbow)} modules) to {exp_out_dir}")

        elbow_embedding = pathway_embedding.compute_pathway_embedding(module_of_elbow, mod_ids_elbow, go_terms)
        elbow_embedding_path = exp_out_dir / "pathway_embedding_elbow.parquet"
        pathway_embedding.save_pathway_embedding(elbow_embedding, elbow_embedding_path)
        print(f"Saved {elbow_embedding_path}")

        elbow_distances = pathway_embedding.compute_module_distance_fast(elbow_embedding, metric='euclidean')
        elbow_distances_path = exp_out_dir / "pathway_module_distances_elbow.parquet"
        pathway_embedding.save_module_distances(elbow_distances, elbow_distances_path)
        print(f"Saved {elbow_distances_path}")
    else:
        print("History has < 3 rows - can't compute an elbow; module_of_elbow.json not written.")

    fig.tight_layout()
    fig_path = exp_out_dir / "connectivity_vs_split.pdf"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(1, len(history_df) + 1)
    ax.plot(x, history_df["sub_ari_stability"], "-o", color="indianred", label="ARI stability")
    ax.plot(x, history_df["sub_nmi_stability"], "-s", color="seagreen", label="NMI stability (reference only)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"mod {int(m)}" for m in history_df["split_module"]], fontsize=7, rotation=90)
    ax.set(xlabel="split step (least- to most-connected original module)",
           ylabel="sub-benchmark stability",
           title="Reproducibility of each module's own resolution sweep (mean pairwise ARI/NMI across seeds)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    stability_fig_path = exp_out_dir / "sub_benchmark_stability_vs_split.pdf"
    fig.savefig(stability_fig_path)
    plt.close(fig)
    print(f"Saved {stability_fig_path}")

    io_utils.save_json(new_module_of, exp_out_dir / "module_of_refined.json")
    io_utils.save_json(new_mod_ids, exp_out_dir / "mod_ids_refined.json")
    print(f"Saved module_of_refined.json, mod_ids_refined.json to {exp_out_dir}")

    refined_embedding = pathway_embedding.compute_pathway_embedding(new_module_of, new_mod_ids, go_terms)
    refined_embedding_path = exp_out_dir / "pathway_embedding_refined.parquet"
    pathway_embedding.save_pathway_embedding(refined_embedding, refined_embedding_path)
    print(f"Saved {refined_embedding_path}")

    refined_distances = pathway_embedding.compute_module_distance_fast(refined_embedding, metric='euclidean')
    refined_distances_path = exp_out_dir / "pathway_module_distances_refined.parquet"
    pathway_embedding.save_module_distances(refined_distances, refined_distances_path)
    print(f"Saved {refined_distances_path}")
    print(f"\nStarting modules: {len(mod_ids)} -> refined modules: {len(new_mod_ids)}")

    # --- Make the chosen (elbow or fully-refined) partition INTERMEDIATE_DIR's
    # canonical one, so stages 06-10 (unmodified) run on it instead of stage
    # 05's original partition. ---
    if config.HIERARCHICAL_RECLUSTER_STOP_AT_ELBOW and elbow_x is not None:
        chosen_module_of, chosen_mod_ids, chosen_label = module_of_elbow, mod_ids_elbow, "elbow"
        chosen_embedding, chosen_distances = elbow_embedding, elbow_distances
    else:
        if config.HIERARCHICAL_RECLUSTER_STOP_AT_ELBOW:
            print("HIERARCHICAL_RECLUSTER_STOP_AT_ELBOW=True but history has < 3 rows "
                  "(no elbow could be computed) - falling back to the fully-refined partition.")
        chosen_module_of, chosen_mod_ids, chosen_label = new_module_of, new_mod_ids, "refined"
        chosen_embedding, chosen_distances = refined_embedding, refined_distances

    # Back up stage 05's ORIGINAL partition before overwriting it below -
    # nothing about it is lost, it's just no longer what stages 06-10 read
    # by default.
    io_utils.save_json(module_of, config.INTERMEDIATE_DIR / "module_of_pre_recluster.json")
    io_utils.save_json(mod_ids, config.INTERMEDIATE_DIR / "mod_ids_pre_recluster.json")

    nx.set_node_attributes(giant, chosen_module_of, "module")
    io_utils.save_graph(giant, config.INTERMEDIATE_DIR / "giant.pickle")
    io_utils.save_json(chosen_module_of, config.INTERMEDIATE_DIR / "module_of.json")
    io_utils.save_json(chosen_mod_ids, config.INTERMEDIATE_DIR / "mod_ids.json")
    io_utils.save_json(build_module_palette(chosen_mod_ids), config.INTERMEDIATE_DIR / "mod_palette.json")
    pathway_embedding.save_pathway_embedding(chosen_embedding, config.INTERMEDIATE_DIR / "pathway_embedding.parquet")
    pathway_embedding.save_module_distances(
        chosen_distances, config.INTERMEDIATE_DIR / "pathway_module_distances.parquet"
    )
    print(f"\nINTERMEDIATE_DIR now holds the {chosen_label} partition ({len(chosen_mod_ids)} modules) - "
          f"stages 06-10 will run on this, not stage 05's original {len(mod_ids)}-module partition "
          f"(backed up as module_of_pre_recluster.json/mod_ids_pre_recluster.json).")


if __name__ == "__main__":
    main()
