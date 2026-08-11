"""Stage 05 - Leiden resolution sweep (modularity / ARI-stability trade-off)
and final module assignment.

Input:  INTERMEDIATE_DIR/giant.pickle (stage 04)
Output: OUT_DIR/optimisation_resolution_leiden.pdf (skipped if --resolution given)
        INTERMEDIATE_DIR/resolution_sweep.csv (skipped if --resolution given)
        INTERMEDIATE_DIR/giant.pickle (overwritten - now has a `module` node attribute)
        INTERMEDIATE_DIR/module_of.json, mod_ids.json, mod_palette.json
        INTERMEDIATE_DIR/pathway_embedding.parquet (GO pathway x module embedding)
        INTERMEDIATE_DIR/pathway_module_distances.parquet (module x module cosine
            distance between pathway-embedding columns - see
            pathway_embedding.compute_module_distance_fast)

Usage:
    python 05_cluster_leiden.py                  # run the full sweep, use its suggestion
    python 05_cluster_leiden.py --resolution 1.5  # skip the sweep, cluster at this resolution
"""
import argparse

import matplotlib.pyplot as plt
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import clustering, config, io_utils, pathway_embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=float, default=None,
                         help="Skip the sweep and cluster directly at this resolution.")
    parser.add_argument("--seed", type=int, default=42, help="Leiden seed for the final clustering.")
    args = parser.parse_args()

    config.ensure_dirs()
    giant = io_utils.load_graph(config.INTERMEDIATE_DIR / "giant.pickle")

    if args.resolution is not None:
        best_resolution = args.resolution
        print(f"Using manually specified resolution: {best_resolution} (sweep skipped)")
    else:
        best_resolution, sweep_df = clustering.benchmark_resolution(
            giant, resolutions=config.RESOLUTION_SWEEP, n_runs=config.N_STABILITY_RUNS,
            n_iterations=config.LEIDEN_N_ITERATIONS, modularity_weight=config.TRADEOFF_MODULARITY_WEIGHT,
        )
        fig_path = config.OUT_DIR / "optimisation_resolution_leiden.pdf"
        plt.savefig(fig_path)
        plt.close("all")
        print(f"Saved {fig_path}")

        sweep_path = config.INTERMEDIATE_DIR / "resolution_sweep.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(f"Saved {sweep_path}")
        print(f"Suggested resolution from modularity/ARI-stability trade-off: {best_resolution}")

    module_of, mod_ids, mod_palette = clustering.assign_modules(giant, resolution=best_resolution, seed=args.seed)

    io_utils.save_graph(giant, config.INTERMEDIATE_DIR / "giant.pickle")
    io_utils.save_json(module_of, config.INTERMEDIATE_DIR / "module_of.json")
    io_utils.save_json(mod_ids, config.INTERMEDIATE_DIR / "mod_ids.json")
    io_utils.save_json(mod_palette, config.INTERMEDIATE_DIR / "mod_palette.json")
    io_utils.save_json({"resolution": best_resolution}, config.INTERMEDIATE_DIR / "chosen_resolution.json")
    print(f"Saved module_of.json, mod_ids.json, mod_palette.json to {config.INTERMEDIATE_DIR}")

    go_terms = pathway_embedding.load_go_terms(config.GO_TERMS_CSV)
    embedding = pathway_embedding.compute_pathway_embedding(module_of, mod_ids, go_terms)
    embedding_path = config.INTERMEDIATE_DIR / "pathway_embedding.parquet"
    pathway_embedding.save_pathway_embedding(embedding, embedding_path)
    print(f"Saved {embedding_path}")

    distances = pathway_embedding.compute_module_distance_fast(embedding, metric='euclidean')
    distances_path = config.INTERMEDIATE_DIR / "pathway_module_distances.parquet"
    pathway_embedding.save_module_distances(distances, distances_path)
    print(f"Saved {distances_path}")


if __name__ == "__main__":
    main()
