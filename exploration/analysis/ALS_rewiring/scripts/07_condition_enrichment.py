"""Stage 07 - per-module, per-condition enrichment (known vs new, CTRL vs VCP,
d22 vs d35), plus the four modules-of-interest buckets used by stages 09/10.

Input:  INTERMEDIATE_DIR/giant.pickle, mod_ids.json, mod_palette.json (stage 05)
        OUT_DIR/module_enrichment_results.csv (stage 06, for panel titles)
Output: OUT_DIR/distribution_novel_edges_per_module.pdf
        OUT_DIR/enrichment_per_module_compared_known.pdf
        OUT_DIR/enrichment_per_module_vcp_ct.pdf
        OUT_DIR/FE_significance_d22.pdf, OUT_DIR/FE_significance_d35.pdf
        OUT_DIR/FE_significance_d22.csv, OUT_DIR/FE_significance_d35.csv
            (module, FE, Pval - the exact table each volcano plot draws)
        OUT_DIR/FE_significance_d22_small_modules.pdf,
        OUT_DIR/FE_significance_d35_small_modules.pdf (same, restricted to
        modules with < 1000 total internal edges)
        INTERMEDIATE_DIR/counts.parquet, enrichment_vcp_ct.parquet (needed by
        stages 09/10), distribution_shift.csv, moi.json
"""
import pandas as pd
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import condition_enrichment as ce
from exploration.analysis.ALS_rewiring import config, io_utils, plotting


def main():
    config.ensure_dirs()

    giant = io_utils.load_graph(config.INTERMEDIATE_DIR / "giant.pickle")
    mod_ids = io_utils.load_json(config.INTERMEDIATE_DIR / "mod_ids.json")
    mod_palette = io_utils.load_json_int_keys(config.INTERMEDIATE_DIR / "mod_palette.json")
    module_enrichment_results = pd.read_csv(config.OUT_DIR / "module_enrichment_results.csv")

    counts = ce.count_edges_per_module(giant, mod_ids)
    enrichment_new_known = ce.enrichment_new_vs_known(counts)
    enrichment_cond_known = ce.enrichment_condition_vs_known(counts, config.CONDITION_COLS)
    enrichment_vcp_ct = ce.enrichment_vcp_vs_ct(counts)
    distribution_shift = ce.global_distribution_shift(counts, [
        ("all", "known"), ("all", "new"), ("known", "new"),
        ("all", "d22"), ("all", "d35"),
        ("all", "ct"), ("all", "vcp"), ("ct", "vcp"),
        ("all", "ct_d22"), ("all", "vcp_22"), ("all", "ct_35"), ("all", "vcp_35"),
    ])

    plotting.plot_distribution_novel_edges(
        enrichment_new_known, config.OUT_DIR / "distribution_novel_edges_per_module.pdf"
    )
    plotting.plot_enrichment_compared_known(
        enrichment_cond_known, mod_palette, config.CONDITION_COLS,
        config.OUT_DIR / "enrichment_per_module_compared_known.pdf",
    )
    n_nodes_map = module_enrichment_results.set_index("module")["n_nodes"].to_dict()
    plotting.plot_enrichment_vcp_ct(
        enrichment_vcp_ct, mod_palette, config.OUT_DIR / "enrichment_per_module_vcp_ct.pdf", n_nodes_map
    )
    plotting.plot_module_volcano(
        enrichment_vcp_ct, "22", config.OUT_DIR / "FE_significance_d22.pdf",
        config.MOI_FE_CUTOFF, config.MOI_PADJ_CUTOFF,
    )
    plotting.save_fe_significance_csv(enrichment_vcp_ct, "22", config.OUT_DIR / "FE_significance_d22.csv")
    plotting.plot_module_volcano(
        enrichment_vcp_ct, "35", config.OUT_DIR / "FE_significance_d35.pdf",
        config.MOI_FE_CUTOFF, config.MOI_PADJ_CUTOFF,
    )
    plotting.save_fe_significance_csv(enrichment_vcp_ct, "35", config.OUT_DIR / "FE_significance_d35.csv")
    plotting.plot_module_volcano(
        enrichment_vcp_ct, "22", config.OUT_DIR / "FE_significance_d22_small_modules.pdf",
        config.MOI_FE_CUTOFF, config.MOI_PADJ_CUTOFF, max_edges=1000,
    )
    plotting.plot_module_volcano(
        enrichment_vcp_ct, "35", config.OUT_DIR / "FE_significance_d35_small_modules.pdf",
        config.MOI_FE_CUTOFF, config.MOI_PADJ_CUTOFF, max_edges=1000,
    )

    moi = ce.modules_of_interest(enrichment_vcp_ct, config.MOI_FE_CUTOFF, config.MOI_PADJ_CUTOFF)
    print("Modules of interest:", moi)

    counts.to_parquet(config.INTERMEDIATE_DIR / "counts.parquet")
    enrichment_vcp_ct.to_parquet(config.INTERMEDIATE_DIR / "enrichment_vcp_ct.parquet")
    distribution_shift.to_csv(config.INTERMEDIATE_DIR / "distribution_shift.csv", index=False)
    io_utils.save_json(moi, config.INTERMEDIATE_DIR / "moi.json")
    print(f"Saved intermediate tables to {config.INTERMEDIATE_DIR}")


if __name__ == "__main__":
    main()
