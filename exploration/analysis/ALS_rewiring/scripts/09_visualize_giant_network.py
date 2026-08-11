"""Stage 09 - network / module visualisation.

Input:  INTERMEDIATE_DIR/giant.pickle, module_of.json, mod_ids.json,
        mod_palette.json (stage 05), enrichment_vcp_ct.parquet, moi.json
        (stage 07)
        OUT_DIR/string_enrichment_per_module.csv (stage 08)
Output: OUT_DIR/plotGiant_network.pdf, module_density_known.csv,
        module_density_known_and_new.csv
        OUT_DIR/modules_oi_{vcp,ct}_d{22,35}.pdf
        OUT_DIR/modules_oi_{vcp,ct}_d{22,35}.csv (full RCTM+GO enriched-pathway
            list for every module in that bucket - one row per module/term)
        OUT_DIR/modules_oi_{vcp,ct}_d{22,35}_log2fc.pdf (log2FC-vs-D0
            distribution across every gene in each module, one box per
            condition, full D3/D7/D14/D22/D35 time course from the
            Cytoplasmic-only expression matrix - see
            gene_expression.cyto_log2fc_all_timepoints)
        OUT_DIR/modules_oi_{vcp,ct}_d{22,35}_log2fc_heatmap.pdf (same log2FC-
            vs-D0, but per-gene detail as a heatmap)
"""
import pandas as pd
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import config, gene_expression, io_utils, layout as layout_mod, plotting, string_api


def main():
    config.ensure_dirs()

    giant = io_utils.load_graph(config.INTERMEDIATE_DIR / "giant.pickle")
    module_of = io_utils.load_json(config.INTERMEDIATE_DIR / "module_of.json")
    mod_ids = io_utils.load_json(config.INTERMEDIATE_DIR / "mod_ids.json")
    mod_palette = io_utils.load_json_int_keys(config.INTERMEDIATE_DIR / "mod_palette.json")
    enrichment_vcp_ct = pd.read_parquet(config.INTERMEDIATE_DIR / "enrichment_vcp_ct.parquet")
    moi = io_utils.load_json(config.INTERMEDIATE_DIR / "moi.json")
    all_enrichment = pd.read_csv(config.OUT_DIR / "string_enrichment_per_module.csv")

    expr_cyto = gene_expression.load_cyto_gene_expression()
    log2fc_cyto_df = gene_expression.cyto_log2fc_all_timepoints(expr_cyto)

    focus_modules = sorted(set(moi["vcp_d22"]) | set(moi["vcp_d35"]) | set(moi["ct_d22"]) | set(moi["ct_d35"]))

    giant_layout = layout_mod.community_layout(giant, module_of)
    plotting.save_giant_network_pdf(
        giant, module_of, mod_ids, mod_palette, giant_layout,
        config.OUT_DIR / "plotGiant_network.pdf", focus_modules=focus_modules,
    )

    for bucket in ("vcp_d22", "vcp_d35", "ct_d22", "ct_d35"):
        plotting.save_modules_of_interest_pdf(
            moi[bucket], enrichment_vcp_ct, all_enrichment, mod_palette,
            giant, module_of, config.OUT_DIR / f"modules_oi_{bucket}.pdf",
        )
        string_api.save_module_pathway_csv(
            moi[bucket], all_enrichment, config.OUT_DIR / f"modules_oi_{bucket}.csv",
        )
        plotting.save_module_log2fc_boxplots(
            moi[bucket], log2fc_cyto_df, module_of, mod_palette,
            config.OUT_DIR / f"modules_oi_{bucket}_log2fc.pdf",
        )
        plotting.save_module_log2fc_heatmaps(
            moi[bucket], log2fc_cyto_df, module_of,
            config.OUT_DIR / f"modules_oi_{bucket}_log2fc_heatmap.pdf",
        )

    # Cache the layout too - handy for ad-hoc re-plots that want the exact same
    # node positions without re-running the (slower) community_layout call.
    layout_json = {n: [float(p[0]), float(p[1])] for n, p in giant_layout.items()}
    io_utils.save_json(layout_json, config.INTERMEDIATE_DIR / "giant_layout.json")


if __name__ == "__main__":
    main()
