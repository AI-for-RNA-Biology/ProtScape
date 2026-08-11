"""Stage 10 - connected components of a single new-edge condition within each
module of interest, with STRING enrichment for components with >10 genes.

Matches the notebook's section 9, which covers `vcp_35`/`ct_35` (d35) - extend
`CONDITION_PAIRS` below to also do `vcp_22`/`ct_22` if you want the d22 side too.

Input:  INTERMEDIATE_DIR/giant.pickle, module_of.json, mod_palette.json
        (stage 05), moi.json (stage 07)
Output: OUT_DIR/{vcp,ct}_d35_new_edge_components.pdf/.csv
        OUT_DIR/{vcp,ct}_d35_new_edge_components_string_enrichment.csv (when any
        component has >10 genes)
"""
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import config, edge_components, io_utils, string_api

# (edge_col, moi bucket key, output basename)
CONDITION_PAIRS = [
    ("vcp_35", "vcp_d35", "vcp_d35_new_edge_components"),
    ("ct_35", "ct_d35", "ct_d35_new_edge_components"),
]


def main():
    config.ensure_dirs()

    giant = io_utils.load_graph(config.INTERMEDIATE_DIR / "giant.pickle")
    module_of = io_utils.load_json(config.INTERMEDIATE_DIR / "module_of.json")
    mod_palette = io_utils.load_json_int_keys(config.INTERMEDIATE_DIR / "mod_palette.json")
    moi = io_utils.load_json(config.INTERMEDIATE_DIR / "moi.json")

    for edge_col, moi_key, basename in CONDITION_PAIRS:
        components_df = edge_components.extract_edge_components(giant, module_of, moi[moi_key], edge_col)
        enrichment_df = string_api.run_component_enrichment(components_df)
        edge_components.save_edge_components(
            components_df, giant, module_of, mod_palette, config.OUT_DIR,
            basename=basename, edge_col=edge_col, enrichment_df=enrichment_df,
        )


if __name__ == "__main__":
    main()
