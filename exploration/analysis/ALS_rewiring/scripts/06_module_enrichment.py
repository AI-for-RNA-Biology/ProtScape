"""Stage 06 - per-module enrichment: are *new* edges over-represented in a
module vs the rest?

Input:  INTERMEDIATE_DIR/giant.pickle, mod_palette.json, module_of.json,
        mod_ids.json (stage 05)
Output: OUT_DIR/module_enrichment_results.csv
        INTERMEDIATE_DIR/module_members.json (used by stage 08's STRING calls)
"""
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import config, io_utils, module_enrichment


def main():
    config.ensure_dirs()

    giant = io_utils.load_graph(config.INTERMEDIATE_DIR / "giant.pickle")
    mod_palette = io_utils.load_json_int_keys(config.INTERMEDIATE_DIR / "mod_palette.json")
    module_of = io_utils.load_json(config.INTERMEDIATE_DIR / "module_of.json")
    mod_ids = io_utils.load_json(config.INTERMEDIATE_DIR / "mod_ids.json")

    results = module_enrichment.get_enrichment_module(giant, mod_palette)
    out_path = config.OUT_DIR / "module_enrichment_results.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    module_members = {m: [n for n, mm in module_of.items() if mm == m] for m in mod_ids}
    members_path = config.INTERMEDIATE_DIR / "module_members.json"
    io_utils.save_json(module_members, members_path)
    print(f"Saved {members_path}")


if __name__ == "__main__":
    main()
