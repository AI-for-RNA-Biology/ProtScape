"""Stage 08 - STRING functional enrichment per module (REST API, rate-limited
to ~1 request/second - this stage is the slowest and the only one needing
network access, so it's worth caching its output separately from everything
around it).

Input:  INTERMEDIATE_DIR/module_members.json (stage 06)
Output: OUT_DIR/string_enrichment_per_module.csv
"""
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import config, io_utils, string_api


def main():
    config.ensure_dirs()

    module_members = io_utils.load_json_int_keys(config.INTERMEDIATE_DIR / "module_members.json")
    all_enrichment = string_api.run_string_enrichment_per_module(module_members)

    out_path = config.OUT_DIR / "string_enrichment_per_module.csv"
    all_enrichment.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
