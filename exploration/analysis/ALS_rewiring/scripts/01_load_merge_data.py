"""Stage 01 - load & merge the d22 / d35 tables into `mydat`.

Input:  config.D22_CSV, config.D35_CSV
Output: INTERMEDIATE_DIR/mydat.parquet
"""
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import config, io_utils


def main():
    config.ensure_dirs()

    d22 = io_utils.load_pair_table(config.D22_CSV)
    d35 = io_utils.load_pair_table(config.D35_CSV)
    mydat = io_utils.merge_d22_d35(d22, d35)

    out_path = config.INTERMEDIATE_DIR / "mydat.parquet"
    mydat.to_parquet(out_path)
    print(f"Rows: {len(mydat)}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
