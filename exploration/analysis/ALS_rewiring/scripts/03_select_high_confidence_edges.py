"""Stage 03 - select high-confidence new edges, unique to CTRL vs VCP, per
timepoint, and build the combined (new + already-known) edge table.

Input:  INTERMEDIATE_DIR/mydat.parquet (stage 01)
Output: INTERMEDIATE_DIR/mynewgraph.parquet
"""
import pandas as pd
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import config, high_confidence


def main():
    config.ensure_dirs()

    mydat = pd.read_parquet(config.INTERMEDIATE_DIR / "mydat.parquet")

    A, B, C, D = high_confidence.select_high_confidence_new_edges(mydat, config.HIGH_CONF_THRESHOLD)
    print(len(A), "unique new high-confidence edges at d22, CTRL only")
    print(len(B), "unique new high-confidence edges at d22, VCP only")
    print(len(C), "unique new high-confidence edges at d35, CTRL only")
    print(len(D), "unique new high-confidence edges at d35, VCP only")

    mynewgraph = high_confidence.build_selected_edge_table(mydat, A, B, C, D)

    out_path = config.INTERMEDIATE_DIR / "mynewgraph.parquet"
    mynewgraph.to_parquet(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
