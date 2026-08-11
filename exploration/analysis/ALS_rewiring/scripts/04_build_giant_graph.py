"""Stage 04 - edge typing + giant-component graph construction.

Input:  INTERMEDIATE_DIR/mynewgraph.parquet (stage 03)
Output: INTERMEDIATE_DIR/giant.pickle (networkx Graph, no `module` node
        attribute yet - that's added in stage 05)
"""
import pandas as pd
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import config, graph_build, io_utils


def main():
    config.ensure_dirs()

    mynewgraph = pd.read_parquet(config.INTERMEDIATE_DIR / "mynewgraph.parquet")
    df = graph_build.label_edge_types(mynewgraph)
    giant = graph_build.build_giant_component(df)

    out_path = config.INTERMEDIATE_DIR / "giant.pickle"
    io_utils.save_graph(giant, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
