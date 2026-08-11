"""Section 3: edge typing + giant-component graph construction."""
import networkx as nx
import numpy as np
import pandas as pd


def label_edge_types(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate `protA`/`protB` pairs are collapsed, keeping the
    `present_in_ppi == 1` version if a pair appears twice. Labels each edge per
    day: known (present in the original PPI), other (new edge from another
    day), ct (new edge in cyto at that day), vcp (new edge in vcp at that day)."""
    df = df.sort_values("present_in_ppi", ascending=False).drop_duplicates("pair").copy()

    df["edge_type"] = np.where(df["present_in_ppi"] == 1, "known", "new")

    df["edge_type_d22"] = "other"
    df.loc[df["ct_22"] == 1, "edge_type_d22"] = "ct"
    df.loc[df["vcp_22"] == 1, "edge_type_d22"] = "vcp"
    df.loc[df["present_in_ppi"] == 1, "edge_type_d22"] = "known"

    df["edge_type_d35"] = "other"
    df.loc[df["ct_35"] == 1, "edge_type_d35"] = "ct"
    df.loc[df["vcp_35"] == 1, "edge_type_d35"] = "vcp"
    df.loc[df["present_in_ppi"] == 1, "edge_type_d35"] = "known"
    return df


def build_giant_component(df: pd.DataFrame) -> nx.Graph:
    G = nx.from_pandas_edgelist(
        df, source="protA", target="protB",
        edge_attr=["ct_22", "vcp_22", "ct_35", "vcp_35",
                   "edge_type", "edge_type_d22", "edge_type_d35", "present_in_ppi"],
    )
    for _, _, data in G.edges(data=True):
        data["weight"] = 1

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print(f"Number of connected components: {len(components)}")
    print([len(c) for c in components[:10]])

    giant = G.subgraph(components[0]).copy()
    print(f"Giant component nodes/edges: {giant.number_of_nodes()}, {giant.number_of_edges()}")
    return giant
