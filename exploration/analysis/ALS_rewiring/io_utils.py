"""Loading the raw input tables, and saving/loading intermediate pipeline
state (DataFrames, the giant graph, and small dicts) between stage scripts.

Every stage script reads its inputs and writes its outputs through these
helpers, so the on-disk format for each kind of object is defined in exactly
one place.
"""
import json
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd


def load_pair_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.set_index(df["protA"] + "_" + df["protB"])
    return df


def merge_d22_d35(d22: pd.DataFrame, d35: pd.DataFrame) -> pd.DataFrame:
    """Recreates `mydat` from the R script: one row per protein pair seen in
    either timepoint, with per-condition/per-compartment S2GAE scores."""
    all_pairs = d22.index.union(d35.index)
    d22r = d22.reindex(all_pairs)
    d35r = d35.reindex(all_pairs)

    mydat = pd.DataFrame({
        "pair": all_pairs,
        "protA": d22r["protA"].combine_first(d35r["protA"]).values,
        "protB": d22r["protB"].combine_first(d35r["protB"]).values,
        "present_in_ppi": d22r["present_in_ppi"].combine_first(d35r["present_in_ppi"]).values,
        "stringdb_physical_score": d22r["stringdb_physical_score"]
            .combine_first(d35r["stringdb_physical_score"]).values,
        "stringdb_combined_score": d22r["stringdb_combined_score"]
            .combine_first(d35r["stringdb_combined_score"]).values,
        "ct_d22_nuc": d22r["s2gae_score_CTRL_nuc_d22"].values,
        "ct_d35_nuc": d35r["s2gae_score_CTRL_nuc_d35"].values,
        "ct_d22_cyto": d22r["s2gae_score_CTRL_cyto_d22"].values,
        "ct_d35_cyto": d35r["s2gae_score_CTRL_cyto_d35"].values,
        "vcp_d22_nuc": d22r["s2gae_score_VCP_nuc_d22"].values,
        "vcp_d35_nuc": d35r["s2gae_score_VCP_nuc_d35"].values,
        "vcp_d22_cyto": d22r["s2gae_score_VCP_cyto_d22"].values,
        "vcp_d35_cyto": d35r["s2gae_score_VCP_cyto_d35"].values,
    })
    return mydat


# --- intermediate-state I/O -------------------------------------------------

def save_graph(g: nx.Graph, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(g, f)


def load_graph(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path: Path) -> None:
    """JSON-safe dump for dicts with int/str keys and tuple values (e.g.
    `mod_palette`'s RGBA tuples) - dict keys are coerced to str (JSON has no
    integer keys) and tuples become lists."""
    def default(o):
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(f"not JSON-serializable: {type(o)}")

    with open(path, "w") as f:
        json.dump(obj, f, default=default)


def load_json_int_keys(path: Path) -> dict:
    """Loads a JSON object back into a dict with int keys (module ids) - the
    inverse of what `save_json` had to do to make those keys JSON-legal."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
