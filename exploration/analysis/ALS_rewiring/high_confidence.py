"""Section 2: high-confidence new edges, unique to CTRL vs VCP, per timepoint."""
import pandas as pd


def select_high_confidence_new_edges(mydat: pd.DataFrame, threshold: float):
    mydat_new = mydat[mydat["present_in_ppi"] == 0]

    def pairs_above(col):
        return set(mydat_new.loc[mydat_new[col] >= threshold, "pair"])

    n_cyto_ct_22 = pairs_above("ct_d22_cyto")
    n_cyto_ct_35 = pairs_above("ct_d35_cyto")
    n_cyto_vcp_22 = pairs_above("vcp_d22_cyto")
    n_cyto_vcp_35 = pairs_above("vcp_d35_cyto")
    n_nuc_ct_22 = pairs_above("ct_d22_nuc")
    n_nuc_vcp_22 = pairs_above("vcp_d22_nuc")
    n_nuc_ct_35 = pairs_above("ct_d35_nuc")
    n_nuc_vcp_35 = pairs_above("vcp_d35_nuc")

    n_ct_22 = n_nuc_ct_22 | n_cyto_ct_22
    n_ct_35 = n_nuc_ct_35 | n_cyto_ct_35
    n_vcp_35 = n_nuc_vcp_35 | n_cyto_vcp_35
    n_vcp_22 = n_nuc_vcp_22 | n_cyto_vcp_22

    A = n_ct_22 - n_vcp_22  # unique new high-confidence edges at d22, CTRL only
    B = n_vcp_22 - n_ct_22  # unique new high-confidence edges at d22, VCP only
    C = n_ct_35 - n_vcp_35  # unique new high-confidence edges at d35, CTRL only
    D = n_vcp_35 - n_ct_35  # unique new high-confidence edges at d35, VCP only
    return A, B, C, D


def build_selected_edge_table(mydat: pd.DataFrame, A, B, C, D) -> pd.DataFrame:
    """All edges touching a protein involved in A|B|C|D, restricted to those
    selected edges plus every already-known PPI edge among those proteins."""
    selected_new = A | B | C | D
    seed_rows = mydat[mydat["pair"].isin(selected_new)]
    proteins = set(seed_rows["protA"]) | set(seed_rows["protB"])

    touches_proteins = mydat["protA"].isin(proteins) | mydat["protB"].isin(proteins)
    is_relevant = mydat["pair"].isin(selected_new) | (mydat["present_in_ppi"] == 1)
    mynewgraph = mydat.loc[touches_proteins & is_relevant, ["pair", "protA", "protB", "present_in_ppi"]].copy()

    mynewgraph["ct_22"] = mynewgraph["pair"].isin(A).astype(int)
    mynewgraph["vcp_22"] = mynewgraph["pair"].isin(B).astype(int)
    mynewgraph["ct_35"] = mynewgraph["pair"].isin(C).astype(int)
    mynewgraph["vcp_35"] = mynewgraph["pair"].isin(D).astype(int)
    return mynewgraph
