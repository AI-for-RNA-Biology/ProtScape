"""Section 5: per-module enrichment - are *new* edges over-represented in a
module vs the rest? One-sided Fisher's exact test (module vs rest of the giant
component), no multiple-testing correction - matches `GetEnrichmentModule()`
in the R script, which was deliberately raw p-values "no FDR" per its own
header comment.
"""
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def edges_with_module(giant: nx.Graph) -> pd.DataFrame:
    edges = nx.to_pandas_edgelist(giant)
    module_of = nx.get_node_attributes(giant, "module")
    edges["from_mod"] = edges["source"].map(module_of)
    edges["to_mod"] = edges["target"].map(module_of)
    # only edges internal to a single module count towards that module's stats
    edges["module"] = np.where(edges["from_mod"] == edges["to_mod"], edges["from_mod"], np.nan)
    return edges


def get_enrichment_module(giant: nx.Graph, mod_palette: dict) -> pd.DataFrame:
    edges = edges_with_module(giant)
    module_of = nx.get_node_attributes(giant, "module")

    total_new = (edges["edge_type"] == "new").sum()
    total_known = (edges["edge_type"] == "known").sum()
    print(f"Giant component: total new edges = {total_new}  total known edges = {total_known}")

    rows = []
    for m in sorted(edges["module"].dropna().unique()):
        sub = edges[edges["module"] == m]
        mod_new, mod_known = (sub["edge_type"] == "new").sum(), (sub["edge_type"] == "known").sum()
        mod_total = mod_new + mod_known
        if mod_total == 0:
            continue

        bg_new, bg_known = total_new - mod_new, total_known - mod_known
        _, pval = fisher_exact([[mod_new, mod_known], [bg_new, bg_known]], alternative="greater")

        frac_new_module = mod_new / mod_total
        frac_new_bg = bg_new / (bg_new + bg_known) if (bg_new + bg_known) else np.nan
        fold_enrichment = frac_new_module / frac_new_bg if frac_new_bg else np.nan

        members = sorted(n for n, mm in module_of.items() if mm == m)
        rows.append({
            "module": int(m), "n_nodes": len(members), "n_edges_total": mod_total,
            "n_new_edges": mod_new, "n_known_edges": mod_known,
            "frac_new": frac_new_module, "fold_enrichment": fold_enrichment,
            "p_value": pval, "members": ";".join(members), "color": mod_palette[int(m)],
        })

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
