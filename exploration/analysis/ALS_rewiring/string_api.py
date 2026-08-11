"""Section 7 (+ section 9 reuse): STRING functional enrichment REST calls.

Same REST call as the R script (`POST` so identifier lists aren't limited by
URL length), politely rate-limited to ~1 request/second.
"""
import io
import time

import pandas as pd
import requests

STRING_ENRICHMENT_URL = "https://string-db.org/api/tsv/enrichment"


def query_string_enrichment(genes, species: int = 9606,
                             caller_identity: str = "python_ppi_module_enrichment",
                             min_genes: int = 10) -> "pd.DataFrame | None":
    genes = [g for g in genes if g and not pd.isna(g)]
    if len(genes) < min_genes:
        return None

    resp = requests.post(STRING_ENRICHMENT_URL, data={
        "identifiers": "%0d".join(genes), "species": species, "caller_identity": caller_identity,
    })
    if not resp.ok or not resp.text.strip():
        return None
    return pd.read_csv(io.StringIO(resp.text), sep="\t")


# Columns downstream code (plotting.plot_pathways, and any script re-loading
# this from a saved CSV) reads even when nothing came back from STRING - a
# plain `pd.DataFrame()` has no columns at all, which round-trips through
# `to_csv`/`read_csv` as a *totally empty file* that `pd.read_csv` errors on
# (`EmptyDataError`). Kept minimal (real STRING responses have more columns);
# only matters when every single module comes back empty.
EMPTY_ENRICHMENT_COLUMNS = ["category", "term", "number_of_genes", "p_value", "description",
                            "module", "n_proteins_in_module"]


def run_string_enrichment_per_module(module_members: dict, rate_limit_s: float = 1.0) -> pd.DataFrame:
    results = []
    for m, genes in module_members.items():
        print(f"Module {m} ({len(genes)} proteins): querying STRING...")
        res = query_string_enrichment(genes)
        if res is not None and len(res):
            res["module"] = m
            res["n_proteins_in_module"] = len(genes)
            results.append(res)
        else:
            print(f"  -> no significant enrichment terms for module {m}")
        time.sleep(rate_limit_s)

    if results:
        all_enrichment = pd.concat(results, ignore_index=True)
        all_enrichment = all_enrichment[all_enrichment["number_of_genes"] > 10]
    else:
        all_enrichment = pd.DataFrame(columns=EMPTY_ENRICHMENT_COLUMNS)
    return all_enrichment


def save_module_pathway_csv(module_ids, all_enrichment: pd.DataFrame, out_path,
                             categories=("RCTM", "Process", "Function", "Component")) -> None:
    """Saves the full list of enriched pathway terms for exactly
    `module_ids` as a flat CSV - one row per (module, term), restricted to
    `categories` (default: RCTM + all 3 GO categories - Biological Process,
    Molecular Function, Cellular Component). Columns kept first, in this
    order, if present: category, term, number_of_genes,
    number_of_genes_in_background, p_value, fdr, description, module - any
    other columns STRING's response carries (e.g. inputGenes,
    preferredNames) are appended after. Rows sorted by (module, p_value)."""
    preferred_cols = ["category", "term", "number_of_genes", "number_of_genes_in_background",
                       "p_value", "fdr", "description", "module"]
    if not module_ids:
        pd.DataFrame(columns=preferred_cols).to_csv(out_path, index=False)
        print(f"No modules of interest for {out_path} - saved empty CSV")
        return

    sub = all_enrichment[
        all_enrichment["module"].isin(module_ids) & all_enrichment["category"].isin(categories)
    ].copy()
    cols = [c for c in preferred_cols if c in sub.columns] + [c for c in sub.columns if c not in preferred_cols]
    sub = sub[cols].sort_values(["module", "p_value"])
    sub.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def run_component_enrichment(components_df: pd.DataFrame, min_nodes: int = 11,
                              categories=("Function",), rate_limit_s: float = 1.0) -> pd.DataFrame:
    """STRING functional enrichment for every component with more than 10 genes
    (`min_nodes=11`), restricted to `categories` (STRING's GO Molecular Function
    terms by default - drops Component, Process, RCTM, SMART, WikiPathways,
    KEGG, etc.). Results are keyed by (module, component) so they can be
    matched back to `components_df` and to the right PDF page. Unlike the
    module-level `run_string_enrichment_per_module` (which drops terms with
    `number_of_genes <= 10`), the filter here is just `>= 2` - components are
    small enough that the stricter module-level cutoff would discard almost
    everything."""
    results = []
    for _, row in components_df.iterrows():
        genes = row["genes"]
        if len(genes) < min_nodes:
            continue
        print(f"Module {row['module']} component {row['component']} ({len(genes)} genes): querying STRING...")
        res = query_string_enrichment(genes)
        if res is not None and len(res):
            res = res[res["category"].isin(categories)]
        if res is not None and len(res):
            res["module"] = row["module"]
            res["component"] = row["component"]
            results.append(res)
        else:
            print("  -> no significant enrichment terms")
        time.sleep(rate_limit_s)

    enrichment = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if len(enrichment):
        enrichment = enrichment[enrichment["number_of_genes"] >= 2]
    return enrichment
