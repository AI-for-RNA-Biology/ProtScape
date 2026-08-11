"""Section 6: per-module, per-condition enrichment (known vs new, CTRL vs VCP,
d22 vs d35).

- `count_edges_per_module` - the shared counts table every downstream test reads from,
- three enrichment tables (`enrichment_new_vs_known`, `enrichment_condition_vs_known`,
  `enrichment_vcp_vs_ct`) that mirror the R script's sections A/B/C,
- `global_distribution_shift` - the corrected version of the R script's "global
  shift" chi-square diagnostics (see its docstring for why the R version doesn't
  test what its comments claim),
- `modules_of_interest` - the four FE/padj-cutoff buckets used throughout the
  rest of the pipeline.
"""
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

from .module_enrichment import edges_with_module

CONDITIONS = {
    "known": lambda e: e["present_in_ppi"] == 1,
    "new": lambda e: e["present_in_ppi"] == 0,
    "d22": lambda e: e["edge_type_d22"].isin(["ct", "vcp"]),
    "d35": lambda e: e["edge_type_d35"].isin(["ct", "vcp"]),
    "ct": lambda e: (e["edge_type_d22"] == "ct") | (e["edge_type_d35"] == "ct"),
    "vcp": lambda e: (e["edge_type_d22"] == "vcp") | (e["edge_type_d35"] == "vcp"),
    "ct_d22": lambda e: e["edge_type_d22"] == "ct",
    "vcp_22": lambda e: e["edge_type_d22"] == "vcp",
    "ct_35": lambda e: e["edge_type_d35"] == "ct",
    "vcp_35": lambda e: e["edge_type_d35"] == "vcp",
}


def _fmt_p(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.1e}"


def count_edges_per_module(giant: nx.Graph, mod_ids) -> pd.DataFrame:
    """Per-module edge counts, restricted to edges internal to a module
    (`edge_df_mod` in the R script), one column per condition plus 'all'."""
    edges = edges_with_module(giant)
    edges_in_modules = edges.dropna(subset=["module"])

    counts = pd.DataFrame(
        {"all": edges_in_modules["module"].value_counts().reindex(mod_ids, fill_value=0)}
    )
    for name, cond in CONDITIONS.items():
        mask = cond(edges_in_modules)
        counts[name] = (
            edges_in_modules.loc[mask, "module"].value_counts().reindex(mod_ids, fill_value=0)
        )
    return counts.astype(int)


def _fisher_module_vs_rest(counts: pd.DataFrame, col_a: str, col_b: str) -> np.ndarray:
    pvals = []
    for m in counts.index:
        in_mod = counts.loc[m, [col_a, col_b]].to_numpy()
        rest = counts.drop(index=m)[[col_a, col_b]].sum().to_numpy()
        _, p = fisher_exact([in_mod, rest])
        pvals.append(p)
    return np.array(pvals)


def enrichment_new_vs_known(counts: pd.DataFrame) -> pd.DataFrame:
    """Section A: are new edges enriched in specific modules compared to known?"""
    perc = 100 * counts / counts.sum()
    pvals = _fisher_module_vs_rest(counts, "known", "new")
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")
    return pd.DataFrame({
        "n_edges_tot": counts["all"], "perc_known": perc["known"], "perc_new": perc["new"],
        "FE": perc["new"] / perc["known"], "pval": pvals, "padj": padj,
    })


def enrichment_condition_vs_known(counts: pd.DataFrame, condition_cols) -> pd.DataFrame:
    """Section B: each condition's new-edge fraction vs the known-PPI background,
    fold enrichment + FDR-adjusted p-value per module per condition. `perc_known`
    (like every other `perc_*` column here) is this module's share of ALL known
    edges in the graph - not the fraction of this module's own edges that are
    known - matching R's column-normalized `per_edges_modules`."""
    perc = 100 * counts / counts.sum()
    out = {"n_tot": counts["all"], "n_known": counts["known"], "perc_known": perc["known"]}
    pvals_by_col = {col: _fisher_module_vs_rest(counts, "known", col) for col in condition_cols}

    all_p = np.concatenate(list(pvals_by_col.values()))
    _, all_padj, _, _ = multipletests(all_p, method="fdr_bh")
    padj_by_col = dict(zip(condition_cols, np.split(all_padj, len(condition_cols))))

    for col in condition_cols:
        out[f"perc_{col}"] = perc[col]
        out[f"FE_{col}"] = perc[col] / perc["known"]
        out[f"padj_{col}"] = padj_by_col[col]
    return pd.DataFrame(out)


def enrichment_vcp_vs_ct(counts: pd.DataFrame) -> pd.DataFrame:
    """Section C: compare VCP vs CTRL directly, at each timepoint."""
    perc = 100 * counts / counts.sum()
    min_ct22 = perc.loc[perc["ct_d22"] > 0, "ct_d22"].min()
    min_ct35 = perc.loc[perc["ct_35"] > 0, "ct_35"].min()

    fe_d22 = perc["vcp_22"] / (min_ct22 + perc["ct_d22"])
    fe_d35 = perc["vcp_35"] / (min_ct35 + perc["ct_35"])

    pval_d22 = _fisher_module_vs_rest(counts, "vcp_22", "ct_d22")
    pval_d35 = _fisher_module_vs_rest(counts, "vcp_35", "ct_35")
    _, padj_all, _, _ = multipletests(np.concatenate([pval_d22, pval_d35]), method="fdr_bh")
    padj_d22, padj_d35 = np.split(padj_all, 2)

    return pd.DataFrame({
        "module": counts.index, "n_tot": counts["all"], "n_known": counts["known"],
        "FE_cond_d22": fe_d22.to_numpy(), "FE_cond_d35": fe_d35.to_numpy(),
        "padj_d22": padj_d22, "padj_d35": padj_d35,
        "perc_ct_d22": perc["ct_d22"].to_numpy(), "perc_vcp_22": perc["vcp_22"].to_numpy(),
        "perc_ct_35": perc["ct_35"].to_numpy(), "perc_vcp_35": perc["vcp_35"].to_numpy(),
    })


def global_distribution_shift(counts: pd.DataFrame, cols) -> pd.DataFrame:
    """The R script's "global shift" diagnostics run
    `chisq.test(x=<count vector>, y=<count vector>)` across whole per-module
    count columns (e.g. `all` vs `known`). Passing two numeric vectors this way
    makes R build a contingency table by *tabulating pairs of values* - it does
    not test whether `known`'s distribution across modules differs from `all`'s,
    which is what the comments say it's for. This is the direct replacement: a
    single chi-square test of independence on the module-by-condition
    contingency table itself - is a condition's count distributed across
    modules the same way as another condition's, or as the 'all' background?"""
    rows = []
    for col_a, col_b in cols:
        table = counts[[col_a, col_b]].to_numpy()
        # Modules with zero count in both columns carry no information for
        # this comparison and make chi2_contingency's internally-computed
        # expected-frequency table zero-valued (it raises ValueError on
        # that) - drop them. With finer partitions (e.g. the hierarchical
        # re-clustering experiment's smaller modules) this is common.
        table = table[table.sum(axis=1) > 0]
        if table.shape[0] < 2 or (table.sum(axis=0) == 0).any():
            # Too few informative modules left, or one whole column is now
            # empty (e.g. no module has any edge of that condition) - a
            # chi-square test of independence isn't meaningful here.
            rows.append({"compare": f"{col_a}_vs_{col_b}", "chi2": np.nan, "dof": np.nan, "p_value": np.nan})
            continue
        chi2, p, dof, _ = chi2_contingency(table)
        rows.append({"compare": f"{col_a}_vs_{col_b}", "chi2": chi2, "dof": dof, "p_value": p})
    return pd.DataFrame(rows)


def modules_of_interest(enrichment_vcp_ct: pd.DataFrame, fe_cutoff: float = 1.2,
                         padj_cutoff: float = 0.01) -> dict:
    """Same four buckets as the R script's `modules_of_interest` list."""
    df = enrichment_vcp_ct
    return {
        "vcp_d22": df.index[(df.FE_cond_d22 > fe_cutoff) & (df.padj_d22 < padj_cutoff)].tolist(),
        "vcp_d35": df.index[(df.FE_cond_d35 > fe_cutoff) & (df.padj_d35 < padj_cutoff)].tolist(),
        "ct_d22": df.index[(df.FE_cond_d22 < 1 / fe_cutoff) & (df.padj_d22 < padj_cutoff)].tolist(),
        "ct_d35": df.index[(df.FE_cond_d35 < 1 / fe_cutoff) & (df.padj_d35 < padj_cutoff)].tolist(),
    }
