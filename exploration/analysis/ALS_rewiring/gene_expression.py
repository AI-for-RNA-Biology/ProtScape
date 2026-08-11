"""Per-condition mean gene expression.

Source matrix (already filtered/normalized) - see CYTO_GENE_EXPRESSION_CSV
below (`als_rewiring_cyto_gene_expression_csv` in configs/paths.yaml).
Columns are samples named `<replicate>_<day>_<genotype>`, e.g.
`CTRL1_D22_Wildtype`, `MUT2_D22_VCP-R155C` - multiple replicates per
condition. `condition_means()` log2(x+1)-transforms each sample, then
collapses replicates to one mean column per condition.
"""
import numpy as np
import pandas as pd

from downstream_tasks.config import PATHS

# Nuc/cyto separate columns (`_Cytoplasmic_`/`_Nuclear_` in the sample name),
# every timepoint (D0/D3/D7/D14/D22/D35) - used for the module log2FC-vs-D0
# heatmap, which wants the full time course. `load_cyto_gene_expression`
# keeps only the Cytoplasmic columns (nuclear dropped), as a proxy for
# whole-cell expression.
CYTO_GENE_EXPRESSION_CSV = str(PATHS["als_rewiring_cyto_gene_expression_csv"])

CONDITIONS = ["ctrl_d22", "vcp_d22", "ctrl_d35", "vcp_d35"]

_ALL_DAYS = ["d0", "d3", "d7", "d14", "d22", "d35"]
# D3/D7/D14 (plus D0 as the always-present baseline) only exist in the
# CYTO_GENE_EXPRESSION_CSV time course, not the merged D22/D35-only matrix -
# harmless to declare them here regardless of which matrix is actually in use,
# since `condition_means` only reads the day/genotype entries it's asked for.
CYTO_CONDITIONS = [f"{geno}_{day}" for day in _ALL_DAYS[1:] for geno in ("ctrl", "vcp")]  # D3..D35, no D0
CYTO_CONDITION_LABELS = {f"{geno}_{day}": f"{geno.upper()} {day.upper()}"
                         for day in _ALL_DAYS[1:] for geno in ("ctrl", "vcp")}

# D0 (baseline) entries exist only to compute `delta_vs_d0`'s delta-from-D0 -
# they're deliberately NOT in `CONDITIONS` above, since that drives the
# 4-condition box column in plotting.py and D0 isn't one of the 4 conditions
# ever displayed directly.
_DAY_BY_CONDITION = {f"{geno}_{day}": day.upper() for day in _ALL_DAYS for geno in ("ctrl", "vcp")}
# Sample names carry the genotype as "Wildtype" (CTRL) or "VCP-R155C"/"VCP-R191Q"
# (the VCP mutant lines) - "VCP" alone is enough to match either mutant column.
_GENOTYPE_KEYWORD_BY_CONDITION = {f"{geno}_{day}": ("Wildtype" if geno == "ctrl" else "VCP")
                                  for day in _ALL_DAYS for geno in ("ctrl", "vcp")}


def load_cyto_gene_expression(csv_path: str = CYTO_GENE_EXPRESSION_CSV) -> pd.DataFrame:
    """Raw gene x sample expression matrix from the NON-merged (nuc/cyto
    separate) preprocessing output, keeping only the Cytoplasmic columns
    (`_Cytoplasmic_` in the sample name) - nuclear samples dropped, cyto used
    as a proxy for whole-cell expression. Covers all 6 timepoints
    (D0/D3/D7/D14/D22/D35)."""
    expr = pd.read_csv(csv_path, index_col=0)
    cyto_cols = [c for c in expr.columns if "_Cytoplasmic_" in c]
    return expr[cyto_cols]


def condition_means(expr: pd.DataFrame, conditions: list = CONDITIONS) -> pd.DataFrame:
    """Collapses `expr`'s per-sample columns into one mean column per
    condition (`conditions`, default `CONDITIONS`). Each sample is
    `log2(x + 1)`-transformed BEFORE averaging over its condition's
    replicates - i.e. the mean-of-logs (~ geometric mean of the raw values),
    not the log of the arithmetic mean - the usual convention for expression
    data, since it's less skewed by a single high-expressing replicate than
    an arithmetic-then-log summary would be."""
    log_expr = np.log2(expr + 1)
    out = {}
    for cond in conditions:
        day, genotype_kw = _DAY_BY_CONDITION[cond], _GENOTYPE_KEYWORD_BY_CONDITION[cond]
        cols = [c for c in expr.columns if f"_{day}_" in c and genotype_kw in c]
        if not cols:
            raise ValueError(f"No sample columns matched condition {cond!r} (day={day}, genotype={genotype_kw!r})")
        out[cond] = log_expr[cols].mean(axis=1)
    return pd.DataFrame(out)


def delta_vs_d0(expr: pd.DataFrame, conditions: list = CONDITIONS) -> pd.DataFrame:
    """For each of `conditions`, mean log2(expr+1) at that condition/day
    minus that SAME genotype's Day 0 baseline - a log2 fold-change vs D0,
    per gene. Default `conditions=CONDITIONS`; pass `CYTO_CONDITIONS` for the
    full D3..D35 time course instead (see `cyto_log2fc_all_timepoints`)."""
    means = condition_means(expr, conditions=["ctrl_d0", "vcp_d0"] + list(conditions))
    out = {}
    for cond in conditions:
        baseline = "ctrl_d0" if cond.startswith("ctrl") else "vcp_d0"
        out[cond] = means[cond] - means[baseline]
    return pd.DataFrame(out)


def cyto_log2fc_all_timepoints(expr_cyto: pd.DataFrame) -> pd.DataFrame:
    """Log2 fold-change vs D0, for every CTRL/VCP x D3/D7/D14/D22/D35
    timepoint (`CYTO_CONDITIONS`) - the full time course, from the
    Cytoplasmic-only expression matrix (`load_cyto_gene_expression`), for
    `plotting.plot_module_log2fc_heatmap`."""
    return delta_vs_d0(expr_cyto, conditions=CYTO_CONDITIONS)
