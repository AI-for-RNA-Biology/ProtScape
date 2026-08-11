"""Pathway (GO term) embedding of Leiden modules.

For each (GO pathway, module) pair:

    value = (# module genes annotated to that GO term / total # genes
              annotated to that GO term) * (1 / total # genes in that module)

This is a fixed weighting, not a hypergeometric/Fisher enrichment test -
modules that cover a large share of a small pathway score highly, scaled down
for large modules so a big module doesn't score highly on every pathway just
by virtue of containing many genes.

Reused at the end of stage 05 (`scripts/05_cluster_leiden.py`) and after every
split step of the hierarchical re-clustering (`scripts/05b_hierarchical_recluster.py`,
`hierarchical_recluster.recursive_refine_by_density`), since both
produce the same `(module_of, mod_ids)` shape this module operates on.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist, squareform


def load_go_terms(csv_path: Path) -> pd.DataFrame:
    """Loads the GO term annotation table (columns `GO_id`, `Description`,
    `Gene_Symbols`) into a DataFrame indexed by `GO_id`, with a `genes` column
    holding each term's gene symbols as a set (comma-separated in the source
    CSV, and not de-duplicated there - e.g. GO:0000002 lists SLC25A36 twice)."""
    df = pd.read_csv(csv_path)
    df["genes"] = df["Gene_Symbols"].apply(lambda s: frozenset(g.strip() for g in s.split(",")))
    return df.set_index("GO_id")[["Description", "genes"]]


def compute_pathway_embedding(module_of: dict, mod_ids, go_terms: pd.DataFrame) -> pd.DataFrame:
    """Pathway x module embedding matrix (see module docstring for the
    formula), returned as a DataFrame indexed by GO_id with one column per
    entry of `mod_ids`.

    `module_of` maps gene -> module id; `mod_ids` is the module id list for
    the partition being embedded (kept as an explicit argument, rather than
    inferred as `set(module_of.values())`, so a module with zero surviving
    genes still gets an all-zero column instead of silently vanishing).

    Built as a single sparse matrix product (GO-term x gene incidence times
    gene x module incidence) rather than a per-module/per-term Python loop,
    since this is expected to be called once per stage-05 run and once per
    hierarchical-recluster split step - both GO-term count (~19k) and repeat
    calls make a quadratic Python loop too slow.
    """
    all_genes = sorted(set().union(*go_terms["genes"]) | set(module_of))
    gene_idx = {g: i for i, g in enumerate(all_genes)}

    go_ids = go_terms.index.to_numpy()
    go_rows, go_cols = [], []
    for gi, genes in enumerate(go_terms["genes"]):
        for g in genes:
            go_rows.append(gi)
            go_cols.append(gene_idx[g])
    go_incidence = sparse.csr_matrix(
        (np.ones(len(go_rows)), (go_rows, go_cols)),
        shape=(len(go_ids), len(all_genes)),
    )
    pathway_totals = np.asarray(go_incidence.sum(axis=1)).ravel()

    mod_index = {m: i for i, m in enumerate(mod_ids)}
    mod_rows, mod_cols = [], []
    for gene, m in module_of.items():
        if m not in mod_index:
            continue
        mod_rows.append(gene_idx[gene])
        mod_cols.append(mod_index[m])
    module_incidence = sparse.csr_matrix(
        (np.ones(len(mod_rows)), (mod_rows, mod_cols)),
        shape=(len(all_genes), len(mod_ids)),
    )
    module_sizes = np.asarray(module_incidence.sum(axis=0)).ravel()

    overlap = (go_incidence @ module_incidence).toarray()

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_of_pathway = overlap / pathway_totals[:, None]
        embedding = frac_of_pathway / module_sizes[None, :]
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame(embedding, index=go_ids, columns=mod_ids)


def compute_module_distance_fast(embedding: pd.DataFrame, metric: str = "cosine") -> pd.DataFrame:
    """Fast module x module distance matrix, straight from the pathway
    embedding columns - one vectorized `scipy.spatial.distance.pdist` call.
    No per-pathway ground-cost matrix, no per-pair optimization needed, so
    this is cheap enough to call once per split step of the hierarchical
    re-clustering (05b).

    Default `metric="cosine"` treats each module's embedding column as a
    plain vector (not renormalized into a probability distribution) and
    measures the angle between two modules' pathway-weight vectors - unlike
    an optimal-transport (Wasserstein) distance, it has no notion of two
    *different* pathways being "close" (e.g. via shared genes), so it can't
    detect similarity between modules that enrich for related-but-non-identical
    pathways. Any metric `scipy.spatial.distance.pdist` accepts works instead
    (e.g. "euclidean", "jensenshannon" - the latter also distribution-aware,
    just without a ground cost between pathways).

    Rows (pathways) with zero mass across every module are dropped first -
    they don't change cosine/euclidean distance (a shared all-zero
    coordinate contributes nothing to either dot products or norms), but
    `embedding` has one row per GO term (~19k), almost all irrelevant to any
    given partition, so this keeps the vectors `pdist` actually works with
    small.

    Returns a symmetric module x module DataFrame (zero diagonal). A module
    with an all-zero column has no vector to compare and gets NaN against
    every other module.
    """
    active_pathways = (embedding != 0).any(axis=1)
    mod_ids = embedding.columns.tolist()
    X = embedding.loc[active_pathways].to_numpy().T  # modules x active pathways
    has_mass = X.sum(axis=1) > 0

    n = len(mod_ids)
    dist = np.full((n, n), np.nan)
    np.fill_diagonal(dist, 0.0)
    active = np.flatnonzero(has_mass)
    if len(active) > 1:
        dist[np.ix_(active, active)] = squareform(pdist(X[active], metric=metric))

    return pd.DataFrame(dist, index=mod_ids, columns=mod_ids)


def save_pathway_embedding(embedding: pd.DataFrame, path: Path) -> None:
    """Parquet requires string column labels; module ids come in as ints from
    `mod_ids`/`module_of`, so stringify them here rather than pushing that
    concern onto every caller of `compute_pathway_embedding`."""
    embedding = embedding.set_axis(embedding.columns.astype(str), axis="columns")
    embedding.to_parquet(path)


def save_module_distances(distances: pd.DataFrame, path: Path) -> None:
    """Same string-column-label requirement as `save_pathway_embedding` -
    `compute_module_distance_fast`'s module x module DataFrame also comes
    back with int column labels."""
    distances = distances.set_axis(distances.columns.astype(str), axis="columns")
    distances.to_parquet(path)


def pairwise_distance_stats(distances: pd.DataFrame) -> "tuple[float, float, int]":
    """Mean and standard deviation of the pairwise distances in a symmetric
    module x module `distances` matrix (as returned by
    `compute_module_distance_fast`), taken over the upper triangle
    (excluding the zero diagonal) so each pair is counted once. Module pairs
    where one side had no distribution/vector (NaN) are dropped. Returns
    `(mean, std, n_pairs)`; `(nan, nan, 0)` if no pair has a finite
    distance."""
    upper = distances.to_numpy()[np.triu_indices(len(distances), k=1)]
    upper = upper[~np.isnan(upper)]
    if len(upper) == 0:
        return float("nan"), float("nan"), 0
    return float(upper.mean()), float(upper.std()), len(upper)
