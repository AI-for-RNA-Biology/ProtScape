"""Build STRING mappings, enrichments, and the Parkinson interaction network."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from downstream_tasks.config import PATHS


STRING_INFO = Path(PATHS["string_protein_info"])
STRING_ALIASES = Path(PATHS["string_protein_aliases"])
STRING_TERMS = Path(PATHS["string_enrichment_terms"])
STRING_LINKS = Path(PATHS["string_links_detailed"])

ALIAS_PRIORITY = (
    "KEGG_NAME",
    "UniProt_GN_Name",
    "Ensembl_HGNC_alias_symbol",
    "Ensembl_HGNC_symbol",
)
STRING_CATEGORIES = {
    "Biological Process (Gene Ontology)": (
        "Process",
        "GO biological process",
    ),
    "Molecular Function (Gene Ontology)": (
        "Function",
        "GO molecular function",
    ),
    "Protein Domains (Pfam)": ("Pfam", "Pfam"),
    "Reactome Pathways": ("RCTM", "Reactome"),
}


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required Parkinson analysis input: {path}")
    return path


def benjamini_hochberg(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    order = np.argsort(array)
    adjusted = array[order] * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def build_string_mapping(genes: list[str]) -> pd.DataFrame:
    info = pd.read_csv(require_file(STRING_INFO), sep="\t").rename(
        columns={"#string_protein_id": "string_id"}
    )
    preferred = (
        info.assign(symbol=info["preferred_name"].str.upper())
        .groupby("symbol")["string_id"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    chosen = {}
    unresolved = set(genes)
    for gene in genes:
        hits = preferred.get(gene, [])
        if len(hits) == 1:
            chosen[gene] = (hits[0], "preferred_name")
            unresolved.remove(gene)

    alias_hits = {
        gene: {source: set() for source in ALIAS_PRIORITY} for gene in unresolved
    }
    with gzip.open(require_file(STRING_ALIASES), "rt") as handle:
        next(handle)
        for line in handle:
            string_id, alias, source = line.rstrip("\n").split("\t", 2)
            symbol = alias.upper()
            if symbol in alias_hits and source in alias_hits[symbol]:
                alias_hits[symbol][source].add(string_id)

    rows = []
    for gene in genes:
        string_id, method = chosen.get(gene, ("", "none"))
        if not string_id:
            for source in ALIAS_PRIORITY:
                hits = alias_hits[gene][source]
                if len(hits) == 1:
                    string_id, method = next(iter(hits)), source
                    break
        rows.append(
            {
                "protein": gene,
                "string_id": string_id,
                "mapping_method": method,
                "mapped": bool(string_id),
            }
        )
    return pd.DataFrame(rows)


def read_string_annotations(allowed_ids: set[str]) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(
        require_file(STRING_TERMS), sep="\t", chunksize=500_000
    ):
        chunk = chunk[
            chunk["#string_protein_id"].isin(allowed_ids)
            & chunk["category"].isin(STRING_CATEGORIES)
        ].copy()
        if chunk.empty:
            continue
        chunk[["category_code", "source"]] = (
            chunk["category"].map(STRING_CATEGORIES).apply(pd.Series)
        )
        parts.append(
            chunk.rename(columns={"#string_protein_id": "string_id"})[
                ["string_id", "category_code", "source", "term", "description"]
            ]
        )
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def enrichment_table(
    annotations: pd.DataFrame,
    string_to_gene: dict[str, str],
    set_name: str,
    query_ids: set[str],
    background_ids: set[str],
) -> pd.DataFrame:
    annotations = annotations[annotations["string_id"].isin(background_ids)]
    keys = ["category_code", "source", "term", "description"]
    result = (
        annotations.groupby(keys, as_index=False)["string_id"]
        .nunique()
        .rename(columns={"string_id": "term_size"})
    )
    result = result[result["term_size"].between(10, 500)].copy()
    hits = (
        annotations[annotations["string_id"].isin(query_ids)]
        .groupby(keys)["string_id"]
        .agg(lambda values: sorted(set(values)))
    )
    result["hit_string_ids"] = [
        hits.get(tuple(row), [])
        for row in result[keys].itertuples(index=False, name=None)
    ]
    result["gene_count"] = result["hit_string_ids"].map(len)
    result["p_value"] = hypergeom.sf(
        result["gene_count"] - 1,
        len(background_ids),
        result["term_size"],
        len(query_ids),
    )
    result["fdr"] = result.groupby("source")["p_value"].transform(
        benjamini_hochberg
    )
    result["fold_enrichment"] = (
        result["gene_count"] / len(query_ids)
    ) / (result["term_size"] / len(background_ids))
    result["log2_fold_enrichment"] = np.nan
    positive = result["fold_enrichment"].gt(0)
    result.loc[positive, "log2_fold_enrichment"] = np.log2(
        result.loc[positive, "fold_enrichment"]
    )
    result["minus_log10_fdr"] = -np.log10(result["fdr"].clip(lower=1e-300))
    result["set"] = set_name
    result["category"] = result["category_code"]
    result["number_of_genes"] = result["gene_count"]
    result["hit_gene_symbols"] = result["hit_string_ids"].map(
        lambda ids: ";".join(
            sorted(string_to_gene[string_id] for string_id in ids)
        )
    )
    return result


def build_main_string_enrichment(
    membership: pd.DataFrame,
    candidates: pd.DataFrame,
    mapping: pd.DataFrame,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_to_string = (
        mapping[mapping["mapped"]]
        .set_index("protein")["string_id"]
        .to_dict()
    )
    string_to_gene = {
        string_id: gene for gene, string_id in gene_to_string.items()
    }
    labelled = set(
        membership.loc[
            ~membership["benchmark_split"].eq("label_excluded"), "protein"
        ]
    )
    label_excluded = set(
        membership.loc[
            membership["benchmark_split"].eq("label_excluded"), "protein"
        ]
    )
    positives = set(
        membership.loc[membership["benchmark_label"].eq(1), "protein"]
    )
    protscape_candidates = set(
        candidates.loc[candidates["model"].eq("protscape"), "protein"]
    )
    specifications = [
        ("known_targets", positives, labelled),
        ("protscape_candidates", protscape_candidates, label_excluded),
    ]
    tables = []
    for set_name, query_genes, background_genes in specifications:
        tables.append(
            enrichment_table(
                annotations,
                string_to_gene,
                set_name,
                {gene_to_string[g] for g in query_genes if g in gene_to_string},
                {
                    gene_to_string[g]
                    for g in background_genes
                    if g in gene_to_string
                },
            )
        )
    enrichment = pd.concat(tables, ignore_index=True)

    known = enrichment[
        enrichment["set"].eq("known_targets")
        & enrichment["category"].isin(["Process", "Function", "Pfam"])
    ]
    selected = []
    for category_order, category in enumerate(["Process", "Function", "Pfam"]):
        rows = (
            known[known["category"].eq(category)]
            .sort_values(
                ["fdr", "fold_enrichment", "term"],
                ascending=[True, False, True],
            )
            .head(3)
            .copy()
        )
        rows["category_order"] = category_order
        rows["term_order"] = np.arange(len(rows))
        selected.append(rows)
    selected = pd.concat(selected, ignore_index=True)[
        ["category", "term", "description", "category_order", "term_order"]
    ]
    display = selected.merge(
        enrichment[
            enrichment["set"].isin(
                ["known_targets", "protscape_candidates"]
            )
        ],
        on=["category", "term", "description"],
        how="left",
    )
    return selected, display


def build_string_network(
    membership: pd.DataFrame,
    candidates: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    known = set(
        membership.loc[membership["benchmark_label"].eq(1), "protein"]
    )
    novel = set(
        candidates.loc[candidates["model"].eq("protscape"), "protein"]
    )
    submitted = known | novel
    gene_to_string = (
        mapping[mapping["mapped"]]
        .set_index("protein")["string_id"]
        .to_dict()
    )
    string_to_gene = {
        string_id: gene for gene, string_id in gene_to_string.items()
    }
    ids = {
        gene_to_string[gene] for gene in submitted if gene in gene_to_string
    }
    parts = []
    usecols = ["protein1", "protein2", "experimental", "combined_score"]
    for chunk in pd.read_csv(
        require_file(STRING_LINKS),
        sep=r"\s+",
        usecols=usecols,
        chunksize=500_000,
    ):
        selected = chunk[
            chunk["experimental"].gt(0)
            & chunk["protein1"].isin(ids)
            & chunk["protein2"].isin(ids)
        ].copy()
        if selected.empty:
            continue
        first = selected["protein1"].map(string_to_gene)
        second = selected["protein2"].map(string_to_gene)
        selected["protein_a"] = [min(a, b) for a, b in zip(first, second)]
        selected["protein_b"] = [max(a, b) for a, b in zip(first, second)]
        selected["experimental_score"] = selected["experimental"] / 1000
        selected["combined_score"] = selected["combined_score"] / 1000
        parts.append(
            selected[
                [
                    "protein_a",
                    "protein_b",
                    "experimental_score",
                    "combined_score",
                ]
            ]
        )
    edges = (
        pd.concat(parts, ignore_index=True)
        .sort_values(
            ["experimental_score", "combined_score"], ascending=False
        )
        .drop_duplicates(["protein_a", "protein_b"])
    )
    degree = pd.concat([edges["protein_a"], edges["protein_b"]]).value_counts()
    nodes = pd.DataFrame({"protein": sorted(submitted)})
    nodes["node_role"] = np.where(
        nodes["protein"].isin(known), "benchmark_positive", "candidate"
    )
    nodes["string_id"] = nodes["protein"].map(gene_to_string)
    nodes["mapped"] = nodes["string_id"].notna()
    nodes["degree"] = nodes["protein"].map(degree).fillna(0).astype(int)
    nodes["connected"] = nodes["degree"].gt(0)
    return nodes, edges
