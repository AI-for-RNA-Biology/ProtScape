"""Align STRING scores and save across-context loss-score moments."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from exploration.analysis.loss_consensus_scoring import (
    LOSSES,
    clean_gene,
    require_file,
)


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("r")


def build_string_scores(genes: list[str], paths: dict) -> np.ndarray:
    info_path = require_file(Path(paths["string_protein_info"]).expanduser())
    links_path = require_file(Path(paths["string_links_detailed"]).expanduser())
    protein_to_gene = {}
    with open_text(info_path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        protein_col = header.index("#string_protein_id")
        gene_col = header.index("preferred_name")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            protein_to_gene[fields[protein_col]] = clean_gene(fields[gene_col])

    gene_to_id = {gene: index for index, gene in enumerate(genes)}
    n_genes = len(genes)
    scores = np.zeros(n_genes * (n_genes - 1) // 2, dtype=np.float32)
    with open_text(links_path) as handle:
        header = handle.readline().split()
        p1_col = header.index("protein1")
        p2_col = header.index("protein2")
        score_col = header.index("combined_score")
        for line in handle:
            fields = line.split()
            gene_a = gene_to_id.get(protein_to_gene.get(fields[p1_col], ""))
            gene_b = gene_to_id.get(protein_to_gene.get(fields[p2_col], ""))
            if gene_a is None or gene_b is None or gene_a == gene_b:
                continue
            lo, hi = sorted((gene_a, gene_b))
            pair = lo * n_genes - (lo * (lo + 1)) // 2 + (hi - lo - 1)
            score = int(fields[score_col]) / 1000.0
            if score > scores[pair]:
                scores[pair] = score
    print(f"Aligned STRING scores for {(scores > 0).sum():,} protein pairs", flush=True)
    return scores


def save_string_moments(
    results: dict, string_scores: np.ndarray, output_dir: Path
):
    keep = results["string_count"] >= 2
    counts = results["string_count"][keep]
    sums = results["string_sum"][:, keep]
    sum_squares = results["string_sum_sq"][:, keep]
    means = sums / counts
    variances = np.maximum(sum_squares / counts - means * means, 0.0)
    pair_ids_selected = results["string_pairs"][keep]
    np.savez_compressed(
        output_dir / "string_positive_loss_score_moments.npz",
        pair_scope=np.asarray("string_positive_copresent_at_least_2"),
        losses=np.asarray(LOSSES),
        pair_ids=pair_ids_selected,
        string_combined=string_scores[pair_ids_selected],
        mean_scores=means.astype(np.float32),
        std_scores=np.sqrt(variances).astype(np.float32),
        n_copresent_contexts=counts,
        n_observed_contexts=results["string_observed"][keep],
    )
