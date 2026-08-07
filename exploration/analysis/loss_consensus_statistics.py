"""Summarize agreement between ProtScape loss variants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from exploration.analysis.loss_consensus_scoring import (
    CLASS_NAMES,
    LABEL_NAMES,
    LOSSES,
    SCORE_CUTOFF,
)


LOSS_PAIRS = ((0, 1), (0, 2))
SIMILARITY_GROUPS = {1: "Labelled positives", 0: "Labelled negatives"}


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def exact_similarity_rows(
    context: str, group: str, scores: np.ndarray
) -> list[dict]:
    """Compute exact score and thresholded-call similarities."""
    n = scores.shape[1]
    if n == 0:
        return []

    ranks = np.empty((len(LOSSES), n), dtype=np.float64)
    for loss_index in range(len(LOSSES)):
        ranks[loss_index] = rankdata(scores[loss_index], method="average")

    calls = scores >= SCORE_CUTOFF
    rows = []
    for first, second in LOSS_PAIRS:
        first_call = calls[first]
        second_call = calls[second]
        intersection = int((first_call & second_call).sum())
        union = int((first_call | second_call).sum())
        agreement = int((first_call == second_call).sum())
        first_positive = int(first_call.sum())
        second_positive = int(second_call.sum())
        rows.append(
            {
                "context": context,
                "group": group,
                "comparison": f"{LOSSES[first]} vs {LOSSES[second]}",
                "edge_contexts": n,
                "pearson_r": safe_corrcoef(scores[first], scores[second]),
                "spearman_rho": safe_corrcoef(ranks[first], ranks[second]),
                "intersection_positive": intersection,
                "union_positive": union,
                "binary_agreement_count": agreement,
                "first_positive_count": first_positive,
                "second_positive_count": second_positive,
            }
        )
    return rows


def pooled_binary_phi(
    n: int, first_positive: int, second_positive: int, intersection: int
) -> float:
    first_fraction = first_positive / n
    second_fraction = second_positive / n
    variance = (
        first_fraction
        * (1 - first_fraction)
        * second_fraction
        * (1 - second_fraction)
    )
    if variance == 0:
        return np.nan
    return (
        intersection / n - first_fraction * second_fraction
    ) / np.sqrt(variance)


def summarize_exact_similarity(
    by_context: pd.DataFrame, pooled_positive: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    grouped = by_context.groupby(["group", "comparison"], sort=False)
    for (group, comparison), table in grouped:
        n = int(table["edge_contexts"].sum())
        first_positive = int(table["first_positive_count"].sum())
        second_positive = int(table["second_positive_count"].sum())
        intersection = int(table["intersection_positive"].sum())
        union = int(table["union_positive"].sum())
        agreement = int(table["binary_agreement_count"].sum())
        rows.append(
            {
                "group": group,
                "comparison": comparison,
                "contexts": int(table["context"].nunique()),
                "edge_contexts": n,
                "pearson_context_mean": table["pearson_r"].mean(),
                "pearson_context_q25": table["pearson_r"].quantile(0.25),
                "pearson_context_median": table["pearson_r"].median(),
                "pearson_context_q75": table["pearson_r"].quantile(0.75),
                "spearman_context_mean": table["spearman_rho"].mean(),
                "spearman_context_q25": table["spearman_rho"].quantile(0.25),
                "spearman_context_median": table["spearman_rho"].median(),
                "spearman_context_q75": table["spearman_rho"].quantile(0.75),
                "pooled_binary_phi": pooled_binary_phi(
                    n, first_positive, second_positive, intersection
                ),
                "pooled_positive_call_jaccard": (
                    intersection / union if union else np.nan
                ),
                "pooled_binary_agreement": agreement / n,
                "pooled_first_positive_fraction": first_positive / n,
                "pooled_second_positive_fraction": second_positive / n,
                "pooled_exact_pearson": np.nan,
                "pooled_exact_spearman": np.nan,
                "score_summary_estimand": (
                    "macro distribution of exact within-context correlations"
                ),
            }
        )

    summary = pd.DataFrame(rows)
    pooled_lookup = pooled_positive.set_index("comparison")
    positive_rows = summary["group"] == SIMILARITY_GROUPS[1]
    summary.loc[positive_rows, "pooled_exact_pearson"] = summary.loc[
        positive_rows, "comparison"
    ].map(pooled_lookup["pearson_r"])
    summary.loc[positive_rows, "pooled_exact_spearman"] = summary.loc[
        positive_rows, "comparison"
    ].map(pooled_lookup["spearman_rho"])
    return summary


def similarity_table(exact_summary: pd.DataFrame) -> pd.DataFrame:
    """Build the compact per-class score summary."""
    rows = []
    for record in exact_summary.to_dict("records"):
        other_loss = record["comparison"].replace("BCE vs ", "", 1)
        # The first label is retained because the plotting/source-table code
        # already uses it; the value is the exact within-context macro mean.
        metrics = {
            "Binned Spearman score correlation": record[
                "spearman_context_mean"
            ],
            "Positive-call Jaccard": record["pooled_positive_call_jaccard"],
            "Binary agreement": record["pooled_binary_agreement"],
            "BCE positive-call fraction": record[
                "pooled_first_positive_fraction"
            ],
            f"{other_loss} positive-call fraction": record[
                "pooled_second_positive_fraction"
            ],
            "Thresholded Spearman call correlation": record[
                "pooled_binary_phi"
            ],
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "group": record["group"],
                    "comparison": record["comparison"],
                    "metric": metric,
                    "value": value,
                    "edge_contexts": int(record["edge_contexts"]),
                }
            )
    return pd.DataFrame(rows)


def population_summary(population_counts, positive_calls, totals) -> pd.DataFrame:
    rows = []
    for label in (1, 0):
        for cls, class_name in enumerate(CLASS_NAMES):
            count = int(population_counts[label, cls])
            rows.append(
                {
                    "edge_label": LABEL_NAMES[label],
                    "summary": class_name,
                    "edge_contexts": count,
                    "fraction": count / totals[label],
                }
            )
        for loss_index, loss in enumerate(LOSSES):
            count = int(positive_calls[label, loss_index])
            rows.append(
                {
                    "edge_label": LABEL_NAMES[label],
                    "summary": f"{loss} score >= {SCORE_CUTOFF}",
                    "edge_contexts": count,
                    "fraction": count / totals[label],
                }
            )
    return pd.DataFrame(rows)
