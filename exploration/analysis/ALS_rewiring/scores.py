"""Section 1: validate the S2GAE score against known physical-interaction
confidence, before picking `HIGH_CONF_THRESHOLD` in `high_confidence.py`.

Every plotting function here takes a `string_score_col` (default
`"stringdb_physical_score"`) so the exact same analysis can be re-run
against `"stringdb_combined_score"` instead - see `SCORE_LABELS` for the
column -> human-readable-label mapping used in titles/axis labels.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SCORE_LABELS = {
    "stringdb_physical_score": "STRING physical score",
    "stringdb_combined_score": "STRING combined score",
}


def _score_label(string_score_col: str) -> str:
    return SCORE_LABELS.get(string_score_col, string_score_col)


def get_high_confident(mydat: pd.DataFrame, cat: str, bins=np.arange(0, 1.01, 0.05),
                        n_last_bins: int = 21, string_score_col: str = "stringdb_physical_score",
                        ax_bar=None, ax_box=None):
    """`getHighConfident()` in the R script. Bins pairs by their S2GAE score column
    `cat`; the bar panel is, per bin, the % of pairs whose STRING
    `string_score_col` exceeds 0.75; the box panel is the distribution of
    (nonzero) scores in the last `n_last_bins` bins (i.e. near the top of
    the S2GAE range). Returns the pairs in the top bin with nonzero score.

    The bar panel here averages over non-NaN scores per bin; the R
    version's `sum(Y > 0.75)` (no `na.rm`) instead returns NA for any bin
    containing one, silently leaving that bar blank - not reproduced here."""
    subdat = mydat.dropna(subset=[cat])
    bin_of = pd.cut(subdat[cat], bins=bins, include_lowest=True)
    categories = bin_of.cat.categories

    frac_high, nonzero_scores, nonzero_pairs = [], [], []
    for b in categories:
        scores = subdat.loc[bin_of == b, string_score_col]
        pairs = subdat.loc[bin_of == b, "pair"]
        frac_high.append(100 * (scores > 0.75).mean() if len(scores) else np.nan)
        keep = scores > 0
        nonzero_scores.append(scores[keep].to_numpy())
        nonzero_pairs.append(pairs[keep])

    ax_bar = ax_bar or plt.gca()
    ax_bar.bar(range(len(frac_high)), frac_high, color="white", edgecolor="black")
    ax_bar.set(ylabel="% of high confident new edges", title=cat)
    ax_bar.set_xticks([1, len(frac_high)], labels=["0", "1"])
    ax_bar.set_xlabel("S2GAE score")

    if ax_box is not None:
        last_bins = nonzero_scores[-n_last_bins:]
        ax_box.boxplot(last_bins, showfliers=False)
        ax_box.set(ylabel=_score_label(string_score_col))
        ax_box.set_xticks([1, len(last_bins)], labels=["0", "1"])
        ax_box.set_xlabel("S2GAE score")

    return nonzero_pairs[-1]


def plot_score_distribution_pooled(all_edges_for_string: pd.DataFrame, out_path: Path,
                                    string_score_col: str = "stringdb_physical_score") -> None:
    """Single-panel bar plot of % high-confidence edges vs. S2GAE score,
    pooling all score categories together (via `all_edges_for_string`, same
    pooling as the boxplot below) - `distribution_string_scores.pdf`."""
    fig, ax = plt.subplots(figsize=(6, 5))
    get_high_confident(all_edges_for_string, "s2gae_score", bins=np.arange(0, 1.01, 0.05),
                        string_score_col=string_score_col, ax_bar=ax)
    ax.set_title(f"{_score_label(string_score_col)} high-confidence fraction vs. S2GAE score for new edges")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_all_edges_for_string(mydat_new: pd.DataFrame, score_categories,
                                string_score_col: str = "stringdb_physical_score") -> pd.DataFrame:
    """Pools all 8 per-condition S2GAE score columns into one long-format table
    with a unified `s2gae_score` column, dropping zero `string_score_col`
    values (no known evidence at all, in that STRING score, for that pair)
    before the pooled boxplot below."""
    all_subs = []
    for score in score_categories:
        sub = mydat_new[~mydat_new[score].isna()].copy()
        sub = sub.rename(columns={score: "s2gae_score"})
        sub = sub.drop(columns=[c for c in score_categories if c != score])
        all_subs.append(sub)

    all_edges_for_string = pd.concat(all_subs, ignore_index=True)
    all_edges_for_string = all_edges_for_string[all_edges_for_string[string_score_col] != 0]
    return all_edges_for_string


def plot_score_vs_string_boxplot(df: pd.DataFrame, bins=np.linspace(0, 1, 21),
                                  string_score_col: str = "stringdb_physical_score", ax=None):
    """Boxplot of STRING `string_score_col` per S2GAE-score bin (20 bins
    across [0, 1]), pooling all 8 score categories together via
    `all_edges_for_string` - a single-axis, coarser view than the per-category
    bar+box panels in `get_high_confident()` above."""
    ax = ax or plt.gca()
    bin_of = pd.cut(df["s2gae_score"], bins=bins, include_lowest=True)

    data, positions = [], []
    for interval in bin_of.cat.categories:
        vals = df.loc[bin_of == interval, string_score_col].dropna().to_numpy()
        if len(vals):
            data.append(vals)
            positions.append(interval.mid)

    width = (bins[1] - bins[0]) * 0.8
    ax.boxplot(data, positions=positions, widths=width, showfliers=False)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{b:.2f}" for b in bins], rotation=90, fontsize=6)
    ax.set(xlabel="S2GAE score", ylabel=_score_label(string_score_col))


def plot_score_vs_string_boxplot_to_file(all_edges_for_string: pd.DataFrame, out_path: Path,
                                          string_score_col: str = "stringdb_physical_score") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_score_vs_string_boxplot(all_edges_for_string, string_score_col=string_score_col, ax=ax)
    fig.tight_layout()
    ax.set_title(f"{_score_label(string_score_col)} vs. S2GAE score for new edges")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_score_vs_string_boxplot_binary(df: pd.DataFrame, threshold: float,
                                         string_score_col: str = "stringdb_physical_score", ax=None) -> float:
    """Two-box version of `plot_score_vs_string_boxplot` above: pools STRING
    `string_score_col` into "S2GAE score < threshold" vs. ">= threshold"
    (rather than one box per S2GAE-score bin), and Mann-Whitney U tests the two
    STRING-score distributions (unpaired, distribution-free - scores are bounded
    in [0, 1] and not expected to be normal). Returns the p-value."""
    ax = ax or plt.gca()
    below = df.loc[df["s2gae_score"] < threshold, string_score_col].dropna().to_numpy()
    above = df.loc[df["s2gae_score"] >= threshold, string_score_col].dropna().to_numpy()

    _, pvalue = mannwhitneyu(below, above, alternative="two-sided")

    ax.boxplot([below, above], showfliers=False)
    ax.set_xticks([1, 2], labels=[f"< {threshold}", f">= {threshold}"])
    ax.set(xlabel="S2GAE score", ylabel=_score_label(string_score_col))

    return pvalue


def plot_score_vs_string_boxplot_binary_to_file(
    all_edges_for_string: pd.DataFrame, threshold: float, out_path: Path,
    string_score_col: str = "stringdb_physical_score",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    pvalue = plot_score_vs_string_boxplot_binary(all_edges_for_string, threshold,
                                                  string_score_col=string_score_col, ax=ax)
    # With N in the hundreds of thousands the true p-value can underflow float64
    # (rounds to exactly 0.0) well before it's actually zero.
    pvalue_label = f"{pvalue:.2e}" if pvalue > 0 else "< 1e-300"
    ax.set_title(
        f"{_score_label(string_score_col)} vs. S2GAE score for new edges\nMann-Whitney p = {pvalue_label}"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path} (Mann-Whitney p = {pvalue_label})")
