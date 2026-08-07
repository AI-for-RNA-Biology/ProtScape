#!/usr/bin/env python3
"""Compute STRING summaries for the exhaustive loss-consensus analysis."""

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

# Keep the held-out OLS calculation deterministic and avoid excessive BLAS
# threading on shared machines. These must be set before importing NumPy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

from downstream_tasks.config import PATHS


ANALYSIS_DIR = Path(PATHS["output_root"]) / "analysis/string_validation"
CONSENSUS_DIR = Path(PATHS["output_root"]) / "analysis/consensus_analysis"
PAIR_CLASS_COUNTS = CONSENSUS_DIR / "pair_class_counts_uint16.npy"
GLOBAL_GENE_ORDER = CONSENSUS_DIR / "global_gene_order.txt"
STRING_SCORE_STATISTICS = CONSENSUS_DIR / "string_positive_loss_score_moments.npz"
STRING_SCORES = CONSENSUS_DIR / "string_combined_scores.npy"


LOSSES = ("BCE", "pHuber", "L1")
CLASS_NAMES = [
    "Consensus negative",
    "Weak disagreement negative",
    "Strong disagreement negative",
    "Strong disagreement positive",
    "Weak disagreement positive",
    "Consensus positive",
]
LABEL_NAMES = {1: "Labelled positive", 0: "Labelled negative"}
SCORE_BANDS = [
    ("no_string_match", "No STRING match", 0, 1),
    ("low", "0 < score < 0.4", 1, 400),
    ("medium", "0.4 <= score < 0.8", 400, 800),
    ("high", "score >= 0.8", 800, 1001),
]
SPLIT_SEED = 13
TEST_FRACTION = 0.2
PAIR_CHUNK_SIZE = 2_000_000


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required analysis input: {path}")
    return path


def read_gene_order() -> list[str]:
    require_file(GLOBAL_GENE_ORDER)
    genes = [
        line.strip().upper()
        for line in GLOBAL_GENE_ORDER.read_text().splitlines()
        if line.strip()
    ]
    if not genes or len(genes) != len(set(genes)):
        raise ValueError(f"{GLOBAL_GENE_ORDER} must contain a non-empty unique order")
    return genes


def load_pair_inputs() -> tuple[np.ndarray, np.ndarray]:
    require_file(PAIR_CLASS_COUNTS)
    counts = np.load(PAIR_CLASS_COUNTS, mmap_mode="r")
    if counts.ndim != 3 or counts.shape[:2] != (2, len(CLASS_NAMES)):
        raise ValueError(
            f"Expected pair-class counts with shape (2, {len(CLASS_NAMES)}, n_pairs); "
            f"found {counts.shape}"
        )
    if counts.dtype != np.uint16:
        raise ValueError(f"Expected uint16 pair-class counts; found {counts.dtype}")

    genes = read_gene_order()
    n_pairs = len(genes) * (len(genes) - 1) // 2
    if counts.shape[2] != n_pairs:
        raise ValueError("Pair-class counts do not match the global gene order")
    require_file(STRING_SCORES)
    string_scores = np.load(STRING_SCORES, mmap_mode="r")
    if string_scores.shape != (n_pairs,):
        raise ValueError("STRING scores do not match the global gene order")
    return counts, string_scores


def empty_string_stats() -> dict[str, float]:
    return {
        "n_items": 0,
        "score_sum": 0.0,
        "n_score_gt0": 0,
        "n_score_ge0p4": 0,
        "n_score_ge0p8": 0,
    }


def add_string_stats(stats: dict[str, float], scores: np.ndarray) -> None:
    stats["n_items"] += int(scores.size)
    stats["score_sum"] += float(scores.sum())
    stats["n_score_gt0"] += int((scores > 0).sum())
    stats["n_score_ge0p4"] += int((scores >= 0.4).sum())
    stats["n_score_ge0p8"] += int((scores >= 0.8).sum())


def string_stats_row(stats: dict[str, float]) -> dict[str, float]:
    total = int(stats["n_items"])
    supported = int(stats["n_score_gt0"])
    return {
        "n_unique_pairs": total,
        "mean_string_combined": stats["score_sum"] / total if total else np.nan,
        "mean_string_combined_supported_only": (
            stats["score_sum"] / supported if supported else np.nan
        ),
        "fraction_string_combined_gt0": supported / total if total else np.nan,
        "fraction_string_combined_ge0p4": (
            int(stats["n_score_ge0p4"]) / total if total else np.nan
        ),
        "fraction_string_combined_ge0p8": (
            int(stats["n_score_ge0p8"]) / total if total else np.nan
        ),
        "n_string_combined_gt0": supported,
        "n_string_combined_ge0p4": int(stats["n_score_ge0p4"]),
        "n_string_combined_ge0p8": int(stats["n_score_ge0p8"]),
    }


def build_string_support_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce exhaustive pair classes to STRING-support tables."""
    counts, string_scores = load_pair_inputs()
    string_stats = {
        (label_id, class_id): empty_string_stats()
        for label_id in [1, 0]
        for class_id in range(len(CLASS_NAMES))
    }
    histograms = {
        (label_id, class_name): np.zeros(1001, dtype=np.int64)
        for label_id in [1, 0]
        for class_name in ["Total", *CLASS_NAMES]
    }

    n_pairs = counts.shape[2]
    for start in range(0, n_pairs, PAIR_CHUNK_SIZE):
        end = min(start + PAIR_CHUNK_SIZE, n_pairs)
        score_chunk = string_scores[start:end]
        for label_id in [1, 0]:
            block = counts[label_id, :, start:end].astype(np.int32)
            present = block.sum(axis=0, dtype=np.int32) > 0
            if not present.any():
                continue
            observed = block[:, present]
            # Use the lower class for exact modal ties.
            majority = observed.argmax(axis=0)
            scores = score_chunk[present]
            score_bins = np.rint(scores * 1000).astype(np.int16)
            histograms[(label_id, "Total")] += np.bincount(
                score_bins, minlength=1001
            )
            for class_id, class_name in enumerate(CLASS_NAMES):
                selected_scores = scores[majority == class_id]
                add_string_stats(string_stats[(label_id, class_id)], selected_scores)
                selected_bins = np.rint(selected_scores * 1000).astype(np.int16)
                histograms[(label_id, class_name)] += np.bincount(
                    selected_bins, minlength=1001
                )
        if end % (10 * PAIR_CHUNK_SIZE) == 0 or end == n_pairs:
            print(f"Reduced STRING support for {end:,} / {n_pairs:,} pairs", flush=True)

    coverage_rows = []
    for label_id in [1, 0]:
        for class_id, class_name in enumerate(CLASS_NAMES):
            coverage_rows.append(
                {
                    "label": LABEL_NAMES[label_id],
                    "loss_class": class_name,
                    **string_stats_row(string_stats[(label_id, class_id)]),
                }
            )

    band_rows = []
    for label_id in [1, 0]:
        for class_name in ["Total", *CLASS_NAMES]:
            histogram = histograms[(label_id, class_name)]
            total = int(histogram.sum())
            supported = total - int(histogram[0])
            for band_id, band_label, start, end in SCORE_BANDS:
                n_in_band = int(histogram[start:end].sum())
                band_rows.append(
                    {
                        "label": LABEL_NAMES[label_id],
                        "loss_class": class_name,
                        "score_kind": "combined",
                        "score_band": band_id,
                        "score_band_label": band_label,
                        "n_unique_pairs_in_band": n_in_band,
                        "n_unique_pairs": total,
                        "n_string_supported_pairs": supported,
                        "fraction_all_unique_pairs": (
                            n_in_band / total if total else np.nan
                        ),
                        "fraction_string_supported_pairs": (
                            n_in_band / supported if supported and start > 0 else np.nan
                        ),
                    }
                )
    return pd.DataFrame(coverage_rows), pd.DataFrame(band_rows)


def make_test_mask(n_pairs: int) -> np.ndarray:
    folds = np.arange(n_pairs, dtype=np.int64) % int(1 / TEST_FRACTION)
    np.random.default_rng(SPLIT_SEED).shuffle(folds)
    return folds == 0


def standardized_ols_predictions(
    features: np.ndarray, target: np.ndarray, test: np.ndarray
) -> np.ndarray:
    train_features = features[~test]
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    target_mean = float(target[~test].mean())

    centered_features = train_features - feature_mean
    centered_target = target[~test] - target_mean
    xtx = centered_features.T @ centered_features
    xty = centered_features.T @ centered_target
    coefficients = np.linalg.lstsq(
        xtx / np.outer(feature_scale, feature_scale),
        xty / feature_scale,
        rcond=None,
    )[0]
    return target_mean + (
        (features[test] - feature_mean) / feature_scale
    ) @ coefficients


def build_string_correlation_table() -> pd.DataFrame:
    """Fit held-out mean and standard-deviation descriptors."""
    require_file(STRING_SCORE_STATISTICS)
    with np.load(STRING_SCORE_STATISTICS, allow_pickle=False) as statistics:
        required = {
            "pair_scope",
            "losses",
            "string_combined",
            "mean_scores",
            "std_scores",
            "n_copresent_contexts",
        }
        missing = required.difference(statistics.files)
        if missing:
            raise ValueError(
                f"{STRING_SCORE_STATISTICS} is missing arrays: {sorted(missing)}"
            )
        if statistics["pair_scope"].item() != "string_positive_copresent_at_least_2":
            raise ValueError("Unexpected protein-pair universe in STRING statistics")
        if statistics["losses"].tolist() != list(LOSSES):
            raise ValueError(f"Expected loss order {list(LOSSES)}")
        target = statistics["string_combined"].astype(np.float64)
        mean_scores = statistics["mean_scores"].copy()
        std_scores = statistics["std_scores"].copy()
        n_copresent = statistics["n_copresent_contexts"].copy()

    n_pairs = len(target)
    expected_shape = (len(LOSSES), n_pairs)
    if mean_scores.shape != expected_shape or std_scores.shape != expected_shape:
        raise ValueError("STRING score-moment arrays have unexpected shapes")
    if np.any(target <= 0) or np.any(n_copresent < 2):
        raise ValueError("Correlation analysis requires STRING > 0 and recurrence >=2")

    test = make_test_mask(n_pairs)
    observed = target[test]
    observed_ranks = rankdata(observed, method="average")
    descriptors = {"mean": mean_scores, "std": std_scores}
    rows = []
    for n_losses in range(1, len(LOSSES) + 1):
        for loss_combination in combinations(LOSSES, n_losses):
            features = np.column_stack(
                [
                    descriptors[descriptor][LOSSES.index(loss)]
                    for loss in loss_combination
                    for descriptor in ("mean", "std")
                ]
            ).astype(np.float64, copy=False)
            predictions = standardized_ols_predictions(features, target, test)
            rows.append(
                {
                    "loss_combination": "+".join(loss_combination),
                    "pearson": float(pearsonr(predictions, observed)[0]),
                    "spearman": float(
                        pearsonr(
                            rankdata(predictions, method="average"), observed_ranks
                        )[0]
                    ),
                    "n_pairs": n_pairs,
                    "n_test": int(test.sum()),
                    "split_seed": SPLIT_SEED,
                    "features": "across-context mean and standard deviation",
                    "string_score": "STRING v12 combined_score",
                }
            )
    return pd.DataFrame(rows)


def write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"Saved {path}", flush=True)


def render_plots() -> None:
    """Render the plots from the computed analysis tables."""
    from exploration.plotting.string_support_plots import (
        plot_string as plot_model_evaluation,
    )
    from exploration.plotting.string_correlation_plots import (
        plot_string as plot_model_diagnostics,
    )

    plot_model_evaluation(ANALYSIS_DIR, ANALYSIS_DIR)
    plot_model_diagnostics(ANALYSIS_DIR, ANALYSIS_DIR)


def write_results() -> None:
    coverage, score_bands = build_string_support_tables()
    write_table(coverage, ANALYSIS_DIR / "string_coverage.csv")
    write_table(score_bands, ANALYSIS_DIR / "string_score_bands.csv")
    write_table(
        build_string_correlation_table(),
        ANALYSIS_DIR / "string_correlations.csv",
    )
    render_plots()


def main() -> None:
    # Run inference before reducing the STRING statistics.
    from exploration.analysis.consensus_analysis import main as run_analysis

    run_analysis()


if __name__ == "__main__":
    main()
