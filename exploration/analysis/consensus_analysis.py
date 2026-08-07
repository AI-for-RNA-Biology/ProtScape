#!/usr/bin/env python3
"""Compute exhaustive BCE, pHuber and L1 consensus and STRING statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from downstream_tasks.config import PATHS


ANALYSIS_DIR = Path(PATHS["output_root"]) / "analysis/consensus_analysis"
PAIR_CLASS_COUNTS = ANALYSIS_DIR / "pair_class_counts_uint16.npy"
SCORE_SAMPLE = ANALYSIS_DIR / "whole_ppi_score_sample.npz"
GLOBAL_GENE_ORDER = ANALYSIS_DIR / "global_gene_order.txt"
STRING_SCORES = ANALYSIS_DIR / "string_combined_scores.npy"
STRING_SCORE_MOMENTS = ANALYSIS_DIR / "string_positive_loss_score_moments.npz"
SCORE_SAMPLE_SUMMARY = ANALYSIS_DIR / "whole_ppi_score_sample_summary.csv"
SCORE_POPULATION_SUMMARY = ANALYSIS_DIR / "whole_ppi_score_population_summary.csv"
LOSS_SIMILARITY = ANALYSIS_DIR / "loss_similarity_to_bce_metrics.csv"

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
EDGE_LABELS = ["Labelled positive", "Labelled negative"]
RECURRENCE_BIN_STARTS = np.array([2, 3, 4, 6, 11, 21, 51, 101], dtype=np.int64)
N_RECURRENCE_BINS = len(RECURRENCE_BIN_STARTS)
SAMPLE_SEED = 13
SAMPLE_PER_STRATUM = 50_000
PAIR_CHUNK_SIZE = 2_000_000


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required analysis output: {path}\n"
            "The exhaustive scoring pass did not produce this file."
        )
    return path


def read_csv(
    path: Path, columns: set[str], *, float_precision: str | None = None
) -> pd.DataFrame:
    require_file(path)
    table = pd.read_csv(path, float_precision=float_precision)
    missing = columns.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return table


def load_score_sample() -> pd.DataFrame:
    require_file(SCORE_SAMPLE)
    with np.load(SCORE_SAMPLE, allow_pickle=False) as sample:
        required = {"labels", "class_ids", *LOSSES}
        missing = required.difference(sample.files)
        if missing:
            raise ValueError(f"{SCORE_SAMPLE} is missing arrays: {sorted(missing)}")
        arrays = {name: sample[name].copy() for name in required}

    if len({len(values) for values in arrays.values()}) != 1:
        raise ValueError(f"{SCORE_SAMPLE} arrays have inconsistent lengths")
    labels = arrays["labels"].astype(np.int8, copy=False)
    class_ids = arrays["class_ids"].astype(np.int8, copy=False)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{SCORE_SAMPLE} contains labels outside {{0, 1}}")
    if not np.isin(class_ids, np.arange(len(CLASS_NAMES))).all():
        raise ValueError(f"{SCORE_SAMPLE} contains invalid consensus-class IDs")

    scores = np.column_stack([arrays[loss] for loss in LOSSES])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"{SCORE_SAMPLE} contains invalid prediction scores")
    return pd.DataFrame(
        {
            "edge_label": np.where(
                labels == 1, "Labelled positive", "Labelled negative"
            ),
            "consensus_class": np.asarray(CLASS_NAMES, dtype=object)[class_ids],
            "BCE": scores[:, 0],
            "pHuber": scores[:, 1],
            "L1": scores[:, 2],
            "prediction_mean": scores.mean(axis=1),
            "prediction_std": scores.std(axis=1, ddof=0),
        }
    )


def build_score_sample_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the Supplementary Figure 2c-d sample and its summary."""
    sample = load_score_sample()
    sampled = (
        sample.groupby(["edge_label", "consensus_class"], sort=False)
        .size()
        .rename("edge_contexts_sampled")
        .reset_index()
        .rename(columns={"edge_label": "label", "consensus_class": "loss_class"})
    )

    population = read_csv(
        SCORE_POPULATION_SUMMARY, {"edge_label", "summary", "edge_contexts"}
    )
    population = population[population["summary"].isin(CLASS_NAMES)].rename(
        columns={
            "edge_label": "label",
            "summary": "loss_class",
            "edge_contexts": "edge_contexts_seen",
        }
    )[["label", "loss_class", "edge_contexts_seen"]]
    summary = population.merge(
        sampled, on=["label", "loss_class"], how="left", validate="one_to_one"
    )
    if summary["edge_contexts_sampled"].isna().any():
        raise ValueError("One or more label/class strata were not sampled")
    summary["edge_contexts_sampled"] = summary["edge_contexts_sampled"].astype(int)

    summary["label"] = pd.Categorical(
        summary["label"], categories=EDGE_LABELS, ordered=True
    )
    summary["loss_class"] = pd.Categorical(
        summary["loss_class"], categories=CLASS_NAMES, ordered=True
    )
    summary = summary.sort_values(["label", "loss_class"]).reset_index(drop=True)
    summary["label"] = summary["label"].astype(object)
    summary["loss_class"] = summary["loss_class"].astype(object)

    expected = np.minimum(SAMPLE_PER_STRATUM, summary["edge_contexts_seen"])
    if not np.array_equal(summary["edge_contexts_sampled"].to_numpy(), expected):
        raise ValueError(
            f"Expected the seed-{SAMPLE_SEED} sample of up to "
            f"{SAMPLE_PER_STRATUM:,} rows per label/class stratum"
        )

    recorded = read_csv(
        SCORE_SAMPLE_SUMMARY,
        {"label", "loss_class", "edge_contexts_seen", "edge_contexts_sampled"},
    )[summary.columns].reset_index(drop=True)
    if not summary.equals(recorded):
        raise ValueError("Score sample counts do not match its inference summary")
    return sample, summary


def build_loss_similarity_table() -> pd.DataFrame:
    """Load exact population-level loss similarities for Supplementary Fig. 2e."""
    table = read_csv(
        LOSS_SIMILARITY,
        {"group", "comparison", "metric", "value", "edge_contexts"},
        float_precision="round_trip",
    )
    if table.duplicated(["group", "comparison", "metric"]).any():
        raise ValueError("Loss-similarity output has duplicate statistic rows")
    required = {
        "Binned Spearman score correlation",
        "Thresholded Spearman call correlation",
    }
    if not required.issubset(table["metric"]):
        raise ValueError("Loss-similarity output is missing a displayed statistic")
    return table


def recurrence_bin_ids(recurrence: np.ndarray) -> np.ndarray:
    return np.searchsorted(
        RECURRENCE_BIN_STARTS[1:], recurrence, side="right"
    ).astype(np.int8)


def load_pair_class_counts() -> np.ndarray:
    require_file(PAIR_CLASS_COUNTS)
    counts = np.load(PAIR_CLASS_COUNTS, mmap_mode="r")
    expected_prefix = (2, len(CLASS_NAMES))
    if counts.ndim != 3 or counts.shape[:2] != expected_prefix:
        raise ValueError(
            f"Expected pair-class counts with shape (2, {len(CLASS_NAMES)}, n_pairs); "
            f"found {counts.shape}"
        )
    if counts.dtype != np.uint16:
        raise ValueError(f"Expected uint16 pair-class counts; found {counts.dtype}")
    return counts


def reduce_pair_class_counts() -> tuple[dict[int, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    """Reduce exhaustive pair counts for Figure 2e-f in one chunked pass."""
    counts = load_pair_class_counts()
    transitions = {
        label_id: np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
        for label_id in [1, 0]
    }
    stability = {
        label_id: {
            "reference_bin_counts": np.zeros(N_RECURRENCE_BINS, dtype=np.int64),
            "bin_class_weights": np.zeros(
                (len(CLASS_NAMES), N_RECURRENCE_BINS), dtype=np.float64
            ),
            "bin_class_unstable_weights": np.zeros(
                (len(CLASS_NAMES), N_RECURRENCE_BINS), dtype=np.float64
            ),
        }
        for label_id in [1, 0]
    }

    n_pairs = counts.shape[2]
    for start in range(0, n_pairs, PAIR_CHUNK_SIZE):
        end = min(start + PAIR_CHUNK_SIZE, n_pairs)
        for label_id in [1, 0]:
            block = counts[label_id, :, start:end].astype(np.int32)
            recurrence = block.sum(axis=0, dtype=np.int32)
            present = recurrence > 0
            if not present.any():
                continue

            observed = block[:, present]
            # Historical context-independent class: lower class wins exact ties.
            majority = observed.argmax(axis=0)
            for majority_id in range(len(CLASS_NAMES)):
                selected = majority == majority_id
                if selected.any():
                    transitions[label_id][majority_id] += observed[:, selected].sum(
                        axis=1, dtype=np.int64
                    )

            # Stability requires recurrence >=2. Exact co-modal ties receive
            # fractional 1/k membership rather than an arbitrary class label.
            eligible = recurrence >= 2
            if not eligible.any():
                continue
            eligible_block = block[:, eligible]
            bin_ids = recurrence_bin_ids(recurrence[eligible])
            accumulator = stability[label_id]
            accumulator["reference_bin_counts"] += np.bincount(
                bin_ids, minlength=N_RECURRENCE_BINS
            )
            co_modal = eligible_block == eligible_block.max(axis=0)
            n_co_modal = co_modal.sum(axis=0)
            unstable = (eligible_block > 0).sum(axis=0) > 1
            for class_id in range(len(CLASS_NAMES)):
                class_weights = co_modal[class_id] / n_co_modal
                accumulator["bin_class_weights"][class_id] += np.bincount(
                    bin_ids, weights=class_weights, minlength=N_RECURRENCE_BINS
                )
                accumulator["bin_class_unstable_weights"][class_id] += np.bincount(
                    bin_ids,
                    weights=class_weights * unstable,
                    minlength=N_RECURRENCE_BINS,
                )
        if end % (10 * PAIR_CHUNK_SIZE) == 0 or end == n_pairs:
            print(f"Reduced {end:,} / {n_pairs:,} protein pairs", flush=True)
    return transitions, stability


def build_transition_tables(transitions: dict[int, np.ndarray]) -> dict[str, pd.DataFrame]:
    tables = {}
    for label_id, slug in [(1, "positive"), (0, "negative")]:
        table = pd.DataFrame(
            transitions[label_id], index=CLASS_NAMES, columns=CLASS_NAMES
        )
        table.index.name = "context_independent_consensus_class"
        tables[f"labelled_{slug}_class_counts.csv"] = table.reset_index()
    return tables


def build_stability_table(stability: dict[int, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for label_id in [1, 0]:
        accumulator = stability[label_id]
        reference_counts = accumulator["reference_bin_counts"]
        if reference_counts.sum() == 0:
            raise ValueError(f"No recurrence >=2 pairs for {LABEL_NAMES[label_id]}")
        reference_weights = reference_counts / reference_counts.sum()
        class_weights = accumulator["bin_class_weights"]
        unstable_weights = accumulator["bin_class_unstable_weights"]
        bin_unstable = np.divide(
            unstable_weights,
            class_weights,
            out=np.full_like(unstable_weights, np.nan),
            where=class_weights > 0,
        )
        for class_id, class_name in enumerate(CLASS_NAMES):
            supported = (class_weights[class_id] > 0) & (reference_weights > 0)
            supported_fraction = float(reference_weights[supported].sum())
            standardized_unstable = float(
                np.sum(reference_weights[supported] * bin_unstable[class_id, supported])
                / supported_fraction
            )
            effective_n = float(class_weights[class_id].sum())
            raw_unstable_n = float(unstable_weights[class_id].sum())
            raw_unstable = raw_unstable_n / effective_n
            rows.append(
                {
                    "label": LABEL_NAMES[label_id],
                    "majority_class": class_name,
                    "modal_class": class_name,
                    "stable_unique_pairs": effective_n - raw_unstable_n,
                    "unstable_unique_pairs": raw_unstable_n,
                    "n_unique_pairs": effective_n,
                    "stable_fraction": 1.0 - standardized_unstable,
                    "unstable_fraction": standardized_unstable,
                    "raw_stable_fraction": 1.0 - raw_unstable,
                    "raw_unstable_fraction": raw_unstable,
                    "supported_reference_fraction": supported_fraction,
                    "minimum_label_recurrence": 2,
                    "modal_rule": "fractional 1/k membership across exact co-modal classes",
                    "stability_rule": "one observed loss-consensus class within the label stratum",
                    "standardization_reference": (
                        "full recurrence-bin distribution within the label stratum"
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_table(table: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, **kwargs)
    print(f"Saved {path}", flush=True)


def render_plots() -> None:
    """Render this analysis through the same functions used by plot-only scripts."""
    from exploration.plotting.plot_figure_2 import (
        plot_consensus as plot_figure_2_consensus,
    )
    from exploration.plotting.plot_supplementary_figure_2 import (
        plot_consensus as plot_supplementary_figure_2_consensus,
    )

    plot_figure_2_consensus(ANALYSIS_DIR, ANALYSIS_DIR)
    plot_supplementary_figure_2_consensus(
        ANALYSIS_DIR, ANALYSIS_DIR
    )


def main() -> None:
    from exploration.analysis.loss_consensus_inference import run

    run(ANALYSIS_DIR)
    sample, sample_summary = build_score_sample_tables()
    write_table(
        sample,
        ANALYSIS_DIR / "score_sample.csv.gz",
        float_format="%.8g",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    write_table(
        sample_summary,
        ANALYSIS_DIR / "score_sample_summary.csv",
    )
    write_table(
        build_loss_similarity_table(),
        ANALYSIS_DIR / "loss_similarity.csv",
    )

    transitions, stability = reduce_pair_class_counts()
    for filename, table in build_transition_tables(transitions).items():
        write_table(table, ANALYSIS_DIR / filename)
    write_table(
        build_stability_table(stability),
        ANALYSIS_DIR / "edge_stability.csv",
    )
    render_plots()

    # STRING validation uses the predictions made above; it is part of this
    # same exhaustive scientific analysis, not a separate precomputed prerequisite.
    from exploration.analysis.string_validation import write_results as write_string_results

    write_string_results()

    # These arrays only connect the two exhaustive reductions above.
    for path in (
        PAIR_CLASS_COUNTS,
        SCORE_SAMPLE,
        GLOBAL_GENE_ORDER,
        STRING_SCORES,
        STRING_SCORE_MOMENTS,
    ):
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
