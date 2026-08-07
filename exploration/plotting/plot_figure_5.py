#!/usr/bin/env python3
"""Plot the standalone Figure 5 panels."""

from pathlib import Path

import matplotlib
import pandas as pd

from exploration.plotting import parkinson_target_plots as parkinson
from exploration.plotting import therapeutic_target_core_plots as targets


# Input and output directories.
SOURCE_DATA_DIR = Path(
    "/storage/research/dbmr_luisierlab/temp/athomas/outputs_protscape_repo/"
    "figure_source_data/figure_5"
)
FIGURE_OUTPUT_DIR = Path(
    "/storage/research/dbmr_luisierlab/temp/athomas/outputs_protscape_repo/"
    "figures/figure_5"
)


def main() -> None:
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context(targets.PAPER_RC):
        performance = pd.read_csv(SOURCE_DATA_DIR / "mean_performance.csv")
        targets.plot_average_metric(performance, "auprc", FIGURE_OUTPUT_DIR)
        targets.plot_average_legend(performance, FIGURE_OUTPUT_DIR)
        targets.plot_disease_comparisons(
            pd.read_csv(SOURCE_DATA_DIR / "disease_model_comparison.csv"),
            FIGURE_OUTPUT_DIR,
        )
        targets.plot_signed_heatmap(
            pd.read_csv(
                SOURCE_DATA_DIR / "cell_class_signed_contribution_percent.csv"
            ),
            FIGURE_OUTPUT_DIR,
        )
        targets.plot_focal_targets(
            pd.read_csv(SOURCE_DATA_DIR / "focal_target_top_contexts.csv"),
            FIGURE_OUTPUT_DIR,
        )

    with matplotlib.rc_context(parkinson.PLOT_RC):
        parkinson.plot_candidate_recovery(SOURCE_DATA_DIR, FIGURE_OUTPUT_DIR)
        parkinson.plot_external_support(SOURCE_DATA_DIR, FIGURE_OUTPUT_DIR)
        parkinson.plot_synaptic_completion(SOURCE_DATA_DIR, FIGURE_OUTPUT_DIR)
        parkinson.plot_leiden_network(SOURCE_DATA_DIR, FIGURE_OUTPUT_DIR)
        parkinson.plot_leiden_legend(FIGURE_OUTPUT_DIR)
        parkinson.plot_string_enrichment(SOURCE_DATA_DIR, FIGURE_OUTPUT_DIR)

    print(f"Wrote Figure 5 source plots to {FIGURE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
