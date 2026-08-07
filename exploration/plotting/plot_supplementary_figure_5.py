#!/usr/bin/env python3
"""Plot the standalone Supplementary Figure 5 panels."""

from pathlib import Path

import matplotlib
import pandas as pd

from exploration.plotting import parkinson_target_plots as parkinson
from exploration.plotting import therapeutic_target_core_plots as targets


# Input and output directories.
SOURCE_DATA_DIR = Path(
    "/storage/research/dbmr_luisierlab/temp/athomas/outputs_protscape_repo/"
    "figure_source_data/supplementary_figure_5"
)
FIGURE_OUTPUT_DIR = Path(
    "/storage/research/dbmr_luisierlab/temp/athomas/outputs_protscape_repo/"
    "figures/supplementary_figure_5"
)


def main() -> None:
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context(targets.PAPER_RC):
        targets.plot_absolute_relevance(
            pd.read_csv(
                SOURCE_DATA_DIR / "absolute_contextual_relevance_diseases.csv"
            ),
            FIGURE_OUTPUT_DIR,
        )
        targets.plot_disease_top_contexts(
            pd.read_csv(SOURCE_DATA_DIR / "disease_top_contexts.csv"),
            FIGURE_OUTPUT_DIR,
        )

    with matplotlib.rc_context(parkinson.PLOT_RC):
        parkinson.plot_role_network(
            SOURCE_DATA_DIR,
            FIGURE_OUTPUT_DIR,
            "parkinson_known_candidates_experimental_network",
            "#B8B8B8",
        )
        parkinson.plot_leiden_network(SOURCE_DATA_DIR, FIGURE_OUTPUT_DIR)
        parkinson.plot_leiden_legend(FIGURE_OUTPUT_DIR)
        parkinson.plot_reactome_modules(SOURCE_DATA_DIR, FIGURE_OUTPUT_DIR)

    print(f"Wrote Supplementary Figure 5 source plots to {FIGURE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
