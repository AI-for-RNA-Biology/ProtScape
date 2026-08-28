#!/usr/bin/env python3
"""Compute therapeutic-target evaluation and LRP tables and plots."""

from __future__ import annotations

import pandas as pd

from exploration.analysis.therapeutic_target.benchmark import (
    dataset_statistics,
    disease_model_comparison,
    mean_performance,
)
from exploration.analysis.therapeutic_target.checkpoint_lrp import (
    recompute_performance,
)
from exploration.analysis.therapeutic_target.context_attribution import (
    ANALYSIS_DIR,
    LRP_MODELS,
    absolute_contextual_relevance,
    disease_top_contexts,
    focal_target_contexts,
    generate_checkpoint_lrp,
    load_cell_metadata,
    prot_scape_lrp_tables,
)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    statistics = dataset_statistics()
    statistics.to_csv(ANALYSIS_DIR / "dataset_statistics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model,
                "inference_key": inference_key,
                "readout_key": readout_key,
            }
            for model, (inference_key, readout_key) in LRP_MODELS.items()
        ]
    ).to_csv(ANALYSIS_DIR / "lrp_model_readouts.csv", index=False)

    performance = recompute_performance()
    performance.to_csv(ANALYSIS_DIR / "held_out_performance.csv", index=False)
    mean_performance(performance).to_csv(
        ANALYSIS_DIR / "mean_performance.csv", index=False
    )
    disease_model_comparison(performance).to_csv(
        ANALYSIS_DIR / "disease_model_comparison.csv", index=False
    )

    generate_checkpoint_lrp()
    metadata = load_cell_metadata()
    (
        lrp_data,
        cohort,
        positive_matrix,
        signed_matrix,
        max_positive_matrix,
        max_positive_targets,
    ) = prot_scape_lrp_tables(metadata)
    cohort.to_csv(ANALYSIS_DIR / "target_recovery_cohort.csv", index=False)
    positive_matrix.to_csv(
        ANALYSIS_DIR / "cell_class_positive_contribution.csv",
        index=False,
    )
    signed_matrix.to_csv(
        ANALYSIS_DIR / "cell_class_signed_contribution_percent.csv", index=False
    )
    max_positive_matrix.to_csv(
        ANALYSIS_DIR / "cell_class_max_positive_contribution.csv",
        index=False,
    )
    max_positive_targets.to_csv(
        ANALYSIS_DIR / "cell_class_max_positive_targets.csv",
        index=False,
    )
    disease_top_contexts(lrp_data, cohort, metadata).to_csv(
        ANALYSIS_DIR / "disease_top_contexts.csv", index=False
    )
    focal_top, focal_summary = focal_target_contexts(lrp_data)
    focal_top.to_csv(ANALYSIS_DIR / "focal_target_top_contexts.csv", index=False)
    focal_summary.to_csv(ANALYSIS_DIR / "focal_target_summary.csv", index=False)

    targets, diseases, relevance_summary = absolute_contextual_relevance()
    targets.to_csv(
        ANALYSIS_DIR / "absolute_contextual_relevance_targets.csv", index=False
    )
    diseases.to_csv(
        ANALYSIS_DIR / "absolute_contextual_relevance_diseases.csv", index=False
    )
    relevance_summary.to_csv(
        ANALYSIS_DIR / "absolute_contextual_relevance_summary.csv", index=False
    )

    from exploration.plotting.therapeutic_target_attribution_plots import (
        plot_attributions,
    )
    from exploration.plotting.therapeutic_target_performance_plots import (
        plot_performance,
    )

    plot_performance(ANALYSIS_DIR, ANALYSIS_DIR)
    plot_attributions(ANALYSIS_DIR, ANALYSIS_DIR)
    print(f"Wrote therapeutic-target analysis to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
