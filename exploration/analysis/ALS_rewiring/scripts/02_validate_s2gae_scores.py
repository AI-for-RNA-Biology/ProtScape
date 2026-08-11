"""Stage 02 - validate the S2GAE score against known physical-interaction
confidence, before picking `HIGH_CONF_THRESHOLD` (stage 03).

Purely diagnostic: nothing downstream depends on this stage's output, so it's
safe to skip if you already trust the threshold.

Runs the same analysis twice - once against `stringdb_physical_score`, once
against `stringdb_combined_score` (the "_combined" suffixed files) - see
scores.py's module docstring.

Input:  INTERMEDIATE_DIR/mydat.parquet (stage 01) - must have both
        stringdb_physical_score and stringdb_combined_score columns
Output: OUT_DIR/distribution_string_scores.pdf
        OUT_DIR/s2gae_vs_string_score_boxplot.pdf
        OUT_DIR/s2gae_vs_string_score_boxplot_binary.pdf
        OUT_DIR/distribution_string_scores_combined.pdf
        OUT_DIR/s2gae_vs_string_score_boxplot_combined.pdf
        OUT_DIR/s2gae_vs_string_score_boxplot_binary_combined.pdf
"""
import pandas as pd
import _bootstrap  # noqa: F401

from exploration.analysis.ALS_rewiring import config, scores


def main():
    config.ensure_dirs()

    mydat = pd.read_parquet(config.INTERMEDIATE_DIR / "mydat.parquet")
    mydat_new = mydat[mydat["present_in_ppi"] == 0]

    for string_score_col, suffix in (
        ("stringdb_physical_score", ""),
        ("stringdb_combined_score", "_combined"),
    ):
        all_edges_for_string = scores.build_all_edges_for_string(
            mydat_new, config.SCORE_CATEGORIES, string_score_col=string_score_col
        )

        scores.plot_score_distribution_pooled(
            all_edges_for_string, config.OUT_DIR / f"distribution_string_scores{suffix}.pdf",
            string_score_col=string_score_col,
        )

        scores.plot_score_vs_string_boxplot_to_file(
            all_edges_for_string, config.OUT_DIR / f"s2gae_vs_string_score_boxplot{suffix}.pdf",
            string_score_col=string_score_col,
        )

        scores.plot_score_vs_string_boxplot_binary_to_file(
            all_edges_for_string,
            config.HIGH_CONF_THRESHOLD,
            config.OUT_DIR / f"s2gae_vs_string_score_boxplot_binary{suffix}.pdf",
            string_score_col=string_score_col,
        )


if __name__ == "__main__":
    main()
