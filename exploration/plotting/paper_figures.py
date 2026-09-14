"""Render the paper's quantitative panels from released source tables."""

from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def plot_all(source: Path, output: Path):
    from exploration.plotting import (
        consensus_diagnostic_plots,
        consensus_evaluation_plots,
        corum_benchmark_plots,
        corum_context_plots,
        data_bulk_statistics_plots,
        parkinson_target_analysis_plots,
        pretraining_diagnostic_plots,
        pretraining_evaluation_plots,
        string_correlation_plots,
        string_support_plots,
        therapeutic_target_attribution_plots,
        therapeutic_target_performance_plots,
    )

    panels = [
        ("data_bulk_statistics", data_bulk_statistics_plots.plot_all),
        ("pretraining_evaluation", pretraining_evaluation_plots.plot_pretraining),
        ("pretraining_evaluation", pretraining_diagnostic_plots.plot_pretraining),
        ("consensus_analysis", consensus_diagnostic_plots.plot_consensus),
        ("consensus_analysis", consensus_evaluation_plots.plot_consensus),
        ("string_validation", string_correlation_plots.plot_string),
        ("string_validation", string_support_plots.plot_string),
        ("corum_analysis", corum_benchmark_plots.plot_benchmark),
        ("corum_analysis", corum_context_plots.plot_context_analysis),
        ("therapeutic_target_analysis", therapeutic_target_performance_plots.plot_performance),
        ("therapeutic_target_analysis", therapeutic_target_attribution_plots.plot_attributions),
        ("parkinson_target_analysis", parkinson_target_analysis_plots.plot_all),
    ]
    for name, plot in panels:
        print(f"Plotting {name}: {plot.__module__}", flush=True)
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        with matplotlib.rc_context(matplotlib.rcParamsDefault):
            plot(source / name, destination)
    print(f"Wrote panels to {output}")


def main():
    config_file = Path(__file__).resolve().parents[2] / "configs/paths.yaml"
    with config_file.open() as handle:
        paths = yaml.safe_load(handle)
    plot_all(Path(paths["paper_source_data"]), Path(paths["output_root"]) / "paper_figures")


if __name__ == "__main__":
    main()
