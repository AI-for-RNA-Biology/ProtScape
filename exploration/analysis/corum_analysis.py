#!/usr/bin/env python3
"""Compute the CORUM analyses and plots."""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from downstream_tasks.config import PATHS
from exploration.analysis.corum_complex_properties import (
    CORUM_COMPLEXES,
    build_topology_and_cell_coverage,
    dataset_tables,
    driver_tables,
)
from exploration.analysis.corum_context_relevance import (
    add_observed_context_coverage,
    xmil_lrp_values,
    xmil_tables,
)
from exploration.analysis.corum_model_evaluation import (
    ABLATION_MODEL_ORDER,
    CONTEXT_READOUTS,
    LOSS_READOUTS,
    MAIN_CONTEXT_MODEL_ORDER,
    MODEL_LABELS,
    append_modeling_statistics,
    checkpoint_cell_representations,
    evaluate,
    load_run_data,
    performance_rows,
    per_complex_rows,
    run_spec,
    selected_run,
)


ANALYSIS_DIR = Path(PATHS["output_root"]) / "analysis/corum_analysis"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    stats, size_distribution, protein_distribution = dataset_tables()
    checkpoint_inputs = checkpoint_cell_representations()
    topology, cell_coverage = build_topology_and_cell_coverage()

    main_rows = []
    loss_rows = []
    ablation_rows = []
    per_complex_parts = []
    lrp_parts = []

    # Aggregate ESM2 baseline.
    run_dir = selected_run("aggregate", "lr_esm", "lr_esm")
    split_file = run_dir / "split_indices.npz"
    data = load_run_data(run_spec("aggregate", "lr_esm", "lr_esm"), split_file)
    result = evaluate(
        data,
        run_dir / "models",
        split_file,
        "lr_esm",
    )
    main_rows.extend(performance_rows("lr_esm", "lr_esm", result))
    stats = append_modeling_statistics(stats, data, split_file)
    del data, result
    gc.collect()

    # Contextual models: aggregate performance, per-complex performance and xMIL.
    for model_key in MAIN_CONTEXT_MODEL_ORDER:
        print(f"CORUM analysis: {MODEL_LABELS[model_key]}", flush=True)
        per_complex_dir = selected_run(
            "per_complex", model_key, "abmil8_pdl_id2_dropout"
        )
        per_complex_split = per_complex_dir / "split_indices.npz"
        per_complex_spec = run_spec(
            "per_complex", model_key, "abmil8_pdl_id2_dropout"
        )
        data = load_run_data(per_complex_spec, per_complex_split)
        for readout in CONTEXT_READOUTS:
            run_dir = selected_run("aggregate", model_key, readout)
            aggregate_spec = run_spec("aggregate", model_key, readout)
            if (
                aggregate_spec["cell_embedding_file"]
                == per_complex_spec["cell_embedding_file"]
            ):
                readout_data = data
            else:
                readout_data = load_run_data(
                    aggregate_spec,
                    run_dir / "split_indices.npz",
                )
            result = evaluate(
                readout_data,
                run_dir / "models",
                run_dir / "split_indices.npz",
                readout,
            )
            rows = performance_rows(model_key, readout, result)
            main_rows.extend(rows)
            if model_key == "s2gae_bce_uni":
                loss_rows.extend(rows)
            if readout_data is not data:
                del readout_data
                gc.collect()

        result = evaluate(
            data,
            per_complex_dir / "models",
            per_complex_split,
            "abmil8_pdl_id2_dropout",
        )
        per_complex_parts.append(
            per_complex_rows(
                model_key,
                result,
                data["class_names"],
            )
        )
        lrp_parts.append(
            xmil_lrp_values(
                model_key,
                data,
                per_complex_dir / "models",
                per_complex_split,
            )
        )

        if model_key == "s2gae_bce_uni":
            baseline_dir = selected_run("per_complex", "lr_esm", "lr_esm")
            result = evaluate(
                data,
                baseline_dir / "models",
                baseline_dir / "split_indices.npz",
                "lr_esm",
            )
            per_complex_parts.append(
                per_complex_rows("lr_esm", result, data["class_names"])
            )
        del data, result
        gc.collect()

    # pHuber and L1 were trained with the pooled pre-CCI cell export.
    for model_key in ("s2gae_phuber_uni", "s2gae_l1_uni"):
        data_by_embedding = {}
        for readout in LOSS_READOUTS[model_key]:
            run_dir = selected_run("aggregate", model_key, readout)
            split_file = run_dir / "split_indices.npz"
            aggregate_spec = run_spec("aggregate", model_key, readout)
            embedding_file = aggregate_spec["cell_embedding_file"]
            if embedding_file not in data_by_embedding:
                data_by_embedding[embedding_file] = load_run_data(
                    aggregate_spec, split_file
                )
            data = data_by_embedding[embedding_file]
            result = evaluate(
                data,
                run_dir / "models",
                split_file,
                readout,
            )
            loss_rows.extend(performance_rows(model_key, readout, result))
        del data, data_by_embedding, result
        gc.collect()

    for model_key in ABLATION_MODEL_ORDER:
        readout = "abmil8_pdl_id2_dropout"
        run_dir = selected_run("aggregate", model_key, readout)
        split_file = run_dir / "split_indices.npz"
        data = load_run_data(run_spec("aggregate", model_key, readout), split_file)
        result = evaluate(data, run_dir / "models", split_file, readout)
        ablation_rows.extend(performance_rows(model_key, readout, result))
        del data, result
        gc.collect()

    # ProstT5 sequence baseline (aggregate and split-fixed per-complex results).
    run_dir = selected_run("aggregate", "lr_prostt5", "lr_prostt5")
    split_file = run_dir / "split_indices.npz"
    data = load_run_data(
        run_spec("aggregate", "lr_prostt5", "lr_prostt5"), split_file
    )
    result = evaluate(
        data,
        run_dir / "models",
        split_file,
        "lr_prostt5",
    )
    main_rows.extend(performance_rows("lr_prostt5", "lr_prostt5", result))
    per_complex_dir = selected_run("per_complex", "lr_prostt5", "lr_prostt5")
    result = evaluate(
        data,
        per_complex_dir / "models",
        per_complex_dir / "split_indices.npz",
        "lr_prostt5",
    )
    per_complex_parts.append(
        per_complex_rows("lr_prostt5", result, data["class_names"])
    )
    del data, result
    gc.collect()

    main_performance = pd.DataFrame(main_rows)
    loss_performance = pd.DataFrame(loss_rows)
    per_complex = pd.concat(per_complex_parts, ignore_index=True)
    metadata = pd.read_csv(
        CORUM_COMPLEXES,
        dtype={"complex_id": str},
        usecols=["complex_id", "complex_name", "n_members_in_ppi_universe"],
    )
    per_complex = (
        per_complex.merge(metadata, on="complex_id", how="left")
        .merge(topology, on=["complex_id", "complex_name", "n_members_in_ppi_universe"])
        .merge(cell_coverage, on="complex_id")
    )
    driver_summary, driver_bins, driver_correlations = driver_tables(per_complex)

    raw_lrp = pd.concat(lrp_parts, ignore_index=True)
    xmil_values = add_observed_context_coverage(raw_lrp)
    (
        xmil_summary,
        xmil_bin_definitions,
        xmil_per_complex_bins,
        xmil_correlations,
    ) = xmil_tables(xmil_values)

    tables = {
        "corum_dataset_statistics.csv": stats,
        "corum_checkpoint_cell_representations.csv": checkpoint_inputs,
        "corum_complex_size_distribution.csv": size_distribution,
        "corum_complexes_per_protein_distribution.csv": protein_distribution,
        "corum_main_performance.csv": main_performance,
        "corum_loss_performance.csv": loss_performance,
        "corum_ablation_performance.csv": pd.DataFrame(ablation_rows),
        "corum_per_complex_performance.csv": per_complex,
        "corum_complex_topology_metrics.csv": topology,
        "corum_complex_cell_ppi_metrics.csv": cell_coverage,
        "corum_per_complex_driver_summary.csv": driver_summary,
        "corum_per_complex_driver_bins.csv": driver_bins,
        "corum_per_complex_driver_correlations.csv": driver_correlations,
        "corum_xmil_context_values.csv": xmil_values,
        "corum_xmil_bin_definitions.csv": xmil_bin_definitions,
        "corum_xmil_per_complex_bins.csv": xmil_per_complex_bins,
        "corum_xmil_binned_summary.csv": xmil_summary,
        "corum_xmil_correlation_summary.csv": xmil_correlations,
    }
    for filename, table in tables.items():
        table.to_csv(ANALYSIS_DIR / filename, index=False)

    from exploration.plotting.corum_benchmark_plots import plot_benchmark
    from exploration.plotting.corum_context_plots import plot_context_analysis

    plot_benchmark(ANALYSIS_DIR, ANALYSIS_DIR)
    plot_context_analysis(ANALYSIS_DIR, ANALYSIS_DIR)
    print(f"Wrote CORUM analysis tables and plots to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
