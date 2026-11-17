#!/usr/bin/env python3
"""Evaluate pretraining checkpoints and write analysis tables and plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml

from exploration.analysis.pretraining_benchmark import (
    evaluate_pinnacle,
    evaluate_protscape_metagraph,
    evaluate_protscape_ppi,
    load_dataset,
)
from exploration.analysis.pretraining_model_specs import MODELS
from exploration.analysis.pretraining_tables import (
    add_context_metadata,
    build_output_tables,
    ppi_metric_rows,
)
from pretraining.checkpoints import load_pinnacle_model, load_protscape_model


REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS_FILE = REPO_ROOT / "configs" / "paths.yaml"


def load_checkpoint(path: Path) -> dict:
    """Memory-map the checkpoint so weights are loaded as needed."""
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def main() -> None:
    """Evaluate the configured checkpoints and export pretraining tables and figures."""
    with PATHS_FILE.open("r", encoding="utf-8") as handle:
        paths = yaml.safe_load(handle)
    checkpoint_root = Path(paths["checkpoint_root"])
    output_dir = Path(paths["output_root"]) / "analysis" / "pretraining_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Pretraining evaluation requires CUDA for the PINNACLE "
            "FP16 evaluation protocol."
        )
    device = torch.device("cuda")

    robust = []
    metagraph = {}
    cci = {}
    parameters = {}
    contextwise = []

    # Reuse network inputs while the checkpoint's data settings stay the same.
    loaded_settings = None
    for model_key in [
        "gae_att",
        "gae_att_uni",
        "gae_vn",
        "gae_vn_uni",
        "gae_learnedvn",
        "gae_learnedvn_uni",
        "s2gae_att_k1_no_uni",
        "s2gae_att_k1_uni",
        "s2gae_att_k1_phuber",
        "s2gae_att_k1_l1_do00",
    ]:
        print(f"\nEvaluating {MODELS[model_key]['name']}", flush=True)
        checkpoint = load_checkpoint(checkpoint_root / MODELS[model_key]["checkpoint"])
        config = checkpoint["config"]
        data_settings = (
            config["features_mode"],
            config.get("esm2_pooling", "bos"),
            config.get("input_dim") or config.get("protein_config", {}).get("input_dim"),
            config.get("seed", 0),
            config.get("symmetric_ppi", True),
            config["dataset_mode"],
            config["split_mode"],
            config.get("weighted_ppi_loss", False),
        )
        if data_settings != loaded_settings:
            factored_data, id_to_name = load_dataset(checkpoint, paths, defer_features=False)
            loaded_settings = data_settings
        cell_ids = [int(cell_id) for cell_id in checkpoint["cell_ids"]]
        model_data = (
            {cell_id: factored_data[0][cell_id] for cell_id in cell_ids},
            *factored_data[1:],
        )
        expected_names = [id_to_name[cell_id] for cell_id in cell_ids]
        if checkpoint["cell_names"] != expected_names:
            raise ValueError(f"Cell names do not match for {model_key}.")
        model = load_protscape_model(checkpoint, model_data[0], device=device)
        parameters[model_key] = sum(parameter.numel() for parameter in model.parameters())
        ppi_metrics, local_context, pooled_cells = evaluate_protscape_ppi(
            model_key, model, model_data, id_to_name, device
        )
        robust.extend(ppi_metric_rows(model_key, ppi_metrics))
        contextwise.extend(local_context)
        if MODELS[model_key].get("metagraph"):
            cci[model_key] = evaluate_protscape_metagraph(
                model, model_data, pooled_cells, device
            )
            metagraph[model_key] = cci[model_key]
            print(f"  {model_key} CCI metrics: {cci[model_key]}", flush=True)
        del checkpoint, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del factored_data

    # Adapted PINNACLE random inputs are generated with the checkpoint seed.
    for model_key in ["pinnacle_random", "pinnacle_esm2_acm", "pinnacle_esm2"]:
        print(f"\nEvaluating {MODELS[model_key]['name']}", flush=True)
        checkpoint = load_checkpoint(checkpoint_root / MODELS[model_key]["checkpoint"])
        data, id_to_name = load_dataset(checkpoint, paths, defer_features=True)
        model = load_pinnacle_model(checkpoint, data[0], device="cpu")
        parameters[model_key] = sum(parameter.numel() for parameter in model.parameters())
        ppi_metrics, metagraph_metrics, cci_metrics, local_context = evaluate_pinnacle(
            model_key, model, data, id_to_name, device
        )
        robust.extend(ppi_metric_rows(model_key, ppi_metrics))
        metagraph[model_key] = metagraph_metrics
        cci[model_key] = cci_metrics
        print(f"  {model_key} metagraph metrics: {metagraph_metrics}", flush=True)
        print(f"  {model_key} cell-cell metrics: {cci_metrics}", flush=True)
        contextwise.extend(local_context)
        del checkpoint, data, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    contextwise = add_context_metadata(
        pd.DataFrame(contextwise), Path(paths["celltype_class_mapping"])
    )
    outputs = build_output_tables(robust, metagraph, cci, parameters, contextwise)
    for filename, table in outputs.items():
        path = output_dir / filename
        table.to_csv(path, index=False)
        print(f"Saved {path}", flush=True)

    from exploration.plotting.pretraining_evaluation_plots import (
        plot_pretraining as plot_model_evaluation,
    )
    from exploration.plotting.pretraining_diagnostic_plots import (
        plot_pretraining as plot_model_diagnostics,
    )

    plot_model_evaluation(output_dir, output_dir)
    plot_model_diagnostics(output_dir, output_dir)


if __name__ == "__main__":
    main()
