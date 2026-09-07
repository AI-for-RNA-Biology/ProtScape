"""Exploratory high-k validation reranking of the 54 retained AP@1-best snapshots.

This does not replace the original AP@1-selected winner or search training history
for high-k-optimal checkpoints. No test predictions or training updates are made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .diagnose_global_s2gae import (
    BASELINE_AP_TOLERANCE, state_hash, tensor_hash, validate_checkpoint_data,
)
from .evaluate_global_s2gae import (
    _validate_checkpoint, build_model, load_checkpoint, select_checkpoint, write_rows,
)
from .global_s2gae import (
    GLOBAL_K_VALUES, build_global_negative_bank, evaluate_global_edges, load_global_ppi_data,
)
from .run_global_s2gae_sweep import load_sweep


INTERPRETATION = (
    "Exploratory validation reranking among each configuration's retained AP@1-best "
    "snapshot; not a search for high-k-optimal training checkpoints. The original "
    "AP@1-selected winner is unchanged. Global high-k ranks require merging all shards."
)


def shard_runs(configurations: list[dict], index: int, count: int) -> list[dict]:
    if not 1 <= count <= len(configurations) or not 0 <= index < count:
        raise ValueError("Require 1 <= shard count <= run count and 0 <= index < count.")
    return configurations[index::count]


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    started = perf_counter()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite validation reranking: {output_dir}")
    _, configurations = load_sweep(args.sweep_config.expanduser())
    if len(configurations) != 54 or any(c["experiment_role"] != "sweep" for c in configurations):
        raise ValueError("This analysis requires the complete 54-configuration primary sweep.")
    assigned = shard_runs(configurations, args.shard_index, args.shard_count)

    # The existing selector checks all completions, model/training parameters,
    # protocol, and shared code/data fingerprints before any GPU model is built.
    selection_args = argparse.Namespace(
        checkpoint=None, runs_root=args.runs_root, sweep_config=args.sweep_config,
    )
    selected_path, reference, selection_rows = select_checkpoint(selection_args)
    if len(selection_rows) != 54 or any(row["status"] != "complete" for row in selection_rows):
        raise ValueError("All 54 configurations must be complete before reranking.")
    original_rows = {row["run_name"]: row for row in selection_rows}
    original_selection = {
        "checkpoint": str(selected_path), "run_name": selected_path.parent.name,
        "global_val_ap": float(reference["best_global_val_ap"]),
        "selection_metric": "global_val_ap_at_1_to_1", "unchanged": True,
    }
    split_seed = int(reference["training_config"]["split_seed"])
    training_commit = reference["git_commit"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    data = load_global_ppi_data(
        args.networks_dir, args.esm2_embeddings, seed=split_seed, verbose=False,
    )
    validate_checkpoint_data(reference, data)
    del reference
    bank = build_global_negative_bank(data, split="val", max_k=500, seed=split_seed)
    features, train_edges = data.features.to(device), data.train_edge_index.to(device)
    audit = {
        "negative_bank_sha256": tensor_hash(bank),
        "validation_pairs_sha256": tensor_hash(data.val_edge_index),
        "train_topology_sha256": tensor_hash(data.train_edge_index),
        "baseline_ap_absolute_tolerance": BASELINE_AP_TOLERANCE,
    }

    metric_rows, run_rows = [], []
    for config in assigned:
        path = (args.runs_root / config["name"] / "best_model_state_dict.pt").expanduser().resolve()
        checkpoint_hash = file_hash(path)
        checkpoint = load_checkpoint(path)
        _validate_checkpoint(path, checkpoint)
        validate_checkpoint_data(checkpoint, data)
        if checkpoint.get("model_type") != "global_s2gae" or checkpoint["git_commit"] != training_commit:
            raise ValueError(f"Model type or training source changed: {path}")
        expected_ap = float(original_rows[config["name"]]["best_global_val_ap"])
        if float(checkpoint["best_global_val_ap"]) != expected_ap:
            raise ValueError(f"Checkpoint changed after sweep preflight: {path}")
        model = build_model(checkpoint, device).eval()
        initial_state = state_hash(model)
        model_started = perf_counter()
        embeddings, layers = model.encode(features, train_edges)
        del embeddings
        metrics = evaluate_global_edges(
            model, data, split="val", k_values=GLOBAL_K_VALUES, device=device,
            seed=split_seed, layer_embeddings=layers, negative_bank=bank,
        )
        elapsed = perf_counter() - model_started
        observed_ap = float(metrics[1]["ap"])
        if not np.isclose(observed_ap, expected_ap, rtol=0, atol=BASELINE_AP_TOLERANCE):
            raise ValueError(
                f"Baseline validation AP does not reproduce checkpoint {config['name']}: "
                f"observed={observed_ap:.10f}, expected={expected_ap:.10f}."
            )
        if state_hash(model) != initial_state or file_hash(path) != checkpoint_hash:
            raise RuntimeError(f"Frozen checkpoint or model state changed: {path}")
        common = {
            "run_name": config["name"], "checkpoint": str(path),
            "checkpoint_sha256": checkpoint_hash,
            "hidden_dim": config["hidden_dim"], "num_layers": config["num_layers"],
            "dropout": config["dropout"], "best_update": int(checkpoint["best_epoch"]) + 1,
            "original_validation_rank": original_rows[config["name"]]["validation_rank"],
        }
        for k in GLOBAL_K_VALUES:
            metric_rows.append({**common, "split": "val", "k_negatives": k, **metrics[k]})
        run_rows.append({
            **common, **{f"validation_ap_{k}": float(metrics[k]["ap"]) for k in GLOBAL_K_VALUES},
            "exploratory_ap500_rank": None, "rank_status": "pending_all_shard_merge",
            "baseline_ap_abs_error": abs(observed_ap - expected_ap),
            "model_state_sha256": initial_state, "model_state_unchanged": True,
            "elapsed_seconds": elapsed,
        })
        print(f"{config['name']}: val AP@1={observed_ap:.8f}; AP@500={metrics[500]['ap']:.8f}; {elapsed:.1f}s", flush=True)
        del layers, model, checkpoint

    summary = {
        "status": "complete", "interpretation": INTERPRETATION,
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "n_expected_runs": 54, "n_scored_runs": len(run_rows),
        "expected_run_names": [config["name"] for config in configurations],
        "original_selection": original_selection,
        "sweep_config_sha256": file_hash(args.sweep_config.expanduser()),
        "training_commit": training_commit, "split_seed": split_seed,
        "graph_fingerprint": data.graph_fingerprint,
        "feature_fingerprint": data.feature_fingerprint,
        "split_fingerprint": data.split_fingerprint,
        "device": str(device), "elapsed_seconds": perf_counter() - started,
        "protocol": {
            "version": "global_s2gae_exploratory_validation_reranking_v1",
            "split": "validation_only", "evaluation_unit": "unique_undirected_global_pair",
            "message_topology": "training_edges_only",
            "negative_scope": "global_reference_nonedge",
            "negative_bank": "one_fixed_bank_per_process_shared_across_models_and_k_prefixes",
            "k_values": list(GLOBAL_K_VALUES),
            "f1": "macro_binary_classes_at_probability_0.5",
            "weights_and_batchnorm": "frozen_eval_mode_no_grad",
            "snapshot_selection": "retained_global_validation_AP_at_1_to_1_best",
            "original_winner_replaced": False,
        },
        "audit": audit, "runs": run_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_rows(output_dir / "metrics.csv", metric_rows)
    write_rows(output_dir / "run_summary.csv", run_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs-root", "sweep-config", "networks-dir", "esm2-embeddings", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
