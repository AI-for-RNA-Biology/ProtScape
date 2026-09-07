"""Validation-only inference controls for a frozen global S2GAE checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch_geometric.utils import add_remaining_self_loops

from .evaluate_global_s2gae import _validate_checkpoint, build_model
from .global_s2gae import (
    GLOBAL_K_VALUES,
    GlobalPPIData,
    GlobalS2GAE,
    build_global_negative_bank,
    evaluate_global_edges,
    load_global_ppi_data,
)


MODES = ("train_topology", "train_plus_self_loops", "self_loops_only", "permuted_features")
BASELINE_AP_TOLERANCE = 1e-6


def tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def state_hash(model: GlobalS2GAE) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def validate_checkpoint_data(checkpoint: dict, data: GlobalPPIData) -> None:
    checks = {
        "protein order": checkpoint["protein_names"] == data.protein_names,
        "graph": checkpoint["graph_fingerprint"] == data.graph_fingerprint,
        "features": checkpoint["feature_fingerprint"] == data.feature_fingerprint,
        "split": checkpoint["split_fingerprint"] == data.split_fingerprint,
        "feature mean": torch.equal(checkpoint["feature_mean"], data.feature_mean),
        "feature std": torch.equal(checkpoint["feature_std"], data.feature_std),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Checkpoint/data mismatch: " + ", ".join(failed))


def diagnostic_inputs(
    features: torch.Tensor,
    train_edges: torch.Tensor,
    permutation: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Change one inference input, without altering features or graph in place."""
    if mode == "train_topology":
        return features, train_edges
    if mode == "train_plus_self_loops":
        edges, _ = add_remaining_self_loops(train_edges, num_nodes=features.size(0))
        return features, edges
    if mode == "self_loops_only":
        nodes = torch.arange(features.size(0), device=features.device)
        return features, torch.stack([nodes, nodes])
    if mode == "permuted_features":
        return features[permutation.to(features.device)], train_edges
    raise ValueError(f"Unknown diagnostic mode: {mode}")


@torch.no_grad()
def evaluate_modes(
    model: GlobalS2GAE,
    data: GlobalPPIData,
    *,
    device: torch.device,
    k_values: list[int],
    split_seed: int,
    permutation_seed: int,
    expected_baseline_ap: float,
) -> tuple[list[dict], dict]:
    k_values = sorted(set(k_values) | {1})
    if not set(k_values).issubset(GLOBAL_K_VALUES):
        raise ValueError(f"k values must be a subset of {GLOBAL_K_VALUES}.")
    model.eval()
    initial_state_hash = state_hash(model)
    bank = build_global_negative_bank(data, split="val", max_k=max(k_values), seed=split_seed)
    permutation = torch.randperm(
        data.features.size(0), generator=torch.Generator().manual_seed(permutation_seed)
    )
    features = data.features.to(device)
    train_edges = data.train_edge_index.to(device)
    rows = []
    baseline_ap = None
    for mode in MODES:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        mode_features, message_edges = diagnostic_inputs(features, train_edges, permutation, mode)
        embeddings, layers = model.encode(mode_features, message_edges)
        del embeddings
        metrics = evaluate_global_edges(
            model,
            data,
            split="val",
            k_values=k_values,
            device=device,
            seed=split_seed,
            layer_embeddings=layers,
            negative_bank=bank,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        )
        if mode == "train_topology":
            baseline_ap = float(metrics[1]["ap"])
            if not np.isclose(baseline_ap, expected_baseline_ap, rtol=0, atol=BASELINE_AP_TOLERANCE):
                raise ValueError(
                    f"Baseline validation AP does not reproduce checkpoint: "
                    f"observed={baseline_ap:.10f}, expected={expected_baseline_ap:.10f}."
                )
        for k, values in metrics.items():
            rows.append({
                "mode": mode,
                "k": k,
                **values,
                "elapsed_seconds": elapsed,
                "peak_gpu_allocated_gib": peak_memory,
            })
        print(f"{mode}: validation AP@1={metrics[1]['ap']:.8f}; {elapsed:.1f}s", flush=True)
        del layers, mode_features, message_edges
    final_state_hash = state_hash(model)
    if initial_state_hash != final_state_hash:
        raise RuntimeError("Frozen diagnostic changed model parameters or buffers.")
    return rows, {
        "k_values": k_values,
        "negative_bank_sha256": tensor_hash(bank),
        "permutation_sha256": tensor_hash(permutation),
        "validation_pairs_sha256": tensor_hash(data.val_edge_index),
        "train_topology_sha256": tensor_hash(data.train_edge_index),
        "model_state_sha256": initial_state_hash,
        "model_state_unchanged": True,
        "baseline_ap": baseline_ap,
        "expected_baseline_ap": expected_baseline_ap,
        "baseline_ap_absolute_tolerance": BASELINE_AP_TOLERANCE,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--networks-dir", type=Path, required=True)
    parser.add_argument("--esm2-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-values", default="1,10,100,500")
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = perf_counter()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_dir}")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    with checkpoint_path.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint(checkpoint_path, checkpoint)
    if checkpoint.get("model_type") != "global_s2gae":
        raise ValueError("Expected a global S2GAE checkpoint.")
    split_seed = int(checkpoint["training_config"]["split_seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    data = load_global_ppi_data(
        args.networks_dir, args.esm2_embeddings, seed=split_seed, verbose=False
    )
    validate_checkpoint_data(checkpoint, data)
    model = build_model(checkpoint, device).eval()
    rows, audit = evaluate_modes(
        model,
        data,
        device=device,
        k_values=[int(value) for value in args.k_values.split(",")],
        split_seed=split_seed,
        permutation_seed=args.permutation_seed,
        expected_baseline_ap=float(checkpoint["best_global_val_ap"]),
    )
    with checkpoint_path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != checkpoint_hash:
            raise RuntimeError("Checkpoint file changed while diagnostic was running.")
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_training_commit": checkpoint["git_commit"],
        "model_config": checkpoint["model_config"],
        "graph_fingerprint": data.graph_fingerprint,
        "feature_fingerprint": data.feature_fingerprint,
        "split_fingerprint": data.split_fingerprint,
        "split_seed": split_seed,
        "permutation_seed": args.permutation_seed,
        "device": str(device),
        "elapsed_seconds": perf_counter() - started,
        "protocol": {
            "version": "global_s2gae_frozen_validation_controls_v1",
            "split": "validation_only",
            "evaluation_unit": "unique_undirected_global_pair",
            "message_topology": "training_edges_only",
            "negative_scope": "global_reference_nonedge",
            "negative_bank": "shared_fixed_bank500_prefixes",
            "f1": "macro_binary_classes_at_probability_0.5",
            "modes": list(MODES),
            "weights_and_batchnorm": "frozen_eval_mode_no_grad",
            "feature_permutation": "fixed_seeded_global_row_permutation",
            "interpretation": "Inference perturbations, not separately trained ablations; controls may be out of distribution.",
        },
        "audit": audit,
        "metrics": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved validation-only diagnostic: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
