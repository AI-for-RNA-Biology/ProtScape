"""Controlled, validation-only learning-rate refinement from a global S2GAE latest checkpoint."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .diagnose_global_s2gae import BASELINE_AP_TOLERANCE, validate_checkpoint_data
from .evaluate_global_s2gae import build_model
from .global_s2gae import (
    GLOBAL_K_VALUES,
    build_global_negative_bank,
    evaluate_global_edges,
    load_global_ppi_data,
    masked_reconstruction_step,
    protocol_metadata,
)
from .train_global_s2gae import (
    capture_rng_state,
    cpu_state_dict,
    restore_rng_state,
    save_checkpoint,
    write_history,
    write_json,
)


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def state_hash(value) -> str:
    """Device-independent provenance for nested Adam and RNG state."""
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous().numpy()
        if isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            for element in item:
                add(element)
        else:
            digest.update(repr(item).encode())

    add(value)
    return digest.hexdigest()


def restore_training_state(checkpoint: dict, device: torch.device, lr: float):
    """Restore everything first; changing learning rate is the only optimizer edit."""
    model = build_model(checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=checkpoint["training_config"]["lr"])
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state_dict"]))
    if state_hash(optimizer.state_dict()) != state_hash(checkpoint["optimizer_state_dict"]):
        raise ValueError("Adam state did not restore exactly.")
    restore_rng_state(checkpoint["rng_state"])
    if state_hash(capture_rng_state()) != state_hash(checkpoint["rng_state"]):
        raise ValueError("RNG state did not restore exactly; check visible CUDA devices.")
    for group in optimizer.param_groups:
        group["lr"] = lr
    return model, optimizer


def make_checkpoint(parent, config, model, optimizer, history, best_step, best_ap):
    """Keep the portable model schema, with a separate sensitivity-phase provenance."""
    payload = {
        key: parent[key]
        for key in (
            "format_version", "model_type", "model_config", "parameter_count",
            "protein_names", "feature_mean", "feature_std", "graph_fingerprint",
            "feature_fingerprint", "split_fingerprint", "split_counts", "protocol",
        )
    }
    step = int(history[-1]["phase_step"])
    payload.update(
        experiment_role="sensitivity",
        git_commit=config["git_commit"],
        training_config={
            **parent["training_config"],
            "epochs": config["parent_update"] + config["additional_updates"],
            "lr": config["lr"],
            "early_stopping_patience": 0,
            "early_stopping_min_delta": 0.0,
        },
        model_state_dict=cpu_state_dict(model),
        optimizer_state_dict=optimizer.state_dict(),
        rng_state=capture_rng_state(),
        epoch=int(parent["epoch"]) + step,
        best_epoch=int(parent["epoch"]) + best_step,
        best_global_val_ap=best_ap,
        val_metrics={"ap": history[-1]["global_val_ap"]},
        refinement_config=config,
        refinement_history=history,
        best_phase_step=best_step,
    )
    return payload


def save_best_checkpoint(path, payload):
    """Export weights-only-safe inference artifacts; keep Adam/RNG in latest only."""
    portable = {key: value for key, value in payload.items()
                if key not in {"optimizer_state_dict", "rng_state"}}
    save_checkpoint(path, portable)


def run_refinement(
    checkpoint_path, data, output_dir, *, lr, additional_updates, device, resume=False
):
    checkpoint_path, output_dir = Path(checkpoint_path).resolve(), Path(output_dir).resolve()
    if not np.isfinite(lr) or lr <= 0 or additional_updates < 1:
        raise ValueError("Learning rate and additional update count must be positive.")
    if output_dir.exists() != bool(resume):
        raise FileExistsError("Use a new output directory, or explicit --resume for an existing run.")
    parent_hash = file_hash(checkpoint_path)
    parent = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if parent.get("format_version") != 3 or parent.get("model_type") != "global_s2gae":
        raise ValueError("Expected a portable global S2GAE latest checkpoint.")
    if parent.get("protocol") != protocol_metadata():
        raise ValueError("Parent evaluation protocol mismatch.")
    if not parent.get("optimizer_state_dict") or not parent.get("rng_state"):
        raise ValueError("Parent must retain Adam and RNG state; use the latest checkpoint.")
    validate_checkpoint_data(parent, data)
    config = {
        "experiment_role": "sensitivity",
        "parent_checkpoint": str(checkpoint_path),
        "parent_checkpoint_sha256": parent_hash,
        "parent_update": int(parent["epoch"]) + 1,
        "parent_best_global_val_ap": float(parent["best_global_val_ap"]),
        "parent_best_update": int(parent["best_epoch"]) + 1,
        "parent_training_config": parent["training_config"],
        "parent_training_commit": parent["git_commit"],
        "initial_model_state_sha256": state_hash(parent["model_state_dict"]),
        "initial_adam_state_sha256": state_hash(parent["optimizer_state_dict"]),
        "initial_rng_state_sha256": state_hash(parent["rng_state"]),
        "lr": float(lr),
        "additional_updates": int(additional_updates),
        "device": str(device),
        "source_file_sha256": file_hash(Path(__file__)),
        "git_commit": os.environ.get("PROTSCAPE_GIT_COMMIT", "uncommitted"),
    }
    output_dir.mkdir(parents=True, exist_ok=resume)
    with (output_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output_dir / "config.json"
        if resume:
            if json.loads(config_path.read_text()) != config:
                raise ValueError("Cannot resume with a changed refinement configuration.")
            if (output_dir / "completed.json").exists():
                return json.loads((output_dir / "completed.json").read_text())
        else:
            write_json(config_path, config)
        return _run_phase(parent, config, data, output_dir, device, resume)


def _run_phase(parent, config, data, output_dir, device, resume):
    started = perf_counter()
    latest_path = output_dir / "latest_checkpoint.pt"
    best_path = output_dir / "best_model_state_dict.pt"
    source = torch.load(latest_path, map_location="cpu", weights_only=False) if resume else parent
    if resume and source.get("refinement_config") != config:
        raise ValueError("Latest checkpoint refinement configuration mismatch.")
    model, optimizer = restore_training_state(source, device, config["lr"])
    seed = int(parent["training_config"]["split_seed"])
    bank = build_global_negative_bank(data, split="val", max_k=max(GLOBAL_K_VALUES), seed=seed)
    data.features = data.features.to(device)

    def evaluate(k_values):
        model.eval()
        with torch.no_grad():
            return evaluate_global_edges(
                model, data, split="val", k_values=k_values, device=device,
                seed=seed, negative_bank=bank,
            )

    initial_ap = float(evaluate([1])[1]["ap"])
    if not np.isclose(initial_ap, source["val_metrics"]["ap"], rtol=0, atol=BASELINE_AP_TOLERANCE):
        raise ValueError("Loaded latest weights do not reproduce their own validation AP.")
    if resume:
        history = source["refinement_history"]
        best_step, best_ap = int(source["best_phase_step"]), float(source["best_global_val_ap"])
        if [int(row["phase_step"]) for row in history] != list(range(len(history))):
            raise ValueError("Resume history is not contiguous from phase step zero.")
        if int(source["epoch"]) != int(parent["epoch"]) + len(history) - 1:
            raise ValueError("Resume history does not match the latest optimizer update.")
        if not best_path.exists():
            raise FileNotFoundError("Missing retained best checkpoint.")
    else:
        best_step, best_ap = 0, initial_ap
        history = [{"phase_step": 0, "update": config["parent_update"], "train_loss": None,
                    "global_val_ap": initial_ap, "lr": config["lr"]}]
        initial = make_checkpoint(parent, config, model, optimizer, history, best_step, best_ap)
        save_best_checkpoint(best_path, initial)
        save_checkpoint(latest_path, initial)
    write_history(output_dir / "history.csv", history)
    training = parent["training_config"]
    for step in range(len(history), config["additional_updates"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, logits, labels = masked_reconstruction_step(
            model, data, device=device, mask_ratio=training["mask_ratio"],
            mask_type=training["mask_type"], k_negatives=training["k_negatives"],
            validate_targets=step == 1,
        )
        loss.backward()
        optimizer.step()
        ap = float(evaluate([1])[1]["ap"])
        if not np.isfinite(ap):
            raise ValueError("Validation AP is not finite.")
        history.append({"phase_step": step, "update": config["parent_update"] + step,
                        "train_loss": float(loss.item()), "global_val_ap": ap, "lr": config["lr"]})
        improved = ap > best_ap
        if improved:
            best_step, best_ap = step, ap
        payload = make_checkpoint(parent, config, model, optimizer, history, best_step, best_ap)
        if improved:
            save_best_checkpoint(best_path, payload)
        save_checkpoint(latest_path, payload)
        write_history(output_dir / "history.csv", history)
        print(f"phase_update={step}/{config['additional_updates']} lr={config['lr']} "
              f"val_ap={ap:.8f} best={best_ap:.8f}", flush=True)
        del loss, logits, labels, payload
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best["refinement_config"] != config or best["best_global_val_ap"] != best_ap:
        raise ValueError("Retained best checkpoint does not match phase history.")
    model.load_state_dict(best["model_state_dict"])
    metrics = evaluate(list(GLOBAL_K_VALUES))
    if not np.isclose(metrics[1]["ap"], best_ap, rtol=0, atol=BASELINE_AP_TOLERANCE):
        raise ValueError("Retained best checkpoint validation AP changed.")
    if file_hash(Path(config["parent_checkpoint"])) != config["parent_checkpoint_sha256"]:
        raise RuntimeError("Parent checkpoint changed during refinement.")
    summary = {
        "config": config,
        "protocol": {
            "split": "validation_only", "selection_metric": "global_val_ap_at_k1",
            "initial_candidate": "parent_latest_weights_not_parent_best_weights",
            "negative_bank": "fixed_shared_global_bank500_prefixes",
            "training_budget": "fixed_additional_updates_without_early_stopping",
            "original_54_config_sweep": "unchanged_experiment_role_sensitivity",
        },
        "completed_additional_updates": len(history) - 1,
        "initial_latest_val_ap": history[0]["global_val_ap"],
        "parent_best_global_val_ap": parent["best_global_val_ap"],
        "best_phase_step": best_step,
        "best_global_val_ap": best_ap,
        "best_validation_metrics": metrics,
        "elapsed_seconds_this_invocation": perf_counter() - started,
        "parent_checkpoint_unchanged": True,
    }
    write_json(output_dir / "completed.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--networks-dir", type=Path, required=True)
    parser.add_argument("--esm2-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--additional-updates", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data = load_global_ppi_data(
        args.networks_dir, args.esm2_embeddings,
        seed=int(parent["training_config"]["split_seed"]), verbose=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    run_refinement(
        args.checkpoint, data, args.output_dir, lr=args.lr,
        additional_updates=args.additional_updates, device=device, resume=args.resume,
    )


if __name__ == "__main__":
    main()
