"""Train the context-free global-PPI ProtScape baseline."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import random
import resource
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from .global_s2gae import (
    GlobalS2GAE,
    binary_metrics,
    build_global_negative_bank,
    evaluate_global_edges,
    load_global_ppi_data,
    masked_reconstruction_step,
    protocol_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ESM2 -> ACM -> S2GAE on the released global PPI."
    )
    parser.add_argument("--networks-dir", type=Path, required=True)
    parser.add_argument("--esm2-embeddings", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--experiment-role", choices=("sweep", "sensitivity"), default="sweep"
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--decode-channels", type=int, default=512)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-dropout", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--mask-type", choices=("dm", "um"), default="dm")
    parser.add_argument("--k-negatives", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after this many updates without meaningful validation AP improvement; 0 disables.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                        help="Required validation AP improvement on the 0-to-1 scale.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--residue-config", help="Optional learned pooling YAML; mean uses frozen mean features")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--extend-completed",
        action="store_true",
        help="Continue a completed shorter run without replaying finished epochs.",
    )
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--wandb-entity", default="cedricvincentcuaz")
    parser.add_argument("--wandb-project", default="pinnacle")
    parser.add_argument("--wandb-group", default="protscape_global_ppi_baseline")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(
            [value.detach().cpu() for value in state["torch_cuda"]]
        )


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


@dataclass
class EarlyStopping:
    """Validation-only patience; independent of the absolute-best checkpoint."""

    patience: int = 0
    min_delta: float = 0.0
    reference_ap: float = -np.inf
    last_improvement_epoch: int = -1
    last_epoch: int = -1

    def observe(self, ap: float, epoch: int) -> None:
        if not np.isfinite(ap):
            raise ValueError("Validation AP must be finite.")
        self.last_epoch = int(epoch)
        if ap > self.reference_ap + self.min_delta:
            self.reference_ap = float(ap)
            self.last_improvement_epoch = int(epoch)

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and (
            self.last_epoch - self.last_improvement_epoch >= self.patience
        )


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    dropout = str(args.dropout).replace(".", "")
    return (
        f"global_s2gae_acm_h{args.hidden_dim}_l{args.num_layers}"
        f"_do{dropout}_seed{args.seed}"
    )


def training_config(args: argparse.Namespace) -> dict:
    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "mask_ratio": args.mask_ratio,
        "mask_type": args.mask_type,
        "k_negatives": args.k_negatives,
        "seed": args.seed,
        "split_seed": args.split_seed,
    }
    # Preserve the schema and tracking identity of existing fixed-budget runs.
    if getattr(args, "early_stopping_patience", 0) > 0:
        config.update(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )
    if getattr(args, "residue_config", None):
        config["residue_pooling"] = yaml.safe_load(Path(args.residue_config).read_text())
    return config


def tracking_config(
    args: argparse.Namespace,
    name: str,
    data,
    model_config: dict,
) -> dict:
    identity = json.dumps(
        {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "group": args.wandb_group,
            "name": name,
            "experiment_role": args.experiment_role,
            "model": model_config,
            "training": training_config(args),
            "protocol": protocol_metadata(),
            "git_commit": os.environ.get("PROTSCAPE_GIT_COMMIT", "uncommitted"),
            "graph_fingerprint": data.graph_fingerprint,
            "feature_fingerprint": data.feature_fingerprint,
            "split_fingerprint": data.split_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "group": args.wandb_group,
        "run_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
    }


def checkpoint_payload(
    args: argparse.Namespace,
    model: GlobalS2GAE,
    data,
    *,
    epoch: int,
    best_epoch: int,
    best_global_val_ap: float,
    val_metrics: dict,
    optimizer=None,
    tracking: dict | None = None,
) -> dict:
    payload = {
        "format_version": 3,
        "model_type": "global_s2gae",
        "experiment_role": args.experiment_role,
        "model_config": model.model_config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_config": training_config(args),
        "wandb": tracking
        if tracking is not None
        else tracking_config(args, run_name(args), data, model.model_config),
        "model_state_dict": cpu_state_dict(model),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_global_val_ap": best_global_val_ap,
        "val_metrics": val_metrics,
        "protein_names": data.protein_names,
        "feature_mean": data.feature_mean,
        "feature_std": data.feature_std,
        "graph_fingerprint": data.graph_fingerprint,
        "feature_fingerprint": data.feature_fingerprint,
        "split_fingerprint": data.split_fingerprint,
        "split_counts": data.split_counts,
        "protocol": protocol_metadata(),
        "git_commit": os.environ.get("PROTSCAPE_GIT_COMMIT", "uncommitted"),
    }
    continuations = getattr(args, "continuations", None)
    if continuations:
        payload["continuations"] = continuations
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["rng_state"] = capture_rng_state()
    return payload


def validate_completed_extension(
    checkpoint: dict,
    completion: dict,
    previous_config: dict,
    current_training: dict,
    current_git_commit: str,
) -> dict:
    """Validate and describe a continuation of a completed shorter run."""
    if completion.get("stop_reason") == "early_stopping":
        raise ValueError("Cannot extend: run already satisfied its early-stopping rule.")
    previous_epochs = int(completion["completed_epochs"])
    if int(completion["last_epoch"]) != previous_epochs - 1:
        raise ValueError("Cannot extend: completion has an invalid last epoch.")
    if int(checkpoint["epoch"]) + 1 != previous_epochs:
        raise ValueError("Cannot extend: latest checkpoint is not the completed epoch.")
    if int(current_training["epochs"]) <= previous_epochs:
        raise ValueError("Cannot extend: target epochs must exceed completed epochs.")
    stopping_keys = {"early_stopping_patience", "early_stopping_min_delta"}
    previous_training = checkpoint["training_config"]
    for key, value in current_training.items():
        if key not in {"epochs", *stopping_keys} and previous_training[key] != value:
            raise ValueError(f"Cannot extend: training parameter {key} changed.")
    if previous_training.get("early_stopping_patience", 0) > 0:
        for key in stopping_keys:
            if previous_training.get(key, 0) != current_training.get(key, 0):
                raise ValueError(f"Cannot extend: stopping parameter {key} changed.")
    if int(checkpoint["training_config"]["epochs"]) != previous_epochs:
        raise ValueError("Cannot extend: checkpoint epoch budget is inconsistent.")
    if int(previous_config["epochs"]) != previous_epochs:
        raise ValueError("Cannot extend: config epoch budget is inconsistent.")

    if checkpoint["git_commit"] != completion["git_commit"]:
        raise ValueError("Cannot extend: completion/checkpoint commit mismatch.")
    if previous_config["git_commit"] != checkpoint["git_commit"]:
        raise ValueError("Cannot extend: config/checkpoint commit mismatch.")
    if completion["protocol"] != checkpoint["protocol"]:
        raise ValueError("Cannot extend: completion/checkpoint protocol mismatch.")
    if previous_config["protocol"] != checkpoint["protocol"]:
        raise ValueError("Cannot extend: config/checkpoint protocol mismatch.")
    for key in ("graph_fingerprint", "feature_fingerprint", "split_fingerprint"):
        if completion[key] != checkpoint[key] or previous_config[key] != checkpoint[key]:
            raise ValueError(f"Cannot extend: {key} provenance mismatch.")
    if int(completion["best_epoch"]) != int(checkpoint["best_epoch"]):
        raise ValueError("Cannot extend: completion/checkpoint best epoch mismatch.")
    if float(completion["best_global_val_ap"]) != float(
        checkpoint["best_global_val_ap"]
    ):
        raise ValueError("Cannot extend: completion/checkpoint best metric mismatch.")

    continuation = {
        "from_completed_epochs": previous_epochs,
        "to_epochs": int(current_training["epochs"]),
        "from_git_commit": checkpoint["git_commit"],
        "to_git_commit": current_git_commit,
    }
    if current_training.get("early_stopping_patience", 0) > 0:
        continuation["early_stopping"] = {
            "patience": current_training["early_stopping_patience"],
            "min_delta": current_training["early_stopping_min_delta"],
            "initialization": "full_validation_history",
        }
    return continuation


def validate_archived_best(best: dict, latest: dict) -> None:
    """Ensure the prior best artifact agrees with the completed latest state."""
    if best.get("format_version") != 3:
        raise ValueError("Cannot extend: unsupported best-checkpoint format.")
    for key in (
        "experiment_role",
        "model_config",
        "training_config",
        "wandb",
        "graph_fingerprint",
        "feature_fingerprint",
        "split_fingerprint",
        "protocol",
        "git_commit",
        "best_epoch",
        "best_global_val_ap",
    ):
        if best[key] != latest[key]:
            raise ValueError(f"Cannot extend: best/latest {key} mismatch.")
    if int(best["epoch"]) != int(best["best_epoch"]):
        raise ValueError("Cannot extend: best artifact is not from the best epoch.")


def main() -> None:
    args = parse_args()
    if args.extend_completed and not args.resume:
        raise ValueError("--extend-completed requires --resume.")
    if args.hidden_dim <= 0 or args.num_layers <= 0:
        raise ValueError("hidden_dim and num_layers must be positive.")
    if args.decode_channels <= 0 or args.decoder_layers < 2:
        raise ValueError("decode_channels must be positive and decoder_layers at least 2.")
    if not 0.0 <= args.dropout < 1.0 or not 0.0 <= args.decoder_dropout < 1.0:
        raise ValueError("encoder and decoder dropout must lie in [0, 1).")
    if not 0.0 < args.mask_ratio < 1.0:
        raise ValueError("mask_ratio must lie strictly between 0 and 1.")
    if args.k_negatives < 1:
        raise ValueError("k_negatives must be positive.")
    if args.epochs < 1:
        raise ValueError("epochs must be positive.")
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be nonnegative.")
    if not np.isfinite(args.early_stopping_min_delta) or args.early_stopping_min_delta < 0:
        raise ValueError("early-stopping-min-delta must be finite and nonnegative.")
    if args.early_stopping_patience == 0 and args.early_stopping_min_delta != 0:
        raise ValueError("early-stopping-min-delta requires a positive patience.")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive.")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if args.wandb_mode == "online" and device.type != "cuda":
        raise RuntimeError("Online W&B mode is reserved for real GPU training runs.")
    if (
        device.type == "cuda"
        and os.environ.get("SLURM_JOB_ID")
        and torch.cuda.device_count() != 1
    ):
        raise RuntimeError(
            f"Each Slurm training task must see one GPU, found {torch.cuda.device_count()}."
        )
    if device.type == "cuda":
        print(
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
            f"cuda_device_count={torch.cuda.device_count()} "
            f"cuda_device={torch.cuda.get_device_name(device)}",
            flush=True,
        )

    name = run_name(args)
    output_root = args.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_root / f".{name}.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Run is already active: {name}") from error

    run_dir = output_root / name
    latest_path = run_dir / "latest_checkpoint.pt"
    best_path = run_dir / "best_model_state_dict.pt"
    completed_path = run_dir / "completed.json"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if args.resume and not latest_path.is_file():
        raise FileNotFoundError(f"No resumable checkpoint at {latest_path}")
    if args.resume and completed_path.is_file() and not args.extend_completed:
        raise RuntimeError(f"Run is already complete: {run_dir}")
    if args.extend_completed and not completed_path.is_file():
        raise FileNotFoundError(
            f"--extend-completed requires {completed_path}; use ordinary --resume "
            "after an interrupted extension."
        )

    set_seed(args.seed)
    data = load_global_ppi_data(
        args.networks_dir,
        args.esm2_embeddings,
        seed=args.split_seed,
        verbose=False,
    )
    print(
        f"loaded proteins={data.features.size(0)} "
        f"global_edges={data.all_edge_index.size(1) // 2} "
        f"split_source_contexts={data.source_context_count} "
        f"splits={data.split_counts}",
        flush=True,
    )
    pooling, residue_ids, pooler_lr = None, None, None
    if args.residue_config:
        pooling = yaml.safe_load(Path(args.residue_config).read_text())
        pooler_lr = float(pooling.pop("lr"))
        manifest = json.loads((Path(pooling["cache_root"]) / "manifest.json").read_text())
        lookup = {}
        for index, gene in enumerate(manifest["genes"]):
            lookup.setdefault(gene, index)
        residue_ids = [lookup[gene] for gene in data.protein_names]
    model = GlobalS2GAE(
        data.features.size(1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        decode_channels=args.decode_channels,
        decoder_layers=args.decoder_layers,
        decoder_dropout=args.decoder_dropout,
        device=device,
        residue_pooling=pooling,
        residue_ids=residue_ids,
    ).to(device)
    if any(not key.startswith(("encoder.", "decoder.", "residue_pooler.", "residue_ids")) for key in model.state_dict()):
        raise RuntimeError("The baseline contains parameters outside the encoder/decoder.")
    if pooling:
        torch.testing.assert_close(model.residue_pooler.feature_mean.cpu(), data.feature_mean, rtol=0, atol=0)
        torch.testing.assert_close(model.residue_pooler.feature_std.cpu(), data.feature_std, rtol=0, atol=0)
        groups = [dict(params=list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=args.lr),
                  dict(params=model.residue_pooler.parameters(), lr=pooler_lr)]
    else:
        groups = model.parameters()
    optimizer = torch.optim.Adam(groups, lr=args.lr, weight_decay=0.0)

    if not args.resume:
        run_dir.mkdir(parents=False, exist_ok=False)

    config_path = run_dir / "config.json"
    previous_config = None
    if args.resume:
        with config_path.open(encoding="utf-8") as handle:
            previous_config = json.load(handle)

    if previous_config is None:
        tracking = tracking_config(args, name, data, model.model_config)
    else:
        for key in ("wandb_entity", "wandb_project", "wandb_group"):
            if previous_config[key] != getattr(args, key):
                raise ValueError(f"Cannot resume: {key} changed.")
        tracking = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "group": args.wandb_group,
            "run_id": previous_config["wandb_run_id"],
        }

    config = {
        **vars(args),
        "networks_dir": str(args.networks_dir),
        "esm2_embeddings": str(args.esm2_embeddings),
        "output_root": str(args.output_root),
        "resolved_device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "split_counts": data.split_counts,
        "graph_fingerprint": data.graph_fingerprint,
        "feature_fingerprint": data.feature_fingerprint,
        "split_fingerprint": data.split_fingerprint,
        "protocol": protocol_metadata(),
        "git_commit": os.environ.get("PROTSCAPE_GIT_COMMIT", "uncommitted"),
    }
    wandb_run_id = tracking["run_id"]
    config["wandb_run_id"] = wandb_run_id
    if args.residue_config:
        config["residue_pooling"] = training_config(args)["residue_pooling"]
    if not args.resume:
        write_json(config_path, config)

    start_epoch = 0
    best_epoch = -1
    best_global_val_ap = -np.inf
    history = []
    continuation = None
    if args.resume:
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint.get("format_version") != 3:
            raise ValueError("Cannot resume: unsupported checkpoint format.")
        if checkpoint["graph_fingerprint"] != data.graph_fingerprint:
            raise ValueError("Cannot resume: global graph fingerprint changed.")
        if checkpoint["split_fingerprint"] != data.split_fingerprint:
            raise ValueError("Cannot resume: edge split fingerprint changed.")
        if checkpoint["feature_fingerprint"] != data.feature_fingerprint:
            raise ValueError("Cannot resume: ESM2 features changed.")
        if not torch.equal(checkpoint["feature_mean"].cpu(), data.feature_mean):
            raise ValueError("Cannot resume: ESM2 feature means changed.")
        if not torch.equal(checkpoint["feature_std"].cpu(), data.feature_std):
            raise ValueError("Cannot resume: ESM2 feature scales changed.")
        if checkpoint["protocol"] != protocol_metadata():
            raise ValueError("Cannot resume: evaluation protocol changed.")
        if checkpoint["experiment_role"] != args.experiment_role:
            raise ValueError("Cannot resume: experiment role changed.")
        if checkpoint["model_config"] != model.model_config:
            raise ValueError("Cannot resume: model configuration changed.")
        current_training = training_config(args)
        if args.extend_completed:
            with completed_path.open(encoding="utf-8") as handle:
                previous_completion = json.load(handle)
            continuation = validate_completed_extension(
                checkpoint,
                previous_completion,
                previous_config,
                current_training,
                config["git_commit"],
            )
            prior_best = torch.load(
                best_path,
                map_location="cpu",
                weights_only=False,
            )
            validate_archived_best(prior_best, checkpoint)
        else:
            if checkpoint["git_commit"] != config["git_commit"]:
                raise ValueError("Cannot resume: Git commit changed.")
            if checkpoint["training_config"] != current_training:
                raise ValueError("Cannot resume: training configuration changed.")
        if checkpoint["wandb"]["run_id"] != wandb_run_id:
            raise ValueError("Cannot resume: W&B run identity changed.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_global_val_ap = float(checkpoint["best_global_val_ap"])
        history_path = run_dir / "history.csv"
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
            history = [
                row for row in history if int(row["epoch"]) <= int(checkpoint["epoch"])
            ]
            write_history(history_path, history)

        if continuation is not None:
            archive_suffix = f"epoch{continuation['from_completed_epochs']}"
            archives = {
                completed_path: run_dir / f"completed_{archive_suffix}.json",
                config_path: run_dir / f"config_{archive_suffix}.json",
                best_path: run_dir / f"best_model_state_dict_{archive_suffix}.pt",
            }
            existing = [str(path) for path in archives.values() if path.exists()]
            if existing:
                raise FileExistsError(
                    "Cannot extend: provenance archive already exists: "
                    + ", ".join(existing)
                )
            shutil.copy2(config_path, archives[config_path])
            shutil.copy2(best_path, archives[best_path])
            os.replace(completed_path, archives[completed_path])

            prior_continuations = list(checkpoint.get("continuations", []))
            args.continuations = [*prior_continuations, continuation]
            for checkpoint_path in (latest_path, best_path):
                saved = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                saved["training_config"] = current_training
                saved["git_commit"] = config["git_commit"]
                saved["wandb"] = tracking
                saved["continuations"] = args.continuations
                save_checkpoint(checkpoint_path, saved)
            config["continuations"] = args.continuations
            write_json(config_path, config)
        elif previous_config is not None:
            args.continuations = list(previous_config.get("continuations", []))
    else:
        save_checkpoint(
            latest_path,
            checkpoint_payload(
                args,
                model,
                data,
                epoch=-1,
                best_epoch=-1,
                best_global_val_ap=-np.inf,
                val_metrics={},
                optimizer=optimizer,
                tracking=tracking,
            ),
        )

    stopping = EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta)
    if stopping.patience > 0:
        if [int(row["epoch"]) for row in history] != list(range(start_epoch)):
            raise ValueError("Early stopping requires the complete validation history on resume.")
        for row in history:
            stopping.observe(float(row["global_val_ap"]), int(row["epoch"]))

    global_val_bank = build_global_negative_bank(
        data, split="val", max_k=1, seed=args.split_seed
    )

    tracker = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=name,
        id=wandb_run_id,
        config=None if args.resume else config,
        mode=args.wandb_mode,
        dir=str(run_dir),
        resume=(
            "must"
            if args.resume and start_epoch > 0
            else "allow"
            if args.resume
            else "never"
        ),
        tags=[
            "protscape",
            "global-ppi",
            "context-free",
            "s2gae",
            "acm",
            args.mask_type,
            args.experiment_role,
        ],
        settings=wandb.Settings(_disable_stats=True),
    )
    if args.resume:
        tracker.config.update(config, allow_val_change=True)

    data.features = data.features.to(device)

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        if stopping.should_stop:
            break
        started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, train_logits, train_labels = masked_reconstruction_step(
            model,
            data,
            device=device,
            mask_ratio=args.mask_ratio,
            mask_type=args.mask_type,
            k_negatives=args.k_negatives,
            validate_targets=epoch == start_epoch,
        )
        loss.backward()
        optimizer.step()

        train_metrics = binary_metrics(
            torch.sigmoid(train_logits.detach()).cpu().numpy(),
            train_labels.detach().cpu().numpy(),
        )
        model.eval()
        with torch.no_grad():
            _, val_layers = model.encode(
                data.features, data.train_edge_index.to(device)
            )
        global_val_metrics = evaluate_global_edges(
            model,
            data,
            split="val",
            k_values=[1],
            device=device,
            seed=args.split_seed,
            layer_embeddings=val_layers,
            negative_bank=global_val_bank,
        )
        val_metrics = global_val_metrics[1]
        stopping.observe(float(val_metrics["ap"]), epoch)
        row = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_ap": train_metrics["ap"],
            "train_f1": train_metrics["f1"],
            "global_val_ap": val_metrics["ap"],
            "global_val_f1": val_metrics["f1"],
            "global_val_roc": val_metrics["roc"],
            "global_val_acc": val_metrics["acc"],
            "cpu_peak_rss_gib": peak_rss_gib(),
            "update_seconds": time.monotonic() - started,
        }
        if device.type == "cuda":
            row.update(
                {
                    "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                    / 2**30,
                    "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device)
                    / 2**30,
                }
            )
        history.append(row)
        write_history(run_dir / "history.csv", history)

        if val_metrics["ap"] > best_global_val_ap:
            best_global_val_ap = float(val_metrics["ap"])
            best_epoch = epoch
            save_checkpoint(
                best_path,
                checkpoint_payload(
                    args,
                    model,
                    data,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_global_val_ap=best_global_val_ap,
                    val_metrics=val_metrics,
                    tracking=tracking,
                ),
            )

        save_checkpoint(
            latest_path,
            checkpoint_payload(
                args,
                model,
                data,
                epoch=epoch,
                best_epoch=best_epoch,
                best_global_val_ap=best_global_val_ap,
                val_metrics=val_metrics,
                optimizer=optimizer,
                tracking=tracking,
            ),
        )
        tracker.log(
            {**row, "best_global_val_ap": best_global_val_ap},
            step=epoch,
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={loss.item():.5f} "
            f"val_ap={val_metrics['ap']:.5f} "
            f"best={best_global_val_ap:.5f}",
            flush=True,
        )
        del loss, train_logits, train_labels
        del val_layers
        last_epoch = epoch

    stop_reason = "early_stopping" if stopping.should_stop else "max_epochs"
    print(f"Training finished after {last_epoch + 1} updates: {stop_reason}", flush=True)
    completion = {
        "run_name": name,
        "experiment_role": args.experiment_role,
        "completed_epochs": last_epoch + 1,
        "last_epoch": last_epoch,
        "max_epochs": args.epochs,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_global_val_ap": best_global_val_ap,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selection_metric": protocol_metadata()["selection_metric"],
        "graph_fingerprint": data.graph_fingerprint,
        "feature_fingerprint": data.feature_fingerprint,
        "split_fingerprint": data.split_fingerprint,
        "protocol": protocol_metadata(),
        "git_commit": config["git_commit"],
    }
    if getattr(args, "continuations", None):
        completion["continuations"] = args.continuations
    if stopping.patience > 0:
        completion["early_stopping"] = {
            "patience": stopping.patience,
            "min_delta": stopping.min_delta,
            "reference_ap": stopping.reference_ap,
            "last_improvement_epoch": stopping.last_improvement_epoch,
            "wait_updates": stopping.last_epoch - stopping.last_improvement_epoch,
        }
    tracker.summary["best_global_val_ap"] = best_global_val_ap
    tracker.summary["best_epoch"] = best_epoch
    tracker.summary["best_checkpoint"] = str(best_path)
    tracker.finish()
    write_json(run_dir / "completed.json", completion)


if __name__ == "__main__":
    main()
