"""Run one configuration from the global-S2GAE sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PARAMETERS = (
    "hidden_dim",
    "num_layers",
    "dropout",
    "decode_channels",
    "decoder_layers",
    "decoder_dropout",
    "mask_ratio",
    "mask_type",
    "k_negatives",
    "epochs",
    "lr",
    "seed",
    "split_seed",
    "wandb_entity",
    "wandb_project",
    "wandb_group",
    "experiment_role",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--networks-dir", type=Path, required=True)
    parser.add_argument("--esm2-embeddings", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="online"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sweep(path: Path) -> tuple[dict, list[dict]]:
    with path.open(encoding="utf-8") as handle:
        sweep = yaml.safe_load(handle)
    defaults = sweep["defaults"]
    runs = sweep["runs"]
    names = [run["name"] for run in runs]
    if len(names) != len(set(names)):
        raise ValueError("Sweep run names must be unique.")
    configurations = []
    for run in runs:
        config = {**defaults, **run}
        missing = set(PARAMETERS).difference(config)
        unknown = set(config).difference({"name", *PARAMETERS})
        if missing or unknown:
            raise ValueError(f"Invalid run {run['name']}: missing={missing}, unknown={unknown}")
        configurations.append(config)
    signatures = [tuple(config[key] for key in PARAMETERS) for config in configurations]
    if len(signatures) != len(set(signatures)):
        raise ValueError("Sweep contains duplicate parameter configurations.")
    return sweep, configurations


def command(args: argparse.Namespace, config: dict) -> list[str]:
    output_root = args.output_root.expanduser()
    run_dir = output_root / config["name"]
    cmd = [
        sys.executable,
        "-m",
        "pretraining.train_global_s2gae",
        "--networks-dir",
        str(args.networks_dir.expanduser()),
        "--esm2-embeddings",
        str(args.esm2_embeddings.expanduser()),
        "--output-root",
        str(output_root),
        "--run-name",
        config["name"],
        "--wandb-mode",
        args.wandb_mode,
        "--device",
        args.device,
    ]
    for key in PARAMETERS:
        cmd.extend([f"--{key.replace('_', '-')}", str(config[key])])
    if (run_dir / "latest_checkpoint.pt").is_file():
        cmd.append("--resume")
    elif run_dir.exists():
        raise FileExistsError(f"Non-resumable run directory exists: {run_dir}")
    return cmd


def main() -> None:
    args = parse_args()
    sweep, configurations = load_sweep(args.config.expanduser())
    if args.index < 0 or args.index >= len(configurations):
        raise IndexError(f"Sweep index {args.index} is outside [0, {len(configurations)}).")
    if sweep.get("selection_metric") != "global_val_ap":
        raise ValueError("The primary sweep must select on global_val_ap.")
    config = configurations[args.index]
    completed_path = args.output_root.expanduser() / config["name"] / "completed.json"
    if completed_path.is_file():
        print(f"skip completed run: {config['name']}", flush=True)
        return
    cmd = command(args, config)
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
