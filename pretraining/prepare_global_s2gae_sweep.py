"""Seed a larger global-S2GAE sweep with compatible completed shorter runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .run_global_s2gae_sweep import load_sweep


REQUIRED_FILES = (
    "best_model_state_dict.pt",
    "completed.json",
    "config.json",
    "history.csv",
    "latest_checkpoint.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_reusable(source: Path, target_config: dict) -> int:
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete reusable run {source}: {missing}")
    previous = _read_json(source / "config.json")
    completion = _read_json(source / "completed.json")
    previous_epochs = int(completion["completed_epochs"])
    target_epochs = int(target_config["epochs"])
    if previous_epochs >= target_epochs:
        raise ValueError(
            f"Reusable run {source} has {previous_epochs} epochs; target has "
            f"{target_epochs}."
        )
    if int(completion["last_epoch"]) != previous_epochs - 1:
        raise ValueError(f"Reusable run has an invalid last epoch: {source}")
    if completion["run_name"] != target_config["name"]:
        raise ValueError(f"Reusable run name mismatch: {source}")
    for key, expected in target_config.items():
        if key in {"name", "epochs"}:
            continue
        if previous.get(key) != expected:
            raise ValueError(
                f"Reusable run {source} changed {key}: "
                f"{previous.get(key)!r} != {expected!r}"
            )
    if int(previous["epochs"]) != previous_epochs:
        raise ValueError(f"Reusable config/completion epoch mismatch: {source}")
    return previous_epochs


def prepare(config_path: Path, source_root: Path, output_root: Path) -> list[dict]:
    _, configurations = load_sweep(config_path)
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_root == output_root:
        raise ValueError("Source and output roots must differ to preserve old runs.")
    output_root.mkdir(parents=True, exist_ok=True)

    reused = []
    for config in configurations:
        source = source_root / config["name"]
        target = output_root / config["name"]
        if target.exists() or not source.exists():
            continue
        previous_epochs = _validate_reusable(source, config)
        shutil.copytree(source, target, copy_function=shutil.copyfile)
        reused.append(
            {
                "run_name": config["name"],
                "from_epochs": previous_epochs,
                "to_epochs": int(config["epochs"]),
            }
        )
    return reused


def main() -> None:
    args = parse_args()
    reused = prepare(args.config, args.source_root, args.output_root)
    print(json.dumps({"reused_runs": reused}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
