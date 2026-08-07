"""Train the downstream configurations selected on validation data."""

import csv
import shlex
import subprocess
import sys
from pathlib import Path

from .config import DEFAULT_OUTPUT_ROOT


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "downstream"
TABLES = {
    "corum": CONFIG_DIR / "corum_selected_hyperparameters.csv",
    "therapeutic_targets": CONFIG_DIR
    / "therapeutic_target_selected_hyperparameters.csv",
}


def load_selected_runs(benchmark):
    with TABLES[benchmark].open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_run_dir(row):
    return (
        DEFAULT_OUTPUT_ROOT
        / row["task"]
        / row["source_inference_name"]
        / row["selected_output_model_key"]
    )


def find_selected_run(benchmark, **filters):
    matches = [
        row
        for row in load_selected_runs(benchmark)
        if all(row.get(column) == value for column, value in filters.items())
    ]
    if len(matches) != 1:
        criteria = ", ".join(f"{key}={value}" for key, value in filters.items())
        raise ValueError(
            f"Expected one selected {benchmark} run for {criteria}; found {len(matches)}"
        )
    return matches[0], selected_run_dir(matches[0])


def build_command(row):
    command = [
        sys.executable,
        "-m",
        "downstream_tasks.run",
        "--inference-model",
        row["source_inference_name"],
        "--embedding-inference-model",
        row["embedding_inference_name"],
        "--task",
        row["task"],
        "--model",
        row["base_model_key"],
        "--output-model-key",
        row["selected_output_model_key"],
        "--embedding-source",
        row["embedding_source"],
        "--dataset-mode",
        row["dataset_mode"],
        "--lr",
        row["lr"],
        "--weight-decay",
        row["weight_decay"],
        "--batch-size",
        row["batch_size"],
        "--seed",
        row["seed"],
        "--train-selection-metric",
        "auprc",
        "--scope",
        row["scope"],
        "--inference-key",
        row["inference_key"],
        "--inference-label",
        row["inference_label"],
        "--readout-key",
        row["readout_key"],
        "--readout-label",
        row["readout_label"],
    ]
    if row["disease"]:
        command.extend(["--disease", row["disease"]])
    if row["cell_embedding_file"]:
        command.extend(["--cell-embedding-file", row["cell_embedding_file"]])
    if row["att_hidden_dim"]:
        command.extend(["--att-dim", str(int(float(row["att_hidden_dim"])))])
    if row["dropout"]:
        command.extend(["--dropout", row["dropout"]])
    if row["pdl_pmax"]:
        command.extend(["--pdl-pmax", row["pdl_pmax"]])
    if row["use_class_weights"].lower() == "false":
        command.append("--no-class-weight")
    return command


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in TABLES:
        choices = "|".join(TABLES)
        raise SystemExit(
            f"Usage: python -m downstream_tasks.run_selected {choices}"
        )

    rows = load_selected_runs(sys.argv[1])

    for module in (
        "pretraining.generate_esm2_embeddings",
        "pretraining.generate_prostt5_embeddings",
    ):
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=REPO_ROOT,
            check=True,
        )

    commands = []
    seen = set()
    for row in rows:
        command = build_command(row)
        key = selected_run_dir(row)
        if key not in seen:
            commands.append(command)
            seen.add(key)

    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    for task in dict.fromkeys(row["task"] for row in rows):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "downstream_tasks.run",
                "--task",
                task,
                "--aggregate-only",
            ],
            cwd=REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
