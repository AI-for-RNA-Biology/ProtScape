"""Run the fixed-setting contextual / repeated-mean / CF-instance ablation."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from .data.loaders import load_global_protein_embeddings
from .data.task_loaders import get_task_loader

RELEASE_CODE = "334dd2ab8454d669ec9e9319ee3eac93ffa47608"
MODES = ("contextual", "mean", "global")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def selected_settings():
    result = []
    for filename, key, model in (
        ("corum_selected_hyperparameters.csv", "s2gae_bce_uni", "abmil_hc_cell_gated_8"),
        ("therapeutic_target_selected_hyperparameters.csv", "s2gae_att_k1_fixed_do04_uni", "abmil_hc_cell_ext_embed_gated_8_pdl"),
    ):
        content = subprocess.check_output(["git", "show", f"{RELEASE_CODE}:configs/downstream/{filename}"], text=True)
        result += [r for r in csv.DictReader(io.StringIO(content))
                   if r["inference_key"] == key and r["base_model_key"] == model and r["scope"] in {"aggregate", "held_out"}]
    assert len(result) == 16 and len({r["task"] for r in result}) == 16
    return result


def label_paths(release, task):
    base = release / "data/downstream_tasks"
    if task == "corum":
        root = base / "corum_dataset"
        return root / "corum_memberships_filtered.csv", root / "split_indices.npz"
    root = base / "therapeutic_target_dataset"
    identifier = task.removeprefix("therapeutic_target_").upper()
    return root / f"therapeutic_target_{identifier}.csv", root / "splits" / f"{task}.npz"


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "inputs.json"
    if target.exists():
        raise FileExistsError("Prepared protocol exists; reuse it rather than overwriting")
    settings = selected_settings()
    published = {}
    for line in (args.release / "SHA256SUMS").read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        published[name.lstrip("* ")] = sha
    files = {}
    # Reuse old large files only after verifying them against the new release.
    relative = {"data/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk"}
    for row in settings:
        for name in ("protein_embeddings.pt", "cell_embeddings.pt"):
            relative.add(f"embeddings/{row['embedding_inference_name']}/{name}")
    for name in sorted(relative):
        path = args.existing / name
        sha = digest(path)
        if sha != published[name]:
            raise ValueError(f"Existing file differs from new release: {path}")
        files[name] = {"path": str(path), "sha256": sha}
        print(f"Verified release input: {name}", flush=True)
    global_vectors = load_global_protein_embeddings(args.global_embeddings)
    cohorts = {}
    for row in settings:
        task = row["task"]
        labels, partition = label_paths(args.release, task)
        for path in (labels, partition):
            name = str(path.relative_to(args.release))
            if digest(path) != published[name]:
                raise ValueError(f"Release checksum mismatch: {path}")
        genes, y, classes = get_task_loader(task, labels).load()
        with np.load(partition, allow_pickle=False) as saved:
            cohort = saved["genes"].astype(str).tolist()
            positions = {g: i for i, g in enumerate(genes)}
            selected = [positions[g] for g in cohort]
            assert classes == saved["class_names"].astype(str).tolist()
            np.testing.assert_array_equal(y[selected], saved["labels"])
            assert not (set(cohort) - set(global_vectors)), f"CF misses {task} cohort proteins"
            folds = [saved[f"fold_{i}"] for i in range(6)]
            np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(len(cohort)))
            cohorts[task] = {"proteins": len(cohort), "classes": len(classes), "test_proteins": len(folds[0]),
                             "label_sha256": digest(labels), "split_sha256": digest(partition)}
    info = {"release_record": "22645081", "settings_source_commit": RELEASE_CODE,
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "release": str(args.release), "existing": str(args.existing), "settings": settings,
            "verified_release_files": files, "cohorts": cohorts, "modes": MODES,
            "global_embeddings": str(args.global_embeddings), "global_embedding_sha256": digest(args.global_embeddings),
            "context_protein_dim": 1536, "global_protein_dim": len(next(iter(global_vectors.values()))),
            "capacity_matched_comparison": "contextual versus repeated mean only"}
    target.write_text(json.dumps(info, indent=2) + "\n")
    print("Prepared all 16 tasks with released cohorts and splits", flush=True)


def worker(args):
    info = json.loads((args.output / "inputs.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert commit == info["code_commit"], "Code changed after preflight"
    release, existing = Path(info["release"]), Path(info["existing"])
    jobs = [(row, mode) for row in info["settings"] for mode in MODES]
    for index in range(args.lane, len(jobs), 4):
        row, mode = jobs[index]
        task = row["task"]
        identifier = f"release22645081_{mode}"
        results = list((args.output / "downstream_tasks" / task / identifier).glob("*/results.csv"))
        if results:
            print(f"Reuse completed {task} {mode}", flush=True)
            continue
        labels, partition = label_paths(release, task)
        command = [sys.executable, "-m", "downstream_tasks.run",
                   "--task", task, "--task-csv", str(labels), "--released-split", str(partition),
                   "--inference-model", identifier, "--embedding-inference-model", row["embedding_inference_name"],
                   "--inference-root", str(existing / "embeddings"),
                   "--esm2-embeddings", str(existing / "data/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk"),
                   "--output-root", str(args.output / "downstream_tasks"),
                   "--model", row["base_model_key"], "--protein-context-mode", mode,
                   "--control-global-embeddings", info["global_embeddings"],
                   "--att-dim", "256", "--train-selection-metric", "auprc"]
        for field, flag in (("lr", "--lr"), ("weight_decay", "--weight-decay"), ("batch_size", "--batch-size"),
                            ("seed", "--seed"), ("dropout", "--dropout"), ("pdl_pmax", "--pdl-pmax")):
            if row[field]:
                command += [flag, row[field]]
        log = args.output / "logs" / f"{task}_{mode}.log"
        print(f"Starting {task} {mode}", flush=True)
        with log.open("a") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT,
                           env={**os.environ, "PROTSCAPE_GIT_COMMIT": commit})
        print(f"Finished {task} {mode}", flush=True)


def summarize(args):
    rows = []
    reference = {}
    for path in sorted((args.output / "downstream_tasks").glob("*/*/*/results.csv")):
        frame = pd.read_csv(path)
        assert len(frame) == 1
        row = frame.iloc[0]
        task, mode = row["task"], row["protein_context_mode"]
        with np.load(path.parent / "split_indices.npz", allow_pickle=False) as split:
            current = {k: split[k].copy() for k in ["genes"] + [f"fold_{i}" for i in range(6)]}
        with np.load(path.parent / "test_predictions.npz", allow_pickle=False) as predicted:
            current["y_true"] = predicted["y_true"].copy()
            current["test_idx"] = predicted["test_idx"].copy()
        if task in reference:
            for key in current:
                np.testing.assert_array_equal(current[key], reference[task][key])
        else:
            reference[task] = current
        rows.append({"task": task, "mode": mode, "auprc": float(row["test_auprc_macro_mean"]) * 100})
    values = pd.DataFrame(rows)
    assert len(values) == 48 and not values.duplicated(["task", "mode"]).any()
    values.to_csv(args.output / "task_results.csv", index=False)
    table = values.pivot(index="task", columns="mode", values="auprc")
    table.to_csv(args.output / "disease_results.csv")
    overview = pd.DataFrame({"TT mean": table.loc[table.index.str.startswith("therapeutic_target_")].mean(),
                             "CORUM": table.loc["corum"]}).T
    overview.to_csv(args.output / "comparison.csv")
    print(overview.round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "worker", "summarize"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--existing", type=Path)
    parser.add_argument("--global-embeddings", type=Path)
    parser.add_argument("--lane", type=int, choices=range(4))
    args = parser.parse_args()
    {"prepare": prepare, "worker": worker, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
