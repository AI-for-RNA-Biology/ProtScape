"""CF pooling ablation; reuse the existing residue cache and resumable queue.

prepare ROOT CONTEXTUAL_ROOT, then submit ROOT. Each stage checks its outputs
before automatically submitting the next stage. Tests never select a model.
"""
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from .run_residue_ablation import (
    GNN, DOWNSTREAM, MODES, REPO, grid, read_json, save_json, replace_arg,
    run, worker, sequence_tasks,
)

STAGES = ["smoke", "train", "evaluate", "downstream"]
BOS = Path("/users/aloistho/projects/outputs/global_s2gae_convergence/selection.json")


def settings(root):
    name = read_json(root / "inputs.json").get("config_file", "cf_residue_ablation.yaml")
    return yaml.safe_load((root / "source/configs" / name).read_text())


def modes(root):
    return read_json(root / "inputs.json").get("modes", ["bos", *MODES])


def trial_grid(config):
    if "modes" not in config:
        return grid(config)
    trials = []
    for mode in config["modes"]:
        widths = [None] if mode == "mean" else config["pooler_hidden_dim"]
        if "paired_seeds" in config and mode in {"mlp", "attention"}:
            widths = [config["baseline_hidden_dim"]]
        if mode == "swe":
            widths = config["num_ref_points"]
        rates = [None] if mode == "mean" else config["pooler_lr"]
        for lr, width, rate in itertools.product(config["backbone_lr"], widths, rates):
            tag = "ref" if mode == "swe" else "h"
            name = f"{mode}_lr{lr}" + (f"_{tag}{width}_plr{rate}" if width else "")
            trial = dict(name=name, mode=mode, backbone_lr=lr, hidden_dim=width, lr=rate)
            if mode == "swe":
                trial["num_ref_points"] = width
            trials.append(trial)
    return trials


def features(root, mode):
    if mode == "bos":
        return Path(read_json(root / "inputs.json")["release"]) / "data/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk"
    return root / "residues/mean.plk"


def train_command(root, trial, *, smoke_name=None):
    config = settings(root)["pretraining"]
    release = Path(read_json(root / "inputs.json")["release"])
    name = smoke_name or trial["name"]
    destination = root / ("smoke" if smoke_name else "pretraining")
    args = [GNN, "-m", "pretraining.train_global_s2gae",
            "--networks-dir", str(release / "data/networks_bulk"),
            "--esm2-embeddings", str(features(root, trial["mode"])),
            "--output-root", str(destination), "--run-name", name,
            "--device", "cuda", "--wandb-mode", "online"]
    for key in ["hidden_dim", "num_layers", "dropout", "seed", "early_stopping_patience",
                "early_stopping_min_delta", "wandb_entity", "wandb_project", "wandb_group"]:
        value = config[key]
        if key == "seed":
            value = trial.get("seed", value)
        if key == "wandb_group" and smoke_name:
            value += "_smoke"
        args += ["--" + key.replace("_", "-"), str(value)]
    args += ["--epochs", str(4 if smoke_name else config["epochs"]),
             "--lr", str(trial["backbone_lr"]), "--split-seed", "0",
             "--decode-channels", "512", "--decoder-layers", "2",
             "--decoder-dropout", "0", "--mask-ratio", "0.5", "--mask-type", config.get("mask_type", "dm"),
             "--k-negatives", "1"]
    if trial["mode"] != "mean":
        args += ["--residue-config", str(root / "configs" / f"{trial['name']}.yaml")]
    if (destination / name / "latest_checkpoint.pt").exists():
        args.append("--resume")
    return args


def downstream_command(root, mode, task, *, smoke=False):
    config = settings(root)["downstream"]
    release = Path(read_json(root / "inputs.json")["release"])
    return [DOWNSTREAM, "-m", "downstream_tasks.run", "--task", task["task"],
            "--task-csv", task["csv"], "--model", "lr_global,lr_global_ext_embed",
            "--inference-model", mode, "--global-inference", str(root / "inference" / mode),
            "--output-root", str(root / ("smoke_downstream" if smoke else "downstream")),
            "--context-ppi-edgelists", str(release / "data/networks_bulk/ppi_edgelists"),
            "--esm2-embeddings", str(features(root, "bos")),
            "--embedding-source", "esm", "--dataset-mode", "bulk", "--lr", "0.0001",
            "--weight-decay", "0.0001", "--batch-size", "512", "--seed", str(config["seed"]),
            "--epochs", "2" if smoke else str(config["epochs"]),
            "--patience", "1" if smoke else str(config["patience"]),
            "--train-selection-metric", "auprc"]


def prepare(root, previous, panel="sequence"):
    from pretraining.cache_esm2_residues import sha256
    if root.exists():
        raise FileExistsError(f"Use the existing queue, not a second preparation: {root}")
    if not (previous / "residues/COMPLETE.json").exists():
        raise ValueError("A complete residue cache is required")
    root.mkdir(parents=True)
    for name in ["source", "configs", "logs", "failures", "queue", "pretraining", "smoke", "inference", "ppi"]:
        (root / name).mkdir()
    for name in ["residues", "python-deps", "localization"]:
        (root / name).symlink_to((previous / name).resolve(), target_is_directory=True)
    with subprocess.Popen(["git", "archive", "HEAD"], cwd=REPO, stdout=subprocess.PIPE) as archive:
        subprocess.run(["tar", "-x", "-C", str(root / "source")], stdin=archive.stdout, check=True)
        if archive.wait():
            raise RuntimeError("Source snapshot failed")
    inputs = read_json(previous / "inputs.json")
    inputs.update(workflow="context_free_residue_pooling", contextual_runs_preserved=str(previous),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  bos_checkpoint=read_json(BOS)["selected_checkpoint"])
    if panel in {"partner", "recycle"}:
        filename = "cf_partner_ablation.yaml" if panel == "partner" else "cf_recycling_ablation.yaml"
        inputs.update(workflow="cf_partner_pairmasked" if panel == "partner" else "cf_recycling_pairmasked",
                      config_file=filename, reuse_sequence_baselines_from=str(previous))
        inputs["modes"] = yaml.safe_load((root / "source/configs" / filename).read_text())["pretraining"]["modes"]
    elif panel != "sequence":
        raise ValueError(f"Unknown experiment panel: {panel}")
    save_json(root / "inputs.json", inputs)
    # Shared downstream pooler code reads this filename; scientific settings are
    # copied from CF config, never from the cancelled contextual experiment.
    config = settings(root)
    (root / "source/configs/residue_ablation.yaml").write_text(yaml.safe_dump(config))
    shutil.copy2(previous / "source/configs/paths.yaml", root / "source/configs/paths.yaml")
    for mode in ["lr_esm_bos", "lr_esm_mean"]:
        (root / "inference" / mode).symlink_to((previous / "inference" / mode).resolve(), target_is_directory=True)
    trials = trial_grid(config["pretraining"])
    # Round-robin families: attention and SWE start alongside mean/MLP.
    grouped = [[t for t in trials if t["mode"] == mode] for mode in modes(root) if mode != "bos"]
    trials = [t for group in itertools.zip_longest(*grouped) for t in group if t]
    save_json(root / "trials.json", trials)
    fingerprint = sha256(root / "residues/manifest.json")
    for trial in trials:
        if trial["mode"] == "mean":
            continue
        pool = dict(mode=trial["mode"], cache_root=str(root / "residues"),
                    manifest_sha256=fingerprint, lr=trial["lr"], residue_budget=32768)
        if trial["mode"] == "swe":
            pool["num_ref_points"] = trial["num_ref_points"]
        else:
            pool["hidden_dim"] = trial["hidden_dim"]
        if panel == "partner" and trial["mode"] not in MODES:
            pool.update(max_partners=config["pretraining"]["max_partners"], query_chunk_size=config["pretraining"]["query_chunk_size"])
        (root / "configs" / f"{trial['name']}.yaml").write_text(yaml.safe_dump(pool))
    def command(action, *args):
        return [GNN, "-m", "scripts.cscs.run_cf_residue_ablation", action, str(root), *map(str, args)]
    queues = {
        "smoke": [dict(key=m, command=command("partner_smoke" if panel != "sequence" else "smoke_one", m)) for m in modes(root) if m != "bos"],
        "train": [dict(key=t["name"], command=command("train_one", i)) for i, t in enumerate(trials)],
        "evaluate": [dict(key=m, command=command("evaluate_one", m)) for m in modes(root)],
        "downstream": [dict(key=f"{m}_{t['task']}", command=downstream_command(root, m, t))
                       for m, t in itertools.product(modes(root), inputs["tasks"])],
    }
    if panel == "sequence":
        queues["downstream"].extend(sequence_tasks(root, inputs["tasks"], config["downstream"]))
    else:
        save_json(root / "execution.json", dict(train=1, train_gpu_cache=True, evaluate=1, evaluate_gpu_cache=True,
            reason="One frozen-residue GPU bank per process; validated by panel smoke. No persistent graph-derived cache."))
    if panel == "recycle":
        # Reuse only genuinely matching completed UM baseline configurations.
        inputs["model_families"] = inputs["modes"]
        inputs["modes"] = [f"{m}_seed{s}" for m in inputs["model_families"] for s in config["pretraining"]["paired_seeds"]]
        save_json(root / "inputs.json", inputs)
        reuse = reuse_pairmasked_baselines(root, previous.parent / "protscape-cf-partner-pooling", trials)
        queues["train"] = [task for task in queues["train"] if task["key"] not in reuse]
        queues["repeat"] = []  # populated after validation selection, never from test scores
        queues["evaluate"] = [dict(key=m, command=command("evaluate_one", m)) for m in modes(root)]
        queues["downstream"] = [dict(key=f"{m}_{t['task']}", command=downstream_command(root, m, t))
                                for m, t in itertools.product(modes(root), inputs["tasks"])]
    for stage, tasks in queues.items():
        (root / "queue" / stage).mkdir()
        save_json(root / "queue" / stage / "tasks.json", tasks)
    n_downstream = len(inputs["tasks"]) * (2 * len(modes(root)) + (14 if panel == "sequence" else 0))
    save_json(root / "PREPARED.json", dict(pretraining_runs=len(trials), downstream_configs=n_downstream,
                                          downstream_queue_tasks=len(queues["downstream"]),
                                          repeat_runs=2 * len(inputs.get("model_families", [])),
                                          reused_pretraining_runs=len(trials) - len(queues["train"])))
    print(f"Prepared {len(trials)} CF training configurations; {n_downstream} downstream configurations", flush=True)


def train_one(root, index):
    trial = read_json(root / "trials.json")[int(index)]
    if not (root / "pretraining" / trial["name"] / "completed.json").exists():
        run(train_command(root, trial), root / "logs" / f"fit_{trial['name']}.log")


def partner_smoke(root, mode):
    """Largest-width real-graph fit, pair-hiding audit, export and LR smoke."""
    import pandas as pd
    import torch
    from pretraining.export_global_s2gae_embeddings import export_embeddings
    trial = next(t for t in reversed(read_json(root / "trials.json")) if t["mode"] == mode)
    name = f"pairmasked_{mode}"
    directory = root / "smoke" / name
    # This execution change only caches frozen residue inputs; pooler weights
    # and graph-dependent outputs are never cached across optimizer updates.
    os.environ["PROTSCAPE_RESIDUE_GPU_CACHE"] = "1"
    if not (directory / "completed.json").exists():
        run(train_command(root, trial, smoke_name=name), root / "logs" / f"{name}.log")
    saved = torch.load(directory / "best_model_state_dict.pt", weights_only=True, map_location="cpu")
    assert saved["training_config"]["mask_type"] == saved["protocol"]["primary_mask_type"] == "um"
    if mode != "mean":
        key = ("residue_pooler.output_projection.weight" if mode.startswith("recycle_") or mode == "slots4"
               else "residue_pooler.swe.combination" if mode == "swe"
               else "residue_pooler.slot_output.weight" if mode == "pma4" else "residue_pooler.output.weight")
        weights = saved["model_state_dict"][key]
        assert torch.isfinite(weights).all() and weights.abs().sum() > 0
        if mode in {"partner", "self_query", "partner_slots", "partner_dispersion", "partner_mean_mlp"}:
            assert saved["model_state_dict"]["residue_pooler.query.weight"].abs().sum() > 0
        if mode.startswith("recycle_"):
            latest = torch.load(directory / "latest_checkpoint.pt", weights_only=False, map_location="cpu")
            assert latest["model_state_dict"]["residue_pooler.query_update.2.weight"].abs().sum() > 0
            del latest
    history = pd.read_csv(directory / "history.csv")
    peak = float(history.gpu_peak_reserved_gib.max())
    if peak >= 88:
        raise RuntimeError(f"{mode} exceeds the reserved-memory safety budget: {peak:.2f} GiB")
    if mode == "recycle_residues":
        run(["/usr/bin/env", "-u", "PROTSCAPE_RESIDUE_CACHE", "-u", "PROTSCAPE_RESIDUE_GPU_CACHE",
             GNN, "-m", "pytest", "tests/test_recycling_pooling.py", "-q"],
            root / "logs" / "recycling_gpu_unit_tests.log")
    release = Path(read_json(root / "inputs.json")["release"])
    export_embeddings(directory / "best_model_state_dict.pt", release / "data/networks_bulk",
                      features(root, mode), root / "inference" / name, torch.device("cuda"))
    torch.cuda.empty_cache()
    task = read_json(root / "inputs.json")["tasks"][0]
    run(downstream_command(root, name, task, smoke=True), root / "logs" / f"{name}_downstream.log")
    save_json(root / "smoke" / f"{mode}_checked.json", dict(
        mask_type="um", peak_reserved_gib=peak, update_seconds=float(history.update_seconds.iloc[1:].median()),
        completed_updates=len(history), export_checked=True, downstream_checked=True))


def reuse_pairmasked_baselines(root, old_root, trials):
    """Retired experiments remain recoverable; imported checkpoints are read-only."""
    import torch
    reused = {}
    config = settings(root)["pretraining"]
    for trial in trials:
        if trial["mode"] not in {"mean", "attention"}:
            continue
        folder = old_root / "pretraining" / trial["name"]
        if not (folder / "completed.json").exists():
            continue
        saved = torch.load(folder / "best_model_state_dict.pt", weights_only=True, map_location="cpu")
        train = saved["training_config"]
        expected = dict(mask_type="um", seed=0, split_seed=0, lr=trial["backbone_lr"],
                        mask_ratio=.5, k_negatives=1, epochs=5000,
                        early_stopping_patience=config["early_stopping_patience"],
                        early_stopping_min_delta=config["early_stopping_min_delta"])
        if any(train.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Reusable baseline does not match settings: {folder}")
        expected_model = dict(hidden_dim=512, num_layers=2, dropout=.4, decode_channels=512,
                              decoder_layers=2, decoder_dropout=0.)
        if any(saved["model_config"].get(k) != v for k, v in expected_model.items()):
            raise ValueError(f"Reusable backbone does not match: {folder}")
        if trial["mode"] != "mean":
            old_pool = train["residue_pooling"]
            new_pool = yaml.safe_load((root / "configs" / f"{trial['name']}.yaml").read_text())
            if {k: v for k, v in old_pool.items() if k != "cache_root"} != {k: v for k, v in new_pool.items() if k != "cache_root"}:
                raise ValueError(f"Reusable pooler config differs: {folder}")
        (root / "pretraining" / trial["name"]).symlink_to(folder)
        reused[trial["name"]] = str(folder)
    save_json(root / "reused_baselines.json", reused)
    return reused


def smoke_one(root, mode):
    """Measure the largest pooler per family: one then two concurrent fits."""
    import pandas as pd
    import torch
    from pretraining.export_global_s2gae_embeddings import export_embeddings
    trial = next(t for t in reversed(read_json(root / "trials.json")) if t["mode"] == mode)
    measurements = []
    for count in [1, 2]:
        processes = []
        names = [f"{mode}_pack{count}_{i}" for i in range(count)]
        for name in names:
            folder = root / "smoke" / name
            if (folder / "completed.json").exists():
                continue
            log = (root / "logs" / f"{name}.log").open("a")
            process = subprocess.Popen(train_command(root, trial, smoke_name=name), stdout=log, stderr=subprocess.STDOUT)
            processes.append((process, log))
        codes = []
        for process, log in processes:
            codes.append(process.wait())
            log.close()
        if any(codes):
            if count == 1:
                raise RuntimeError(f"CF {mode} smoke failed; inspect logs")
            # A failed packed benchmark must not block a viable single-worker
            # experiment, but it is explicitly recorded, never treated as success.
            measurements.append(dict(workers=count, passed=False, exit_codes=codes))
            break
        histories = [pd.read_csv(root / "smoke" / name / "history.csv") for name in names]
        seconds = max(float(h.update_seconds.iloc[1:].median()) for h in histories)
        peak = max(float(h.gpu_peak_reserved_gib.max()) for h in histories)
        measurements.append(dict(workers=count, passed=True, update_seconds=seconds,
                                 updates_per_second=count / seconds, peak_reserved_gib=peak))
    save_json(root / "smoke" / f"{mode}_packing.json", measurements)
    name = f"{mode}_pack1_0"
    saved = torch.load(root / "smoke" / name / "best_model_state_dict.pt", weights_only=True, map_location="cpu")
    if mode != "mean":
        key = "residue_pooler.swe.combination" if mode == "swe" else "residue_pooler.output.weight"
        assert torch.isfinite(saved["model_state_dict"][key]).all()
        if mode != "swe":
            assert saved["model_state_dict"][key].abs().sum() > 0
    release = Path(read_json(root / "inputs.json")["release"])
    export_embeddings(root / "smoke" / name / "best_model_state_dict.pt", release / "data/networks_bulk",
                      features(root, mode), root / "inference" / name, torch.device("cuda"))
    torch.cuda.empty_cache()
    task = read_json(root / "inputs.json")["tasks"][0]
    run(downstream_command(root, name, task, smoke=True), root / "logs" / f"{mode}_downstream_smoke.log")
    if mode != "mean":
        benchmark_probes(root, mode)


def probe_one(root, mode):
    from downstream_tasks.residue_probe import run as probe
    trial = next(t for t in reversed(read_json(root / "trials.json")) if t["mode"] == mode)
    task = read_json(root / "inputs.json")["tasks"][0]
    size = trial["num_ref_points"] if mode == "swe" else trial["hidden_dim"]
    probe(root, mode, task["task"], size, trial["lr"], smoke=True)


def benchmark_probes(root, mode):
    """Separate short downstream fits; never count smoke results in the table."""
    results = []
    for count in [1, 2]:
        children, folders = [], []
        for index in range(count):
            folder = root / "smoke" / f"probe_{mode}_pack{count}_{index}"
            folder.mkdir(exist_ok=True)
            for name in ["source", "residues", "inputs.json", "trials.json"]:
                link = folder / name
                if not link.exists():
                    link.symlink_to(root / name)
            folders.append(folder)
            log = (root / "logs" / f"{folder.name}.log").open("a")
            child = subprocess.Popen([GNN, "-m", "scripts.cscs.run_cf_residue_ablation", "probe_one", str(folder), mode], stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log))
        codes = []
        for child, log in children:
            codes.append(child.wait())
            log.close()
        if any(codes):
            if count == 1:
                raise RuntimeError(f"Independent {mode} downstream smoke failed")
            results.append(dict(workers=count, passed=False, exit_codes=codes))
            break
        resources = [read_json(next(folder.glob("smoke_downstream/*/*/*/resources.json"))) for folder in folders]
        seconds = max(r["seconds"] for r in resources)
        results.append(dict(workers=count, passed=True, seconds=seconds, fits_per_second=count / seconds,
                            peak_reserved_gib=max(r["peak_reserved_gib"] for r in resources)))
    save_json(root / "smoke" / f"{mode}_downstream_packing.json", results)


def selection_row(root, trial):
    import torch
    folder = root / "pretraining" / trial["name"]
    completed = read_json(folder / "completed.json")
    assert completed["selection_metric"] == "global_val_ap"
    saved = torch.load(folder / "best_model_state_dict.pt", map_location="cpu", weights_only=True)
    assert saved["best_global_val_ap"] == completed["best_global_val_ap"]
    assert saved["training_config"]["seed"] == trial.get("seed", 0)
    return dict(**trial, score=saved["best_global_val_ap"], epoch=saved["epoch"] + 1,
                completed_updates=completed["completed_epochs"], stop_reason=completed["stop_reason"],
                checkpoint=str(folder / "best_model_state_dict.pt"))


def select(root):
    rows = [selection_row(root, t) for t in read_json(root / "trials.json")]
    save_json(root / "pretraining_grid.json", rows)
    families = read_json(root / "inputs.json").get("model_families", modes(root))
    selected = {m: max((r for r in rows if r["mode"] == m), key=lambda r: r["score"]) for m in families if m != "bos"}
    if "model_families" not in read_json(root / "inputs.json"):
        save_json(root / "selected.json", selected)
        return
    save_json(root / "selected_seed0.json", selected)
    repeats = []
    for family, winner in selected.items():
        trial = next(t for t in read_json(root / "trials.json") if t["name"] == winner["name"])
        for seed in settings(root)["pretraining"]["paired_seeds"][1:]:
            name = f"{trial['name']}_seed{seed}"
            repeats.append({**trial, "name": name, "seed": seed})
            if family != "mean":
                path = root / "configs" / f"{name}.yaml"
                if not path.exists():
                    path.symlink_to(root / "configs" / f"{trial['name']}.yaml")
    save_json(root / "repeats.json", repeats)
    save_json(root / "queue/repeat/tasks.json", [dict(key=t["name"], command=[GNN, "-m",
        "scripts.cscs.run_cf_residue_ablation", "repeat_one", str(root), str(i)]) for i, t in enumerate(repeats)])


def repeat_one(root, index):
    trial = read_json(root / "repeats.json")[int(index)]
    if not (root / "pretraining" / trial["name"] / "completed.json").exists():
        run(train_command(root, trial), root / "logs" / f"fit_{trial['name']}.log")


def select_repeats(root):
    selected = {f"{m}_seed0": {**row, "seed": 0} for m, row in read_json(root / "selected_seed0.json").items()}
    for trial in read_json(root / "repeats.json"):
        selected[f"{trial['mode']}_seed{trial['seed']}"] = selection_row(root, trial)
    assert set(selected) == set(modes(root))
    save_json(root / "selected.json", selected)


def evaluate_one(root, mode):
    import hashlib
    import numpy as np
    import torch
    from pretraining.evaluate_global_s2gae import build_model
    from pretraining.export_global_s2gae_embeddings import export_embeddings, _validate_checkpoint_data
    from pretraining.global_s2gae import load_global_ppi_data, evaluate_global_edges
    from pretraining.compare_inference_topologies import load_context_data, encode_context_free_local, score_decoder, split_masks
    from pretraining.contextwise_ppi import sample_structured_negatives, metrics_from_pos_neg
    inputs = read_json(root / "inputs.json")
    release = Path(inputs["release"])
    path = Path(inputs["bos_checkpoint"] if mode == "bos" else read_json(root / "selected.json")[mode]["checkpoint"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    data = load_global_ppi_data(release / "data/networks_bulk", features(root, mode), verbose=False)
    _validate_checkpoint_data(saved, data)
    contextual = torch.load(release / "models/pretraining/protscape_main_state_dict.pt", weights_only=True, map_location="cpu")
    graphs, values, names, cells = load_context_data(contextual, release / "data/networks_bulk", features(root, mode))
    assert names == saved["protein_names"]
    device = torch.device("cuda")
    model = build_model(saved, device)
    rows, digest = [], hashlib.sha256()
    with torch.no_grad():
        _, global_layers = model.encode(data.features.to(device), data.train_val_edge_index.to(device))
        unique_test = evaluate_global_edges(model, data, split="test", k_values=[1, 10, 50, 100, 500],
                                            device=device, seed=0, layer_embeddings=global_layers)
        for cell, graph in graphs.items():
            message, query = split_masks(graph, "test")
            positive = graph.edge_index[:, query]
            # Identical per-context query banks in global/local encoding and in
            # every pooling variant; topology never includes held-out positives.
            np.random.seed(cell)
            negative = sample_structured_negatives(positive, graph.edge_index, graph.num_nodes, 500)
            digest.update(str(cell).encode())
            digest.update(positive.numpy().tobytes())
            digest.update(negative.numpy().tobytes())
            local_layers = encode_context_free_local(model, values, graph, message, device)
            for inference, layers in [("global", [x[graph.feature_index.to(device)] for x in global_layers]), ("cell", local_layers)]:
                scores = score_decoder(model.decoder, layers, torch.cat([positive, negative], 1), device, 100000)
                n = positive.shape[1]
                positives, negatives = scores[:n], scores[n:].reshape(n, 500)
                for k in [1, 10, 50, 100, 500]:
                    metrics = metrics_from_pos_neg(positives, negatives[:, :k].ravel())
                    rows.append(dict(cell=cells[cell], inference=inference, k=k,
                                     auprc=metrics["ap"], f1=metrics["f1"]))
    result = dict(query_sha256=digest.hexdigest(), rows=rows, global_unique_test=unique_test)
    save_json(root / "ppi" / f"{mode}.json", result)
    log_test_summary(saved, result)
    del model, graphs, global_layers, local_layers, data, values
    torch.cuda.empty_cache()
    export_embeddings(path, release / "data/networks_bulk", features(root, mode), root / "inference" / mode, device)


def log_test_summary(checkpoint, result):
    """Final selected-checkpoint test metrics, not selection signals or curves."""
    import pandas as pd
    import wandb
    tracking = checkpoint["wandb"]
    tracked = wandb.Api().run(f"{tracking['entity']}/{tracking['project']}/{tracking['run_id']}")
    tracked.summary["test/selected_checkpoint"] = True
    tracked.summary["test/checkpoint_update"] = checkpoint["epoch"] + 1
    for k, values in result["global_unique_test"].items():
        for metric in ["ap", "f1"]:
            tracked.summary[f"test/global_unique/{metric}_k{k}"] = float(values[metric])
    frame = pd.DataFrame(result["rows"]).groupby(["inference", "k"])[["auprc", "f1"]].mean()
    for (inference, k), values in frame.iterrows():
        for metric in ["auprc", "f1"]:
            tracked.summary[f"test/cell_macro_{inference}_encoding/{metric}_k{k}"] = float(values[metric])
    tracked.update()


def summarize_ppi(root):
    import pandas as pd
    frames, hashes = [], set()
    for mode in modes(root):
        value = read_json(root / "ppi" / f"{mode}.json")
        hashes.add(value["query_sha256"])
        frames.append(pd.DataFrame(value["rows"]).assign(pooling=mode))
    assert len(hashes) == 1, "PPI query banks changed between pooling modes"
    ppi = pd.concat(frames).groupby(["pooling", "inference", "k"], as_index=False)[["auprc", "f1"]].mean()
    ppi.to_csv(root / "ppi_summary.csv", index=False)
    inputs = read_json(root / "inputs.json")
    plot_modes = modes(root)
    if "model_families" in inputs:
        ppi["seed"] = ppi.pooling.str.rsplit("_seed", n=1).str[-1].astype(int)
        ppi["pooling"] = ppi.pooling.str.rsplit("_seed", n=1).str[0]
        ppi.to_csv(root / "ppi_per_seed.csv", index=False)
        paired_differences(ppi, ["inference", "k"], "auprc").to_csv(root / "ppi_paired_differences.csv", index=False)
        averaged = ppi.groupby(["pooling", "inference", "k"])[["auprc", "f1"]].agg(["mean", "std"])
        averaged.to_csv(root / "ppi_three_seed_summary.csv")
        ppi = ppi.groupby(["pooling", "inference", "k"], as_index=False)[["auprc", "f1"]].mean()
        plot_modes = inputs["model_families"]
        unique_rows = []
        for mode in modes(root):
            for k, values in read_json(root / "ppi" / f"{mode}.json")["global_unique_test"].items():
                unique_rows.append(dict(pooling=mode.rsplit("_seed", 1)[0], seed=int(mode.rsplit("_seed", 1)[1]),
                                        k=int(k), auprc=values["ap"], f1=values["f1"]))
        pd.DataFrame(unique_rows).to_csv(root / "ppi_global_unique_per_seed.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in ["auprc", "f1"]:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        for ax, inference in zip(axes, ["global", "cell"]):
            for mode in plot_modes:
                frame = ppi[(ppi.pooling == mode) & (ppi.inference == inference)]
                ax.plot(range(5), 100 * frame[metric], marker="o", label=mode)
            ax.set(xticks=range(5), xticklabels=[1, 10, 50, 100, 500], title=f"CF / {inference} encoding", xlabel="Negatives per positive", ylabel=metric.upper() + " (%)")
        axes[-1].legend()
        fig.tight_layout()
        fig.savefig(root / f"ppi_{metric}.png", dpi=200)
        plt.close(fig)


def paired_differences(frame, keys, metric):
    """C−B and C−D within each training seed; no unpaired fold pseudo-replicates."""
    import pandas as pd
    selected = frame[frame.pooling.isin(["recycle_memory", "recycle_residues", "recycle_no_graph"])]
    wide = selected.pivot(index=[*keys, "seed"], columns="pooling", values=metric)
    if wide.isna().any().any():
        raise ValueError("Missing paired recycling control")
    return pd.DataFrame({"C_minus_B": wide.recycle_residues - wide.recycle_memory,
                         "C_minus_D": wide.recycle_residues - wide.recycle_no_graph}).reset_index()


def summarize(root):
    import numpy as np
    import pandas as pd
    from downstream_tasks.data.partitions import partition_path
    summarize_ppi(root)
    rows = []
    for task in read_json(root / "inputs.json")["tasks"]:
        files = list((root / "downstream" / task["task"]).glob("*/*/results.csv"))
        expected = 2 * len(modes(root)) + (0 if "reuse_sequence_baselines_from" in read_json(root / "inputs.json") else 14)
        assert len(files) == expected, f"Incomplete downstream configs for {task['task']}: {len(files)}/{expected}"
        with np.load(partition_path(task["task"], Path(task["csv"])), allow_pickle=False) as original:
            for file in files:
                with np.load(file.parent / "split_indices.npz", allow_pickle=False) as saved:
                    for key in ["genes", *[f"fold_{i}" for i in range(6)]]:
                        np.testing.assert_array_equal(saved[key], original[key])
                frame = pd.read_csv(file)
                for record in frame.to_dict("records"):
                    rows.append({**record, "task": task["task"], "pooling": file.parent.parent.name, "result_path": str(file)})
    results = pd.DataFrame(rows)
    results.to_csv(root / "results_all_configs.csv", index=False)
    results["readout"] = results.base_model_key.fillna(results.pooling)
    best = results.loc[results.groupby(["task", "pooling", "readout"]).val_auprc_macro_mean.idxmax()]
    best.to_csv(root / "results.csv", index=False)
    overview = best.assign(auprc=100 * best.test_auprc_macro_mean)[["task", "pooling", "readout", "auprc"]]
    tt = overview[overview.task.str.startswith("therapeutic_target_")].groupby(["pooling", "readout"], as_index=False).auprc.mean().assign(task="TT mean (15)")
    overview = pd.concat([overview[~overview.task.str.startswith("therapeutic_target_")], tt])
    if "model_families" in read_json(root / "inputs.json"):
        overview["seed"] = overview.pooling.str.rsplit("_seed", n=1).str[-1].astype(int)
        overview["pooling"] = overview.pooling.str.rsplit("_seed", n=1).str[0]
        overview.to_csv(root / "downstream_per_seed.csv", index=False)
        paired_differences(overview, ["task", "readout"], "auprc").to_csv(root / "downstream_paired_differences.csv", index=False)
        overview = overview.groupby(["task", "pooling", "readout"], as_index=False).agg(
            auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"), seeds=("seed", "nunique"))
        assert (overview.seeds == 3).all()
        overview.to_csv(root / "downstream_three_seed_summary.csv", index=False)
        resource_rows = []
        for mode, selected in read_json(root / "selected.json").items():
            directory = Path(selected["checkpoint"]).parent
            history = pd.read_csv(directory / "history.csv")
            resource_rows.append(dict(model=mode, selected_update=selected["epoch"],
                completed_updates=selected["completed_updates"],
                parameters=read_json(directory / "completed.json")["parameter_count"],
                median_train_val_seconds=float(history.update_seconds.iloc[1:].median()),
                peak_reserved_gib=float(history.gpu_peak_reserved_gib.max())))
        pd.DataFrame(resource_rows).to_csv(root / "pretraining_resources.csv", index=False)
    partner_panel = read_json(root / "inputs.json").get("workflow") == "cf_partner_pairmasked"
    title = "Pair-masked CF partner pooling" if partner_panel else "CF residue pooling ablation"
    protocol = ("All models here use unordered-pair training masking (UM); compare additions to the matched mean/gated controls, not directly to the earlier DM sweep. Independent sequence baselines are reused from the original pipeline."
                if partner_panel else "Independent ESM poolers are freshly learned on downstream training folds.")
    if "model_families" in read_json(root / "inputs.json"):
        title = "CF graph-guided residue recycling"
        protocol = "Matched UM masking throughout. Mean ± sample SD over three paired pretraining seeds; readout seed/folds are fixed. C−B tests residue access; C−D tests graph feedback. See paired-difference CSVs and resource table. Earlier DM results are historical, not matched controls."
    (root / "RESULTS.md").write_text(f"# {title}\n\nAUPRC (%). Same released task cohorts/folds; localization is the preserved 37-label development benchmark. CF poolers are learned on global PPI only. {protocol}\n\n" + overview.round(2).to_markdown(index=False) + "\n")
    save_json(root / "COMPLETE.json", read_json(root / "PREPARED.json"))


def gpu_worker(root, stage):
    count = 1
    if stage in {"train", "repeat", "downstream", "evaluate"}:
        policy = read_json(root / "packing.json")
        # Optional measured execution choice: one GPU-resident residue bank is
        # faster than two CPU-fed fits. Never allocate two copies of that bank.
        override = root / "execution.json"
        if override.exists():
            policy.update(read_json(override))
        policy_stage = "train" if stage == "repeat" else stage
        count = policy.get(policy_stage, 1)
        if policy.get(policy_stage + "_gpu_cache", False):
            if count != 1:
                raise ValueError("GPU residue caching requires one process per GPU")
            os.environ["PROTSCAPE_RESIDUE_GPU_CACHE"] = "1"
    children = [subprocess.Popen([GNN, "-m", "scripts.cscs.run_cf_residue_ablation", "worker", str(root), stage]) for _ in range(count)]
    codes = [child.wait() for child in children]
    if any(codes):
        raise RuntimeError(f"GPU worker failed: {codes}")


def submit(root, stage="smoke", allocation=1):
    if (root / "RETIRED.json").exists():
        raise RuntimeError("This experiment was retired; no further submissions are allowed")
    if int(allocation) > settings(root)["max_allocations"][stage]:
        raise RuntimeError(f"{stage} allocation cap reached; manual review required")
    workflow = read_json(root / "inputs.json").get("workflow")
    prefix = {"cf_partner_pairmasked": "cf-partner", "cf_recycling_pairmasked": "cf-recycle"}.get(workflow, "cf-pool")
    job = subprocess.check_output(["sbatch", "--parsable", "--job-name", f"{prefix}-{stage}",
        "--output", str(root / "logs" / f"{stage}_%j.slurm.log"),
        str(root / "source/scripts/cscs/run_cf_residue_ablation.sbatch"), str(root), stage, str(allocation)], text=True).strip().split(";")[0]
    save_json(root / "ACTIVE.json", dict(stage=stage, allocation=int(allocation), job=job))
    print(f"Submitted {stage}: {job}", flush=True)


def finish(root, stage, allocation):
    if (root / "RETIRED.json").exists():
        raise RuntimeError("This experiment was retired")
    if list((root / "failures").glob("*.json")):
        raise RuntimeError("Worker failure: inspect failures/ before continuing")
    folder = root / "queue" / stage
    if not all((folder / f"{t['key']}.done.json").exists() for t in read_json(folder / "tasks.json")):
        submit(root, stage, int(allocation) + 1)
        return
    if stage == "smoke":
        if read_json(root / "inputs.json").get("workflow") in {"cf_partner_pairmasked", "cf_recycling_pairmasked"}:
            families = read_json(root / "inputs.json").get("model_families", modes(root))
            checks = {m: read_json(root / "smoke" / f"{m}_checked.json") for m in families}
            assert all(c["peak_reserved_gib"] < 88 for c in checks.values())
            save_json(root / "packing.json", dict(train=1, downstream=2, measured_pretraining=checks,
                downstream_policy="Two frozen-vector LR fits per GPU; no learned residue pooler or residue bank downstream"))
            submit(root, "train")
            return
        measurements = {m: read_json(root / "smoke" / f"{m}_packing.json") for m in MODES}
        fits = all(len(v) == 2 and v[1]["passed"] and 2 * v[1]["peak_reserved_gib"] + 6 < 90 for v in measurements.values())
        gains = [v[1]["updates_per_second"] / v[0]["updates_per_second"] for v in measurements.values() if len(v) == 2 and v[1]["passed"]]
        packed = fits and min(gains, default=0) > 1.05
        probes = {m: read_json(root / "smoke" / f"{m}_downstream_packing.json") for m in ["mlp", "attention", "swe"]}
        probe_fits = all(len(v) == 2 and v[1]["passed"] and 2 * v[1]["peak_reserved_gib"] + 6 < 90 for v in probes.values())
        probe_gains = [v[1]["fits_per_second"] / v[0]["fits_per_second"] for v in probes.values() if len(v) == 2 and v[1]["passed"]]
        probe_packed = probe_fits and min(probe_gains, default=0) > 1.05
        save_json(root / "packing.json", dict(train=2 if packed else 1, downstream=2 if probe_packed else 1,
                                               measurements=measurements, downstream_measurements=probes, memory_margin_gib=6))
    elif stage == "train":
        select(root)
    elif stage == "repeat":
        select_repeats(root)
    elif stage == "evaluate":
        summarize_ppi(root)
    elif stage == "downstream":
        summarize(root)
        return
    stages = ["smoke", "train", "repeat", "evaluate", "downstream"] if "model_families" in read_json(root / "inputs.json") else STAGES
    submit(root, stages[stages.index(stage) + 1])


if __name__ == "__main__":
    action, root, *args = sys.argv[1:]
    root = Path(root).resolve()
    if action == "prepare":
        prepare(root, Path(args[0]).resolve(), *args[1:])
    else:
        globals()[action](root, *args)
