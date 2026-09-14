"""One resumable queue: cache -> smoke -> pretrain -> select/export -> downstream.

Usage: python -m scripts.cscs.run_residue_ablation prepare ROOT RELEASE
       python -m scripts.cscs.run_residue_ablation submit ROOT
Slurm calls the worker/finish stages. No test metric is used for selection.
"""
import fcntl
import itertools
import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

import yaml

GNN = "/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae/bin/python"
DOWNSTREAM = "/iopsstor/scratch/cscs/aloistho/protscape/conda-protscape-downstream/bin/python"
REPO = Path(__file__).resolve().parents[2]
STAGES = ["cache", "smoke", "train", "evaluate", "downstream"]
MODES = ["mean", "mlp", "attention", "swe"]


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def replace_arg(args, key, value):
    if key in args:
        args[args.index(key) + 1] = str(value)
    else:
        args.extend([key, str(value)])


def grid(config):
    result = []
    for mode in MODES:
        widths = [None] if mode == "mean" else config["swe_ref_points" if mode == "swe" else "pooler_hidden_dim"]
        rates = [None] if mode == "mean" else config["pooler_lr"]
        for lr, width, rate in itertools.product(config["backbone_lr"], widths, rates):
            size = "ref" if mode == "swe" else "h"
            name = f"{mode}_lr{lr}" + (f"_{size}{width}_plr{rate}" if width else "")
            trial = dict(name=name, mode=mode, backbone_lr=lr, hidden_dim=None if mode == "swe" else width, lr=rate)
            if mode == "swe":
                trial["num_ref_points"] = width
            result.append(trial)
    return result


def readouts(config):
    for fusion in [False, True]:
        stem = "hc_cell_ext_embed" if fusion else "hc_cell"
        yield dict(model=f"lr_{stem}", dropout=0)
        for dropout in config["dropout"]:
            yield dict(model=f"abmil_{stem}_gated_8", dropout=dropout)
        for pmax in config["pdl_pmax"]:
            yield dict(model=f"abmil_{stem}_gated_8_pdl", dropout=0, pdl_pmax=pmax)


def train_command(root, trial, epochs=None, destination=None):
    settings = yaml.safe_load((root / "source/configs/residue_ablation.yaml").read_text())["pretraining"]
    args = shlex.split(yaml.safe_load((root / "source/configs/pretraining.yaml").read_text())["protscape"]["args"])
    for key, value in {"--lr": trial["backbone_lr"], "--epochs": epochs or settings["epochs"],
                       "--seed": settings["seed"], "--checkpoint-selection": settings["selection"],
                       "--residue-config": root / "configs" / f"{trial['name']}.yaml",
                       "--eval-save-prefix": destination or root / "pretraining" / trial["name"],
                       "--wandb-mode": "disabled"}.items():
        replace_arg(args, key, value)
    return [GNN, "-m", "pretraining.train_protscape", *args]


def downstream_command(root, mode, task, readout, *, smoke=False):
    config = yaml.safe_load((root / "source/configs/residue_ablation.yaml").read_text())["downstream"]
    release = Path(read_json(root / "inputs.json")["release"])
    command = [DOWNSTREAM, "-m", "downstream_tasks.run", "--task", task["task"],
               "--task-csv", task["csv"], "--model", readout["model"],
               "--inference-model", mode, "--inference-root", str(root / "inference"),
               "--output-root", str(root / ("smoke_downstream" if smoke else "downstream")),
               "--esm2-embeddings", str(release / "data/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk"),
               "--embedding-source", "esm", "--dataset-mode", "bulk", "--att-dim", "256",
               "--dropout", str(readout["dropout"]), "--lr", "0.0001", "--weight-decay", "0.0001",
               "--batch-size", "512", "--seed", str(config["seed"]),
               "--epochs", "2" if smoke else str(config["epochs"]),
               "--patience", "1" if smoke else str(config["patience"]),
               "--train-selection-metric", "auprc"]
    if "pdl_pmax" in readout:
        command += ["--pdl-pmax", str(readout["pdl_pmax"])]
    if task["task"] != "protein_localization" and readout["model"] != "lr_ext_embed":
        benchmark = "corum" if task["task"] == "corum" else "therapeutic_target"
        model_key = "s2gae_bce_uni" if benchmark == "corum" else "s2gae_att_k1_fixed_do04_uni"
        table = root / "source/configs/downstream" / f"{benchmark}_selected_hyperparameters.csv"
        with table.open() as handle:
            choices = {row["cell_embedding_file"] for row in csv.DictReader(handle)
                       if row["task"] == task["task"] and row["inference_key"] == model_key
                       and row["base_model_key"] == readout["model"]
                       and row["scope"] in {"aggregate", "held_out"}}
        if len(choices) != 1:
            raise ValueError(f"Ambiguous released cell representation: {task}/{readout}")
        command += ["--cell-embedding-file", choices.pop()]
    return command


def sequence_command(root, mode, task):
    if mode not in {"bos", "mean"}:
        raise ValueError("Learned sequence poolers must be fitted independently downstream")
    command = downstream_command(root, f"lr_esm_{mode}", task, dict(model="lr_ext_embed", dropout=0))
    if mode == "mean":
        replace_arg(command, "--esm2-embeddings", root / "residues/mean.plk")
    return command


def sequence_tasks(root, tasks, config):
    for mode, task in itertools.product(["bos", "mean"], tasks):
        yield dict(key=f"lr_esm_{mode}_{task['task']}", command=sequence_command(root, mode, task))
    for mode in ["mlp", "attention", "swe"]:
        sizes = config["swe_ref_points" if mode == "swe" else "pooler_hidden_dim"]
        prefix = "ref" if mode == "swe" else "h"
        for task, width, lr in itertools.product(tasks, sizes, config["pooler_lr"]):
            yield dict(key=f"esm_{mode}_linear_{task['task']}_{prefix}{width}_lr{lr:g}",
                       command=[GNN, "-m", "downstream_tasks.residue_probe", str(root), mode,
                                task["task"], str(width), str(lr)])


def refresh_sequence(root, commit):
    """Apply the user's independent-baseline correction and append SWE-Simple."""
    from pretraining.cache_esm2_residues import sha256
    folder = root / "queue/downstream"
    if read_json(root / "ACTIVE.json")["stage"] == "downstream" or list(folder.glob("*.lock")):
        raise RuntimeError("Only amend this queue before downstream workers start")
    inputs = read_json(root / "inputs.json")
    settings = yaml.safe_load((root / "source/configs/residue_ablation.yaml").read_text())
    config = settings["downstream"]
    old = read_json(folder / "tasks.json")
    if any(task["key"].startswith("esm_") for task in old):
        raise RuntimeError("Independent sequence baselines are already queued")
    old_trials = read_json(root / "trials.json")
    trials = grid(settings["pretraining"])
    assert trials[:len(old_trials)] == old_trials
    new_trials = trials[len(old_trials):]
    assert len(new_trials) == 8 and all(t["mode"] == "swe" for t in new_trials)
    tasks = [task for task in old if not task["key"].startswith("lr_esm_")]
    tasks.extend(sequence_tasks(root, inputs["tasks"], config))
    for task, (index, readout) in itertools.product(inputs["tasks"], enumerate(readouts(config))):
        tasks.append(dict(key=f"swe_{task['task']}_{index}", command=downstream_command(root, "swe", task, readout)))
    assert len({task["key"] for task in tasks}) == len(tasks)
    shutil.copy2(folder / "tasks.json", folder / "tasks.before_independent_baselines.json")
    save_json(folder / "tasks.json", tasks)
    # Append, never renumber or repeat the already running mean/MLP/attention work.
    fingerprint = sha256(root / "residues/manifest.json")
    for trial in new_trials:
        pool = {key: trial[key] for key in ["mode", "hidden_dim", "lr", "num_ref_points"]}
        pool.update(cache_root=str(root / "residues"), manifest_sha256=fingerprint)
        (root / "configs" / f"{trial['name']}.yaml").write_text(yaml.safe_dump(pool))
    save_json(root / "trials.json", trials)
    for stage, additions in [
            ("train", [dict(key=t["name"], command=train_command(root, t)) for t in new_trials]),
            ("smoke", [dict(key=f"swe_{i}", command=[GNN, "-m", "scripts.cscs.run_residue_ablation", "smoke_one", str(root), str(trials.index(t))]) for i, t in enumerate(new_trials[:4])]),
            ("evaluate", [dict(key="swe", command=[GNN, "-m", "scripts.cscs.run_residue_ablation", "evaluate_one", str(root), "swe"])])]:
        file = root / "queue" / stage / "tasks.json"
        save_json(file, read_json(file) + additions)
    prepared = read_json(root / "PREPARED.json")
    prepared["downstream_runs"] = len(tasks)
    prepared["pretraining_runs"] = len(trials)
    save_json(root / "PREPARED.json", prepared)
    save_json(root / "SOURCE_AMENDMENT.json", dict(original_commit=inputs["git_commit"],
              amended_commit=commit, reason="Independent downstream-only ESM pooler baselines; add SWE-Simple only",
              cache_and_existing_pretraining_protocol_unchanged=True, added_swe_pretraining_runs=len(new_trials)))
    inputs["initial_git_commit"] = inputs["git_commit"]
    inputs["git_commit"] = commit
    save_json(root / "inputs.json", inputs)
    print(f"Amended queue: {len(trials)} pretraining, {len(tasks)} downstream configurations; existing work untouched")


def preserve_localization(root):
    import numpy as np
    from downstream_tasks.data.task_loaders import get_task_loader
    source = Path("/iopsstor/scratch/cscs/aloistho/protscape/topology-baselines/data/processed/hpa_v25_1_localization_per_location_confidence_min50.csv")
    split = Path("/capstor/scratch/cscs/aloistho/protscape/topology-baselines/results/protscape_downstream/protein_localization/bce_uni_hpa_v25_1_per_location_confidence_min50/abmil_hc_cell_gated_8__hp_lr0p0001_att256_do0_wd0p0001_selauprc_cw1__emb_esm/split_indices.npz")
    from pretraining.cache_esm2_residues import sha256
    assert sha256(source) == "9ca15894787a8c0ab9d9ab20527c4ff16e119bc4c246c3f63275e0a811715a1e"
    target = root / "localization" / source.name
    (target.parent / "splits").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    shutil.copy2(source.with_suffix(".manifest.json"), target.with_suffix(".manifest.json"))
    genes, labels, classes = get_task_loader("protein_localization", target).load()
    assert len(classes) == 37
    with np.load(split, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    lookup = {gene: index for index, gene in enumerate(genes)}
    arrays["labels"] = labels[[lookup[gene] for gene in arrays["genes"]]]
    arrays["class_names"] = np.array(classes)
    np.savez_compressed(target.parent / "splits/protein_localization.npz", **arrays)
    save_json(target.parent / "provenance.json", dict(label_sha256=sha256(source), source_split=str(split),
                                                      source_split_sha256=sha256(split), labels=37))
    return target


def preserve_references(root, release):
    """Reuse published TT/CORUM scores and the completed 37-label BOS sweep."""
    import numpy as np
    import pandas as pd
    from exploration.plotting.paper_tables import write_tables
    from downstream_tasks.data.partitions import partition_path
    write_tables(release / "data/paper_source_data", root / "paper_tables")
    rows = []
    for benchmark, model in [("corum", "s2gae_bce_uni"), ("therapeutic_target", "s2gae_att_k1_fixed_do04_uni")]:
        filename = "table_corum.csv" if benchmark == "corum" else "table_therapeutic_diseases.csv"
        # The public disease table contains only the primary readout. Use all
        # released per-fold rows to keep the six readout families comparable.
        settings = pd.read_csv(root / "source/configs/downstream" / f"{benchmark}_selected_hyperparameters.csv")
        mapping = settings[settings.inference_key.eq(model)].set_index("readout_key").base_model_key.to_dict()
        if benchmark == "corum":
            table = pd.read_csv(root / "paper_tables" / filename)
            table = table[table.inference_key.eq(model)]
            for row in table.itertuples():
                rows.append(dict(pooling="bos", task="corum", readout=mapping[row.readout_key], auprc=row.auprc_percent, source="Zenodo paper source table"))
        else:
            table = pd.read_csv(release / "data/paper_source_data/therapeutic_target_analysis/held_out_performance.csv")
            table = table[table.inference_key.eq(model)]
            for (task, readout), values in table.groupby(["task", "readout_key"]):
                assert sorted(values.fold) == list(range(5))
                rows.append(dict(pooling="bos", task=task, readout=mapping[readout], auprc=100 * values.auprc.mean(), source="Zenodo paper source table"))
    old = Path("/capstor/scratch/cscs/aloistho/protscape/topology-baselines/results/protscape_downstream/protein_localization/bce_uni_hpa_v25_1_per_location_confidence_min50")
    table = pd.concat([pd.read_csv(path).assign(result_path=str(path)) for path in old.glob("*/results.csv")], ignore_index=True)
    task = next(t for t in read_json(root / "inputs.json")["tasks"] if t["task"] == "protein_localization")
    wanted = {r["model"] for r in readouts(yaml.safe_load((root / "source/configs/residue_ablation.yaml").read_text())["downstream"])}
    with np.load(partition_path("protein_localization", Path(task["csv"])), allow_pickle=False) as original:
        for family in sorted(wanted):
            group = table[table.base_model_key.eq(family)]
            if group.empty:
                continue  # schedule only missing BOS localization readout families
            best = group.loc[group.val_auprc_macro_mean.idxmax()]
            assert best.n_label_classes == 37
            with np.load(Path(best.result_path).parent / "split_indices.npz", allow_pickle=False) as saved:
                for key in ["genes", *[f"fold_{i}" for i in range(6)]]:
                    np.testing.assert_array_equal(saved[key], original[key])
            rows.append(dict(pooling="bos", task="protein_localization", readout=family, auprc=100 * best.test_auprc_macro_mean,
                             validation_auprc=100 * best.val_auprc_macro_mean, result_path=best.result_path,
                             source="Completed 37-label development sweep (same saved split)"))
    pd.DataFrame(rows).to_csv(root / "reference_results.csv", index=False)


def prepare(root, release):
    from pretraining.cache_esm2_residues import prepare as prepare_cache, sha256
    from downstream_tasks.data.partitions import load_task_partition
    from downstream_tasks.config import THERAPEUTIC_TARGET_IDS
    if (root / "PREPARED.json").exists() or (root / "ACTIVE.json").exists():
        raise FileExistsError("Already prepared: reuse this queue, do not regenerate it")
    for folder in ["configs", "logs", "queue", "failures", "pretraining", "inference", "ppi", "sequence"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    if REPO != root / "source":
        # Snapshot tracked code, including committed development extensions.
        with subprocess.Popen(["git", "archive", "HEAD"], cwd=REPO, stdout=subprocess.PIPE) as archive:
            (root / "source").mkdir(exist_ok=True)
            subprocess.run(["tar", "-x", "-C", str(root / "source")], stdin=archive.stdout, check=True)
            if archive.wait():
                raise RuntimeError("Could not snapshot source")
    paths_file = root / "source/configs/paths.yaml"
    paths = yaml.safe_load(paths_file.read_text())
    for key, value in paths.items():
        if isinstance(value, str) and value.startswith("../ProtScape_release/"):
            paths[key] = str(release / value.removeprefix("../ProtScape_release/"))
    paths["output_root"] = str(root)
    paths_file.write_text(yaml.safe_dump(paths, sort_keys=False))
    localization = preserve_localization(root)
    tasks = [dict(task="corum", csv=str(release / "data/downstream_tasks/corum_dataset/corum_memberships_filtered.csv"))]
    tt = sorted(release / "data/downstream_tasks/therapeutic_target_dataset" / f"therapeutic_target_{disease}.csv" for disease in THERAPEUTIC_TARGET_IDS)
    assert len(tt) == 15, f"Expected all 15 released TT diseases, got {len(tt)}"
    tasks += [dict(task=path.stem.lower(), csv=str(path)) for path in tt]
    tasks += [dict(task="protein_localization", csv=str(localization))]
    for task in tasks:
        load_task_partition(task["task"], Path(task["csv"]))
    save_json(root / "inputs.json", dict(release=str(release), tasks=tasks,
                                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()))
    preserve_references(root, release)
    prepare_cache(root / "residues", release)
    # Sequence-only readouts use the same contextual node coverage to retain the
    # frozen paper cohort; their classifiers never receive these graph features.
    for name in ["bos", "lr_esm_bos", "lr_esm_mean"]:
        link = root / "inference" / name
        target = release / "embeddings/s2gae_att_k1_fixed_do04_uni5e6"
        if not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)
        assert link.resolve() == target.resolve()
    settings = yaml.safe_load((root / "source/configs/residue_ablation.yaml").read_text())
    trials = grid(settings["pretraining"])
    fingerprint = sha256(root / "residues/manifest.json")
    for trial in trials:
        pool = {key: trial[key] for key in ["mode", "hidden_dim", "lr"]}
        if trial["mode"] == "swe":
            pool["num_ref_points"] = trial["num_ref_points"]
        pool.update(cache_root=str(root / "residues"), manifest_sha256=fingerprint)
        (root / "configs" / f"{trial['name']}.yaml").write_text(yaml.safe_dump(pool))
    save_json(root / "trials.json", trials)
    for stage in STAGES:
        (root / "queue" / stage).mkdir(exist_ok=True)
    save_json(root / "queue/cache/tasks.json", [dict(key=str(i), command=[GNN, "-m", "scripts.cscs.run_residue_ablation", "cache_one", str(root), str(i)]) for i in range(4)])
    smoke = [next(t for t in trials if t["mode"] == mode) for mode in ["mean", "mlp", "attention"]]
    smoke += [next(t for t in reversed(trials) if t["mode"] == "attention")]
    smoke += [t for t in trials if t["mode"] == "swe"][:4]
    save_json(root / "queue/smoke/tasks.json", [dict(key=str(i), command=[GNN, "-m", "scripts.cscs.run_residue_ablation", "smoke_one", str(root), str(trials.index(t))]) for i, t in enumerate(smoke)])
    save_json(root / "queue/train/tasks.json", [dict(key=t["name"], command=train_command(root, t)) for t in trials])
    save_json(root / "queue/evaluate/tasks.json", [dict(key=m, command=[GNN, "-m", "scripts.cscs.run_residue_ablation", "evaluate_one", str(root), m]) for m in ["bos", *MODES]])
    downstream = []
    for mode, task, (index, readout) in itertools.product(MODES, tasks, enumerate(readouts(settings["downstream"]))):
        downstream.append(dict(key=f"{mode}_{task['task']}_{index}", command=downstream_command(root, mode, task, readout)))
    downstream.extend(sequence_tasks(root, tasks, settings["downstream"]))
    with (root / "reference_results.csv").open() as handle:
        hpa_done = {row["readout"] for row in csv.DictReader(handle) if row["task"] == "protein_localization"}
    hpa = next(task for task in tasks if task["task"] == "protein_localization")
    for index, readout in enumerate(readouts(settings["downstream"])):
        if readout["model"] not in hpa_done:
            downstream.append(dict(key=f"bos_protein_localization_{index}", command=downstream_command(root, "bos", hpa, readout)))
    save_json(root / "queue/downstream/tasks.json", downstream)
    save_json(root / "PREPARED.json", dict(pretraining_runs=len(trials), downstream_runs=len(downstream)))
    print(f"Prepared {len(trials)} pretraining runs and {len(downstream)} downstream readout runs")


def run(command, log, timeout=None, new_session=False):
    env = os.environ.copy()
    inputs = REPO.parent / "inputs.json"
    if inputs.is_file():
        env["PROTSCAPE_GIT_COMMIT"] = read_json(inputs)["git_commit"]
    if str(command[0]) == DOWNSTREAM:
        env.pop("PYTHONPATH", None)
    with Path(log).open("a") as handle:
        print(shlex.join(map(str, command)), file=handle, flush=True)
        with subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=new_session) as process:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return False  # reuse completed runs; pretraining resumes its last epoch
            if code:
                raise subprocess.CalledProcessError(code, command)
    return True


def worker(root, stage):
    folder = root / "queue" / stage
    deadline = float(os.environ["RESIDUE_DEADLINE"])
    for task in read_json(folder / "tasks.json"):
        if time.time() > deadline - 180 or any((root / "failures").glob("*.json")):
            break
        done = folder / f"{task['key']}.done.json"
        with (folder / f"{task['key']}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if done.exists():
                continue
            started = time.time()
            log = root / "logs" / f"{stage}_{task['key']}.log"
            try:
                complete = run(task["command"], log, timeout=max(1, deadline - time.time()), new_session=True)
            except Exception as error:
                save_json(root / "failures" / f"{stage}_{task['key']}.json", dict(error=str(error), log=str(log)))
                raise
            if complete:
                save_json(done, dict(seconds=time.time() - started, job=os.environ.get("SLURM_JOB_ID"), log=str(log)))
            else:
                break


def smoke_one(root, trial_index):
    import torch
    trial = read_json(root / "trials.json")[int(trial_index)]
    name = f"smoke_{trial['name']}"
    destination = root / "smoke" / trial["name"]
    checkpoint = destination / "best_model_state_dict.pt"
    if not checkpoint.is_file():
        run(train_command(root, trial, epochs=1, destination=destination), root / "logs" / f"{name}_train.log")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if trial["mode"] == "swe":
        reference = next(value for key, value in saved["model_state_dict"].items() if key.endswith("residue_pooler.swe.reference"))
        initial = torch.linspace(-1, 1, len(reference))[:, None].expand_as(reference)
        torch.testing.assert_close(reference, initial, rtol=0, atol=0)  # SWE-Simple reference stays frozen
        combination = next(value for key, value in saved["model_state_dict"].items() if key.endswith("residue_pooler.swe.combination"))
        assert torch.isfinite(combination).all()
    elif trial["mode"] != "mean":
        weights = [v for k, v in saved["model_state_dict"].items() if k.endswith("residue_pooler.output.weight")]
        assert len(weights) == 1 and weights[0].abs().sum() > 0, "Pooler did not learn"
    # The mappings file is written last by inference; reuse completed exports.
    if not (root / "inference" / name / "mappings.pkl").is_file():
        run([GNN, "-m", "pretraining.inference", str(checkpoint), "--output-dir", str(root / "inference" / name), "--overwrite"], root / "logs" / f"{name}_export.log")
    task = read_json(root / "inputs.json")["tasks"][0]
    run(downstream_command(root, name, task, dict(model="abmil_hc_cell_gated_8", dropout=0), smoke=True), root / "logs" / f"{name}_downstream.log")
    if trial["mode"] in {"mlp", "attention", "swe"}:
        from downstream_tasks.residue_probe import run as probe
        size = trial["num_ref_points"] if trial["mode"] == "swe" else trial["hidden_dim"]
        probe(root, trial["mode"], task["task"], size, trial["lr"], smoke=True)


def select(root):
    import torch
    rows = []
    for trial in read_json(root / "trials.json"):
        path = root / "pretraining" / trial["name"] / "best_model_state_dict.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["score_metric"] == "ap_ppi_plus_ap_meta"
        rows.append(dict(**trial, score=float(saved["score"]), epoch=int(saved["epoch"]) + 1, checkpoint=str(path)))
    selected = {mode: max([row for row in rows if row["mode"] == mode], key=lambda r: r["score"]) for mode in MODES}
    save_json(root / "pretraining_grid.json", rows)
    save_json(root / "selected.json", selected)


def evaluate_one(root, mode):
    import hashlib
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score
    from pretraining.checkpoints import load_protscape_model
    from pretraining.compare_inference_topologies import load_context_data, encode_protscape_local, prepare_decoder_input, score_decoder, split_masks
    from pretraining.contextwise_ppi import sample_structured_negatives
    release = Path(read_json(root / "inputs.json")["release"])
    checkpoint = release / "models/pretraining/protscape_main_state_dict.pt" if mode == "bos" else Path(read_json(root / "selected.json")[mode]["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    features_path = release / "data/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk" if mode == "bos" else root / "residues/mean.plk"
    graphs, features, _, names = load_context_data(saved, release / "data/networks_bulk", features_path)
    model = load_protscape_model(saved, graphs, device=torch.device("cuda")).eval()
    digest, rows = hashlib.sha256(), []
    with torch.no_grad():
        for cell, graph in graphs.items():
            message, query = split_masks(graph, "test")
            positive = graph.edge_index[:, query]
            np.random.seed(cell)
            negative = sample_structured_negatives(positive, graph.edge_index, graph.num_nodes, 1)
            digest.update(str(cell).encode())
            digest.update(positive.numpy().tobytes())
            digest.update(negative.numpy().tobytes())
            layers, raw = encode_protscape_local(model, features, graph, cell, message, torch.device("cuda"))
            decoder_input = prepare_decoder_input(model.s2gae_decoder, layers, raw)
            scores = score_decoder(model.s2gae_decoder, decoder_input, torch.cat([positive, negative], dim=1), torch.device("cuda"), 100000)
            labels = np.r_[np.ones(positive.shape[1]), np.zeros(negative.shape[1])]
            rows.append(dict(cell=names[cell], auprc=float(average_precision_score(labels, scores))))
    save_json(root / "ppi" / f"{mode}.json", dict(query_sha256=digest.hexdigest(), auprc=float(np.mean([r["auprc"] for r in rows])), contexts=rows))
    del model, graphs, features
    torch.cuda.empty_cache()
    if mode != "bos":
        run([GNN, "-m", "pretraining.inference", str(checkpoint), "--output-dir", str(root / "inference" / mode), "--overwrite"], root / "logs" / f"{mode}_export.log")


def summarize(root):
    import numpy as np
    import pandas as pd
    from downstream_tasks.data.partitions import partition_path
    inputs = read_json(root / "inputs.json")
    rows = pd.read_csv(root / "reference_results.csv").to_dict("records")
    reference_hash = None
    for mode in ["bos", *MODES]:
        ppi = read_json(root / "ppi" / f"{mode}.json")
        reference_hash = reference_hash or ppi["query_sha256"]
        assert ppi["query_sha256"] == reference_hash, "PPI evaluation queries differ"
        rows.append(dict(pooling=mode, task="PPI 1:1 context mean", readout="S2GAE", auprc=100 * ppi["auprc"]))
    for mode, task in itertools.product(MODES, inputs["tasks"]):
        files = sorted((root / "downstream" / task["task"] / mode).glob("*/results.csv"))
        assert len(files) == 22, f"Incomplete downstream sweep: {mode}/{task['task']}"
        frame = pd.concat([pd.read_csv(file).assign(result_path=str(file)) for file in files], ignore_index=True)
        assert frame.val_auprc_macro_mean.notna().all()
        # Check the saved identities/folds, not merely their sizes.
        with np.load(partition_path(task["task"], Path(task["csv"])), allow_pickle=False) as original:
            for file in files:
                with np.load(file.parent / "split_indices.npz", allow_pickle=False) as saved:
                    for key in ["genes", *[f"fold_{i}" for i in range(6)]]:
                        np.testing.assert_array_equal(saved[key], original[key])
        # Report readout families separately; selection uses validation only.
        for family, group in frame.groupby("base_model_key", sort=True):
            best = group.loc[group.val_auprc_macro_mean.idxmax()]
            rows.append(dict(pooling=mode, task=task["task"], readout=family,
                             auprc=100 * best.test_auprc_macro_mean,
                             validation_auprc=100 * best.val_auprc_macro_mean,
                             result_path=best.result_path))
    for mode, task in itertools.product(["bos", *MODES], inputs["tasks"]):
        learned = mode in {"mlp", "attention", "swe"}
        name = f"esm_{mode}_linear" if learned else f"lr_esm_{mode}"
        files = sorted((root / "downstream" / task["task"] / name).glob("*/results.csv"))
        config = yaml.safe_load((root / "source/configs/residue_ablation.yaml").read_text())["downstream"]
        sizes = config["swe_ref_points" if mode == "swe" else "pooler_hidden_dim"]
        expected = len(sizes) * len(config["pooler_lr"]) if learned else 1
        assert len(files) == expected, f"Incomplete independent ESM baseline: {name}/{task['task']}"
        with np.load(partition_path(task["task"], Path(task["csv"])), allow_pickle=False) as original:
            for file in files:
                with np.load(file.parent / "split_indices.npz", allow_pickle=False) as saved:
                    for key in ["genes", *[f"fold_{i}" for i in range(6)]]:
                        np.testing.assert_array_equal(saved[key], original[key])
                if learned:
                    metadata = read_json(file.parent / "provenance.json")
                    assert metadata["protscape_checkpoint_used"] is False
                    assert metadata["pooler_initialization"] == "fresh_each_fold"
                    assert metadata["supervision"] == "downstream_train_fold"
                    if mode == "swe":
                        assert metadata["swe_variant"] == "simple"
                        assert metadata["slicers_frozen"] and metadata["reference_frozen"]
        frame = pd.concat([pd.read_csv(file).assign(result_path=str(file)) for file in files], ignore_index=True)
        assert frame.val_auprc_macro_mean.notna().all()
        value = frame.loc[frame.val_auprc_macro_mean.idxmax()]
        rows.append(dict(pooling=mode, task=task["task"], readout=f"ESM_{mode}_linear" if learned else f"LR_ESM_{mode}",
                         auprc=100 * value.test_auprc_macro_mean,
                         validation_auprc=100 * value.val_auprc_macro_mean,
                         result_path=value.result_path))
    additional = list((root / "downstream/protein_localization/bos").glob("*/results.csv"))
    if additional:
        frame = pd.concat([pd.read_csv(path).assign(result_path=str(path)) for path in additional], ignore_index=True)
        for family, group in frame.groupby("base_model_key"):
            best = group.loc[group.val_auprc_macro_mean.idxmax()]
            rows.append(dict(pooling="bos", task="protein_localization", readout=family,
                             auprc=100 * best.test_auprc_macro_mean, validation_auprc=100 * best.val_auprc_macro_mean,
                             result_path=best.result_path, source="Missing BOS localization readout (released encoder frozen)"))
    result = pd.DataFrame(rows)
    result["pooling"] = result.pooling.replace({"swe": "SWE-Simple"})
    result["readout"] = result.readout.replace({"ESM_swe_linear": "ESM_SWE_simple_linear"})
    tt = result[result.task.str.startswith("therapeutic_target_")]
    averages = tt.groupby(["pooling", "readout"], as_index=False).auprc.mean().assign(task="TT mean (15 diseases)")
    result = pd.concat([result, averages], ignore_index=True)
    result.to_csv(root / "results.csv", index=False)
    overview = result[~result.task.str.startswith("therapeutic_target_")].pivot(index=["task", "readout"], columns="pooling", values="auprc").round(2)
    (root / "RESULTS.md").write_text("# Residue pooling ablation\n\nAUPRC (%); validation-selected configurations. BOS is the released model (not retrained). LR_ESM_BOS and LR_ESM_mean are frozen-feature LR probes. ESM_mlp_linear, ESM_attention_linear and ESM_SWE_simple_linear fit poolers + linear heads on downstream training folds only; they use no ProtScape weights. SWE-Simple freezes slicers/reference and learns only the reference-combination vector.\n\n" + overview.to_markdown() + "\n\nFull disease detail: results.csv. Localization uses the preserved 37-label dev dataset, not a paper-released benchmark.\n")
    save_json(root / "COMPLETE.json", dict(**read_json(root / "PREPARED.json"), ppi_query_sha256=reference_hash))


def submit(root, stage="cache", allocation=1):
    if not (root / "PREPARED.json").is_file():
        raise ValueError("Finish and validate preparation before submitting GPUs")
    settings = yaml.safe_load((root / "source/configs/residue_ablation.yaml").read_text())
    if allocation > settings["max_allocations"][stage]:
        raise RuntimeError(f"{stage} reached its allocation limit; inspect progress before extending")
    command = ["sbatch", "--parsable", "--exclude=nid007611", "--job-name", f"residue-{stage}",
               "--output", str(root / "logs" / f"{stage}_%j.slurm.log"),
               str(root / "source/scripts/cscs/run_residue_ablation.sbatch"), str(root), stage, str(allocation)]
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    save_json(root / "ACTIVE.json", dict(stage=stage, allocation=allocation, job=job))
    print(f"Submitted {stage}: {job}", flush=True)


def finish(root, stage, allocation):
    if any((root / "failures").glob("*.json")):
        raise RuntimeError("Worker failure: downstream stages were not submitted; see failures/")
    folder = root / "queue" / stage
    tasks = read_json(folder / "tasks.json")
    if not all((folder / f"{task['key']}.done.json").exists() for task in tasks):
        submit(root, stage, int(allocation) + 1)
        return
    if stage == "cache":
        from pretraining.cache_esm2_residues import finalize
        finalize(root / "residues")
    elif stage == "train":
        select(root)
    elif stage == "downstream":
        summarize(root)
        return
    submit(root, STAGES[STAGES.index(stage) + 1])


if __name__ == "__main__":
    action, root, *args = sys.argv[1:]
    root = Path(root).resolve()
    if action == "prepare":
        prepare(root, Path(args[0]).resolve())
    elif action == "cache_one":
        from pretraining.cache_esm2_residues import worker as cache_worker
        cache_worker(root / "residues", int(args[0]))
    else:
        globals()[action](root, *args)
