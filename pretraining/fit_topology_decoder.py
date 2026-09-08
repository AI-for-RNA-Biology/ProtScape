"""Nested pair-held-out correction heads on saved, frozen CF-global scores.

No GNN is trained or re-encoded. Contextual scores are used only to fit an
optional teacher on the fitting pairs, never as an input at student inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .investigate_global_topology import adjacency, pair_fold, BANKS, K_VALUES

VARIANTS = ("linear_context", "mlp_global", "mlp_context", "mlp_context_distilled")
SETTINGS = {"seed": 0, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 2048,
            "hidden_dim": 64, "max_updates": 1000, "eval_every": 25,
            "patience_updates": 100, "min_delta": 0.001,
            "selection": "mean inner context AP500 on random and global-hard banks",
            "fit_mixture": "one positive + ten random negatives", "teacher_weight": 0.5}


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def symmetric_structure(raw, source_degree):
    """Symmetrize directed PPR by detailed balance on an undirected graph.

    PPR(v,u) = degree(u)/degree(v) * PPR(u,v). Other saved features
    (min/max degrees, CN, RA) are already symmetric. Isolated nonself pairs
    have zero PPR. No residue or sequence feature extraction is involved.
    """
    values = raw.copy()
    source_degree = np.asarray(source_degree)[None, :, None]
    target_degree = values[..., 0] + values[..., 1] - source_degree
    ratio = np.divide(source_degree, target_degree, out=np.zeros_like(target_degree), where=target_degree > 0)
    values[..., 4] = np.where(target_degree > 0, .5 * values[..., 4] * (1 + ratio), 0)
    values[..., 4] *= 10000
    return np.log1p(values).astype(np.float32)


class CorrectionHead(nn.Module):
    def __init__(self, variant, mean, scale):
        super().__init__()
        columns = list(range(6 if variant == "mlp_global" else 11))
        self.register_buffer("columns", torch.tensor(columns))
        self.register_buffer("mean", torch.as_tensor(mean[columns], dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale[columns], dtype=torch.float32))
        if variant == "linear_context":
            self.net = nn.Sequential(nn.Linear(len(columns), 1))
        else:
            self.net = nn.Sequential(nn.Linear(len(columns), SETTINGS["hidden_dim"]), nn.ReLU(), nn.Linear(SETTINGS["hidden_dim"], 1))
        # Start at exactly the frozen CF score; the head learns a correction.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, values):
        inputs = (values[..., self.columns] - self.mean) / self.scale
        return values[..., 0] + self.net(inputs).squeeze(-1)


def fold_roles(folds, outer):
    return folds == (outer + 1) % 3, folds == (outer + 2) % 3, folds == outer


def prepare(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "dataset.pt").exists():
        raise FileExistsError("Reuse the existing prepared dataset")
    if args.parent_dir is not None:
        protocol = json.loads((args.parent_dir / "protocol.json").read_text())
        assert protocol["settings"] == SETTINGS
        (args.output_dir / "dataset.pt").symlink_to((args.parent_dir / "dataset.pt").resolve())
        protocol.update(source_file_sha256=file_hash(__file__), parent_dir=str(args.parent_dir), full_budget=args.full_budget)
        (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        print("Reusing prepared dataset; extending saved inner trajectories", flush=True)
        return
    source_protocol = json.loads((args.source_dir / "protocol.json").read_text())
    cache = torch.load(args.source_dir / "cache.pt", map_location="cpu", weights_only=False)
    data = cache["data"]
    global_degree = np.asarray(adjacency(data.train_edge_index.numpy(), len(data.protein_names)).sum(1)).ravel()
    features, context_scores, folds, cells, keys, hashes = [], [], [], [], [], {}
    for cell in source_protocol["selected"]:
        path = args.source_dir / f"cell_{cell}.npz"
        hashes[str(cell)] = file_hash(path)
        with np.load(path) as item:
            graph = cache["contexts"][cell]
            local_degree = np.asarray(adjacency(graph.edge_index[:, graph.train_mask].numpy(), graph.num_nodes).sum(1)).ravel()
            local_degree_by_global = np.zeros(len(data.protein_names), np.float32)
            local_degree_by_global[graph.feature_index.numpy()] = local_degree
            sources = item["positive_global"][0]
            full = symmetric_structure(item["global_features"], global_degree[sources])
            local = symmetric_structure(item["local"], local_degree_by_global[sources])
            features.append(np.concatenate([item["scores"][0, ..., None], full, local], axis=-1))
            context_scores.append(item["scores"][4])
            folds.append(item["folds"])
            cells.append(np.full(len(item["folds"]), cell, np.int32))
            keys.append(item["keys"])
    dataset = {"features": torch.from_numpy(np.concatenate(features, axis=1)),
               "context_scores": torch.from_numpy(np.concatenate(context_scores, axis=1)),
               "folds": torch.from_numpy(np.concatenate(folds)),
               "cells": torch.from_numpy(np.concatenate(cells)),
               "keys": torch.from_numpy(np.concatenate(keys, axis=1))}
    assert np.all(pair_fold(dataset["keys"].numpy()) == dataset["folds"].numpy()[None, :, None])
    assert torch.isfinite(dataset["features"]).all()
    torch.save(dataset, args.output_dir / "dataset.pt")
    protocol = {"source_dir": str(args.source_dir), "source_protocol": source_protocol,
                "source_file_sha256": file_hash(__file__), "full_budget": args.full_budget,
                "source_artifact_hashes": hashes, "settings": SETTINGS, "variants": VARIANTS,
                "feature_order": ["CF global logit", "global min degree", "global max degree", "global CN", "global RA", "global symmetric PPR",
                                  "local min degree", "local max degree", "local CN", "local RA", "local symmetric PPR"],
                "outer_folds": 3, "inner_roles": "fit=(outer+1)%3; stop=(outer+2)%3; refit=both; evaluation=outer",
                "inference_inputs_excluded": ["CF Cell-PPI logits", "contextual ProtScape logits", "cell embeddings", "RNA", "residue features"],
                "status": "exploratory validation-only correction heads; not new pretraining/test results"}
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"prepared_shape": list(dataset["features"].shape), "source_artifacts": len(hashes)}), flush=True)


def fit_arrays(dataset, positive_mask):
    features = dataset["features"][0, positive_mask, :11].reshape(-1, 11).numpy()
    labels = np.broadcast_to(np.arange(11) == 0, (int(positive_mask.sum()), 11)).ravel().copy()
    teacher_inputs = np.stack([features[:, 0], dataset["context_scores"][0, positive_mask, :11].numpy().ravel()], 1)
    return features, labels, teacher_inputs


def make_fit(variant, dataset, mask, device):
    values, labels, teacher_inputs = fit_arrays(dataset, mask)
    mean, scale = values.mean(0, dtype=np.float64), values.std(0, dtype=np.float64)
    scale[scale < 1e-6] = 1
    torch.manual_seed(SETTINGS["seed"])
    model = CorrectionHead(variant, mean, scale).to(device)
    target = labels.astype(np.float32)
    if variant.endswith("distilled"):
        teacher = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, tol=1e-7, random_state=0))
        teacher.fit(teacher_inputs, labels)
        assert teacher[-1].n_iter_.max() < 2000
        weight = SETTINGS["teacher_weight"]
        target = (1 - weight) * target + weight * teacher.predict_proba(teacher_inputs)[:, 1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["lr"], weight_decay=SETTINGS["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(0)
    return model, optimizer, generator, torch.as_tensor(values, device=device), torch.as_tensor(target, dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model, values, device):
    flat = values.reshape(-1, values.shape[-1])
    result = []
    for start in range(0, len(flat), 131072):
        result.append(model(flat[start:start + 131072].to(device)).cpu().numpy())
    return np.concatenate(result).reshape(values.shape[:-1])


def ap_by_cell(scores, cells, k=500):
    results = []
    for cell in np.unique(cells):
        values = scores[cells == cell, :k + 1]
        labels = np.broadcast_to(np.arange(k + 1) == 0, values.shape).ravel()
        results.append(average_precision_score(labels, values.ravel()))
    return float(np.mean(results))


def one_update(model, optimizer, generator, features, targets):
    indices = torch.randint(len(features), (SETTINGS["batch_size"],), generator=generator, device=features.device)
    optimizer.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(model(features[indices]), targets[indices])
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite correction-head loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def fit_fold(args, dataset, outer):
    destination = args.output_dir / args.variant / f"fold_{outer}"
    if (destination / "completed.json").exists():
        print(f"Reuse completed {args.variant} fold {outer}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    folds = dataset["folds"]
    train, monitor, evaluation = fold_roles(folds, outer)
    assert not torch.any(train & monitor) and not torch.any((train | monitor) & evaluation)
    model, optimizer, generator, features, targets = make_fit(args.variant, dataset, train, device)
    monitor_values = {bank: dataset["features"][bank, monitor].to(device) for bank in (0, 2)}
    monitor_cells = dataset["cells"][monitor].numpy()

    def score():
        model.eval()
        aps = {BANKS[bank]: ap_by_cell(predict(model, values, device), monitor_cells) for bank, values in monitor_values.items()}
        model.train()
        return aps, float(np.mean(list(aps.values())))

    started = perf_counter()
    aps, initial = score()
    best_score, patience_score, best_step, patience_step = initial, initial, 0, 0
    history = [{"update": 0, "selection_ap": initial, **aps}]
    first_step, replayed = 1, 0
    parent_dir = getattr(args, "parent_dir", None)
    if parent_dir is not None:
        parent = parent_dir / args.variant / f"fold_{outer}"
        history = pd.read_csv(parent / "inner_history.csv").to_dict("records")
        assert abs(history[0]["selection_ap"] - initial) < 1e-8
        last_step = int(history[-1]["update"])
        if (parent / "inner_latest.pt").exists():
            latest = torch.load(parent / "inner_latest.pt", map_location="cpu", weights_only=True)
            assert latest["update"] == last_step
            model.load_state_dict(latest["state_dict"])
            optimizer.load_state_dict(latest["optimizer_state_dict"])
            generator.set_state(latest["batch_rng_state"])
        else:
            # The initial short pilot retained selected heads, not inner Adam.
            # Reconstruct its cheap optimizer prefix only, without repeating
            # graph work or intermediate evaluations; verify stored losses.
            observed_losses = {int(row["update"]): row.get("loss") for row in history}
            for replay in range(1, last_step + 1):
                loss = one_update(model, optimizer, generator, features, targets)
                if replay in observed_losses:
                    assert abs(loss - observed_losses[replay]) < 1e-6
            replayed = last_step
        _, restored_score = score()
        assert abs(restored_score - history[-1]["selection_ap"]) < 1e-7
        best = max(history, key=lambda row: row["selection_ap"])
        best_score, best_step = best["selection_ap"], int(best["update"])
        for row in history:
            if row["selection_ap"] > patience_score + SETTINGS["min_delta"]:
                patience_score, patience_step = row["selection_ap"], int(row["update"])
        first_step = last_step + 1
    for step in range(first_step, SETTINGS["max_updates"] + 1):
        loss = one_update(model, optimizer, generator, features, targets)
        if step % SETTINGS["eval_every"]:
            continue
        aps, current = score()
        history.append({"update": step, "loss": loss, "selection_ap": current, **aps})
        pd.DataFrame(history).to_csv(destination / "inner_history.csv", index=False)
        print(json.dumps({"variant": args.variant, "fold": outer, "update": step,
                          "inner_selection_ap": current}), flush=True)
        torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(), "batch_rng_state": generator.get_state(),
                    "update": step}, destination / "inner_latest.pt")
        if current > best_score:
            best_score, best_step = current, step
        if current > patience_score + SETTINGS["min_delta"]:
            patience_score, patience_step = current, step
        if not getattr(args, "full_budget", False) and step - patience_step >= SETTINGS["patience_updates"]:
            break
    pd.DataFrame(history).to_csv(destination / "inner_history.csv", index=False)
    del model, optimizer, features, targets, monitor_values
    # Only the inner split selected the update count. Refit from initialization
    # on both non-outer folds; the outer fold still cannot affect preprocessing.
    model, optimizer, generator, features, targets = make_fit(args.variant, dataset, train | monitor, device)
    for _ in range(best_step):
        one_update(model, optimizer, generator, features, targets)
    model.eval()
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"variant": args.variant, "outer_fold": outer, "selected_updates": best_step,
                "state_dict": state, "settings": SETTINGS}, destination / "head.pt")
    rows, predictions = [], []
    cells = dataset["cells"][evaluation].numpy()
    for bank in range(3):
        scores = predict(model, dataset["features"][bank, evaluation], device)
        predictions.append(scores)
        for cell in np.unique(cells):
            for k in K_VALUES:
                values = scores[cells == cell, :k + 1]
                labels = np.broadcast_to(np.arange(k + 1) == 0, values.shape).ravel()
                rows.append({"variant": args.variant, "outer_fold": outer, "cell_id": int(cell), "bank": BANKS[bank], "k": k,
                             "auprc": average_precision_score(labels, values.ravel()),
                             "macro_f1": f1_score(labels, values.ravel() >= 0, average="macro", zero_division=0)})
    pd.DataFrame(rows).to_csv(destination / "metrics.csv", index=False)
    np.savez_compressed(destination / "predictions.npz", scores=np.stack(predictions), cells=cells,
                        keys=dataset["keys"][:, evaluation].numpy())
    completed = {"variant": args.variant, "outer_fold": outer, "inner_fit_positive_occurrences": int(train.sum()),
                 "inner_monitor_positive_occurrences": int(monitor.sum()), "outer_positive_occurrences": int(evaluation.sum()),
                 "selected_updates": best_step, "stopped_update": step, "initial_inner_ap": initial, "selected_inner_ap": best_score,
                 "parameter_count": sum(p.numel() for p in model.parameters()), "elapsed_seconds": perf_counter() - started,
                 "inner_prefix_updates_reconstructed": replayed,
                 "full_budget_check": getattr(args, "full_budget", False),
                 "outer_used_for_selection": False}
    (destination / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    print(json.dumps(completed), flush=True)


def fit(args):
    if args.device != "cuda":
        raise ValueError("Run bulk correction-head fitting on a compute GPU")
    torch.set_num_threads(8)
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    if protocol["settings"] != SETTINGS:
        raise ValueError("Prepared protocol settings differ from the code")
    if protocol["source_file_sha256"] != file_hash(__file__):
        raise ValueError("Source changed after data preparation")
    if protocol.get("full_budget", False) != args.full_budget:
        raise ValueError("Full-budget setting differs from prepared protocol")
    dataset = torch.load(args.output_dir / "dataset.pt", weights_only=True, mmap=True, map_location="cpu")
    for outer in range(3):
        fit_fold(args, dataset, outer)


def summarize(args):
    frames, runs = [], []
    for variant in VARIANTS:
        for fold in range(3):
            path = args.output_dir / variant / f"fold_{fold}"
            runs.append(json.loads((path / "completed.json").read_text()))
            frames.append(pd.read_csv(path / "metrics.csv"))
    metrics = pd.concat(frames, ignore_index=True)
    assert len(metrics) == 4 * 3 * 16 * 3 * 5
    context = metrics.groupby(["variant", "cell_id", "bank", "k"], as_index=False)[["auprc", "macro_f1"]].mean()
    summary = context.groupby(["variant", "bank", "k"], as_index=False)[["auprc", "macro_f1"]].mean()
    metrics.to_csv(args.output_dir / "metrics_by_fold.csv", index=False)
    context.to_csv(args.output_dir / "metrics_by_context.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(runs).to_csv(args.output_dir / "runs.csv", index=False)
    print(summary.query("k == 500").pivot(index="variant", columns="bank", values="auprc").mul(100).round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "summarize"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parent-dir", type=Path)
    parser.add_argument("--full-budget", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "fit": fit, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
