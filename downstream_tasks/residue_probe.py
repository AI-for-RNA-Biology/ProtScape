"""Independent ESM-only pooler + linear classifier on the saved downstream folds.

ESM residues are frozen. Pooler and classifier are initialized afresh for each
fold and trained only on that fold's downstream labels. No graph features or
ProtScape checkpoint enter this experiment. These are not frozen linear probes.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from pretraining.models.residue_pooling import ResiduePooler
from .data.partitions import load_task_partition, partition_path
from .data.preprocessing import compute_class_weights
from .training.cv_utils import get_cv_train_val_indices, get_test_indices
from .training.metrics import auprc_macro_ignore_empty
from .utils.stratification import MultilabelStratifiedSampler, _compute_label_clusters


class ResidueClassifier(nn.Module):
    def __init__(self, cache, mode, width, training_means, n_classes):
        super().__init__()
        normalization = dict(mean=training_means.mean(0),
                             std=training_means.std(0).clip(min=1e-8))
        self.pooler = ResiduePooler(cache, mode, width, normalization=normalization,
                                   num_ref_points=width if mode == "swe" else 100)
        self.head = nn.Linear(self.pooler.input_dim, n_classes)

    def forward(self, protein_ids):
        return self.head(self.pooler(protein_ids))


def cache_indices(cache, genes):
    manifest = json.loads((Path(cache) / "manifest.json").read_text())
    # Match the existing downstream loader: last record per uppercased gene.
    lookup = {gene.upper(): index for index, gene in enumerate(manifest["genes"])}
    return np.array([lookup[gene.upper()] for gene in genes], dtype=np.int64)


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def predict(model, ids, batch_size):
    model.eval()
    return torch.cat([model(batch).sigmoid().cpu() for batch in ids.split(batch_size)]).numpy()


def fit_fold(model, ids, labels, train, validation, output, *, lr, epochs, patience,
             seed, batch_size=512):
    """Validation-only early stopping, with an atomic epoch-level restart file."""
    device = next(model.parameters()).device
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(
        compute_class_weights(labels[train]), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    clusters = _compute_label_clusters(labels[train], min(labels.shape[1], len(train)), seed)
    sampler = MultilabelStratifiedSampler(None, batch_size, seed=seed, cluster_labels=clusters)
    restart = output / "restart.pt"
    start, stale, best_score, best_epoch, best_state = 0, 0, -np.inf, 0, None
    history = []
    if restart.exists():
        saved = torch.load(restart, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, stale = saved["epoch"], saved["stale"]
        best_score, best_epoch, best_state = saved["score"], saved["best_epoch"], saved["best"]
        history = saved["history"]
        sampler._iter_count = start
    for epoch in range(start, epochs):
        if stale >= patience:
            break
        model.train()
        order = train[np.array(list(sampler))]
        total_loss = 0.
        for batch in np.array_split(order, np.arange(batch_size, len(order), batch_size)):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids[batch]), target[batch])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite independent-pooler training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        score = auprc_macro_ignore_empty(labels[validation], predict(model, ids[validation], batch_size))
        if not np.isfinite(score):
            raise ValueError("Independent-pooler validation AUPRC is not finite")
        stale += 1
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(dict(epoch=epoch + 1, train_loss=total_loss / len(train), val_auprc=score))
        atomic_checkpoint(restart, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                       epoch=epoch + 1, stale=stale, score=best_score,
                                       best_epoch=best_epoch, best=best_state, history=history))
        print(f"Epoch {epoch + 1}: val AUPRC {score:.5f}, best {best_score:.5f}", flush=True)
    if best_state is None:
        raise ValueError("No validation-selected pooler checkpoint")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    atomic_checkpoint(output / "best.pt", dict(model=best_state, epoch=best_epoch, val_auprc=best_score))
    return best_epoch


def run(root, mode, task_name, width, lr, *, smoke=False):
    root, width, lr = Path(root), int(width), float(lr)
    settings = yaml.safe_load((root / "source/configs/residue_ablation.yaml").read_text())["downstream"]
    inputs = json.loads((root / "inputs.json").read_text())
    task = next(task for task in inputs["tasks"] if task["task"] == task_name)
    cache = root / "residues"
    size = f"ref{width}" if mode == "swe" else f"h{width}"
    output = root / ("smoke_downstream" if smoke else "downstream") / task_name / f"esm_{mode}_linear" / f"{size}_lr{lr:g}"
    output.mkdir(parents=True, exist_ok=True)
    genes, labels, classes, plan = load_task_partition(task_name, Path(task["csv"]))
    shutil.copy2(partition_path(task_name, Path(task["csv"])), output / "split_indices.npz")
    indices = cache_indices(cache, genes)
    features = pd.read_pickle(cache / "mean.plk")
    means = np.stack(features["ESM2-Embeddings"].to_numpy()[indices]).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    ids = torch.as_tensor(indices, device=device)
    provenance = dict(baseline="ESM_SWE_simple_linear" if mode == "swe" else f"ESM_{mode}_linear", lr=lr,
                      seed=settings["seed"], esm_frozen=True, protscape_checkpoint_used=False,
                      pooler_initialization="fresh_each_fold", supervision="downstream_train_fold",
                      normalization="training_fold_mean_vectors", split=str(partition_path(task_name, Path(task["csv"]))),
                      residue_manifest_sha256=hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest())
    provenance["num_ref_points" if mode == "swe" else "hidden_dim"] = width
    if mode == "swe":
        provenance.update(swe_variant="simple", slicers_frozen=True, reference_frozen=True)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if (output / "results.csv").exists():
        return
    test = get_test_indices(plan)
    rows = []
    for fold in range(1 if smoke else plan.n_cv_folds):
        folder = output / f"fold_{fold}"
        folder.mkdir(exist_ok=True)
        if (folder / "metrics.json").exists():
            rows.append(json.loads((folder / "metrics.json").read_text()))
            continue
        train, validation, _ = get_cv_train_val_indices(plan, fold)
        torch.manual_seed(settings["seed"])
        model = ResidueClassifier(cache, mode, width, means[train], len(classes)).to(device)
        best_epoch = fit_fold(model, ids, labels, train, validation, folder, lr=lr,
                              epochs=2 if smoke else settings["epochs"],
                              patience=1 if smoke else settings["patience"], seed=settings["seed"])
        val_prob, test_prob = predict(model, ids[validation], 512), predict(model, ids[test], 512)
        np.savez_compressed(folder / "predictions.npz", val_indices=validation, val_labels=labels[validation],
                            val_prob=val_prob, test_indices=test, test_labels=labels[test], test_prob=test_prob)
        metrics = dict(fold=fold, epoch=best_epoch,
                       val_auprc=auprc_macro_ignore_empty(labels[validation], val_prob),
                       test_auprc=auprc_macro_ignore_empty(labels[test], test_prob))
        temporary = folder / "metrics.tmp"
        temporary.write_text(json.dumps(metrics, indent=2) + "\n")
        temporary.replace(folder / "metrics.json")
        rows.append(metrics)
        del model
    result = dict(**provenance, task=task_name, n_label_classes=len(classes), n_cv_folds=len(rows),
                  selected_epochs=json.dumps([row["epoch"] for row in rows]))
    for split in ["val", "test"]:
        scores = [row[f"{split}_auprc"] for row in rows]
        result.update({f"{split}_auprc_macro_mean": np.mean(scores), f"{split}_auprc_macro_std": np.std(scores)})
    pd.DataFrame([result]).to_csv(output / "results.tmp", index=False)
    (output / "results.tmp").replace(output / "results.csv")


if __name__ == "__main__":
    # The queue passes fixed positional arguments; no extra user-facing CLI.
    run(*sys.argv[1:])
