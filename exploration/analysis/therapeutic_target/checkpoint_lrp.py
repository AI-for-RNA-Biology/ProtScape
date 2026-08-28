"""Held-out inference and xMIL-LRP from released downstream checkpoints."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from downstream_tasks.config import get_hc_embedding_paths, load_config
from downstream_tasks.data.datasets import ABMILDataset
from downstream_tasks.data.loaders import EmbeddingLoader
from downstream_tasks.data.preprocessing import zscore_normalize_bags
from downstream_tasks.data.task_loaders import get_task_loader
from downstream_tasks.models.abmil import ABMIL_LateFusion
from downstream_tasks.models.linear import LinearProbe
from downstream_tasks.models.registry import MODEL_VARIANTS, ModelType
from downstream_tasks.run import _build_shared_split
from downstream_tasks.run_selected import load_selected_runs, selected_run_dir
from downstream_tasks.training.cv_utils import get_cv_train_val_indices, get_test_indices
from downstream_tasks.training.metrics import compute_all_metrics
from downstream_tasks.training.trainer import Trainer

from .xmil_lrp import explain_late_fusion_bag


LRP_FIELDS = [
    "task",
    "inference_key",
    "fold",
    "gene",
    "class_name",
    "label",
    "prob",
    "cell_label",
    "context_evidence",
    "context_positive_evidence",
    "context_abs_evidence",
    "esm_evidence",
    "esm_positive_evidence",
    "esm_abs_evidence",
]


def read_config(run_dir: Path) -> pd.Series:
    return pd.read_csv(run_dir / "model_config.csv").iloc[0]


def apply_hyperparameters(config, row: pd.Series) -> None:
    for column, cast in (
        ("num_heads", int),
        ("att_hidden_dim", int),
        ("dropout", float),
        ("att_dropout", float),
        ("pdl_proj_layers", int),
    ):
        if column in row and not pd.isna(row[column]):
            setattr(config, column, cast(row[column]))
    if "pdl_proj_mode" in row and not pd.isna(row["pdl_proj_mode"]):
        config.pdl_proj_mode = str(row["pdl_proj_mode"])


def load_embeddings(
    config,
    cell_embedding_file: str = "cell_embeddings.pt",
) -> EmbeddingLoader:
    paths = get_hc_embedding_paths(
        config.get_inference_path(),
        cell_embedding_file=cell_embedding_file,
    )
    return EmbeddingLoader(
        config.embeddings.esm,
        paths["protein_embed"],
        paths["cell_embed"],
    )


def load_task(task: str, config, loader: EmbeddingLoader):
    genes, labels, class_names = get_task_loader(
        task, config.tasks[task].label_csv
    ).load()
    genes, labels, split_plan = _build_shared_split(
        genes,
        labels,
        loader,
        seed=config.seed,
        n_splits=config.n_folds,
    )
    return genes, labels, class_names, split_plan


def check_alignment(run_dir: Path, features, genes, split_plan) -> None:
    if list(features["genes"]) != list(genes):
        raise RuntimeError(f"Feature genes changed for {run_dir}")
    expected = np.load(run_dir / "test_idx.npy")
    if not np.array_equal(get_test_indices(split_plan), expected):
        raise RuntimeError(f"Held-out indices do not match {run_dir}")


def abmil_parameters(trainer: Trainer, variant, features, labels) -> dict:
    esm = features.get("esm")
    return {
        "num_classes": labels.shape[1],
        "ctx_dim": features["ctx_bags"][0].shape[1],
        "esm_dim": 0 if esm is None else esm.shape[1],
        "use_ext_embed": "ext_embed" in variant.embedding_sources,
        "num_heads": int(trainer.config.num_heads),
        "att_hidden": trainer.config.att_hidden_dim,
        "dropout": trainer.config.dropout,
        "att_dropout": trainer.config.att_dropout,
        "attention_type": variant.attention_type or "gated",
        "classifier_type": variant.classifier_type,
        "mlp_hidden_dim": trainer._get_abmil_mlp_hidden_dim(variant),
        "use_pdl": variant.use_pdl,
        "pdl_proj_mode": trainer._resolve_pdl_proj_mode(variant),
        "pdl_proj_layers": trainer._resolve_pdl_proj_layers(variant),
    }


def evaluate_folds(run_dir: Path, row: pd.Series, trainer: Trainer, features, labels, split_plan):
    variant = MODEL_VARIANTS[str(row["base_model_key"])]
    test_indices = get_test_indices(split_plan)
    test_labels = labels[test_indices]
    results = []

    if variant.model_type == ModelType.LR:
        values = features["X"]
        for fold in range(split_plan.n_cv_folds):
            train_indices, _, _ = get_cv_train_val_indices(split_plan, fold)
            mean = values[train_indices].mean(axis=0)
            std = np.maximum(values[train_indices].std(axis=0), 1e-8)
            test_values = ((values[test_indices] - mean) / std).astype(np.float32)
            model = LinearProbe(values.shape[1], labels.shape[1]).to(trainer.device)
            model.load_state_dict(
                torch.load(
                    run_dir / "models" / f"fold_{fold}.pt",
                    map_location=trainer.device,
                    weights_only=True,
                )
            )
            model.eval()
            with torch.no_grad():
                probabilities = torch.sigmoid(
                    model(torch.from_numpy(test_values).to(trainer.device))
                ).cpu().numpy()
            results.append(compute_all_metrics(test_labels, probabilities))
        return results

    bags = features["ctx_bags"]
    sequence = features.get("esm")
    parameters = abmil_parameters(trainer, variant, features, labels)
    for fold in range(split_plan.n_cv_folds):
        train_indices, _, _ = get_cv_train_val_indices(split_plan, fold)
        _, bag_mean, bag_std = zscore_normalize_bags(
            [bags[index] for index in train_indices]
        )
        test_bags = [(bags[index] - bag_mean) / bag_std for index in test_indices]
        if sequence is None:
            test_sequence = None
        else:
            mean = sequence[train_indices].mean(axis=0)
            std = np.maximum(sequence[train_indices].std(axis=0), 1e-8)
            test_sequence = ((sequence[test_indices] - mean) / std).astype(np.float32)
        dataset = ABMILDataset(test_bags, test_sequence, test_labels)
        batches = DataLoader(
            dataset,
            shuffle=False,
            **trainer._abmil_loader_kwargs(),
        )
        model = trainer._build_abmil_model(parameters)
        model.load_state_dict(
            torch.load(
                run_dir / "models" / f"fold_{fold}.pt",
                map_location=trainer.device,
                weights_only=True,
            )
        )
        model.eval()
        probabilities = []
        with torch.no_grad():
            for batch, _ in batches:
                batch = trainer._to_device(batch)
                with torch.amp.autocast(
                    device_type=trainer.device.type,
                    enabled=trainer.use_amp,
                ):
                    probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
        results.append(
            compute_all_metrics(test_labels, np.concatenate(probabilities, axis=0))
        )
    return results


def recompute_performance() -> pd.DataFrame:
    configs = []
    for selected in load_selected_runs("therapeutic_targets"):
        run_dir = selected_run_dir(selected)
        row = dict(selected)
        row.update(read_config(run_dir).to_dict())
        row["run_dir"] = str(run_dir)
        configs.append(row)
    configs = pd.DataFrame(configs)
    configs["cell_embedding_file"] = (
        configs["cell_embedding_file"]
        .replace("", pd.NA)
        .fillna("cell_embeddings.pt")
    )
    output = []
    group_columns = [
        "embedding_inference_name",
        "embedding_source",
        "cell_embedding_file",
    ]
    for (
        inference_name,
        embedding_source,
        cell_embedding_file,
    ), group in configs.groupby(group_columns, sort=False):
        config = load_config(str(inference_name), str(embedding_source), "bulk")
        loader = load_embeddings(config, str(cell_embedding_file))
        for task, task_runs in group.groupby("task", sort=False):
            genes, labels, _, split_plan = load_task(task, config, loader)
            for _, row in task_runs.iterrows():
                apply_hyperparameters(config, row)
                run_dir = Path(row["run_dir"])
                trainer = Trainer(config, loader, run_dir, force=False)
                variant = MODEL_VARIANTS[str(row["base_model_key"])]
                features = trainer._build_features(variant, genes.copy())
                check_alignment(run_dir, features, genes, split_plan)
                metrics = evaluate_folds(
                    run_dir, row, trainer, features, labels, split_plan
                )
                for fold, values in enumerate(metrics):
                    output.append(
                        {
                            "task": row["task"],
                            "disease": row["disease"],
                            "readout_key": row["readout_key"],
                            "readout_label": row["readout_label"],
                            "inference_key": row["inference_key"],
                            "inference_label": row["inference_label"],
                            "inference_name": row["inference_name"],
                            "fold": fold,
                            "auprc": values["auprc_macro"],
                            "f1": values["f1_macro"],
                            "auroc": values["auroc_macro"],
                        }
                    )
        del loader
    return pd.DataFrame(output)


def generate_lrp(
    run_dir: Path,
    output_path: Path,
) -> Path:
    row = read_config(run_dir)
    inference_name = str(row["embedding_inference_name"])
    config = load_config(inference_name, str(row["embedding_source"]), "bulk")
    apply_hyperparameters(config, row)
    cell_embedding_file = row.get("cell_embedding_file", "cell_embeddings.pt")
    if pd.isna(cell_embedding_file) or not str(cell_embedding_file).strip():
        cell_embedding_file = "cell_embeddings.pt"
    loader = load_embeddings(config, str(cell_embedding_file))
    genes, labels, class_names, split_plan = load_task(str(row["task"]), config, loader)
    trainer = Trainer(config, loader, run_dir, force=False)
    variant = MODEL_VARIANTS[str(row["base_model_key"])]
    features = trainer._build_features(variant, genes.copy())
    check_alignment(run_dir, features, genes, split_plan)
    protein_embeddings = loader._load_hc_protein_embed()
    id_to_name = {
        str(cell_id): str(cell_name)
        for cell_id, cell_name in zip(
            protein_embeddings.get("cell_ids", []),
            protein_embeddings.get("cell_names", []),
        )
    }
    del protein_embeddings
    del loader

    bags = features["ctx_bags"]
    sequence = features["esm"]
    cell_labels = [
        [id_to_name.get(str(cell_id), str(cell_id)) for cell_id in bag]
        for bag in features["cell_ids"]
    ]
    test_indices = list(map(int, get_test_indices(split_plan)))
    device = trainer.device

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LRP_FIELDS)
        writer.writeheader()
        for fold in range(split_plan.n_cv_folds):
            train_indices, _, _ = get_cv_train_val_indices(split_plan, fold)
            _, bag_mean, bag_std = zscore_normalize_bags(
                [bags[index] for index in train_indices]
            )
            sequence_mean = sequence[train_indices].mean(axis=0)
            sequence_std = np.maximum(sequence[train_indices].std(axis=0), 1e-8)
            model = ABMIL_LateFusion(
                num_classes=labels.shape[1],
                ctx_dim=bags[0].shape[1],
                esm_dim=sequence.shape[1],
                num_heads=int(config.num_heads),
                att_hidden=int(config.att_hidden_dim),
                dropout=float(config.dropout),
                att_dropout=float(config.att_dropout),
                attention_type=variant.attention_type or "gated",
                classifier_type=variant.classifier_type,
                use_pdl=variant.use_pdl,
                pdl_proj_mode=config.pdl_proj_mode,
                pdl_proj_layers=config.pdl_proj_layers,
            )
            model.load_state_dict(
                torch.load(
                    run_dir / "models" / f"fold_{fold}.pt",
                    map_location=device,
                    weights_only=True,
                )
            )
            model.to(device).eval()
            for gene_index in test_indices:
                bag = ((bags[gene_index] - bag_mean) / bag_std).astype(np.float32)
                esm = (
                    (sequence[gene_index] - sequence_mean) / sequence_std
                ).astype(np.float32)
                scores = explain_late_fusion_bag(
                    model,
                    torch.as_tensor(bag, device=device),
                    torch.as_tensor(esm, device=device),
                    np.arange(labels.shape[1]),
                )
                for class_index, score in scores.items():
                    for context_index, cell_label in enumerate(cell_labels[gene_index]):
                        writer.writerow(
                            {
                                "task": row["task"],
                                "inference_key": row["inference_key"],
                                "fold": fold,
                                "gene": genes[gene_index],
                                "class_name": str(class_names[class_index]),
                                "label": int(labels[gene_index, class_index]),
                                "prob": score["prob"],
                                "cell_label": str(cell_label),
                                "context_evidence": score["context_evidence"][context_index],
                                "context_positive_evidence": score["context_positive_evidence"][context_index],
                                "context_abs_evidence": score["context_abs_evidence"][context_index],
                                "esm_evidence": score["esm_evidence"],
                                "esm_positive_evidence": score["esm_positive_evidence"],
                                "esm_abs_evidence": score["esm_abs_evidence"],
                            }
                        )
    return output_path
