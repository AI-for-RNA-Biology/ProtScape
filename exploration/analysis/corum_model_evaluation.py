"""Load and evaluate the released CORUM downstream checkpoints."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader

from downstream_tasks.config import PATHS, get_hc_embedding_paths, load_config
from downstream_tasks.data.datasets import ABMILDataset, collate_abmil
from downstream_tasks.data.loaders import EmbeddingLoader
from downstream_tasks.data.preprocessing import zscore_normalize_bags
from downstream_tasks.data.task_loaders import get_task_loader
from downstream_tasks.models.abmil import ABMIL_ContextOnly, ABMIL_LateFusion
from downstream_tasks.models.linear import LinearProbe
from downstream_tasks.models.registry import MODEL_VARIANTS, ModelType
from downstream_tasks.run_selected import (
    find_selected_run,
    load_selected_runs,
)
from downstream_tasks.training.metrics import compute_all_metrics, summarize_cv_metrics


DATA_ROOT = Path(PATHS["data_root"])
INFERENCE_ROOT = Path(PATHS["inference_root"])
CORUM_MEMBERSHIPS = (
    Path(PATHS["corum_dataset_dir"]) / "corum_memberships_filtered.csv"
)


MAIN_CONTEXT_MODEL_ORDER = [
    "pinnacle_random",
    "pinnacle_esm",
    "pinnacle_acm",
    "gae_bce",
    "s2gae_bce_uni",
]
MAIN_MODEL_ORDER = ["lr_esm", "lr_prostt5"] + MAIN_CONTEXT_MODEL_ORDER

CONTEXT_READOUTS = [
    "lr_hc_cell",
    "lr_hc_cell_esm",
    "abmil8_hc_cell",
    "abmil8",
    "abmil8_pdl_hc_cell",
    "abmil8_pdl_id2_dropout",
]
LOSS_READOUTS = {
    "s2gae_bce_uni": CONTEXT_READOUTS,
    "s2gae_phuber_uni": [
        "lr_hc_cell",
        "lr_hc_cell_esm",
        "abmil8",
        "abmil8_pdl_hc_cell",
        "abmil8_pdl_id2_dropout",
    ],
    "s2gae_l1_uni": [
        "lr_hc_cell",
        "lr_hc_cell_esm",
        "abmil8",
        "abmil8_pdl_hc_cell",
        "abmil8_pdl_id2_dropout",
    ],
}
READOUT_VARIANTS = {
    "lr_esm": "lr_ext_embed",
    "lr_prostt5": "lr_ext_embed",
    "lr_hc_cell": "lr_hc_cell",
    "lr_hc_cell_esm": "lr_hc_cell_ext_embed",
    "abmil8_hc_cell": "abmil_hc_cell_gated_8",
    "abmil8": "abmil_hc_cell_ext_embed_gated_8",
    "abmil8_pdl_hc_cell": "abmil_hc_cell_gated_8_pdl",
    "abmil8_pdl_id2_dropout": "abmil_hc_cell_ext_embed_gated_8_pdl",
}

READOUT_LABELS = {
    "lr_esm": "Linear probe\nESM2 sequence",
    "lr_prostt5": "Linear probe\nProstT5 sequence",
    "lr_hc_cell": "Mean-pool LR\nContextual instances",
    "lr_hc_cell_esm": "Mean-pool LR\nContextual + ESM2",
    "abmil8_hc_cell": "ABMIL\nContextual instances",
    "abmil8": "ABMIL\nContextual + ESM2",
    "abmil8_pdl_hc_cell": "ABMIL-PDL\nContextual instances",
    "abmil8_pdl_id2_dropout": "ABMIL-PDL\nContextual + ESM2",
}
MODEL_LABELS = {
    "lr_esm": "ESM2",
    "lr_prostt5": "ProstT5",
    "pinnacle_random": "Pinnacle",
    "pinnacle_esm": "Pinnacle-ESM2 (GAT)",
    "pinnacle_acm": "Pinnacle-ESM2 (ACM)",
    "gae_bce": "ProtScape-GAE",
    "s2gae_bce_uni": "ProtScape",
    "s2gae_phuber_uni": "ProtScape (pHuber)",
    "s2gae_l1_uni": "ProtScape (L1)",
}


def run_spec(scope: str, model_key: str, readout: str) -> dict[str, str]:
    row, _ = find_selected_run(
        "corum",
        scope=scope,
        inference_key=model_key,
        readout_key=readout,
    )
    return {
        "scope": scope,
        "inference_key": model_key,
        "inference_name": row["embedding_inference_name"],
        "inference_label": row["inference_label"],
        "readout_key": readout,
        "embedding_source": row["embedding_source"],
        "cell_embedding_file": row["cell_embedding_file"],
        "cell_representation": row["cell_representation"],
    }


def selected_run(scope: str, model_key: str, readout: str) -> Path:
    _, run_dir = find_selected_run(
        "corum",
        scope=scope,
        inference_key=model_key,
        readout_key=readout,
    )
    return run_dir


def checkpoint_cell_representations() -> pd.DataFrame:
    """Record the cell representation selected by the executable run specs."""
    columns = [
        "scope",
        "inference_key",
        "inference_label",
        "readout_key",
        "cell_embedding_file",
        "cell_representation",
    ]
    return pd.DataFrame(load_selected_runs("corum"))[columns]


def load_data(
    inference_name: str,
    split_file: Path,
    embedding_source: str = "esm",
    cell_embedding_file: str = "cell_embeddings.pt",
) -> dict[str, object]:
    config = load_config(inference_name, embedding_source=embedding_source)
    config.data_root = DATA_ROOT
    config.inference_root = INFERENCE_ROOT
    config.seed = 42
    config.n_folds = 6

    paths = get_hc_embedding_paths(config.get_inference_path())
    paths["cell_embed"] = config.get_inference_path() / cell_embedding_file
    if paths["protein_embed"] is None or paths["cell_embed"] is None:
        raise FileNotFoundError(
            f"Missing protein or cell embeddings in {config.get_inference_path()}"
        )
    loader = EmbeddingLoader(
        config.embeddings.esm,
        paths["protein_embed"],
        paths["cell_embed"],
        paths["protein_labels"],
        paths["cell_labels"],
    )
    task_loader = get_task_loader("corum", CORUM_MEMBERSHIPS)
    genes, labels, class_names = task_loader.load()
    with np.load(split_file) as saved_split:
        shared_genes = saved_split["genes"].astype(str).tolist()
    gene_index = {gene: index for index, gene in enumerate(genes)}
    shared_labels = labels[[gene_index[gene] for gene in shared_genes]]

    gene_to_bags, gene_to_cells = loader.load_hc_with_cell(set(shared_genes))
    gene_to_means, _ = loader.load_hc_with_cell_mean(set(shared_genes))
    ctx_bags = [
        np.stack(gene_to_bags[gene], axis=0).astype(np.float32)
        for gene in shared_genes
    ]
    esm_dict = loader.load_esm()
    esm = np.stack([esm_dict[gene] for gene in shared_genes]).astype(np.float32)
    context_means = np.stack(
        [gene_to_means[gene] for gene in shared_genes]
    ).astype(np.float32)

    mapping_path = config.get_inference_path() / "mappings.pkl"
    with mapping_path.open("rb") as handle:
        id_to_name = pickle.load(handle)["id_to_name"]

    def context_name(raw_id: object) -> str:
        if raw_id in id_to_name:
            return str(id_to_name[raw_id])
        try:
            return str(id_to_name[int(raw_id)])
        except (KeyError, TypeError, ValueError) as error:
            raise KeyError(
                f"Context ID {raw_id!r} is missing from {mapping_path}"
            ) from error

    data = {
        "genes": shared_genes,
        "labels": shared_labels,
        "class_names": class_names,
        "cell_ids": [
            [context_name(cell_id) for cell_id in gene_to_cells[gene]]
            for gene in shared_genes
        ],
        "ctx_bags": ctx_bags,
        "esm": esm,
        "context_means": context_means,
    }
    return data


def load_run_data(
    spec: dict[str, str],
    split_file: Path,
) -> dict[str, object]:
    """Load the representation declared for one checkpoint group."""
    cell_embedding_file = spec["cell_embedding_file"] or "cell_embeddings.pt"
    return load_data(
        spec["inference_name"],
        split_file,
        spec["embedding_source"],
        cell_embedding_file,
    )


def load_folds(split_file: Path, genes: list[str]) -> list[np.ndarray]:
    with np.load(split_file) as saved:
        if saved["genes"].astype(str).tolist() != genes:
            raise ValueError(f"Gene order does not match {split_file}")
        return [saved[f"fold_{index}"].astype(int) for index in range(6)]


def train_indices(folds: list[np.ndarray], fold: int) -> np.ndarray:
    return np.concatenate(
        [indices for index, indices in enumerate(folds[1:]) if index != fold]
    )


def features_for_readout(data: dict[str, object], readout: str) -> dict[str, object]:
    means = data["context_means"]
    esm = data["esm"]
    if readout in {"lr_esm", "lr_prostt5"}:
        return {"X": esm}
    if readout == "lr_hc_cell":
        return {"X": means}
    if readout == "lr_hc_cell_esm":
        return {"X": np.concatenate([esm, means], axis=1)}
    features = {"ctx_bags": data["ctx_bags"]}
    if "ext_embed" in MODEL_VARIANTS[READOUT_VARIANTS[readout]].embedding_sources:
        features["esm"] = esm
    return features


def build_abmil(
    readout: str,
    data: dict[str, object],
    device: torch.device,
    use_pdl=None,
) -> nn.Module:
    variant = MODEL_VARIANTS[READOUT_VARIANTS[readout]]
    if use_pdl is None:
        use_pdl = variant.use_pdl
    common = {
        "num_classes": data["labels"].shape[1],
        "ctx_dim": data["ctx_bags"][0].shape[1],
        "num_heads": variant.num_heads,
        "att_hidden": 256,
        "dropout": 0.0,
        "att_dropout": 0.0,
        "attention_type": variant.attention_type or "gated",
        "classifier_type": variant.classifier_type,
        "mlp_hidden_dim": variant.mlp_hidden_dim,
        "use_pdl": use_pdl,
        "pdl_proj_mode": "identity",
        "pdl_proj_layers": 2,
    }
    if "ext_embed" in variant.embedding_sources:
        model = ABMIL_LateFusion(esm_dim=data["esm"].shape[1], **common)
    else:
        model = ABMIL_ContextOnly(**common)
    return model.to(device)


def evaluate(
    data: dict[str, object],
    checkpoint_dir: Path,
    split_file: Path,
    readout: str,
) -> dict[str, object]:
    labels = data["labels"]
    folds = load_folds(split_file, data["genes"])
    test_idx = folds[0]
    y_true = labels[test_idx]
    features = features_for_readout(data, readout)
    variant = MODEL_VARIANTS[READOUT_VARIANTS[readout]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_metrics = []
    fold_probs = []

    for fold in range(5):
        train_idx = train_indices(folds, fold)
        state = torch.load(
            checkpoint_dir / f"fold_{fold}.pt",
            map_location=device,
            weights_only=True,
        )
        if variant.model_type == ModelType.LR:
            x = features["X"]
            mean = x[train_idx].mean(axis=0)
            std = np.maximum(x[train_idx].std(axis=0), 1e-8)
            x_test = torch.from_numpy(
                ((x[test_idx] - mean) / std).astype(np.float32)
            ).to(device)
            model = LinearProbe(x.shape[1], labels.shape[1]).to(device)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                probabilities = torch.sigmoid(model(x_test)).cpu().numpy()
        else:
            bags = features["ctx_bags"]
            _, bag_mean, bag_std = zscore_normalize_bags(
                [bags[index] for index in train_idx]
            )
            esm = features.get("esm")
            if esm is not None:
                esm_mean = esm[train_idx].mean(axis=0)
                esm_std = np.maximum(esm[train_idx].std(axis=0), 1e-8)
            model = build_abmil(
                readout,
                data,
                device,
                use_pdl=any(key.startswith("ctx_proj.") for key in state),
            )
            model.load_state_dict(state)
            model.eval()
            test_bags = [
                ((bags[index] - bag_mean) / bag_std).astype(np.float32)
                for index in test_idx
            ]
            test_esm = None
            if esm is not None:
                test_esm = ((esm[test_idx] - esm_mean) / esm_std).astype(np.float32)
            test_dataset = ABMILDataset(test_bags, test_esm, y_true)
            test_loader = DataLoader(
                test_dataset,
                batch_size=512,
                shuffle=False,
                collate_fn=collate_abmil,
            )
            predictions = []
            with torch.no_grad():
                for model_input, _ in test_loader:
                    model_input = {
                        key: value.to(device) if value is not None else None
                        for key, value in model_input.items()
                    }
                    with torch.amp.autocast(
                        device_type=device.type, enabled=device.type == "cuda"
                    ):
                        logits = model(model_input)
                    predictions.append(torch.sigmoid(logits).cpu().numpy())
            probabilities = np.concatenate(predictions).astype(np.float32)

        metrics = compute_all_metrics(y_true, probabilities)
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        fold_probs.append(probabilities)
        del model, state

    return {
        "summary": summarize_cv_metrics(fold_metrics, split="test"),
        "y_true": y_true,
        "fold_probs": np.stack(fold_probs),
        "test_idx": np.asarray(test_idx),
    }


def performance_rows(
    model_key: str,
    readout_key: str,
    result: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    for metric in ("auprc", "f1"):
        mean = result["summary"][f"test_{metric}_macro_mean"]
        std = result["summary"][f"test_{metric}_macro_std"]
        rows.append(
            {
                "metric": metric,
                "readout_key": readout_key,
                "readout_label": READOUT_LABELS[readout_key],
                "inference_key": model_key,
                "inference_label": MODEL_LABELS[model_key],
                "mean": mean,
                "std": std,
                "score_percent": 100.0 * mean,
                "std_percent": 100.0 * std,
                "n_test_samples": len(result["y_true"]),
                "n_folds": result["fold_probs"].shape[0],
            }
        )
    return rows


def per_complex_rows(
    model_key: str,
    result: dict[str, object],
    class_names: list[str],
) -> pd.DataFrame:
    if result["y_true"].shape[1] != len(class_names):
        raise ValueError("Checkpoint class dimension does not match CORUM labels")
    rows = []
    for class_idx, complex_id in enumerate(class_names):
        labels = result["y_true"][:, class_idx]
        fold_auprcs = []
        fold_f1s = []
        for fold_probs in result["fold_probs"]:
            probabilities = fold_probs[:, class_idx]
            fold_auprcs.append(float(average_precision_score(labels, probabilities)))
            fold_f1s.append(
                float(f1_score(labels, probabilities >= 0.5, zero_division=0))
            )
        rows.append(
            {
                "complex_id": complex_id,
                "inference_key": model_key,
                "inference_label": MODEL_LABELS[model_key],
                "n_test_pos": int(labels.sum()),
                "n_test_neg": int(len(labels) - labels.sum()),
                "mean_test_auprc": float(np.mean(fold_auprcs)),
                "mean_test_f1": float(np.mean(fold_f1s)),
            }
        )
    return pd.DataFrame(rows)


def append_modeling_statistics(
    stats: pd.DataFrame,
    data: dict[str, object],
    split_file: Path,
) -> pd.DataFrame:
    test_idx = load_folds(split_file, data["genes"])[0]
    rows = []
    for scope, labels in [
        ("modeling_universe", data["labels"]),
        ("held_out_test", data["labels"][test_idx]),
    ]:
        positives = int(labels.sum())
        total = int(labels.size)
        rows.append(
            {
                "scope": scope,
                "n_proteins": int(labels.shape[0]),
                "n_complexes": int(labels.shape[1]),
                "n_positive_memberships": positives,
                "n_negative_memberships": total - positives,
                "positive_label_fraction": positives / total,
                "mean_complexes_per_protein": float(labels.sum(axis=1).mean()),
                "median_complexes_per_protein": float(
                    np.median(labels.sum(axis=1))
                ),
                "max_complexes_per_protein": int(labels.sum(axis=1).max()),
                "label_density_percent": 100.0 * positives / total,
            }
        )
    return pd.concat([stats, pd.DataFrame(rows)], ignore_index=True)
