#!/usr/bin/env python3
"""Compute the CORUM analyses and plots."""

from __future__ import annotations

import gc
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
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
from downstream_tasks.training.metrics import compute_all_metrics, summarize_cv_metrics
from exploration.analysis.therapeutic_target.xmil_lrp import explain_late_fusion_bag


DATA_ROOT = Path(PATHS["data_root"])
INFERENCE_ROOT = Path(PATHS["inference_root"])
CORUM_DIR = Path(PATHS["corum_dataset_dir"])
CORUM_COMPLEXES = CORUM_DIR / "corum_complexes_filtered.csv"
CORUM_MEMBERSHIPS = CORUM_DIR / "corum_memberships_filtered.csv"
GLOBAL_PPI = Path(PATHS["networks_bulk"]) / "global_ppi_edgelist.txt"
CELL_PPI_DIR = GLOBAL_PPI.parent / "ppi_edgelists"
CHECKPOINT_ROOT = Path(PATHS["downstream_checkpoint_root"]) / "corum"
ANALYSIS_DIR = Path(PATHS["output_root"]) / "analysis/corum_analysis"


PINNACLE_RANDOM = (
    "PINNACLE_model_globalsplit_random_symmetric_PPI-True_"
    "GATv2_H64_lambda001_drop02_out32__ep150_bulk"
)
PINNACLE_ESM = (
    "PINNACLE_model_globalsplit_ESM2_symmetric_PPI-True_"
    "GATv2_H64_lambda001_drop02_out32__ep150_bulk"
)
PINNACLE_ACM = (
    "PINNACLE_model_globalsplit_ESM2_symmetric_PPI-True_"
    "ACM_RandomWalk_H64_lambda001_drop02_out64__ep150_bulk"
)
GAE_BCE = "gae_att_fixed_do06_ep300"
S2GAE_BCE = "s2gae_att_k1_fixed_do04_uni5e6"
S2GAE_PHUBER = (
    "HCCTassignmentglobalsplit_ESM2symmetricPPI_attention128_s2gae_dm_mr05_"
    "dc512_dl2_do00_MG_bulk__lr001_ep300_phubertau100_ACM_RandomWalkconcat_"
    "H512L3_dropout04_regCTassignment1_unil000005_unit20_unid0"
)
S2GAE_L1 = (
    "HCCTassignmentglobalsplit_ESM2symmetricPPI_attention128_s2gae_dm_mr05_"
    "dc512_dl2_do00_MG_bulk__lr001_ep300_l1_ACM_RandomWalkconcat_H512L3_"
    "regCTassignment1_unil000005_unit20_unid0"
)

INFERENCE_NAMES = {
    "lr_esm": PINNACLE_ESM,
    "lr_prostt5": S2GAE_BCE,
    "pinnacle_random": PINNACLE_RANDOM,
    "pinnacle_esm": PINNACLE_ESM,
    "pinnacle_acm": PINNACLE_ACM,
    "gae_bce": GAE_BCE,
    "s2gae_bce_uni": S2GAE_BCE,
    "s2gae_phuber_uni": S2GAE_PHUBER,
    "s2gae_l1_uni": S2GAE_L1,
}

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


DRIVER_SPECS = [
    ("complex_size", "n_members_in_ppi_universe", "Complex size", False),
    (
        "cell_ppi_member_coverage",
        "mean_cell_ppi_member_coverage",
        "Cell-PPI node coverage",
        True,
    ),
    (
        "cell_ppi_edge_coverage",
        "mean_cell_ppi_edge_coverage",
        "Cell-PPI edge coverage",
        True,
    ),
    (
        "global_ppi_edge_density",
        "induced_ppi_edge_density",
        "Global PPI edge density",
        True,
    ),
]


def build_run_specs() -> dict[tuple[str, str, str], dict[str, str]]:
    """Define the embedding export used by every released checkpoint group."""
    specs = {}

    def add(
        scope: str,
        model_key: str,
        readout: str,
        representation: str = "canonical_contextualized",
    ) -> None:
        filenames = {
            "canonical_contextualized": "cell_embeddings.pt",
            "pooled_before_cci": "cell_embeddings_before_pool.pt",
            "not_used_sequence_only": "",
        }
        key = (scope, model_key, readout)
        specs[key] = {
            "scope": scope,
            "inference_key": model_key,
            "inference_name": INFERENCE_NAMES[model_key],
            "inference_label": MODEL_LABELS[model_key],
            "readout_key": readout,
            "embedding_source": "prostt5" if model_key == "lr_prostt5" else "esm",
            "cell_embedding_file": filenames[representation],
            "cell_representation": representation,
        }

    add("aggregate", "lr_esm", "lr_esm", "not_used_sequence_only")
    add("aggregate", "lr_prostt5", "lr_prostt5", "not_used_sequence_only")
    for model_key in MAIN_CONTEXT_MODEL_ORDER:
        for readout in CONTEXT_READOUTS:
            representation = (
                "pooled_before_cci"
                if model_key == "s2gae_bce_uni"
                and readout == "abmil8_pdl_hc_cell"
                else "canonical_contextualized"
            )
            add("aggregate", model_key, readout, representation)
    for model_key in ("s2gae_phuber_uni", "s2gae_l1_uni"):
        for readout in LOSS_READOUTS[model_key]:
            add("aggregate", model_key, readout, "pooled_before_cci")
    for model_key in MAIN_MODEL_ORDER:
        readout = (
            model_key
            if model_key in {"lr_esm", "lr_prostt5"}
            else "abmil8_pdl_id2_dropout"
        )
        representation = (
            "not_used_sequence_only"
            if model_key in {"lr_esm", "lr_prostt5"}
            else "canonical_contextualized"
        )
        add("per_complex", model_key, readout, representation)
    return specs


RUN_SPECS = build_run_specs()


def run_spec(scope: str, model_key: str, readout: str) -> dict[str, str]:
    try:
        return RUN_SPECS[(scope, model_key, readout)]
    except KeyError as error:
        raise KeyError(
            f"No CORUM run specification for {scope}/{model_key}/{readout}"
        ) from error


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
    return pd.DataFrame(RUN_SPECS.values())[columns]


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


def dataset_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complexes = pd.read_csv(CORUM_COMPLEXES, dtype={"complex_id": str})
    memberships = pd.read_csv(CORUM_MEMBERSHIPS, dtype={"complex_id": str})
    valid_ids = set(complexes["complex_id"])
    memberships["protein"] = memberships["protein"].astype(str).str.upper()
    memberships = memberships[memberships["complex_id"].isin(valid_ids)].drop_duplicates(
        ["protein", "complex_id"]
    )
    per_protein = memberships.groupby("protein")["complex_id"].nunique()
    positives = int(len(memberships))
    n_proteins = int(per_protein.size)
    n_complexes = int(complexes["complex_id"].nunique())
    stats = pd.DataFrame(
        [
            {
                "scope": "filtered_corum_dataset",
                "n_proteins": n_proteins,
                "n_complexes": n_complexes,
                "n_positive_memberships": positives,
                "n_negative_memberships": n_proteins * n_complexes - positives,
                "positive_label_fraction": positives / (n_proteins * n_complexes),
                "mean_complexes_per_protein": float(per_protein.mean()),
                "median_complexes_per_protein": float(per_protein.median()),
                "max_complexes_per_protein": int(per_protein.max()),
                "label_density_percent": 100.0 * positives / (n_proteins * n_complexes),
            }
        ]
    )
    size_distribution = (
        complexes["n_members_in_ppi_universe"]
        .value_counts(sort=False)
        .sort_index()
        .rename_axis("n_members")
        .reset_index(name="n_complexes")
    )
    per_protein_distribution = (
        per_protein.value_counts(sort=False)
        .sort_index()
        .rename_axis("complexes_per_protein")
        .reset_index(name="n_proteins")
    )
    return stats, size_distribution, per_protein_distribution


def sorted_edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def read_ppi(path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0].upper(), parts[1].upper()
            nodes.update((a, b))
            if a != b:
                edges.add(sorted_edge(a, b))
    return nodes, edges


def complex_members() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    complexes = pd.read_csv(CORUM_COMPLEXES, dtype={"complex_id": str})
    members = {
        row.complex_id: {
            gene.strip().upper()
            for gene in str(row.member_hgnc).split(";")
            if gene.strip()
        }
        for row in complexes.itertuples(index=False)
    }
    return complexes, members


def build_topology_and_cell_coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    complexes, members_by_complex = complex_members()
    _, global_edges = read_ppi(GLOBAL_PPI)
    pairs = {
        complex_id: {
            sorted_edge(a, b)
            for a, b in combinations(sorted(members), 2)
        }
        for complex_id, members in members_by_complex.items()
    }
    reference_edges = {
        complex_id: member_pairs & global_edges
        for complex_id, member_pairs in pairs.items()
    }

    topology_rows = []
    for row in complexes.itertuples(index=False):
        complex_id = str(row.complex_id)
        n_members = len(members_by_complex[complex_id])
        n_possible = len(pairs[complex_id])
        n_edges = len(reference_edges[complex_id])
        topology_rows.append(
            {
                "complex_id": complex_id,
                "complex_name": row.complex_name,
                "n_members_in_ppi_universe": n_members,
                "n_possible_ppi_edges": n_possible,
                "n_induced_ppi_edges": n_edges,
                "induced_ppi_edge_density": n_edges / n_possible,
            }
        )

    member_coverage_sum = defaultdict(float)
    edge_coverage_sum = defaultdict(float)
    n_cell_ppis = 0
    for ppi_path in sorted(CELL_PPI_DIR.glob("*.txt")):
        nodes, edges = read_ppi(ppi_path)
        n_cell_ppis += 1
        for complex_id, members in members_by_complex.items():
            member_coverage_sum[complex_id] += len(members & nodes) / len(members)
            reference = reference_edges[complex_id]
            edge_coverage_sum[complex_id] += len(reference & edges) / len(reference)
        del nodes, edges

    coverage_rows = [
        {
            "complex_id": complex_id,
            "mean_cell_ppi_member_coverage": member_coverage_sum[complex_id]
            / n_cell_ppis,
            "mean_cell_ppi_edge_coverage": edge_coverage_sum[complex_id]
            / n_cell_ppis,
            "n_cell_ppis": n_cell_ppis,
        }
        for complex_id in members_by_complex
    ]
    return pd.DataFrame(topology_rows), pd.DataFrame(coverage_rows)


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


def equal_count_bins(
    rows: pd.DataFrame,
    value_column: str,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_column]).copy()
    rows = rows.sort_values([value_column, "complex_id"], kind="mergesort")
    labels = [f"bin_{index}" for index in range(n_bins)]
    bin_numbers = np.minimum(
        np.arange(len(rows), dtype=int) * n_bins // len(rows), n_bins - 1
    )
    rows["driver_bin"] = pd.Categorical(
        [labels[index] for index in bin_numbers], categories=labels, ordered=True
    )
    return rows, labels


def driver_tables(
    per_complex: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluable = set(
        per_complex.loc[
            per_complex["inference_key"].eq("s2gae_bce_uni")
            & per_complex["n_test_pos"].gt(0)
            & per_complex["mean_test_auprc"].notna(),
            "complex_id",
        ]
    )
    per_complex = per_complex[
        per_complex["inference_key"].isin(MAIN_MODEL_ORDER)
        & per_complex["complex_id"].isin(evaluable)
    ].copy()
    summaries = []
    bin_tables = []
    correlations = []
    for driver_key, column, _label, is_percent in DRIVER_SPECS:
        unique = per_complex[["complex_id", column]].drop_duplicates("complex_id")
        unique, labels = equal_count_bins(unique, column)
        bin_lookup = dict(zip(unique["complex_id"], unique["driver_bin"].astype(str)))
        rows = per_complex.copy()
        rows["driver_bin"] = pd.Categorical(
            rows["complex_id"].map(bin_lookup), categories=labels, ordered=True
        )
        observed = (
            unique.groupby("driver_bin", observed=True)
            .agg(
                lower=(column, "min"),
                upper=(column, "max"),
                x=(column, "mean"),
                n_complexes=("complex_id", "nunique"),
            )
            .reindex(labels)
        )
        if is_percent:
            display = [
                f"{int(round(100 * row.lower))}-{int(round(100 * row.upper))}%"
                for row in observed.itertuples()
            ]
        else:
            display = [
                f"{int(round(row.lower))}-{int(round(row.upper))}"
                for row in observed.itertuples()
            ]
        bins = pd.DataFrame(
            {
                "driver": driver_key,
                "driver_bin": labels,
                "bin_label": display,
                "lower": observed["lower"].to_numpy(dtype=float),
                "upper": observed["upper"].to_numpy(dtype=float),
                "x": observed["x"].to_numpy(dtype=float),
                "n_complexes": observed["n_complexes"].to_numpy(dtype=int),
                "percentile_midpoint": 100.0
                * (np.arange(len(labels), dtype=float) + 0.5)
                / len(labels),
            }
        )
        bin_tables.append(bins)

        available = rows.dropna(subset=["driver_bin", "mean_test_auprc"])
        summary = (
            available.groupby(["inference_key", "driver_bin"], observed=True)
            .agg(
                mean_score=("mean_test_auprc", "mean"),
                n_complexes=("complex_id", "nunique"),
            )
            .reset_index()
        )
        summary["driver"] = driver_key
        summaries.append(summary)

        for model_key, group in available.groupby("inference_key"):
            sub = group[[column, "mean_test_auprc"]].dropna()
            statistic = spearmanr(sub[column], sub["mean_test_auprc"])
            correlations.append(
                {
                    "driver": driver_key,
                    "inference_key": model_key,
                    "inference_label": MODEL_LABELS[model_key],
                    "n_complexes": len(sub),
                    "spearman": float(statistic.statistic),
                    "spearman_p": float(statistic.pvalue),
                }
            )
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(bin_tables, ignore_index=True),
        pd.DataFrame(correlations),
    )


def build_lrp_model(
    data: dict[str, object],
    checkpoint_dir: Path,
    fold: int,
) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_abmil("abmil8_pdl_id2_dropout", data, device)
    model.load_state_dict(
        torch.load(
            checkpoint_dir / f"fold_{fold}.pt",
            map_location=device,
            weights_only=True,
        )
    )
    return model.eval()


def xmil_lrp_values(
    model_key: str,
    data: dict[str, object],
    checkpoint_dir: Path,
    split_file: Path,
) -> pd.DataFrame:
    """Recompute normalized positive xMIL relevance from fold checkpoints."""
    ctx_bags = data["ctx_bags"]
    esm = data["esm"]
    cell_ids = data["cell_ids"]
    folds = load_folds(split_file, data["genes"])
    test_idx = folds[0]
    sums: dict[tuple[str, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for fold in range(5):
        train_idx = train_indices(folds, fold)
        _, bag_mean, bag_std = zscore_normalize_bags([ctx_bags[i] for i in train_idx])
        esm_mean = esm[train_idx].mean(axis=0)
        esm_std = np.maximum(esm[train_idx].std(axis=0), 1e-8)
        model = build_lrp_model(data, checkpoint_dir, fold)
        device = next(model.parameters()).device

        for gene_idx in test_idx:
            positive_classes = np.flatnonzero(data["labels"][gene_idx] > 0)
            if len(positive_classes) == 0:
                continue
            bag = ((ctx_bags[gene_idx] - bag_mean) / bag_std).astype(
                np.float32, copy=False
            )
            esm_vector = ((esm[gene_idx] - esm_mean) / esm_std).astype(
                np.float32, copy=False
            )
            scores = explain_late_fusion_bag(
                model,
                torch.as_tensor(bag, dtype=torch.float32, device=device),
                torch.as_tensor(esm_vector, dtype=torch.float32, device=device),
                positive_classes,
            )
            gene = data["genes"][gene_idx]
            for class_idx, score in scores.items():
                signed = score["context_evidence"]
                esm_relevance = score["esm_evidence"]
                positive = np.clip(signed, a_min=0.0, a_max=None)
                denominator = positive.sum() + max(esm_relevance, 0.0)
                normalized = (
                    positive / denominator
                    if denominator > 1e-12
                    else np.zeros_like(positive)
                )
                complex_id = data["class_names"][class_idx]
                for context_idx, cell_id in enumerate(cell_ids[gene_idx]):
                    key = (complex_id, gene, str(cell_id))
                    sums[key] += float(normalized[context_idx])
                    counts[key] += 1
        del model

    rows = [
        {
            "inference_key": model_key,
            "complex_id": complex_id,
            "gene": gene,
            "cell_type": cell_type,
            "complete_positive_lrp": sums[key] / counts[key],
            "n_folds": counts[key],
        }
        for key in sorted(sums)
        for complex_id, gene, cell_type in [key]
    ]
    return pd.DataFrame(rows)


def add_observed_context_coverage(lrp: pd.DataFrame) -> pd.DataFrame:
    _, members_by_complex = complex_members()
    _, global_edges = read_ppi(GLOBAL_PPI)
    member_pairs = {
        complex_id: {
            sorted_edge(a, b)
            for a, b in combinations(sorted(members), 2)
        }
        for complex_id, members in members_by_complex.items()
    }
    reference_edges = {
        complex_id: pairs & global_edges
        for complex_id, pairs in member_pairs.items()
    }
    requested = lrp[["complex_id", "cell_type"]].drop_duplicates()
    metrics = []
    for cell_type, group in requested.groupby("cell_type"):
        ppi_path = CELL_PPI_DIR / f"{cell_type}.txt"
        if not ppi_path.is_file():
            raise FileNotFoundError(f"Missing cell PPI for LRP context: {ppi_path}")
        nodes, edges = read_ppi(ppi_path)
        for complex_id in group["complex_id"]:
            members = members_by_complex[complex_id]
            reference = reference_edges[complex_id]
            metrics.append(
                {
                    "complex_id": complex_id,
                    "cell_type": cell_type,
                    "context_member_coverage": len(members & nodes) / len(members),
                    "context_edge_coverage": len(reference & edges) / len(reference),
                }
            )
        del nodes, edges
    metrics = pd.DataFrame(metrics)
    lrp = lrp.merge(metrics, on=["complex_id", "cell_type"], how="inner")
    return (
        lrp.groupby(
            ["inference_key", "complex_id", "cell_type"], as_index=False
        )
        .agg(
            complete_positive_lrp=("complete_positive_lrp", "mean"),
            context_member_coverage=("context_member_coverage", "first"),
            context_edge_coverage=("context_edge_coverage", "first"),
            n_observed_member_contexts=("gene", "nunique"),
        )
    )


def xmil_tables(
    values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binned_parts = []
    definition_parts = []
    per_complex_parts = []
    correlation_parts = []
    for coverage_column in ("context_member_coverage", "context_edge_coverage"):
        reference = (
            values[values["inference_key"] == "s2gae_bce_uni"]
            [["complex_id", "cell_type", coverage_column]]
            .dropna()
            .drop_duplicates(["complex_id", "cell_type"])
            .sort_values([coverage_column, "complex_id", "cell_type"], kind="mergesort")
        )
        labels = [f"bin_{index}" for index in range(10)]
        bin_numbers = np.minimum(
            np.arange(len(reference), dtype=int) * len(labels) // len(reference),
            len(labels) - 1,
        )
        reference["coverage_bin"] = pd.Categorical(
            [labels[index] for index in bin_numbers], categories=labels, ordered=True
        )
        definitions = (
            reference.groupby("coverage_bin", observed=True)
            .agg(
                lower=(coverage_column, "min"),
                upper=(coverage_column, "max"),
                n_complex_context_pairs=("complex_id", "size"),
            )
            .reindex(labels)
            .reset_index()
        )
        definitions["percentile_midpoint"] = 100.0 * (
            np.arange(len(labels), dtype=float) + 0.5
        ) / len(labels)
        definitions["coverage_metric"] = coverage_column
        definition_parts.append(definitions)

        rows = values.dropna(
            subset=[coverage_column, "complete_positive_lrp"]
        ).copy()
        lookup = reference[["complex_id", "cell_type", "coverage_bin"]].copy()
        lookup["coverage_bin"] = lookup["coverage_bin"].astype(str)
        rows = rows.merge(lookup, on=["complex_id", "cell_type"], how="inner")
        rows["coverage_bin"] = pd.Categorical(
            rows["coverage_bin"], categories=labels, ordered=True
        )
        per_complex_bin = (
            rows.groupby(
                ["inference_key", "complex_id", "coverage_bin"], observed=True
            )["complete_positive_lrp"]
            .mean()
            .rename("aggregated_lrp")
            .reset_index()
            .merge(
                definitions[["coverage_bin", "percentile_midpoint"]],
                on="coverage_bin",
                how="left",
            )
        )
        per_complex_bin["coverage_metric"] = coverage_column
        per_complex_bin["evidence_score"] = "complete_positive_lrp"
        per_complex_parts.append(per_complex_bin)

        summary = (
            per_complex_bin.groupby(
                ["inference_key", "coverage_bin"], observed=True
            )
            .agg(
                mean_lrp=("aggregated_lrp", "mean"),
                n_complexes=("complex_id", "nunique"),
            )
            .reset_index()
            .merge(
                definitions[
                    [
                        "coverage_bin",
                        "lower",
                        "upper",
                        "n_complex_context_pairs",
                        "percentile_midpoint",
                    ]
                ],
                on="coverage_bin",
                how="left",
            )
        )
        summary["coverage_metric"] = coverage_column
        summary["evidence_score"] = "complete_positive_lrp"
        summary["within_complex_aggregation"] = "mean"
        binned_parts.append(summary)

        corr_rows = []
        for (model_key, complex_id), group in values.groupby(
            ["inference_key", "complex_id"]
        ):
            sub = group[[coverage_column, "complete_positive_lrp"]].dropna()
            if len(sub) < 3 or sub[coverage_column].nunique() < 2:
                continue
            constant = sub["complete_positive_lrp"].nunique() < 2
            rho = (
                0.0
                if constant
                else float(
                    spearmanr(
                        sub[coverage_column], sub["complete_positive_lrp"]
                    ).statistic
                )
            )
            corr_rows.append(
                {
                    "inference_key": model_key,
                    "complex_id": complex_id,
                    "coverage_metric": coverage_column,
                    "evidence_score": "complete_positive_lrp",
                    "spearman": rho,
                    "constant_evidence": constant,
                }
            )
        correlations = pd.DataFrame(corr_rows)
        for model_key, group in correlations.groupby("inference_key"):
            rhos = group["spearman"].to_numpy(dtype=float)
            correlation_parts.append(
                {
                    "inference_key": model_key,
                    "inference_label": MODEL_LABELS[model_key],
                    "coverage_metric": coverage_column,
                    "evidence_score": "complete_positive_lrp",
                    "n_complexes": len(group),
                    "n_variable_evidence": int((~group["constant_evidence"]).sum()),
                    "n_constant_evidence": int(group["constant_evidence"].sum()),
                    "median_spearman": float(np.median(rhos)),
                    "fraction_positive_spearman": float(np.mean(rhos > 0)),
                }
            )
    all_per_complex_bins = pd.concat(per_complex_parts, ignore_index=True)
    return (
        pd.concat(binned_parts, ignore_index=True),
        pd.concat(
            [part for part in definition_parts if not part.empty], ignore_index=True
        ),
        all_per_complex_bins,
        pd.DataFrame(correlation_parts),
    )


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


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    stats, size_distribution, protein_distribution = dataset_tables()
    checkpoint_inputs = checkpoint_cell_representations()
    topology, cell_coverage = build_topology_and_cell_coverage()

    main_rows = []
    loss_rows = []
    per_complex_parts = []
    lrp_parts = []

    # Aggregate ESM2 baseline.
    checkpoint_dir = CHECKPOINT_ROOT / "aggregate/lr_esm/lr_esm"
    split_file = checkpoint_dir / "split_indices.npz"
    data = load_run_data(run_spec("aggregate", "lr_esm", "lr_esm"), split_file)
    result = evaluate(
        data,
        checkpoint_dir,
        split_file,
        "lr_esm",
    )
    main_rows.extend(performance_rows("lr_esm", "lr_esm", result))
    stats = append_modeling_statistics(stats, data, split_file)
    del data, result
    gc.collect()

    # Contextual models: aggregate performance, per-complex performance and xMIL.
    for model_key in MAIN_CONTEXT_MODEL_ORDER:
        print(f"CORUM analysis: {MODEL_LABELS[model_key]}", flush=True)
        per_complex_split = (
            CHECKPOINT_ROOT / "per_complex" / model_key / "split_indices.npz"
        )
        per_complex_spec = run_spec(
            "per_complex", model_key, "abmil8_pdl_id2_dropout"
        )
        data = load_run_data(per_complex_spec, per_complex_split)
        for readout in CONTEXT_READOUTS:
            checkpoint_dir = CHECKPOINT_ROOT / "aggregate" / model_key / readout
            aggregate_spec = run_spec("aggregate", model_key, readout)
            if (
                aggregate_spec["cell_embedding_file"]
                == per_complex_spec["cell_embedding_file"]
            ):
                readout_data = data
            else:
                readout_data = load_run_data(
                    aggregate_spec,
                    checkpoint_dir / "split_indices.npz",
                )
            result = evaluate(
                readout_data,
                checkpoint_dir,
                checkpoint_dir / "split_indices.npz",
                readout,
            )
            rows = performance_rows(model_key, readout, result)
            main_rows.extend(rows)
            if model_key == "s2gae_bce_uni":
                loss_rows.extend(rows)
            if readout_data is not data:
                del readout_data
                gc.collect()

        result = evaluate(
            data,
            CHECKPOINT_ROOT / "per_complex" / model_key,
            per_complex_split,
            "abmil8_pdl_id2_dropout",
        )
        per_complex_parts.append(
            per_complex_rows(
                model_key,
                result,
                data["class_names"],
            )
        )
        lrp_parts.append(
            xmil_lrp_values(
                model_key,
                data,
                CHECKPOINT_ROOT / "per_complex" / model_key,
                per_complex_split,
            )
        )

        if model_key == "s2gae_bce_uni":
            result = evaluate(
                data,
                CHECKPOINT_ROOT / "per_complex/lr_esm",
                CHECKPOINT_ROOT / "per_complex/lr_esm/split_indices.npz",
                "lr_esm",
            )
            per_complex_parts.append(
                per_complex_rows("lr_esm", result, data["class_names"])
            )
        del data, result
        gc.collect()

    # pHuber and L1 were trained with the pooled pre-CCI cell export.
    for model_key in ("s2gae_phuber_uni", "s2gae_l1_uni"):
        split_file = CHECKPOINT_ROOT / "aggregate" / model_key / "split_indices.npz"
        data_by_embedding = {}
        for readout in LOSS_READOUTS[model_key]:
            checkpoint_dir = CHECKPOINT_ROOT / "aggregate" / model_key / readout
            aggregate_spec = run_spec("aggregate", model_key, readout)
            embedding_file = aggregate_spec["cell_embedding_file"]
            if embedding_file not in data_by_embedding:
                data_by_embedding[embedding_file] = load_run_data(
                    aggregate_spec, split_file
                )
            data = data_by_embedding[embedding_file]
            result = evaluate(
                data,
                checkpoint_dir,
                checkpoint_dir / "split_indices.npz",
                readout,
            )
            loss_rows.extend(performance_rows(model_key, readout, result))
        del data, data_by_embedding, result
        gc.collect()

    # ProstT5 sequence baseline (aggregate and split-fixed per-complex results).
    checkpoint_dir = CHECKPOINT_ROOT / "aggregate/lr_prostt5/lr_prostt5"
    split_file = checkpoint_dir / "split_indices.npz"
    data = load_run_data(
        run_spec("aggregate", "lr_prostt5", "lr_prostt5"), split_file
    )
    result = evaluate(
        data,
        checkpoint_dir,
        split_file,
        "lr_prostt5",
    )
    main_rows.extend(performance_rows("lr_prostt5", "lr_prostt5", result))
    result = evaluate(
        data,
        CHECKPOINT_ROOT / "per_complex/lr_prostt5",
        CHECKPOINT_ROOT / "per_complex/lr_prostt5/split_indices.npz",
        "lr_prostt5",
    )
    per_complex_parts.append(
        per_complex_rows("lr_prostt5", result, data["class_names"])
    )
    del data, result
    gc.collect()

    main_performance = pd.DataFrame(main_rows)
    loss_performance = pd.DataFrame(loss_rows)
    per_complex = pd.concat(per_complex_parts, ignore_index=True)
    metadata = pd.read_csv(
        CORUM_COMPLEXES,
        dtype={"complex_id": str},
        usecols=["complex_id", "complex_name", "n_members_in_ppi_universe"],
    )
    per_complex = (
        per_complex.merge(metadata, on="complex_id", how="left")
        .merge(topology, on=["complex_id", "complex_name", "n_members_in_ppi_universe"])
        .merge(cell_coverage, on="complex_id")
    )
    driver_summary, driver_bins, driver_correlations = driver_tables(per_complex)

    raw_lrp = pd.concat(lrp_parts, ignore_index=True)
    xmil_values = add_observed_context_coverage(raw_lrp)
    (
        xmil_summary,
        xmil_bin_definitions,
        xmil_per_complex_bins,
        xmil_correlations,
    ) = xmil_tables(xmil_values)

    tables = {
        "corum_dataset_statistics.csv": stats,
        "corum_checkpoint_cell_representations.csv": checkpoint_inputs,
        "corum_complex_size_distribution.csv": size_distribution,
        "corum_complexes_per_protein_distribution.csv": protein_distribution,
        "corum_main_performance.csv": main_performance,
        "corum_loss_performance.csv": loss_performance,
        "corum_per_complex_performance.csv": per_complex,
        "corum_complex_topology_metrics.csv": topology,
        "corum_complex_cell_ppi_metrics.csv": cell_coverage,
        "corum_per_complex_driver_summary.csv": driver_summary,
        "corum_per_complex_driver_bins.csv": driver_bins,
        "corum_per_complex_driver_correlations.csv": driver_correlations,
        "corum_xmil_context_values.csv": xmil_values,
        "corum_xmil_bin_definitions.csv": xmil_bin_definitions,
        "corum_xmil_per_complex_bins.csv": xmil_per_complex_bins,
        "corum_xmil_binned_summary.csv": xmil_summary,
        "corum_xmil_correlation_summary.csv": xmil_correlations,
    }
    for filename, table in tables.items():
        table.to_csv(ANALYSIS_DIR / filename, index=False)

    from exploration.plotting.plot_figure_3 import plot_all

    plot_all(ANALYSIS_DIR, ANALYSIS_DIR)
    print(f"Wrote CORUM analysis tables and plots to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
