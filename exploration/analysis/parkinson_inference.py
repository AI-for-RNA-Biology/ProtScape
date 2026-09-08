"""Run ensemble inference across the Parkinson target universe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from downstream_tasks.config import (
    DEFAULT_ESM_EMBEDDINGS,
    DEFAULT_INFERENCE_ROOT,
    PATHS,
    get_hc_embedding_paths,
    load_config,
)
from downstream_tasks.data.datasets import collate_abmil
from downstream_tasks.data.loaders import EmbeddingLoader
from downstream_tasks.models.abmil import ABMIL_LateFusion
from downstream_tasks.models.registry import MODEL_VARIANTS
from downstream_tasks.run_selected import find_selected_run
from downstream_tasks.training.cv_utils import (
    get_cv_train_val_indices,
    get_test_indices,
)
from exploration.analysis.therapeutic_target.checkpoint_lrp import (
    check_saved_split,
    load_task,
)


ESM_PATH = DEFAULT_ESM_EMBEDDINGS
PREDICTIONS_PATH = (
    Path(PATHS["output_root"]) / "analysis/parkinson_target_analysis"
    / "parkinson_global_predictions.csv.gz"
)

TASK = "therapeutic_target_mondo_0005180"
MODEL_KEY = "abmil_hc_cell_ext_embed_gated_8_pdl"
DEVICE = "cuda"
BATCH_SIZE = 64


@dataclass(frozen=True)
class ModelSpec:
    inference_key: str
    readout_key: str


MODEL_SPECS = {
    "protscape": ModelSpec(
        inference_key="s2gae_att_k1_fixed_do04_uni",
        readout_key="abmil8_pdl_id2_dropout",
    ),
    "pinnacle": ModelSpec(
        inference_key="pinnacle_random_fixed",
        readout_key="abmil8_pdl_id2_dropout",
    ),
}
MODEL_LABELS = {"protscape": "ProtScape", "pinnacle": "Pinnacle"}


@dataclass
class InferenceData:
    genes: list[str]
    bags: list[np.ndarray]
    cell_ids: list[list[str]]
    esm: np.ndarray
    task_genes: list[str]
    task_labels: np.ndarray
    task_global_indices: np.ndarray
    split: object
    class_names: list[str]


class NormalizedBagDataset(Dataset):
    def __init__(
        self,
        bags: list[np.ndarray],
        esm: np.ndarray,
        bag_mean: np.ndarray,
        bag_std: np.ndarray,
        esm_mean: np.ndarray,
        esm_std: np.ndarray,
    ):
        self.bags = bags
        self.esm = esm
        self.bag_mean = bag_mean.astype(np.float32, copy=False)
        self.bag_std = bag_std.astype(np.float32, copy=False)
        self.esm_mean = esm_mean.astype(np.float32, copy=False)
        self.esm_std = esm_std.astype(np.float32, copy=False)
        self.labels = torch.zeros((len(bags), 1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, index: int):
        bag = ((self.bags[index] - self.bag_mean) / self.bag_std).astype(
            np.float32
        )
        esm = ((self.esm[index] - self.esm_mean) / self.esm_std).astype(
            np.float32
        )
        return {
            "ctx": torch.from_numpy(bag),
            "esm": torch.from_numpy(esm),
        }, self.labels[index]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required Parkinson analysis input: {path}")
    return path


def build_model(
    config: pd.Series, ctx_dim: int, esm_dim: int, device: torch.device
):
    variant = MODEL_VARIANTS[MODEL_KEY]
    return ABMIL_LateFusion(
        num_classes=1,
        ctx_dim=ctx_dim,
        esm_dim=esm_dim,
        num_heads=int(config["num_heads"]),
        att_hidden=int(config["att_hidden_dim"]),
        dropout=float(config["dropout"]),
        att_dropout=float(config["att_dropout"]),
        attention_type=variant.attention_type,
        classifier_type=variant.classifier_type,
        mlp_hidden_dim=int(config["mil_mlp_dim"]),
        use_pdl=variant.use_pdl,
        pdl_proj_mode=str(config["pdl_proj_mode"]),
        pdl_proj_layers=int(config["pdl_proj_layers"]),
    ).to(device)


def load_inference_data(selected: dict[str, str]) -> InferenceData:
    config = load_config(
        selected["embedding_inference_name"], selected["embedding_source"], "bulk"
    )
    embedding_paths = get_hc_embedding_paths(
        DEFAULT_INFERENCE_ROOT / selected["embedding_inference_name"],
        cell_embedding_file=selected["cell_embedding_file"],
    )
    loader = EmbeddingLoader(
        require_file(ESM_PATH),
        require_file(embedding_paths["protein_embed"]),
        require_file(embedding_paths["cell_embed"]),
    )
    esm_by_gene = loader.load_esm()
    task_genes, task_labels, class_names, split = load_task(TASK, config, loader)
    bags_by_gene, cells_by_gene = loader.load_hc_with_cell(set(esm_by_gene))

    genes = sorted(set(esm_by_gene) & set(bags_by_gene))
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    bags = [
        np.stack(bags_by_gene[gene]).astype(np.float32, copy=False)
        for gene in genes
    ]
    cell_ids = [cells_by_gene[gene] for gene in genes]
    esm = np.stack([esm_by_gene[gene] for gene in genes]).astype(
        np.float32, copy=False
    )
    task_global_indices = np.asarray(
        [gene_to_index[gene] for gene in task_genes]
    )
    if split.n_cv_folds != 5:
        raise RuntimeError(f"Expected five fold models, found {split.n_cv_folds}")
    return InferenceData(
        genes,
        bags,
        cell_ids,
        esm,
        task_genes,
        task_labels,
        task_global_indices,
        split,
        class_names,
    )


def score_ensemble(
    data: InferenceData,
    config: pd.Series,
    run_dir: Path,
    device: torch.device,
) -> np.ndarray:
    fold_probabilities = []
    for fold in range(data.split.n_cv_folds):
        train_indices, _, _ = get_cv_train_val_indices(data.split, fold)
        train_global = data.task_global_indices[train_indices]
        train_contexts = np.concatenate(
            [data.bags[index] for index in train_global]
        )
        bag_mean = train_contexts.mean(axis=0)
        bag_std = np.maximum(train_contexts.std(axis=0), 1e-8)
        esm_mean = data.esm[train_global].mean(axis=0)
        esm_std = np.maximum(data.esm[train_global].std(axis=0), 1e-8)
        batches = DataLoader(
            NormalizedBagDataset(
                data.bags, data.esm, bag_mean, bag_std, esm_mean, esm_std
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_abmil,
        )
        model = build_model(
            config, data.bags[0].shape[1], data.esm.shape[1], device
        )
        checkpoint = require_file(run_dir / f"fold_{fold}.pt")
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        probabilities = []
        with torch.no_grad():
            for features, _ in batches:
                features = {
                    name: value.to(device) for name, value in features.items()
                }
                with torch.amp.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    probabilities.append(
                        torch.sigmoid(model(features)).cpu().numpy()[:, 0]
                    )
        fold_probabilities.append(
            np.concatenate(probabilities).astype(np.float32)
        )
        print(f"Completed global inference fold {fold + 1}/5", flush=True)
    return np.stack(fold_probabilities)


def cohort_membership(data: InferenceData) -> pd.DataFrame:
    test_indices = get_test_indices(data.split)
    label_by_gene = dict(
        zip(data.task_genes, data.task_labels[:, 0].astype(int))
    )
    test_genes = {data.task_genes[index] for index in test_indices}
    membership = pd.DataFrame({"protein": data.genes})
    membership["benchmark_label"] = membership["protein"].map(label_by_gene).astype(
        "Int64"
    )
    membership["benchmark_split"] = "label_excluded"
    membership.loc[
        membership["benchmark_label"].notna(), "benchmark_split"
    ] = "train_validation"
    membership.loc[
        membership["protein"].isin(test_genes), "benchmark_split"
    ] = "permanent_test"
    return membership


def run_global_inference() -> tuple[pd.DataFrame, pd.DataFrame]:
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Parkinson global inference requires a CUDA GPU")
    device = torch.device(DEVICE)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    score_tables = []
    shared_membership = None
    for model_name, spec in MODEL_SPECS.items():
        selected, _ = find_selected_run(
            "therapeutic_targets",
            task=TASK,
            inference_key=spec.inference_key,
            readout_key=spec.readout_key,
        )
        run_dir = (Path(PATHS["parkinson_checkpoint_root"]) / TASK
                   / selected["source_inference_name"] / selected["selected_output_model_key"])
        config = pd.read_csv(require_file(run_dir / "model_config.csv")).iloc[0]
        data = load_inference_data(selected)
        check_saved_split(
            run_dir, data.task_genes, data.task_labels, data.class_names, data.split
        )
        fold_scores = score_ensemble(data, config, run_dir / "models", device)
        membership = cohort_membership(data)
        if shared_membership is None:
            shared_membership = membership
        else:
            pd.testing.assert_frame_equal(shared_membership, membership)

        scores = pd.DataFrame(
            {
                "protein": data.genes,
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "mean_probability": fold_scores.mean(axis=0),
                "std_probability": fold_scores.std(axis=0),
                "n_contexts": [len(ids) for ids in data.cell_ids],
            }
        )
        for fold, values in enumerate(fold_scores):
            scores[f"fold_{fold}_probability"] = values
        score_tables.append(scores)
        del data, fold_scores
        torch.cuda.empty_cache()

    assert shared_membership is not None
    scores = pd.concat(score_tables, ignore_index=True).merge(
        shared_membership, on="protein", how="left", validate="many_to_one"
    )
    return scores, shared_membership


def load_predictions(path: Path = PREDICTIONS_PATH):
    """Load ensemble predictions and their shared benchmark membership."""
    scores = pd.read_csv(require_file(path), float_precision="round_trip")
    scores["benchmark_label"] = scores["benchmark_label"].astype("Int64")
    membership = scores[["protein", "benchmark_label", "benchmark_split"]].drop_duplicates()
    if membership["protein"].duplicated().any():
        raise ValueError("Prediction ensembles use different benchmark memberships.")
    return scores, membership.reset_index(drop=True)


def main():
    scores, _ = run_global_inference()
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved ensemble predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
