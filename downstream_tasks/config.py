from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

PATHS_FILE = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
with PATHS_FILE.open("r", encoding="utf-8") as handle:
    PATHS = yaml.safe_load(handle)

DEFAULT_DATA_ROOT = Path(PATHS["data_root"]).expanduser()
DEFAULT_GLOBAL_PPI = Path(PATHS["global_ppi"]).expanduser()
DEFAULT_INFERENCE_ROOT = Path(PATHS["inference_root"]).expanduser()
DEFAULT_OUTPUT_ROOT = Path(PATHS["output_root"]).expanduser() / "downstream_tasks"
DEFAULT_ESM_EMBEDDINGS = Path(PATHS["esm2_embeddings"]).expanduser()
DEFAULT_PROSTT5_EMBEDDINGS = Path(PATHS["prostt5_embeddings"]).expanduser()
DEFAULT_PINNACLE_PAPER_EMBEDDINGS_DIR = Path(
    PATHS["pinnacle_paper_embeddings_dir"]
).expanduser()
DEFAULT_CORUM_RAW_JSON = Path(PATHS["corum_raw_json"]).expanduser()
DEFAULT_CORUM_DATASET_DIR = Path(PATHS["corum_dataset_dir"]).expanduser()
DEFAULT_THERAPEUTIC_TARGET_DRUGBANK_TARGETS = Path(
    PATHS["therapeutic_target_drugbank_targets"]
).expanduser()
DEFAULT_THERAPEUTIC_TARGET_EVIDENCE_DIR = Path(
    PATHS["therapeutic_target_evidence_dir"]
).expanduser()
DEFAULT_THERAPEUTIC_TARGET_OT_RELEASE = PATHS["therapeutic_target_ot_release"]
DEFAULT_THERAPEUTIC_TARGET_OT_DISEASES_DIR = Path(
    PATHS["therapeutic_target_ot_diseases_dir"]
).expanduser()
DEFAULT_THERAPEUTIC_TARGET_OT_TARGETS_DIR = Path(
    PATHS["therapeutic_target_ot_targets_dir"]
).expanduser()
DEFAULT_THERAPEUTIC_TARGET_OT_ASSOCIATIONS_DIR = Path(
    PATHS["therapeutic_target_ot_associations_dir"]
).expanduser()
DEFAULT_THERAPEUTIC_TARGET_DATASET_DIR = Path(
    PATHS["therapeutic_target_dataset_dir"]
).expanduser()
THERAPEUTIC_TARGET_IDS = (
    "EFO_0003767",
    "EFO_0000685",
    "EFO_0000305",
    "EFO_1001207",
    "EFO_0000571",
    "EFO_0000676",
    "EFO_0001361",
    "MONDO_0005148",
    "MONDO_0007915",
    "EFO_0000274",
    "MONDO_0004979",
    "EFO_0000341",
    "EFO_1001249",
    "EFO_0003884",
    "MONDO_0005180",
)


@dataclass
class TaskConfig:
    label_csv: Path
    name: str


@dataclass
class EmbeddingPaths:
    esm: Path
    pinnacle_paper_protein: Path
    pinnacle_paper_cell: Path
    pinnacle_paper_labels: Path
    pinnacle_paper_cell_labels: Path


@dataclass
class Config:
    data_root: Path
    inference_root: Path
    output_root: Path
    embeddings: EmbeddingPaths
    tasks: Dict[str, TaskConfig]
    inference_model: str
    embedding_source: str = "esm"
    dataset_mode: str = "bulk"

    seed: int = 42
    n_folds: int = 6
    val_fold: int = 0
    batch_size: int = 512
    epochs: int = 300
    patience: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    train_selection_metric: str = "auprc"

    num_heads: int = 1
    att_hidden_dim: int = 256
    dropout: float = 0.0
    att_dropout: float = 0.0
    mil_mlp_dim: int = 256
    ctx_proj_dim: Optional[int] = None
    aem_lambda: float = 0.0
    pdl_pmax: float = 0.0
    pdl_proj_mode: str = "identity"
    pdl_proj_layers: int = 2
    esm_proj_dim: Optional[int] = None
    use_class_weights: bool = True

    def get_inference_path(self) -> Path:
        return self.inference_root / self.inference_model

def get_therapeutic_tasks(dataset_dir: Path) -> Dict[str, TaskConfig]:
    """Return the configured therapeutic-target tasks."""
    return {
        f"therapeutic_target_{disease_id.lower()}": TaskConfig(
            label_csv=dataset_dir / f"therapeutic_target_{disease_id}.csv",
            name=f"Therapeutic Target Prediction ({disease_id})",
        )
        for disease_id in THERAPEUTIC_TARGET_IDS
    }


def load_config(
    inference_model: str,
    embedding_source: str = "esm",
    dataset_mode: str = "bulk",
) -> Config:
    embedding_source = str(embedding_source).strip().lower()
    if embedding_source not in {"esm", "prostt5"}:
        raise ValueError(
            f"Invalid embedding_source='{embedding_source}'. Expected one of: esm, prostt5"
        )
    dataset_mode = str(dataset_mode).strip().lower()
    if dataset_mode not in {"bulk", "legacy"}:
        raise ValueError(
            f"Invalid dataset_mode='{dataset_mode}'. Expected one of: bulk, legacy"
        )

    data_root = DEFAULT_DATA_ROOT
    inference_root = DEFAULT_INFERENCE_ROOT
    output_root = DEFAULT_OUTPUT_ROOT

    embeddings = EmbeddingPaths(
        esm=(
            DEFAULT_ESM_EMBEDDINGS
            if embedding_source == "esm"
            else DEFAULT_PROSTT5_EMBEDDINGS
        ),
        pinnacle_paper_protein=DEFAULT_PINNACLE_PAPER_EMBEDDINGS_DIR
        / "pinnacle_protein_embed.pth",
        pinnacle_paper_cell=DEFAULT_PINNACLE_PAPER_EMBEDDINGS_DIR
        / "pinnacle_mg_embed.pth",
        pinnacle_paper_labels=DEFAULT_PINNACLE_PAPER_EMBEDDINGS_DIR
        / "pinnacle_protein_labels_dict.txt",
        pinnacle_paper_cell_labels=DEFAULT_PINNACLE_PAPER_EMBEDDINGS_DIR
        / "pinnacle_mg_labels_dict.txt",
    )

    tasks = {
        "corum": TaskConfig(
            label_csv=DEFAULT_CORUM_DATASET_DIR / "corum_memberships_filtered.csv",
            name="CORUM Complex Prediction",
        ),
    }
    tasks.update(get_therapeutic_tasks(DEFAULT_THERAPEUTIC_TARGET_DATASET_DIR))

    return Config(
        data_root=data_root,
        inference_root=inference_root,
        output_root=output_root,
        embeddings=embeddings,
        tasks=tasks,
        inference_model=inference_model,
        embedding_source=embedding_source,
        dataset_mode=dataset_mode,
    )


def get_hc_embedding_paths(
    inference_path: Path,
    cell_embedding_file: str = "cell_embeddings.pt",
) -> Dict[str, Optional[Path]]:
    """Return the canonical contextual protein and cell embedding exports."""
    protein_embed = inference_path / "protein_embeddings.pt"
    cell_embed = inference_path / cell_embedding_file
    missing = [path for path in (protein_embed, cell_embed) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing canonical inference embedding export(s): "
            + ", ".join(str(path) for path in missing)
        )

    return {
        "protein_embed": protein_embed,
        "cell_embed": cell_embed,
        "protein_labels": None,
        "cell_labels": None,
    }
