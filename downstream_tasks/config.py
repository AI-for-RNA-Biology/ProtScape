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

ESM_EMBED_PATH = "protein_gene_based_embeddings/gene_protein_embeddings_esm2_650M.plk"
PROSTT5_EMBED_PATH = "protein_gene_based_embeddings/gene_protein_embeddings_prostt5_1024.plk"

CORUM_CSV = "downstream_tasks/corum_dataset/corum_memberships_filtered.csv"
PAPER_THERAPEUTIC_IDS = (
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

def get_therapeutic_tasks(data_root: Path) -> Dict[str, TaskConfig]:
    """Return the therapeutic-target tasks reported in the paper."""
    dataset_dir = data_root / "downstream_tasks" / "therapeutic_target_dataset"
    return {
        f"therapeutic_target_{disease_id.lower()}": TaskConfig(
            label_csv=dataset_dir / f"therapeutic_target_{disease_id}.csv",
            name=f"Therapeutic Target Prediction ({disease_id})",
        )
        for disease_id in PAPER_THERAPEUTIC_IDS
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

    sequence_relpath = ESM_EMBED_PATH if embedding_source == "esm" else PROSTT5_EMBED_PATH
    pinnacle_paper_dir = data_root.parent / "pinnacle_embeds"

    embeddings = EmbeddingPaths(
        esm=data_root / sequence_relpath,
        pinnacle_paper_protein=pinnacle_paper_dir / "pinnacle_protein_embed.pth",
        pinnacle_paper_cell=pinnacle_paper_dir / "pinnacle_mg_embed.pth",
        pinnacle_paper_labels=pinnacle_paper_dir / "pinnacle_protein_labels_dict.txt",
        pinnacle_paper_cell_labels=pinnacle_paper_dir / "pinnacle_mg_labels_dict.txt",
    )

    tasks = {
        "corum": TaskConfig(
            label_csv=data_root / CORUM_CSV,
            name="CORUM Complex Prediction",
        ),
    }
    tasks.update(get_therapeutic_tasks(data_root))

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


def get_hc_embedding_paths(inference_path: Path) -> Dict[str, Path]:
    # Canonical exports are the representations used by the historical
    # downstream runs. Additional diagnostic exports (for example the
    # pre-CCI cell tensor) must not replace them based on directory order.
    canonical_protein = inference_path / "protein_embeddings.pt"
    canonical_cell = inference_path / "cell_embeddings.pt"
    protein_embed = canonical_protein if canonical_protein.exists() else None
    cell_embed = canonical_cell if canonical_cell.exists() else None
    protein_labels = None
    cell_labels = None

    for f in sorted(inference_path.iterdir()):
        name = f.name.lower()
        if (
            protein_embed is None
            and "protein" in name
            and "embed" in name
            and f.suffix in (".pth", ".pt")
        ):
            protein_embed = f
        elif (
            cell_embed is None
            and "cell" in name
            and "embed" in name
            and f.suffix in (".pth", ".pt")
        ):
            cell_embed = f
        elif "protein" in name and "labels" in name and f.suffix == ".txt":
            protein_labels = f
        elif "cell" in name and "labels" in name and f.suffix == ".txt":
            cell_labels = f

    if protein_embed is None:
        for ext in ("*.pth", "*.pt"):
            candidates = list(inference_path.glob(f"*protein*{ext[1:]}"))
            if candidates:
                protein_embed = candidates[0]
                break

    if cell_embed is None:
        for ext in ("*.pth", "*.pt"):
            candidates = list(inference_path.glob(f"*cell*{ext[1:]}"))
            if candidates:
                cell_embed = candidates[0]
                break

    return {
        "protein_embed": protein_embed,
        "cell_embed": cell_embed,
        "protein_labels": protein_labels,
        "cell_labels": cell_labels,
    }
