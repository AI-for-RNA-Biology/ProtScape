"""Paths and analysis parameters for the bulk-data preparation pipeline."""

import os
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

PATHS_FILE = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
with PATHS_FILE.open("r", encoding="utf-8") as handle:
    PATHS = yaml.safe_load(handle)

RAW_DATA_ROOT = Path(PATHS["raw_data_root"]).expanduser()
OUTPUT_ROOT = Path(PATHS["output_root"]).expanduser()
PROCESSING_ROOT = OUTPUT_ROOT / "data_processing_bulk"

# Keep string paths because the original preprocessing code uses os.path.
DATA_DIR = str(RAW_DATA_ROOT)
PINNACLE_BASE = str(PROCESSING_ROOT)

TABULA_BASE = os.path.join(DATA_DIR, "Tabula_Sapiens_v1")
HBCA_BASE = os.path.join(DATA_DIR, "HBCA_v1")

TABULA_INTERMEDIATE = os.path.join(PINNACLE_BASE, "tabula_pseudobulk")
HBCA_INTERMEDIATE = os.path.join(PINNACLE_BASE, "hbca_pseudobulk")
MERGED_INTERMEDIATE = os.path.join(PINNACLE_BASE, "merged_single_cell")

GLOBAL_PPI = str(Path(PATHS["global_ppi"]).expanduser())


# ---------------------------------------------------------------------------
# Dataset-specific resources
# ---------------------------------------------------------------------------

TABULA_H5AD = os.path.join(TABULA_BASE, "GSM6058681_TabulaSapiens.h5ad")
TABULA_METADATA = os.path.join(TABULA_BASE, "GSM6058681_TabulaSapiens_metadata.csv")

HBCA_NEURONS_PATH = os.path.join(HBCA_BASE, "HBCA_v1_all_neurons.h5ad")
HBCA_NONNEURONS_PATH = os.path.join(HBCA_BASE, "HBCA_v1_all_non_neuronal.h5ad")
HBCA_GENE_METADATA = os.path.join(HBCA_BASE, "gene_metadata.csv")

ALS_MN_KALLISTO = str(PATHS["als_motor_neuron_kallisto"])
ALS_MN_METADATA = str(PATHS["als_motor_neuron_metadata"])
ALS_ASTRO_KALLISTO = str(PATHS["als_astrocyte_kallisto"])
ALS_ASTRO_METADATA = str(PATHS["als_astrocyte_metadata"])

ALS_INTERMEDIATE = os.path.join(PINNACLE_BASE, "als_bulk")
ALS_BULK_DIR = ALS_INTERMEDIATE  # Alias for consistency
ALS_MN_INTERMEDIATE = os.path.join(ALS_INTERMEDIATE, "motor_neurons")
ALS_ASTRO_INTERMEDIATE = os.path.join(ALS_INTERMEDIATE, "astrocytes")

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# CellPhoneDB paths
# ---------------------------------------------------------------------------

# CellPhoneDB input/output directories (per dataset)
HBCA_CELLPHONEDB_INPUT = os.path.join(HBCA_INTERMEDIATE, "cellphonedb_input")
HBCA_CELLPHONEDB_OUTPUT = os.path.join(HBCA_INTERMEDIATE, "cellphonedb_results")
HBCA_CCI_EDGELIST = os.path.join(HBCA_INTERMEDIATE, "cci_edgelist.txt")

TABULA_CELLPHONEDB_INPUT = os.path.join(TABULA_INTERMEDIATE, "cellphonedb_input")
TABULA_CELLPHONEDB_OUTPUT = os.path.join(TABULA_INTERMEDIATE, "cellphonedb_results")
TABULA_CCI_EDGELIST = os.path.join(TABULA_INTERMEDIATE, "cci_edgelist.txt")

MERGED_CELLPHONEDB_INPUT = os.path.join(MERGED_INTERMEDIATE, "cellphonedb_input")
MERGED_CELLPHONEDB_OUTPUT = os.path.join(MERGED_INTERMEDIATE, "cellphonedb_results")
MERGED_CCI_EDGELIST = os.path.join(MERGED_INTERMEDIATE, "cci_edgelist.txt")

HBCA_SUPERCLUSTER_TO_CL = os.path.join(MODULE_DIR, "hbca_supercluster_to_cl.csv")

# CellPhoneDB parameters
CELLPHONEDB_PVALUE = 1e-3
CELLPHONEDB_MIN_LR = 15
CELLPHONEDB_THRESHOLD = 0.9

# Dataset-specific defaults
CELLPHONEDB_MIN_LR_BY_DATASET = {
    "tabula": 15,
    "hbca": 15,
    "als": 15,
    "merged": 40,
}

# Ontology paths
ONTOLOGY_DIR = os.path.join(DATA_DIR, "PINNACLE", "ontologies")
BTO_PATH = os.path.join(ONTOLOGY_DIR, "bto.obo")
CL_PATH = os.path.join(ONTOLOGY_DIR, "cl-full.obo")


@dataclass(frozen=True)
class PseudobulkParams:
    min_cells_per_sample: int = 25
    log_offset: float = 1.0
    gmm_quantile: float = 0.99  # Quantile used for the frozen paper networks_bulk build.
    min_donor_fraction: float = 0.5
    min_donor_count: int = 1


DEFAULTS = PseudobulkParams()


DATASET_DEFAULTS = {
    "tabula": replace(
        DEFAULTS,
        min_cells_per_sample=25,
        min_donor_fraction=0.5,
        min_donor_count=1,
    ),
    "hbca": replace(
        DEFAULTS,
        min_cells_per_sample=25,
        min_donor_fraction=0.5,
        min_donor_count=1,
    ),
}

# Cell Ontology mappings for ALS datasets
ALS_CELL_TYPE_MAPPINGS = {
    "motor_neuron": "CL:0000100",
    "astrocyte": "CL:0000127",
}


def resolve_params(dataset: str, **overrides) -> PseudobulkParams:
    """Return dataset defaults with any non-null command-line overrides."""
    base = DATASET_DEFAULTS.get(dataset, DEFAULTS)
    valid_overrides = {k: v for k, v in overrides.items() if v is not None}
    if not valid_overrides:
        return base
    return replace(base, **valid_overrides)
