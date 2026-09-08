# ProtScape: Resolving context-specific protein-protein interactomes for biological discovery and therapeutic target prioritisation

**Authors:** Alois Thomas, Lisa Fournier, Vincent Jung, Rickie Patani, Pascal Frossard, Raphaëlle Luisier and Cédric Vincent-Cuaz<br>
**Correspondence:** [Cédric Vincent-Cuaz](mailto:cedric.vincent-cuaz@unibe.ch)<br>
**Data and pretrained models:** [Zenodo](https://zenodo.org/records/22645081)

[![Overview of ProtScape](assets/protscape_overview.png)](assets/protscape_overview.pdf)

## Overview

ProtScape is a multiscale framework for learning context-specific protein representations across proteins, cells and tissues. It combines protein foundation-model features, graph representation learning over context-specific interactomes, and hierarchical cell–cell and cell–tissue supervision. The repository includes construction of the cellular-context networks, model pretraining and inference, CORUM protein-complex and therapeutic-target prediction.

## Installation

```bash
git clone https://github.com/AI-for-RNA-Biology/ProtScape.git
cd ProtScape

conda env create -f environment.yml
conda activate protscape
```

Dataset processing additionally uses:

```bash
conda env create -f environment_scrna.yml
conda env create -f environment_cellphonedb.yml
```

## Data and configuration

Download the [data release](https://zenodo.org/records/22645081) and set paths in [`configs/paths.yaml`](configs/paths.yaml). Defaults expect a sibling directory named `ProtScape_release` and write results to `outputs/`.

The release includes networks, protein embeddings, pretraining checkpoints, CORUM and therapeutic-target labels and splits, and figure source tables. Downstream checkpoints cover the ProtScape and PINNACLE Parkinson ensembles. Extract the graph bundle with:

```bash
unzip /path/to/ProtScape_release/data/networks_bulk.zip \
    -d /path/to/ProtScape_release/data
```

Use the released inputs for the paper experiments. To rebuild datasets, follow the [network](data_processing_bulk/README.md) and [downstream](downstream_tasks/README.md) processing instructions and obtain their listed inputs.

## Usage

Run commands from the repository root:

Before training all selected downstream configurations, generate the six ablation embedding exports listed in the [inference instructions](pretraining/README.md#architecture-ablation-embeddings).

```bash
# Train the main ProtScape model
bash scripts/run_pretraining.sh

# Generate embeddings from the released main checkpoint
python -m pretraining.inference \
    /path/to/ProtScape_release/models/pretraining/protscape_main_state_dict.pt

# Train the validation-selected downstream configurations
python -m downstream_tasks.run_selected corum
python -m downstream_tasks.run_selected therapeutic_targets

# Redraw quantitative panels from released source tables (excluding ALS rewiring)
python -m exploration.plotting.paper_figures

# Export the five result tables as numeric CSV files
python -m exploration.plotting.paper_tables
```

Detailed instructions:

- [Dataset processing](data_processing_bulk/README.md)
- [Pretraining and inference](pretraining/README.md)
- [Downstream tasks](downstream_tasks/README.md)
- [Analyses and plots](exploration/README.md)

## Repository structure

```text
data_processing_bulk/   Context-specific PPIs and metagraph construction
pretraining/            ProtScape and adapted PINNACLE training and inference
downstream_tasks/       CORUM and therapeutic-target prediction
exploration/            Analysis and plotting code
configs/                Data paths and model configurations
scripts/                Workflow entry points
```
