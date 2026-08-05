# ProtScape

ProtScape learns contextualized protein representations from transcriptome-constrained protein interaction networks.

This repository contains the code used to construct the study graphs, pretrain ProtScape and its baselines, train downstream models, and reproduce the manuscript analyses.

## Repository structure

```text
data_processing_bulk/   Build the context-specific PPIs and metagraph
pretraining/            Train ProtScape and PINNACLE baseline models
downstream_tasks/       Train and evaluate downstream prediction models
exploration/            Manuscript figure and analysis scripts
configs/                Paper model configurations
scripts/                Simple entry points for the main workflows
data/                   Documentation for inputs distributed separately
```

## Environment

The model and downstream workflows use the main environment:

```bash
conda env create -f environment.yml
conda activate protscape
```

Graph preprocessing uses the single-cell environment and, only for the CellPhoneDB step, its legacy-compatible environment:

```bash
conda env create -f environment_scrna.yml
conda env create -f environment_cellphonedb.yml
conda activate protscape-scrna
```

The processing script dispatches each stage to the appropriate environment with `conda run`; activating the single-cell environment also provides the small YAML reader used to load `configs/paths.yaml`.

## Data

Input datasets, expected filenames and the planned public data release are described in [`data/README.md`](data/README.md). Large expression matrices, processed graphs, model checkpoints and embeddings are distributed separately and are not stored in Git.

## Running the models

Input, inference and output locations are set once in `configs/paths.yaml`. `global_ppi` is the source interactome used only when rebuilding the bulk graphs. `networks_bulk` is the ready-to-use processed graph bundle consumed by pretraining; it can point directly to the released paper dataset. Generated files are never written inside this repository by default.

Run the main ProtScape model with the command written explicitly in the script:

```bash
bash scripts/run_pretraining.sh
```

The other paper pretraining configurations are recorded in `configs/pretraining.yaml`. CORUM and therapeutic-target runs have separate, readable scripts:

```bash
bash scripts/run_downstream_corum.sh <inference-model>
bash scripts/run_downstream_tt.sh <inference-model>
```

Each script runs the complete set of downstream readouts and hyperparameter values reported in the paper.

The bulk graph pipeline is similarly direct:

```bash
bash data_processing_bulk/run_pipeline.sh all
```

This creates `<output_root>/data_processing_bulk/networks_bulk`. To train on that rebuilt dataset instead of the released graph bundle, point `networks_bulk` in `configs/paths.yaml` to the new directory.

## Figures

The `exploration/` directory is reserved for the scripts that generate the submitted figures and tables. Once the final panel scripts are frozen, `scripts/run_plots.sh` will run them together.

## Attribution

ProtScape builds on PINNACLE. See [`NOTICE`](NOTICE) for attribution and licensing information.
