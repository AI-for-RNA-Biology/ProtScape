# Analyses and plots

Run commands from the repository root with paths set in `configs/paths.yaml`:

```bash
conda activate protscape
```

Plots use centimetre-scale layouts, Arial and editable PDF text. Set `paper_source_data` to the released `data/paper_source_data/` directory, then redraw the quantitative panels without model training:

```bash
python -m exploration.plotting.paper_figures
python -m exploration.plotting.paper_tables
```

This writes panel PDFs and PNGs below `<output_root>/paper_figures/` for the supplied dataset, pretraining, consensus, CORUM, therapeutic-target and Parkinson source tables. Manuscript layouts, externally generated STRING views and ALS rewiring are separate.
The table command writes five numeric CSV files below `<output_root>/paper_tables/`. AUPRC and sample SD use separate percentage-point columns. SD describes five models for CORUM and individual diseases, or 15 disease means for therapeutic-target averages.
To plot or tabulate newly computed analyses, set `paper_source_data` to `<output_root>/analysis/` and use the same commands.

## Recompute analyses

Pretraining evaluation and Parkinson inference require CUDA.

Pretraining re-evaluation draws structured negatives with per-context seeds and uses a shared held-out CCI set for ProtScape. These scores can differ from the recorded pretraining results supplied for the paper figures and tables.

Run the selected CORUM and therapeutic-target training commands in `downstream_tasks/README.md` before recomputing their analyses.

The STRING analyses use the four human STRING v12 files configured by `string_protein_info`, `string_links_detailed`, `string_protein_aliases` and `string_enrichment_terms`. Download `9606.protein.info.v12.0.txt.gz`, `9606.protein.links.detailed.v12.0.txt.gz`, `9606.protein.aliases.v12.0.txt.gz` and `9606.protein.enrichment.terms.v12.0.txt.gz` from the official [STRING v12 download page](https://version-12-0.string-db.org/cgi/download).

```bash
bash scripts/run_analysis.sh
```

## Run individual analyses

Run these in order when rebuilding all outputs:

```bash
# Bulk-network statistics
python -m exploration.analysis.data_bulk_statistics

# Pretraining and pooling evaluation
python -m exploration.analysis.pretraining_evaluation

# Loss consensus and STRING analyses
python -m exploration.analysis.consensus_analysis

# CORUM evaluation
python -m exploration.analysis.corum_analysis

# Therapeutic-target evaluation and LRP
python -m exploration.analysis.therapeutic_target_analysis

# Parkinson ensemble predictions
python -m exploration.analysis.parkinson_inference

# Target discovery, Leiden clustering and enrichment from those predictions
python -m exploration.analysis.parkinson_target_analysis

# Modularity against 1,000 degree-preserving null networks (connected STRING graph)
python -m exploration.analysis.parkinson_modularity

# ALS rewiring, after preparing the inputs documented in ALS_rewiring/README.md
python -m exploration.analysis.ALS_rewiring_analysis
```

The bulk-network step requires the complete `<output_root>/data_processing_bulk/` directory. Pretraining analyses use `checkpoint_root`; downstream analyses use the selected runs below `<output_root>/downstream_tasks/`. Outputs are written below `<output_root>/analysis/`.

See [ALS rewiring](analysis/ALS_rewiring/README.md) for interaction-matrix and neighbourhood generators, expression inputs and external annotations.

Parkinson inference reads the two ensembles from `parkinson_checkpoint_root`; set this to `<output_root>/downstream_tasks` for new training runs. Candidate analyses use the saved predictions, `opentargets_parkinson_associations.csv` beside the disease labels, and the configured October 2022 DrugBank export.

To generate Parkinson candidates and external support from the saved predictions without the STRING network analyses:

```bash
python -m exploration.analysis.parkinson_candidates
```

Inference writes `parkinson_global_predictions.csv.gz` below `<output_root>/analysis/parkinson_target_analysis/`. Both candidate and network analyses read this file; they do not rerun the models. Plotting scripts read the resulting analysis tables.
