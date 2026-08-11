# Analyses and plots

Run commands from the repository root with paths set in `configs/paths.yaml`:

```bash
conda activate protscape
```

Pretraining evaluation and Parkinson analysis require CUDA. Arial is required for plot rendering.

Run the selected CORUM and therapeutic-target training commands in `downstream_tasks/README.md` before recomputing their analyses.

The STRING analyses use the four human STRING v12 files configured by `string_protein_info`, `string_links_detailed`, `string_protein_aliases` and `string_enrichment_terms`. Download `9606.protein.info.v12.0.txt.gz`, `9606.protein.links.detailed.v12.0.txt.gz`, `9606.protein.aliases.v12.0.txt.gz` and `9606.protein.enrichment.terms.v12.0.txt.gz` from the official [STRING v12 download page](https://version-12-0.string-db.org/cgi/download).

## Run everything

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

# Parkinson target discovery, Leiden clustering and enrichment
python -m exploration.analysis.parkinson_target_analysis

# ALS motor-neuron PPI-module rewiring (CTRL vs. VCP, D22 vs. D35)
python -m exploration.analysis.ALS_rewiring_analysis
```

The bulk-network step requires the complete `<output_root>/data_processing_bulk/` directory. Pretraining analyses use `checkpoint_root`; downstream analyses use the selected runs below `<output_root>/downstream_tasks/`. Outputs are written below `<output_root>/analysis/`.
