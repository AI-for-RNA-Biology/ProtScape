# Analyses and plots

Each analysis module performs the computation and then calls its dedicated plotting module. `run_analysis.sh` runs the complete workflow.

## Requirements

Use the main `protscape` environment. Pretraining evaluation and Parkinson analysis require CUDA; CUDA is recommended for the other model analyses, and exhaustive consensus inference is very slow on CPU. Arial must be installed for plot rendering.

## Run the analyses

From the repository root:

```bash
bash scripts/run_analysis.sh
```

The bulk-network statistics step requires the complete `<output_root>/data_processing_bulk/` tree, including its intermediate gene-selection tables. The remaining analyses use the datasets, embeddings and checkpoints configured in `configs/paths.yaml`. `downstream_checkpoint_root` must point to the released paper-analysis checkpoint bundle; raw sweep outputs are not rearranged automatically.

This runs:

- bulk-network statistics;
- pretraining and pooling evaluation;
- loss-consensus and STRING analyses;
- CORUM evaluation;
- therapeutic-target evaluation and LRP;
- Parkinson disease target discovery, Leiden clustering and enrichment.

Analysis outputs and their plots are written below `<output_root>/analysis/`.

The exhaustive consensus scoring is isolated in `loss_consensus_inference.py`; plotting never recomputes model predictions, correlations or enrichment analyses. Plots are saved separately rather than assembled into multi-panel figures. Matplotlib uses Arial with PDF and PostScript font type 42.
