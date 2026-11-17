# Analyses and plots

Run from the repository root after setting `configs/paths.yaml`.

## Draw figures and tables

Set `paper_source_data` to the source-table directory, then run:

```bash
python -m exploration.plotting.paper_figures
python -m exploration.plotting.paper_tables
```

Figures go to `<output_root>/paper_figures/` and CSV tables to
`<output_root>/paper_tables/`. PDF text stays editable; plots use Arial or Liberation Sans when Arial is absent.

## Therapeutic targets

Evaluate the [selected downstream runs](../downstream_tasks/README.md) and explain
their predictions with xMIL-LRP:

```bash
python -m exploration.analysis.therapeutic_target_analysis
```

For Parkinson target discovery, set `parkinson_checkpoint_root` to the directory
containing the trained ensembles, then run:

```bash
python -m exploration.analysis.parkinson_inference
python -m exploration.analysis.parkinson_candidates
python -m exploration.analysis.parkinson_target_analysis
python -m exploration.analysis.parkinson_modularity
```

Inference requires CUDA. Candidate analysis reads the saved predictions, the Open
Targets association CSV beside the disease labels, and the configured DrugBank
export. Network analyses also need the human [STRING v12 files](https://version-12-0.string-db.org/cgi/download)
listed in `configs/paths.yaml`.

## Other analyses

```bash
python -m exploration.analysis.data_bulk_statistics
python -m exploration.analysis.pretraining_evaluation
python -m exploration.analysis.consensus_analysis
python -m exploration.analysis.corum_analysis
```

Pretraining evaluation requires CUDA and `checkpoint_root`. Bulk statistics need
the completed data-processing outputs. Results go to `<output_root>/analysis/`;
use that directory as `paper_source_data` to plot them. See the separate
[ALS instructions](analysis/ALS_rewiring/README.md) for rewiring analyses.
