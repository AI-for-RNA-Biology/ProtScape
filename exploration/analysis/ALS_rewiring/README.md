# ALS network rewiring

PPI-module analysis for the ALS motor-neuron S2GAE/STRING network
(CTRL versus VCP, days 22 and 35).

## Inputs

Set the `als_rewiring_*` paths in [`configs/paths.yaml`](../../../configs/paths.yaml):

- `als_rewiring_data_root`: D22/D35 subset matrices.
- `als_rewiring_raw_matrix_csv`: full S2GAE/STRING interaction matrix.
- `als_rewiring_go_terms_csv`: gene–GO annotations.
- `als_rewiring_cyto_gene_expression_csv`: cytoplasmic gene-expression matrix.
- `als_rewiring_neighborhood_edges_csv`: context-specific neighbourhood edges.

Set thresholds and Leiden resolution settings in `config.py`.

## Run

From the repository root:

```bash
python -m exploration.analysis.ALS_rewiring_analysis

# Resume at clustering, or run individual stages
python -m exploration.analysis.ALS_rewiring_analysis --from 05
python -m exploration.analysis.ALS_rewiring_analysis --only 07 09
```

The workflow subsets the interaction matrix, selects high-confidence edges,
clusters and refines modules, tests enrichment, and plots networks.
Stage 05b saves the initial partition before replacing it with the refined
partition used by subsequent analyses. Stage 08 queries STRING enrichment
and requires network access.

Stage scripts in `scripts/` can also run individually. Neighbourhood analyses
use the separately supplied edge table and the gene selected in `config.py`.
