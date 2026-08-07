# Dataset processing

This pipeline builds the context-specific protein interaction networks, cell-cell interaction network and metagraph used by ProtScape. Set every path below in `configs/paths.yaml` before running it.

## Required paths

| Config key | Path |
|---|---|
| `output_root` | Directory for all generated files |
| `global_ppi` | Two-column human PPI edge list using gene symbols |
| `tabula_h5ad` | Tabula Sapiens v1 count matrix |
| `tabula_metadata` | Tabula Sapiens cell metadata |
| `hbca_neurons_h5ad` | HBCA neuronal count matrix |
| `hbca_nonneurons_h5ad` | HBCA non-neuronal count matrix |
| `hbca_gene_metadata` | Frozen HBCA Ensembl-to-HGNC mapping |
| `als_gene_metadata` | Frozen ALS Ensembl-to-HGNC mapping |
| `als_motor_neuron_kallisto` | Motor-neuron Kallisto output directories |
| `als_motor_neuron_metadata` | Motor-neuron sample metadata |
| `als_astrocyte_kallisto` | Astrocyte Kallisto output directories |
| `als_astrocyte_metadata` | Astrocyte sample metadata |
| `cell_ontology_obo` | Cell Ontology OBO file |
| `tissue_ontology_obo` | BRENDA Tissue Ontology OBO file |
| `celltype_class_mapping` | Broad cell-class annotations |
| `cellphonedb_database` | CellPhoneDB v3 database file |

`hbca_supercluster_to_cl.csv` is included in this directory. The two gene-mapping tables are fixed inputs so processing does not depend on a live identifier service.

`tabula_metadata` must contain `cell_id`; after joining it to the count matrix, the cell annotations must provide `cell_ontology_class`, `donor` and `organ_tissue`. Each ALS Kallisto directory must contain `<Run>/abundance.tsv` files with `target_id` and `est_counts` columns.

## Run

From the repository root, a complete rebuild is:

```bash
conda env create -f environment_scrna.yml
conda env create -f environment_cellphonedb.yml
conda activate protscape-scrna
bash data_processing_bulk/run_pipeline.sh all
```

Individual branches or step ranges can also be run:

```bash
bash data_processing_bulk/run_pipeline.sh {hbca|tabula|als|merged} [start_step] [end_step]

# Example: run Tabula Sapiens Steps 1 through 7
bash data_processing_bulk/run_pipeline.sh tabula 1 7
```

The `merged` branch uses the previously generated Tabula Sapiens and HBCA intermediates. Use `all` for a complete rebuild from raw inputs.

The complete graph bundle is written to:

```text
<output_root>/data_processing_bulk/networks_bulk/
    global_ppi_edgelist.txt
    ppi_edgelists/
    cci_edgelist.txt
    mg_edgelist.txt
    count_edge_dict.pkl
    celltype_metadata.csv
    celltype_class_mapping.csv
```

## Processing details

Reliable genes are selected with a two-component `BayesianGaussianMixture` fitted to each log2(count + 1) profile. The threshold is the 0.99 quantile of the lower-mean component, using `random_state=0`, `max_iter=1000` and an inclusive `>=` comparison.

Before CellPhoneDB, each single-cell type is capped at 100 cells with seed 7. CellPhoneDB runs with 100 permutations, expression threshold 0.1 and debug seed 1. The single-cell branches use CellPhoneDB subsampling; the replicate-averaged ALS profiles do not.

For compartment-resolved ALS contexts, only nucleus--nucleus and nucleus--cytoplasm CCI edges are removed. Cytoplasm--cytoplasm, cytoplasm--whole-cell, nucleus--whole-cell and whole-cell--whole-cell edges are retained. Autocrine CCI self-loops are also retained.

The released checkpoints use the historical metagraph containing 5,658 cell--cell, 696 cell--tissue and 109 tissue--tissue edges. A fresh corrected rebuild contains 6,767 cell--cell, 696 cell--tissue and 109 tissue--tissue edges. Use the historical graph bundle with the released checkpoints to reproduce their inference outputs; use the rebuilt graph to train models on the corrected dataset.
