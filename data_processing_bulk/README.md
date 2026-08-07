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
| `als_motor_neuron_counts` | Released motor-neuron count matrix |
| `als_motor_neuron_metadata` | Released motor-neuron context metadata |
| `als_astrocyte_counts` | Released astrocyte count matrix |
| `als_astrocyte_metadata` | Released astrocyte context metadata |
| `cell_ontology_obo` | Cell Ontology OBO file |
| `tissue_ontology_obo` | BRENDA Tissue Ontology OBO file |
| `celltype_class_mapping` | Broad cell-class annotations |
| `cellphonedb_database` | CellPhoneDB v3 database file |

`hbca_supercluster_to_cl.csv` is the curated mapping from HBCA superclusters to Cell Ontology IDs used by ProtScape.

`tabula_metadata` must contain `cell_id`; after joining it to the count matrix, the cell annotations must provide `cell_ontology_class`, `donor` and `organ_tissue`.

## Dataset downloads

The processed Tabula Sapiens v1 matrix and metadata are available from [GEO accession GSM6058681](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6058681) and accompany the [Tabula Sapiens atlas](https://doi.org/10.1126/science.abl4896).

The neuronal and non-neuronal Human Brain Cell Atlas v1.0 objects are available from the [Human Cell Atlas data portal](https://data.humancellatlas.org/hca-bio-networks/nervous-system/atlases/brain-v1-0) and accompany [Siletti et al.](https://doi.org/10.1126/science.add7046).

The release already contains the ALS count matrices and context metadata consumed by Step 0. They were derived from the public motor-neuron dataset [GSE152983](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152983) ([Tyzack et al.](https://doi.org/10.1093/brain/awab078)) and astrocyte dataset [GSE160133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE160133) ([Ziff et al.](https://doi.org/10.1093/nar/gkab115)). The GEO pages provide the raw sequencing files.

The CellPhoneDB 3.0.0 database is available from the [CellPhoneDB releases](https://github.com/ventolab/CellphoneDB/releases).

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

The `merged` branch uses the previously generated Tabula Sapiens and HBCA intermediates. Use `all` for a complete rebuild from the configured inputs.

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

The data release contains the exact `networks_bulk` snapshot used to train the released checkpoints. A fresh rebuild can differ slightly after the corrected ALS compartment filtering and upstream dataset updates; this is expected. Use the released graph bundle for checkpoint-equivalent inference.
