# Dataset processing

Build context-specific PPIs and a cell–tissue metagraph from Tabula Sapiens, the Human Brain Cell Atlas (HBCA), and ALS expression profiles.

Set input paths and `output_root` in `configs/paths.yaml`. Inputs are:

- [Tabula Sapiens v1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6058681): count matrix and cell metadata (`cell_id`, `cell_ontology_class`, `donor`, `organ_tissue`).
- [HBCA v1](https://data.humancellatlas.org/hca-bio-networks/nervous-system/atlases/brain-v1-0): neuronal and non-neuronal count matrices.
- ALS count profiles, gene mappings, Cell Ontology/BTO snapshots, cell-class annotations, and the global PPI: included in the [data download](https://zenodo.org/records/22645081).
- [CellPhoneDB 3.0.0 database](https://raw.githubusercontent.com/ventolab/cellphonedb-data/v3.0.0/cellphone.db).

Run from the repository root:

```bash
conda env create -f environment_scrna.yml
conda env create -f environment_cellphonedb.yml
conda activate protscape-scrna
bash data_processing_bulk/run_pipeline.sh all

# Run one dataset or a range of steps
bash data_processing_bulk/run_pipeline.sh tabula 1 7
```

Datasets: `hbca`, `tabula`, `als`, `merged`, or `all`. The `merged` branch needs the HBCA and Tabula intermediates. Set `PIPELINE_THREADS` to change the default of four workers.

| Step | Output |
|---|---|
| 0–1 | Load counts and aggregate pseudobulks |
| 2 | Select reliably expressed genes with a two-component GMM |
| 2b | Rank genes by context enrichment; choose network size from coverage and overlap |
| 3 | Build context PPIs from the global interactome |
| 4–6 | Prepare CellPhoneDB inputs and infer cell–cell interactions |
| 7–8 | Build the metagraph and merge datasets |

The final bundle is `<output_root>/data_processing_bulk/networks_bulk/`, containing `ppi_edgelists/`, `global_ppi_edgelist.txt`, `cci_edgelist.txt`, `mg_edgelist.txt`, edge counts, and cell-type metadata. Set `networks_bulk`, `global_ppi`, and `celltype_class_mapping` to this bundle to use it for pretraining.
