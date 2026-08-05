# Data

Large data files are not committed to this repository. Input and output locations are configured in `configs/paths.yaml`. The `networks_bulk` entry selects the processed bundle used by pretraining, while `global_ppi` supplies the starting interactome only when the bundle is rebuilt.

The intended downloaded-release layout is:

```text
data/
├── raw/
│   ├── Tabula_Sapiens_v1/
│   ├── HBCA_v1/
│   └── PINNACLE/ontologies/
├── processed/
│   ├── networks_bulk/
│   │   ├── global_ppi_edgelist.txt
│   │   ├── count_edge_dict.pkl
│   │   ├── ppi_edgelists/
│   │   ├── cci_edgelist.txt
│   │   └── mg_edgelist.txt
│   ├── protein_embeddings/
│   └── downstream_tasks/
└── README.md
```

All newly generated artifacts belong under the external `output_root` configured in `configs/paths.yaml`.

## Bulk-processing input paths

The pipeline reads these entries directly from `configs/paths.yaml`:

| Config entry | Required input |
|---|---|
| `global_ppi` | Global PPI edge list |
| `als_motor_neuron_kallisto` | Directory containing one Kallisto output folder per motor-neuron sample |
| `als_motor_neuron_metadata` | Motor-neuron sample metadata table |
| `als_astrocyte_kallisto` | Directory containing one Kallisto output folder per astrocyte sample |
| `als_astrocyte_metadata` | Astrocyte sample metadata table |
| `cellphonedb_database` | CellPhoneDB v3 SQLite database |

The remaining files are located relative to `raw_data_root`:

```text
Tabula_Sapiens_v1/GSM6058681_TabulaSapiens.h5ad
Tabula_Sapiens_v1/GSM6058681_TabulaSapiens_metadata.csv
HBCA_v1/HBCA_v1_all_neurons.h5ad
HBCA_v1/HBCA_v1_all_non_neuronal.h5ad
HBCA_v1/gene_metadata.csv                         # optional
PINNACLE/ontologies/cl-full.obo
PINNACLE/ontologies/bto.obo
```

`data_processing_bulk/hbca_supercluster_to_cl.csv` is bundled with the code. If the optional HBCA gene metadata is absent, the loader uses symbols already stored in the AnnData object or queries g:Profiler; the public data snapshot should include the resolved gene mapping and checksum so rebuilding does not depend on a live service.

## Inputs used in the study

- Tabula Sapiens v1 single-cell expression and metadata.
- Human Brain Cell Atlas expression matrices and gene metadata.
- ALS motor-neuron and astrocyte RNA-seq expression and sample metadata.
- The global PPI assembled from BioGRID, HuRI and the Menche interactome.
- Cell Ontology and BRENDA Tissue Ontology files.
- CellPhoneDB v3 ligand--receptor database.
- The protein-language-model embedding snapshots used by pretraining, plus their exact model identifiers and checksums.
- CORUM complex memberships.
- Open Targets therapeutic evidence and disease hierarchy data.
- DrugBank approved-human-drug target annotations.
- STRING v12.0 evidence used for the Parkinson disease analyses.

Exact source versions, accession identifiers, download dates, preprocessing outputs and checksums will be included in the manuscript data archive manifest.

The historical file named `gene_protein_embeddings_esm2_650M.plk` contains 2,560-dimensional vectors (SHA256 `8884ab0a6c31d3a6f0b8e345dbd9e7ce5c7f9b101c0be2b1d4167a453fa6a37d`), whereas ESM-2 t33/650M outputs 1,280 dimensions. The generator extracted layer 33 directly, so the artifact is consistent with t36/3B layer-33 features, but the original weights were not recorded. Do not label the public artifact as 650M until that checkpoint provenance is confirmed.

## What will be released

The public data archive should contain the minimum artifacts needed to reproduce the paper without repeating the expensive raw single-cell preprocessing:

1. The global PPI, 207 context-specific PPI edge lists, CCI edge list and metagraph.
2. Protein-language-model input embeddings used by pretraining.
3. Frozen train, validation and test splits.
4. The main ProtScape and comparison-model checkpoints.
5. Exported protein and context embeddings used by downstream tasks.
6. Processed CORUM and therapeutic-target label tables.
7. Source tables consumed by every manuscript figure.
8. A checksum and provenance manifest linking every artifact to its generating command and code version.
