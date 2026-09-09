# ALS network rewiring

Motor-neuron PPI-module analysis comparing CTRL and VCP at days 22 and 35.
Run commands from the repository root; set paths in [`configs/paths.yaml`](../../../configs/paths.yaml).

## Prepare inputs

The release supplies the separate ALS graphs, main S2GAE checkpoint, protein embeddings
and sample-level expression matrix. For newly trained models, first generate embeddings
with `python -m pretraining.inference <checkpoint> --output-dir <directory>` and set
`als_rewiring_checkpoint` and `als_rewiring_inference_dir` to that matching pair.
Keep the genotype, day and nuclear/cytoplasmic graphs separate during inference.

Download the human [STRING v12](https://version-12-0.string-db.org/cgi/download) files
`9606.protein.info.v12.0.txt.gz`, `9606.protein.links.detailed.v12.0.txt.gz` and
`9606.protein.physical.links.v12.0.txt.gz`; set `string_protein_info`,
`string_links_detailed` and `string_links_physical` to their local paths.

```bash
python -m exploration.analysis.ALS_rewiring.prepare_matrix
python -m exploration.analysis.ALS_rewiring.prepare_neighborhood
```

The matrix generator scores all unordered pairs among ALS motor-neuron proteins
using concatenated S2GAE layer embeddings; pairs without both protein embeddings
in a context receive missing scores. It averages R155C/R191Q probabilities
only where both are available, retaining separate nuclear/cytoplasmic scores.
The rewiring analysis subsequently takes the union of high-confidence edges across
compartments. It does not merge graphs before inference.
The neighbourhood CSV instead describes **observed** PPI unions; it contains no
predicted edges. Select its focal gene in `config.py`.

Outputs follow the `als_rewiring_*` paths. The full paper-sized matrix has about
93 million rows (~16 GB); generate it on a GPU and allow ~30 GB of working space.
These derived CSVs do not need to be downloaded with the release.

For GO annotations, supply `human_GO_terms.csv` with columns
`GO_id,Description,Gene_Symbols`, or generate it using R packages `yaml`,
[`org.Hs.eg.db`](https://bioconductor.org/packages/org.Hs.eg.db/) and `GO.db`:

```r
install.packages(c("yaml", "BiocManager"))
BiocManager::install(c("org.Hs.eg.db", "GO.db"))
```

```bash
Rscript exploration/analysis/ALS_rewiring/prepare_go_terms.R
```

This exports direct GO-to-gene annotations, not ancestor-expanded annotations.
Keep the resulting table fixed within an analysis; annotation versions can change.

For expression plots, `als_rewiring_cyto_gene_expression_csv` points to the released
normalized, unlogged gene-by-sample matrix. New expression inputs use HGNC gene rows
and sample names such as `CTRL1_D22_Cytoplasmic_Wildtype` or
`MUT1_D22_Cytoplasmic_VCP-R155C`. Retain biological replicates: the analysis averages
`log2(x+1)` per sample, which cannot be recovered from the network-building context means.

## Run

Set thresholds and Leiden resolution settings in `config.py`, then run:

```bash
python -m exploration.analysis.ALS_rewiring_analysis

# Resume at clustering, or run individual stages
python -m exploration.analysis.ALS_rewiring_analysis --from 05
python -m exploration.analysis.ALS_rewiring_analysis --only 07 09
```

The workflow subsets the matrix, selects high-confidence edges, clusters and refines
modules, tests enrichment and plots networks. Stage 05b saves the initial partition
before selecting the refined partition used downstream. Stage 08 queries STRING
enrichment and requires network access. Outputs go to `<output_root>/analysis/ALS_rewiring/`.

After regenerating the full matrix, rerun from the start to refresh its day-specific
subsets and downstream analyses. Individual stage scripts also run from `scripts/`.
