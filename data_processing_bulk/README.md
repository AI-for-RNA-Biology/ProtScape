# Bulk data processing

Raw input paths, the source `global_ppi`, the CellPhoneDB database and the external output root are set in `configs/paths.yaml`.

```bash
conda env create -f environment_scrna.yml
conda env create -f environment_cellphonedb.yml
conda activate protscape-scrna
bash data_processing_bulk/run_pipeline.sh all
```

The pipeline runs Steps 0--9: expression loading, pseudobulk construction, reliable and context-specific gene selection, context-specific PPIs, CellPhoneDB, the CCI network, the metagraph, merged graph export and QC. The ready-to-use graph bundle is written to `<output_root>/data_processing_bulk/networks_bulk/`.

Reliable-gene filtering preserves the paper build exactly: a two-component `BayesianGaussianMixture` (`random_state=0`, `max_iter=1000`) is fit to each log2(count + 1) profile, and the 0.99 quantile of the lower-mean component is used with an inclusive `>=` threshold and a fallback of 0.5.

For the ALS compartment-resolved contexts, CCI edges with a nuclear (`_nuc_`) endpoint are removed, including nuclear--cytoplasmic edges. Cytoplasmic--cytoplasmic CCI edges are retained; nuclear and cytoplasmic contexts remain connected to their tissue in the metagraph.

The exact graph snapshot used by the paper checkpoints will be distributed separately with checksums. It is not currently byte-identical to a fresh run of this pipeline.

Pretraining reads the independent `networks_bulk` path in `configs/paths.yaml`. Leave it pointed at the released paper bundle, or change it to the newly generated directory after rebuilding the data.
