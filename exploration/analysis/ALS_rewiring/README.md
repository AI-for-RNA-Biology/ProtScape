# ALS_rewiring

PPI-module rewiring analysis for the ALS motor-neuron S2GAE/STRING network
(CTRL vs. VCP, D22 vs. D35).

## Layout

```
exploration/analysis/ALS_rewiring_analysis.py   # entry point - run as `python -m exploration.analysis.ALS_rewiring_analysis`,
                                                  # Runs every ALS_rewiring/scripts/ stage in order (or a subset - see below).
exploration/analysis/ALS_rewiring/
    config.py                  # every path/threshold/constant, in one place
                                # (paths themselves come from configs/paths.yaml)
    io_utils.py                # loading raw CSVs + save/load helpers for intermediates
    scores.py                  # section 1: S2GAE-vs-STRING score validation
    high_confidence.py         # section 2: high-confidence new edges
    graph_build.py             # section 3: edge typing + giant component
    clustering.py              # section 4-5: Leiden resolution sweep + module assignment
    pathway_embedding.py       # GO-term pathway embedding (end of stage 05, and hierarchical recluster)
    module_enrichment.py       # section 5: per-module new-edge enrichment
    condition_enrichment.py    # section 6: per-module per-condition enrichment
    string_api.py              # section 7 (+9): STRING REST calls
    layout.py                  # shared network-layout helpers (section 8/9)
    plotting.py                # section 6 (plots) + section 8: all figures
    gene_expression.py         # per-condition mean gene expression
    edge_components.py         # section 9: new-edge connected components
    hierarchical_recluster.py  # stage 05b: hierarchical re-clustering (core, not optional)
    scripts/
        0_subset_d22.sh                         # stage 0: generates config.D22_CSV (idempotent, auto-run first)
        0_subset_d35.sh                         # stage 0: generates config.D35_CSV, same idea
        01_load_merge_data.py
        02_validate_s2gae_scores.py
        03_select_high_confidence_edges.py
        04_build_giant_graph.py
        05_cluster_leiden.py
        05b_hierarchical_recluster.py            # re-clusters stage 05's least intra-connected modules,
                                                  # then makes the refined partition canonical (see below)
        06_module_enrichment.py
        07_condition_enrichment.py
        08_string_enrichment.py
        09_visualize_giant_network.py
        10_edge_components.py
        neighborhood_by_context.py              # auto-run last, for config.NEIGHBORHOOD_GENE: per-context ego-network figures
        neighborhood_new_gene_expression.py      # auto-run last, same gene: expression overlay on the above
        _bootstrap.py                            # adds the ProtScape repo root to sys.path
```

Every function lives in the package; every stage script is a thin driver
that reads its inputs, calls into the package, and writes its outputs.
`ALS_rewiring_analysis.py` is the orchestrator, kept at the top level (next
to `corum_analysis.py`, `parkinson_target_analysis.py`, etc.) so it's
invoked the same way as every other analysis in this directory.

## Configuration

All data locations live in ProtScape's shared [`configs/paths.yaml`](../../../configs/paths.yaml),
under the `als_rewiring_*` keys:

- `als_rewiring_data_root` - directory holding the raw D22/D35 subset CSVs
- `als_rewiring_go_terms_csv` - GO-term annotation table for the pathway embedding
- `als_rewiring_cyto_gene_expression_csv` - per-condition mean gene expression matrix (Cytoplasmic samples)
- `als_rewiring_raw_matrix_csv` - full (pre-subsetting, all timepoints) S2GAE/STRING matrix `0_subset_d22.sh`/`0_subset_d35.sh` stream from
- `als_rewiring_neighborhood_edges_csv` - neighborhood-edges CSV for `scripts/neighborhood_by_context.py`/`neighborhood_new_gene_expression.py`'s default gene (`config.NEIGHBORHOOD_GENE`)

`config.py` reads these via `downstream_tasks.config.PATHS`, the same pattern
used elsewhere in `exploration/analysis/`. All other thresholds (Leiden
resolution sweep, high-confidence cutoff, FDR cutoffs, etc.) are set directly
in `config.py`.

## Running it

From the ProtScape repository root:

```bash
conda activate protscape
python -m exploration.analysis.ALS_rewiring_analysis                  # everything, stage 0 through 10 (incl. 05b)
python -m exploration.analysis.ALS_rewiring_analysis --from 05        # resume from stage 05 onward
python -m exploration.analysis.ALS_rewiring_analysis --only 07 09     # just these two
python -m exploration.analysis.ALS_rewiring_analysis --skip 02 08     # everything except these
                                                                        # (02 is diagnostic-only; 08 needs
                                                                        # network access to STRING and is slow)
```

Individual stage scripts (`01_load_merge_data.py`, ...,
`neighborhood_by_context.py`, etc.) can also be re-run on their own:

```bash
python exploration/analysis/ALS_rewiring/scripts/05_cluster_leiden.py --resolution 1.5   # skip the sweep, cluster at this value
python exploration/analysis/ALS_rewiring/scripts/09_visualize_giant_network.py           # just regenerate the network figures
```


## Changing parameters

Edit `config.py` (thresholds, resolution sweep grid, FDR cutoffs, etc.), then
re-run from whichever stage first depends on the changed value.