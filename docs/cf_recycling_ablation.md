# CF graph-guided residue re-pooling

Focused replacement for the retired partner-query panel. Keep the original
mean, residue-wise MLP, gated attention and SWE-Simple results as historical
DM results; do not treat those as matched controls for a UM experiment.
No Light Attention, neighbour-query, dispersion, post-fusion or Cross-BoM runs
are added. The earlier PMA-style block used separate key/value projections;
it is historical, not the exact one-pass control below.

## Eight families

| Family | Computation |
| --- | --- |
| Mean | Frozen mean residues → ACM |
| MLP | Mean of residue-wise residual MLP outputs → ACM (not MLP of the mean) |
| Gated attention | Existing scalar gated residue pooling → ACM |
| SWE-Simple | Existing fixed slicers/reference, learned combination → ACM |
| A: slots4 | Four learned queries read projected residues → ACM |
| B: recycle_memory | A → graph-conditioned query update → read the first four slots → same ACM |
| C: recycle_residues | A → graph-conditioned query update → re-read original projected residues → same ACM |
| D: recycle_no_graph | C with graph projection replaced by zero in the query update |

Frozen ESM layer 33 residues are standardized with the existing fixed mean
statistics. One shared bias-free projection maps residues to 128 dimensions.
Attention is scaled dot-product, one head, four shared learned initial queries;
the **same projected residues serve as both keys and values**. Concatenated slots
are projected back to the input dimension as a residual on frozen sequence mean.
This output projection starts at zero. The query updater concatenates each first
slot with a projection of the first ACM **JK-concatenated output** and applies a
shared 256→128→128 GELU MLP. Its final layer starts at zero.

B/C/D have identical parameters. Attention/input/output projections and ACM are
shared between passes, not separately initialized copies. Gradients flow through
the first slots and, for B/C, the first graph output. D deliberately has no loss
gradient through its first ACM output. It still executes the first pass so shared
batch-normalization updates and graph-pass count match B/C. Ordinary shared ACM
batch normalization is retained: two updates per training forward for B/C/D.
The unchanged S2GAE decoder consumes **second-pass per-layer outputs**; the
export is still one JK embedding per protein, never one embedding per pair.

Only frozen residues are cached across updates. Projected residues are recomputed
in activation-checkpointed, length-bucketed batches; first slots are shared only
within the current forward. There is no stop-gradient, detached feedback, or
graph-derived persistent cache. The entire sequence is used, without subsampling.

## Masking, selection and paired seeds

Every model in this comparison uses UM: remove both orientations of each masked
non-self target. The exact same visible adjacency is passed to **both** ACM calls.
Self-loops are permitted for message passing, never reconstruction targets.
Validation/test pairs are absent from training adjacency. Validation uses the
train graph; PPI test inference uses train+validation, not the full reference.
Downstream-only embedding export uses the full reference graph, as before.

Fixed ACM 512×2, dropout .4; decoder 512×2, dropout 0; BCE at 1:1, mask .5.
The development split/negative-bank seed stays 0 for all training seeds.

- Seed-0 LR grid: backbone {.003, .01} × learned pooler {.0001, .001}.
- MLP/gated bottleneck 64; SWE reference 100 (informed by the previous sweep).
- A/B/C/D: four slots of width 128. No depth/head/slot-count sweep.
- 30 seed-0 configurations, including two mean learning rates.
- Reuse four exactly matching completed UM mean/gated configurations from the
  retired panel; preserve their original checkpoint/source/W&B provenance.
- Select one configuration per family by global validation AUPRC at 1:1;
  run those settings with seeds 1 and 2 (16 further fits, **no seed selection**).
- Thus **42 new fits**, not 46 reruns. Seed-0 baselines are referenced read-only.
- Cap 5,000 full-batch updates, validation patience 200, min-delta .0005.
  Save the absolute-best validation checkpoint, not necessarily the final update.

The three seeds quantify optimization variability, not uncertainty from new data
splits. Historical baseline choices were informed by earlier validation results;
this is not a claim of equally exhaustive architecture tuning for every family.

## Evaluation and reporting

For all 24 selected checkpoints, evaluate identical PPI test queries at
1:{1,10,50,100,500}, with global and per-Cell-PPI encoding. Report AUPRC/F1
macro-averaged across contexts separately from the global unique-pair test.
Final test metrics are uploaded to each source W&B run's summary under `test/`.
They do not affect model selection. New run group: `cf_recycling_pairmasked`
in `cedricvincentcuaz/pinnacle`; smoke runs use a separate `_smoke` group.

Frozen LR and LR+released ESM BOS readouts use all 15 TT diseases, CORUM, and
the preserved HPA37 development task. Cohorts and saved folds remain unchanged;
readout seed 42/settings are fixed across pretraining seeds. This is 816 readout
configurations (408 queue tasks, five-fold fits). Independent ESM-only baselines
from the original pipeline are retained, not retrained by this panel.

Report mean and sample SD across the three pretraining seeds, paired C−B/C−D
differences, and selected update/parameter count/median warm train+validation
step time/peak reserved GPU memory. Warm step time excludes checkpoint writing
and startup. Do not call folds independent pretraining seeds.

An improvement over A alone does not establish the benefit of graph-guided
residue access. C must improve over both B and D, with downstream evidence.

## Running unattended

Use the existing controller, not a second orchestration system:

```bash
python -m scripts.cscs.run_cf_residue_ablation prepare NEW_ROOT ORIGINAL_CF_ROOT recycle
python -m scripts.cscs.run_cf_residue_ablation submit NEW_ROOT
```

Eight full-graph smoke checks gate the queue: train → seed repeats → PPI test
and export → downstream. One frozen residue bank per GPU; four concurrent
training jobs per allocation. Two frozen-vector readout processes per GPU.
Checkpoint resume, allocation caps and failure gates remain in place. Outputs
are under `projects/outputs/cf_recycling` through the storage partition.

## Literature scope

Learned-query set pooling: [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html).
Repeated access to original inputs: [Perceiver](https://proceedings.mlr.press/v139/jaegle21a.html).
The graph-conditioned recycling combination here is a proposed PPI adaptation,
not a reproduction of either full model or a verified novelty/SOTA claim.
