# Pair-masked CF partner-pooling panel

This is a separate experiment, not a change to the running 26-configuration
sequence-pooling sweep. Only this panel removes both orientations of masked
training pairs (`um`). Self-loops remain in ACM message passing but are not
reconstruction targets or observed partner queries. Validation/test pairs retain
the existing released splits and are absent from training topology.

## Models and controls

| Pooler | Definition |
| --- | --- |
| Mean | Matched pair-masked frozen residue-mean reference |
| Gated attention | Matched pair-masked unconditioned gated reference |
| Partner query | Gated residue score + scaled partner-query/key compatibility; mean of partner-conditioned focal-residue summaries |
| Self query | Exact same query/key module using the focal protein's sequence mean |
| Post-pooling fusion | Gated summary + bottleneck MLP of [summary, mean partner descriptor]; approximately matched Q/K parameter budget |
| Partner dispersion | Partner-view mean + zero-output-initialized bottleneck MLP of partner-view population SD |
| Partner mean-MLP | Identical parameterization to dispersion, but the MLP receives the mean; capacity control |
| Four-slot partner queries | Four learned low-dimensional sequence summaries per partner query the focal residues; mean within partner, then across partners |
| Four-query pooling (PMA-style) | Four learned queries pool the focal sequence; concatenate low-dimensional summaries, project back to D as a zero-initialized residual on the mean; no neighbours in the pooler |

The last two borrow learned-query compression, not the complete Set Transformer.
The PMA-style pooler has no neighbour input, but **ACM still uses topology**.
There is no additional Light Attention, Cross-BoM, Pool PaRTI, interface loss,
cell vector, RNA input, or iterative residue/edge architecture in this panel.

All values selected by partner attention are the focal protein's own residues.
Standardized frozen sequence means supply partner descriptors; layer-normalized
descriptors and residues supply queries/keys. The query projection starts at zero
and keys start randomly, so direct partner/self/dispersion models initially match
gated pooling while retaining a query gradient. The four slot seeds start distinct.
Isolated proteins fall back to gated pooling. The dispersion calculation uses
centred moments, excludes padded queries, and is zero for one partner.

## Graph access and efficiency

`GlobalS2GAE.encode` passes the same currently visible graph to both pooling and
ACM. Partner poolers refuse directed-mask training. No candidate edge is appended
to the neighbour lists. Queries use unique non-self visible neighbours, uniformly
subsampled without replacement to at most **16 per protein during training**.
A private, checkpointed sampling-step counter leaves masking/dropout RNG streams
unchanged. At evaluation **all permitted neighbours** are used in query chunks
of 32, including when the model is re-encoded on each Cell-PPI.

Sequence and degree buckets bound padding; activation checkpointing bounds
training memory. Only frozen residue inputs are cached across updates. Learned
four-slot summaries are shared within a forward pass, not across optimization
steps; graph-dependent pooled outputs are never reused across different graphs.

## Search and selection

`configs/cf_partner_ablation.yaml` fixes ACM-RW 512 × 2, dropout 0.4; existing
512 × 2 S2GAE decoder, BCE, mask ratio 0.5; seed/split seed 0. Frozen ESM2 layer
33 cache/extraction and sequence normalization are unchanged.

- Backbone LR: 0.003, 0.01.
- Query/bottleneck width: 64, 128; pooler LR: 0.0001, 0.001.
- One attention head; four slots where applicable; no head/slot-count sweep.
- **66 new pretraining configurations**: 2 mean, 8 gated, 7 × 8 additions/controls.
- At most 5,000 global updates; validation patience 200, min-delta 0.0005.
- Absolute-best global validation AUPRC selects epochs and configurations within
  each family. W&B online group: `cf_partner_pooling_pairmasked` in
  `cedricvincentcuaz/pinnacle`; smoke fits use a separate `_smoke` group.

All nine selected encoders run PPI 1:k (global and per-cell encoding), and frozen
global-vector LR ± the released BOS ESM2 vector on all 15 TT diseases, CORUM,
and preserved HPA37: **306 downstream configurations**, five saved-fold fits each.
Independent sequence baselines from the original pipeline are not duplicated.
Do not attribute a comparison between this UM panel and the original DM sweep
solely to pooling; the matched mean/gated controls are the primary references.

## Unattended queue

```bash
python -m scripts.cscs.run_cf_residue_ablation prepare NEW_ROOT EXISTING_CF_ROOT partner
python -m scripts.cscs.run_cf_residue_ablation submit NEW_ROOT
```

The existing controller/wrapper is reused, with a separate immutable source
snapshot, output directory, queues and W&B group. Nine full-graph, largest-width
smokes validate training gradients, pair-mask provenance, memory, full-neighbour
export and downstream LR before the queue advances. Pretraining/evaluation use
one GPU-resident residue bank per GPU; downstream can run two frozen-vector LR
fits per GPU. Atomic epoch restart and failure-gated continuations are retained.
No checkpoints or jobs from the original experiment are replaced or cancelled.
