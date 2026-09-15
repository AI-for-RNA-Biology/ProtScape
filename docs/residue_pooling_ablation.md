# Residue pooling ablation (development branch)

## Current experiment: context-free ProtScape (supersedes contextual runs)

The user corrected the target to **CF ProtScape**. The contextual queue was
cancelled; its checkpoints and cache are preserved, not treated as CF results.
Use `configs/cf_residue_ablation.yaml` and
`python -m scripts.cscs.run_cf_residue_ablation` for the replacement. The
contextual protocol below is historical documentation only.

The CF ablation fixes the previously validation-selected ACM-RW backbone at
**512 hidden units, 2 layers, dropout 0.4**, with the existing S2GAE decoder
(512 × 2), directed mask 0.5, 1:1 BCE, full-global-graph updates, seed/split seed 0.
No CCI, tissue, uniformity or cell-embedding objective is introduced globally.
The grid is **26 runs**: mean (2), MLP (8), attention (8), SWE-Simple (8), with
the same two backbone LRs and pooler widths/LRs/reference sizes documented below.
Keep CF convergence control: **5,000 updates maximum, patience 200 updates,
minimum validation AUPRC improvement 0.0005**. Selection uses the absolute best
global validation AUPRC, not a test score. Reuse the previous CF BOS checkpoint
only after its graph, features and split fingerprints match the fresh release.

W&B is **online**, entity `cedricvincentcuaz`, project `pinnacle`, group
`cf_residue_pooling`; resource/smoke trials use `cf_residue_pooling_smoke`.
Each run retains a deterministic ID across checkpoint resumes. Pooler config,
train loss/AP/F1, validation metrics, best epoch/AP, update time and GPU memory
are logged. A failed W&B connection is an explicit job failure, not silent
disabled tracking.

The new queue reuses the completed residue cache without another ESM pass.
An initial four-GPU smoke stage measures one versus two concurrent fits per GPU
for each pooling family, tests export, CF LR and independent pooler downstream
training. Packing uses measured throughput and memory margins. Completed runs
are skipped; unfinished runs resume the saved model, optimizer and RNG state.
MLP's last linear layer is applied after mean (algebraically equivalent);
SWE-Simple's fixed transport features are cached across optimization steps.

Downstream uses the released fixed **LR / LR+ESM2** settings, appropriate for
one CF vector per protein (not an artificial ABMIL bag or zero cell vector).
All 15 TT diseases, CORUM and HPA37 remain included. Independent ESM-only
baselines remain as described below. This gives **408 downstream configurations**
(170 CF probes, 34 frozen ESM probes, 204 independent learned poolers), each
with five saved-fold fits. The previous 22-setting contextual sweep is cancelled.
PPI evaluation reports both global and per-Cell-PPI re-encoding on identical
1:k test banks; full-reference exports are exclusively for downstream tasks.

```bash
python -m scripts.cscs.run_cf_residue_ablation prepare NEW_ROOT PREVIOUS_CONTEXTUAL_ROOT
python -m scripts.cscs.run_cf_residue_ablation submit NEW_ROOT
```

## Historical contextual setup (cancelled)

The released ProtScape uses the existing ESM2 BOS generator. It is a valid,
unchanged reference; this experiment does **not** retrain it or modify that
generator. The four new contextual models replace only the protein input
pooling, before the existing GNN:

| Input | Definition | Initialization |
| --- | --- | --- |
| Mean | Mean of frozen residue vectors | No learned pooling parameters |
| MLP → mean | Residual bottleneck `R + W2 GELU(W1 R)` then mean | `W2=0`, starts at mean |
| Gated attention | `softmax(w[tanh(VR) * sigmoid(UR)])`, weighted sum of residues | `w=0`, starts at mean |
| SWE-Simple | Sorted/interpolated 1-D transport to a fixed reference, combined over reference points | Frozen random unit slicers and uniform reference; learned combination only |

Pooling is context-independent. The same protein has the same pooled vector
before the GNN; contextualization remains the responsibility of the existing
cell-PPI encoder. No cross-attention, transcriptomic conditioning, or new loss.

## Frozen inputs and splits

Use Zenodo record **22645081**, archive MD5
`05a39563ccb9ef638eba1739faff45fc`. Cache ESM2 3B **layer 33**, 2560 dimensions,
once in float16 after float32 inference. Exclude BOS/EOS/padding. For long
proteins use consecutive, nonoverlapping 1024-residue windows and keep all
residues. Every new pooling variant uses exactly this cache. This deterministic
long-sequence policy differs from the original generator's OOM-triggered BOS
chunk fallback; a BOS-versus-residue comparison therefore includes that caveat
for long sequences. The mean/MLP/attention comparisons have identical inputs.
The release contains duplicate gene/isoform records and ESM-supported `.`
positions. Preserve them and their order: the original GNN uses the first record
for a gene; the downstream feature dictionary uses the last. The new residue
cache mirrors both conventions rather than silently changing the proteins.

For ProtScape pretraining, standardize using fixed mean-vector statistics on the feature-covered
global interactome, matching the original feature-normalization convention.
Learned pooling receives these standardized residues. Deduplicate proteins in
each sampled batch; use activation checkpointing to limit GPU memory and a
memory-mapped residue bank. Each allocation stages one shared copy in node RAM
to avoid slow random storage reads; the manifest must match the persistent cache.
Validation caches
pooled protein features and invalidates that cache before further training.

TT (all 15 diseases) and CORUM use the release's saved cohorts and six folds,
unchanged. Fold 0 is test; the other five rotate as validation. Localization is
the preserved HPA 25.1 **37-label development extension**, with its existing
10,583-protein split, not a Zenodo/paper benchmark. File hashes are recorded.

## Search and selection

`configs/residue_ablation.yaml` contains the small grid:

- Backbone learning rate: **0.003, 0.01**.
- Learned-pooler width: **64, 128**; pooler learning rate: **0.0001, 0.001**.
- SWE-Simple reference points: **100, 200**; the same two pooler learning rates.
- **26 pretraining runs**: 2 mean, 8 MLP, 8 attention, 8 SWE-Simple; **300 epochs**, seed **0**.
- Fixed official contextual backbone: ACM-RW 512 × 3, dropout 0.4, GraphSAINT,
  original S2GAE mask/decoder/self-loop handling, CCI/tissue/uniformity objectives.
- Epoch and configuration selection: original **validation PPI AP + CCI AP**,
  separately for each pooling family. No test-based selection.

For each selected encoder and each downstream task, reuse the released readout
grid: contextual LR ± ESM; ABMIL ± ESM with dropout 0/0.2/0.4/0.6; ABMIL-PDL ±
ESM with pmax 0.2/0.3/0.4/0.5/0.6/0.7. This is 22 settings per encoder/task.
All use seed 42, learning rate and weight decay 1e-4, batch 512, up to 300 epochs,
patience 50 and validation AUPRC selection. The `+ESM` late-fusion feature stays
the **released BOS vector** for every encoder, to isolate the GNN input pooling.
Cell-feature choice also matches the corresponding released readout: CORUM
context-only ABMIL-PDL uses the pre-CCI pooled cell vector; the other five
readouts and TT use the canonical contextual cell vector. This is fixed across
pooling variants, not another search dimension.

All five ESM-only baselines are **independent of ProtScape pretraining**:

- `LR_ESM_BOS` and `LR_ESM_mean`: fixed ESM vectors → trained linear head.
- `ESM_MLP_linear` and `ESM_attention_linear`: frozen ESM residues → fresh
  pooler + linear head, trained jointly on each downstream training fold only.
  These are learned-pooling classifiers, not strictly linear probes. They reuse
  the pooling architecture, **never the ProtScape-trained weights or GNN**.
  Width 64/128 × learning rate 1e-4/1e-3: four settings per family/task. Use
  train-only mean-vector normalization, class weights and label-cluster sampling;
  fixed seed 42, AdamW weight decay 1e-4, batch 512, 300 epochs, patience 50.
  Select epochs and settings by validation AUPRC, never test scores.
- `ESM_SWE_simple_linear`: a separately initialized SWE-Simple combination vector
  and linear classifier, fitted only on downstream training folds. Frozen ESM,
  slicers and reference; reference points 100/200 × learning rate 1e-4/1e-3.

Total: 1,496 contextual readout settings + 34 frozen-feature LR settings + 204
independent learned-pooler settings + 12 missing BOS localization settings =
**1,746 downstream configurations**, each with five CV fits. Sequence baselines
run once per task/configuration, not once per contextual model. Sharing the frozen
ESM cache does not share ProtScape learning; the sequence baselines need no
selected pretraining checkpoint, even though the queue runs them downstream.
Existing BOS contextual TT/CORUM and localization results are reused. Only the
12 missing BOS localization ABMIL-PDL settings are additionally fitted, using
the released frozen encoder (no BOS pretraining).

The controlled PPI comparison recomputes 1:1 context-mean test AUPRC for the
released BOS model and four selected models, with identical query banks.
Only train+validation topology is available when predicting test pairs.

### SWE-Simple implementation

Use the SWE-Simple variant of [NaderiAlizadeh & Singh (2025)](https://doi.org/10.1093/bioadv/vbaf060),
with the [authors' implementation](https://github.com/navid-naderi/PLM_SWE/blob/b9491235444a463e4b906b5dc8fbfae0d6b95ead/model/architectures.py)
as a reference. There is **no full/trainable-slicer SWE configuration**. Set
`L=d=2560`; initialize frozen unit-normalized Gaussian slicers and a frozen
reference uniform on [-1, 1]. Only the shared length-m combination vector learns:
100 or 200 pooling parameters. Unlike MLP/attention, this does not start at mean.
The output stays 2560-dimensional, leaving the existing GNN unchanged.

Native Torch sort/linear interpolation handles unpadded variable-length proteins;
the interior quantile grid and linear tail extrapolation follow the authors' code.
Use the inverse reference permutation specified in paper Eq. 4 (the published code
uses the forward permutation; they coincide for our fixed, sorted reference).
The displacement sign follows the paper; reversing this overall sign can be
absorbed by the learned combination. Tests cover variable lengths, singleton
proteins, permutation invariance, frozen buffers, gradients and reloads.

## Unattended execution

One Python controller and one Slurm wrapper handle cache → eight short
end-to-end smoke runs → pretraining → validation selection/PPI evaluation/full
embedding export → downstream → `results.csv` / `RESULTS.md`.

Run the prepare command only after verifying/extracting the release and
downloading FAIR's pretrained ESM weights into `ROOT/torch/hub/checkpoints`:

```bash
python -m scripts.cscs.run_residue_ablation prepare ROOT RELEASE
python -m scripts.cscs.run_residue_ablation submit ROOT
```

The controller snapshots committed source and pins absolute data paths. Four
workers share a locked task queue on one four-GPU node; no GPU waits on downloads
or on another job's dependency. Completed tasks are never repeated. Time-limited
pretraining resumes the saved model, optimizer and RNG state. Independent
learned-pooler classifiers resume their last epoch and skip completed folds;
other interrupted downstream readout runs can restart, while completed runs
remain untouched.
At allocation end the wrapper automatically submits remaining work or the next
stage. A worker error stops stage advancement and writes `failures/*.json`.
Per-stage allocation caps prevent unbounded retries. `ACTIVE.json` records the
latest submitted job; `COMPLETE.json` is written only after all results and
cross-model split checks pass.
