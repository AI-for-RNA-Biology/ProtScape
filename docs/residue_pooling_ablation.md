# Residue pooling ablation (development branch)

The released ProtScape uses the existing ESM2 BOS generator. It is a valid,
unchanged reference; this experiment does **not** retrain it or modify that
generator. The three new contextual models replace only the protein input
pooling, before the existing GNN:

| Input | Definition | Initialization |
| --- | --- | --- |
| Mean | Mean of frozen residue vectors | No learned pooling parameters |
| MLP → mean | Residual bottleneck `R + W2 GELU(W1 R)` then mean | `W2=0`, starts at mean |
| Gated attention | `softmax(w[tanh(VR) * sigmoid(UR)])`, weighted sum of residues | `w=0`, starts at mean |

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

Standardize using the fixed mean-vector statistics on the feature-covered
global interactome, matching the original feature-normalization convention.
Learned pooling receives these standardized residues. Deduplicate proteins in
each sampled batch; use activation checkpointing to limit GPU memory and a
memory-mapped residue bank shared through the host page cache. Validation caches
pooled protein features and invalidates that cache before further training.

TT (all 15 diseases) and CORUM use the release's saved cohorts and six folds,
unchanged. Fold 0 is test; the other five rotate as validation. Localization is
the preserved HPA 25.1 **37-label development extension**, with its existing
10,583-protein split, not a Zenodo/paper benchmark. File hashes are recorded.

## Search and selection

`configs/residue_ablation.yaml` contains the small grid:

- Backbone learning rate: **0.003, 0.01**.
- Learned-pooler width: **64, 128**; pooler learning rate: **0.0001, 0.001**.
- **18 pretraining runs**: 2 mean, 8 MLP, 8 attention; **300 epochs**, seed **0**.
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

Add two **frozen ESM-only** linear baselines on each task: `LR_ESM_BOS` and
`LR_ESM_mean`. No learned-pooler-only LR baselines. Total: 1,156 downstream
readout runs, each with five CV fits. Unchanged sequence baselines run once per
task, not once per contextual model.
Existing BOS contextual TT/CORUM and localization results are reused. Only the
12 missing BOS localization ABMIL-PDL settings are additionally fitted, using
the released frozen encoder (no BOS pretraining).

The controlled PPI comparison recomputes 1:1 context-mean test AUPRC for the
released BOS model and three selected models, with identical query banks.
Only train+validation topology is available when predicting test pairs.

## Unattended execution

One Python controller and one Slurm wrapper handle cache → four short
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
pretraining resumes the saved model, optimizer and RNG state; interrupted
downstream readout runs can restart, while completed runs remain untouched.
At allocation end the wrapper automatically submits remaining work or the next
stage. A worker error stops stage advancement and writes `failures/*.json`.
Per-stage allocation caps prevent unbounded retries. `ACTIVE.json` records the
latest submitted job; `COMPLETE.json` is written only after all results and
cross-model split checks pass.
