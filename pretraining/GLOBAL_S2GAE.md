# Global-interactome S2GAE baseline

This experiment trains and evaluates one context-free protein interaction model.
It does not run context-wise reconstruction, protein-complex, therapeutic-target,
pathway, localisation, CCI, or other downstream tasks.

## Model

The model consumes the normalized ESM-2 matrix released with ProtScape. The only
trainable components are the existing ProtScape protein backbone
(`ACM_RandomWalk`, with concatenated jumping knowledge) and the existing
cross-layer S2GAE decoder. Training uses one global PPI graph. Cell, tissue,
metagraph, pooling, entropy, uniformity, and contextual prediction components are
not present.

## Released inputs and split

The loader uses the released global interactome, Cell-PPI files, cross-context
edge-count dictionary, and ESM-2 matrix. It does not regenerate networks or
sequence embeddings. The Cell-PPI files recover ProtScape's leakage-controlled
split and supply targets for the matched contextual evaluation; they are never
inputs to the context-free model.

After applying released ESM-2 coverage, the global graph contains 15,020 proteins
and 204,038 unique undirected reference interactions. Each of the 200,137 pairs
observed in at least one released Cell-PPI is assigned once globally, so a pair
that occurs in several contexts cannot cross splits. The split contains 160,109
training, 20,014 validation, and 20,014 test pairs. The remaining 3,901
reference-only pairs occur in no Cell-PPI and are training-only, giving 164,010
total training pairs.

| Stage | Encoder graph | Positive targets |
|---|---|---|
| Training | 164,010 training pairs | a newly masked subset of the training graph |
| Validation | training graph | 20,014 unique validation pairs, each scored once |
| Test | training plus validation graph | 20,014 unique test pairs, each scored once |

Validation and test negatives are target-corrupted global non-edges. Self-loops
and every known interaction in the complete released global PPI are excluded. A
fixed seeded bank of 500 negatives per positive is generated once and sliced so
the reported 1:1, 1:10, 1:50, 1:100, and 1:500 results are nested and directly
comparable. Test reports AP, macro-F1, accuracy, and AUROC over the complete
unique-pair target set. Model selection uses global validation AP at 1:1 only.

The primary runs preserve ProtScape's published directed masking (`dm`). Because
the global training graph is symmetric, this inherited procedure can leave the
reverse arc of a masked interaction available for message passing; the masking
helper also restores self-loops. Undirected masking (`um`) is retained only as a
sensitivity and is never eligible for primary selection.

## Sweep and execution

`configs/global_s2gae_sweep.yaml` defines the complete 54-point seed-0 encoder
grid: hidden width `{128, 256, 512}`, depth `{2, 3, 4}`, and dropout
`{0, 0.1, 0.2, 0.3, 0.4, 0.5}`. Every run uses 500 epochs; decoder, masking,
negative-sampling, optimizer, and split settings remain fixed. Four independent
training runs are packed onto each four-GPU Clariden node.

The submission helper copies the nine compatible 300-epoch runs from the old
run root into the isolated grid-500 run root and resumes them from epoch 301.
The old artifacts remain unchanged. The other 45 grid points start normally.

Production training logs to `cedricvincentcuaz/pinnacle`, group
`protscape_global_ppi_baseline`. Debug and unit-test runs keep W&B disabled and do
not create remote runs.

Install the pinned standalone Miniconda environment once; shell activation is not
required:

```bash
bash scripts/cscs/setup_global_s2gae_env.sh
/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae/bin/wandb login
```

Then submit the sweep from a clean reviewed commit:

```bash
bash scripts/cscs/submit_global_s2gae_sweep.sh
```

The wrapper verifies the released inputs, sweep manifest, reusable-run
compatibility, Git state, W&B access, and absence of another active sweep. Each
run checkpoints every epoch and uses a deterministic W&B run ID for safe resume.

After all 54 runs complete, evaluate the validation-selected checkpoint:

```bash
sbatch scripts/cscs/evaluate_global_s2gae.sbatch
```

The evaluator requires every completion manifest, ranks runs by `global_val_ap`,
and fixes the selected checkpoint before accessing test targets. It writes two
atomic evaluations below `projects/outputs/global_s2gae_grid500/evaluation/`:

- the unique-global diagnostic, where each held-out pair is scored once against
  global non-edges;
- the directly comparable Cell-PPI evaluation, using the same directed
  edge-context positives, cell-seeded nested negative banks, and unweighted
  macro average as the contextual ProtScape and PINNACLE curves.

The context-free encoder runs once on the global train-plus-validation topology;
its embeddings and decoder are frozen for every Cell-PPI target. The evaluator
also writes `context_ppi_pair_multiplicity.csv` and summary statistics to
quantify how often the same held-out pair appears across contexts.

After verifying the local evaluation, update the selected existing W&B run
without creating a new run:

```bash
bash scripts/cscs/update_global_s2gae_wandb.sh
```

## Comparison with ProtScape

Use `context_ppi_macro_test_metrics.csv` for the cross-model 1:k plot: it matches
the contextual models' test targets, negative sampling, and aggregation. The
unique-global table remains useful for measuring context-free link prediction,
but it answers a different question and must not be overlaid as a comparable
model curve.
