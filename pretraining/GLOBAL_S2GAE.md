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
sequence embeddings. The Cell-PPI files are used only to recover ProtScape's
leakage-controlled split; they are never model inputs and are not evaluated.

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

`configs/global_s2gae_sweep.yaml` defines ten primary ACM configurations over
hidden width, depth, and dropout. It also defines two non-selection sensitivities:
undirected masking at seed 0 and the anchor configuration at seed 1. Four
independent training runs are packed onto each four-GPU Clariden node.

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

The wrapper verifies the released inputs, sweep manifest, Git state, W&B access,
and absence of another active sweep. Each run checkpoints every epoch and uses a
deterministic W&B run ID for safe resume.

After all ten primary runs complete, evaluate the validation-selected checkpoint:

```bash
sbatch scripts/cscs/evaluate_global_s2gae.sbatch
```

The evaluator requires every primary completion manifest, ranks runs by
`global_val_ap`, fixes the selected checkpoint before accessing test targets, and
writes the unique-global test results atomically. Test pairs are evaluated once;
there is no context macro, context sharding, or downstream embedding export.

After verifying the local evaluation, update the selected existing W&B run
without creating a new run:

```bash
bash scripts/cscs/update_global_s2gae_wandb.sh
```

## Comparison with ProtScape

The unique-global result weights every held-out protein pair once. Published
ProtScape PPI numbers instead score occurrences within individual Cell-PPIs and
macro-average across contexts. Those values are not directly comparable. A fair
numerical comparison requires evaluating the released ProtScape checkpoints on
this same unique-global split, topology, and negative bank; otherwise the global
baseline must be reported as a separate pretraining diagnostic.
