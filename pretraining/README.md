# Pretraining and inference

`train_protscape.py` trains the hierarchical factored ProtScape models. `train_pinnacle.py` trains the adapted PINNACLE baselines.

## Inputs

Set these paths in `configs/paths.yaml`:

- `networks_bulk`: processed graph bundle containing the global PPI, context-specific PPIs, metagraph and edge-count dictionary.
- `esm2_embeddings`: protein sequence embeddings used as model input.
- `prostt5_embeddings`: ProstT5 features used by the downstream sequence baseline.
- `output_root`: destination for training and inference outputs.

The graph bundle and checkpoint must belong to the same dataset snapshot. The companion release includes the exact ESM-2 and ProstT5 feature matrices used in the paper; use those matrices for checkpoint-equivalent results.

Optional regeneration requires a `protein_sequences` table with `gene_name` and `fasta_seq` columns.

Generate the ESM-2 features with:

```bash
python -m pretraining.generate_esm2_embeddings
```

The script uses the BOS token from layer 33 of `esm2_t36_3B_UR50D`, producing the 2,560-dimensional inputs expected by the released checkpoints. It first embeds each complete sequence; after a CUDA out-of-memory error, it averages BOS embeddings from 1,024-residue chunks. The model weights are downloaded automatically.

Generate the 1,024-dimensional ProstT5 features used by the downstream sequence baseline with:

```bash
python -m pretraining.generate_prostt5_embeddings
```

This script uses the historical `Rostlab/ProstT5` revision, filters proteins to the configured global PPI, mean-pools residue representations and length-weights 1,000-residue chunks for proteins longer than 1,500 residues.

The pretraining and downstream entry points run the required generator when a configured feature file is absent. Regeneration uses the same model and pooling definitions as the historical scripts, but the released matrices remain the exact inputs used for the reported models.

## Training

The context-free global-interactome S2GAE ablation, including its released
leakage-controlled split, unique-pair evaluation, ACM sweep, and CSCS launchers,
is documented in [`GLOBAL_S2GAE.md`](GLOBAL_S2GAE.md). This experiment is limited
to global PPI pretraining and link prediction; it does not run context-wise or
downstream evaluations.

From the repository root, run the main ProtScape configuration with:

```bash
bash scripts/run_pretraining.sh
```

The remaining ProtScape loss, pooling and uniformity configurations and the adapted PINNACLE configurations are listed in `configs/pretraining.yaml`. Run any named configuration directly, for example:

```bash
python -m pretraining.run_config protscape_phuber
```

Each entry records the Python module and complete model arguments. Training uses seed 0 and disabled Weights & Biases logging by default.

Training outputs are written below `<output_root>/pretraining/`. Each run keeps its full restart checkpoints and also writes `best_model_state_dict.pt`, the portable checkpoint used for inference.

## Inference

```bash
python -m pretraining.inference \
    /path/to/ProtScape_release/models/pretraining/protscape_main_state_dict.pt
```

By default, inference writes to `<output_root>/inference/<checkpoint-name>/`. A different directory can be selected with `--output-dir`.

ProtScape exports:

```text
protein_embeddings.pt
cell_embeddings.pt
cell_embeddings_before_pool.pt
tissue_predictions.pt
mappings.pkl
```

`cell_embeddings.pt` contains the final post-CCI cell representations used by new downstream runs. The historical filename `cell_embeddings_before_pool.pt` is retained for compatibility with existing analyses; it contains the pooled cell representation before CCI refinement.

Adapted PINNACLE inference exports protein, cell and full-metagraph embeddings with their mappings. Inference reconstructs each metagraph relation from its corresponding edge type.
