# Pretraining and inference

`train_protscape.py` trains the hierarchical factored ProtScape models. `train_pinnacle.py` trains the adapted PINNACLE baselines.

## Inputs

Set these paths in `configs/paths.yaml`:

- `networks_bulk`: processed graph bundle containing the global PPI, context-specific PPIs, metagraph and edge-count dictionary.
- `esm2_embeddings`: protein sequence embeddings used as model input.
- `output_root`: destination for training and inference outputs.

The graph bundle and checkpoint must belong to the same dataset snapshot.

Released checkpoints require the matching protein-embedding artifact; do not substitute another embedding model based only on the historical filename.

## Training

From the repository root, run the main ProtScape configuration with:

```bash
bash scripts/run_pretraining.sh
```

The remaining ProtScape loss, pooling and uniformity configurations and the adapted PINNACLE configurations are listed in `configs/pretraining.yaml`. Each entry gives the Python module and its complete arguments.

Training outputs are written below `<output_root>/pretraining/`. Each run keeps its full restart checkpoints and also writes `best_model_state_dict.pt`, the portable checkpoint used for inference.

## Inference

```bash
python -m pretraining.inference /path/to/best_model_state_dict.pt
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
