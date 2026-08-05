# Pretraining

`train_protscape.py` trains the hierarchical factored ProtScape models. `train_pinnacle.py` trains the adapted PINNACLE comparison models.

```bash
bash scripts/run_pretraining.sh
```

The script shows the complete main-model command directly. The remaining paper ablation and PINNACLE commands are recorded together in `configs/pretraining.yaml`; the processed graph bundle is selected by `networks_bulk` in `configs/paths.yaml`.

The two minibatch modules are both required: `minibatch_factored_utils.py` is used by ProtScape, while `minibatch_utils.py` is used by the PINNACLE baselines.

Run inference from a released portable ProtScape or adapted PINNACLE checkpoint with:

```bash
python -m pretraining.inference /path/to/checkpoint.pt
```

This reads the processed graph bundle selected by `networks_bulk` in `configs/paths.yaml` and writes `protein_embeddings.pt`, `cell_embeddings.pt` and mappings under `<output_root>/inference/<checkpoint-name>`. ProtScape additionally writes tissue predictions; adapted PINNACLE writes the complete metagraph embeddings. Use `--output-dir` to select another output directory.

To train downstream models from newly generated embeddings, set `inference_root` in `configs/paths.yaml` to `<output_root>/inference` and pass the checkpoint stem as `<inference-model>` to the downstream scripts.
