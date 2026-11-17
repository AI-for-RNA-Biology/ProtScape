# Pretraining

Train ProtScape or the adapted PINNACLE models on cell-specific PPI graphs and a cell/tissue metagraph.

Set the network, sequence-embedding and output paths in `configs/paths.yaml`. Existing files are used directly.

```bash
# Main ProtScape model; other model names are in configs/pretraining.yaml.
bash scripts/run_pretraining.sh
python -m pretraining.run_config pinnacle_esm_acm

# Generate both ESM-2 representations, or train with residue means.
python -m pretraining.generate_esm2_embeddings --pooling both
python -m pretraining.run_config protscape --esm2-pooling mean

# ProstT5 sequence baseline.
python -m pretraining.generate_prostt5_embeddings
```

ESM-2 uses layer 33 of `esm2_t36_3B_UR50D` (2,560 features). BOS is the default; mean pooling excludes the BOS/EOS tokens. The two files are `esm2_embeddings` and `esm2_mean_embeddings` in the path config. Long sequences fall back to chunks if GPU memory runs out.

PPI graphs and S2GAE masks are undirected. `--s2gae-mask-type dm` selects directed masking. Training options can follow the configuration name, for example `--epochs 10 --device cpu`.

Models and metrics go to `<output_root>/pretraining/`. Use `best_model_state_dict.pt` for inference:

```bash
python -m pretraining.inference /path/to/model_state_dict.pt \
    --output-dir /path/to/embeddings
```

Inference saves protein and cell embeddings plus `mappings.pkl`. ProtScape also saves pooled cell embeddings and tissue predictions; PINNACLE saves the full metagraph embeddings. Set downstream embedding paths to this directory.

## Architecture-ablation embeddings

Before training the selected downstream ablations, run inference for each checkpoint below into `<inference_root>/<directory>`:

```bash
python -m pretraining.inference <checkpoint_root>/<checkpoint> --output-dir <inference_root>/<directory>
```

| Checkpoint | Directory |
|---|---|
| `protscape_gae_uniformity_state_dict.pt` | `gae_att_fixed_do06_ep300_uni5e-5` |
| `protscape_gae_virtual_node_state_dict.pt` | `gae_vn_fixed_do04_ep300` |
| `protscape_gae_virtual_node_uniformity_state_dict.pt` | `gae_vn_fixed_do04_ep300_uni5e-5` |
| `protscape_gae_learned_virtual_node_state_dict.pt` | `gae_learnedvn_fixed_do06_ep300` |
| `protscape_gae_learned_virtual_node_uniformity_state_dict.pt` | `gae_learnedvn_fixed_do06_ep300_uni5e-5` |
| `protscape_no_uniformity_state_dict.pt` | `s2gae_att_k1_fixed_do04` |
