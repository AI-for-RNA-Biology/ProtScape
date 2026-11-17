# Downstream tasks

Predict CORUM complex membership or therapeutic targets for 15 diseases using linear probes, attention pooling (ABMIL), or ABMIL with progressive dropout (PDL).

## Run

Set the embedding, dataset and output paths in [`configs/paths.yaml`](../configs/paths.yaml). The supplied configuration points to the shared data. Each contextual embedding directory contains `protein_embeddings.pt` and `cell_embeddings.pt`; `esm2_embeddings` selects the sequence features (BOS or residue mean).

First export the [architecture-ablation embeddings](../pretraining/README.md#architecture-ablation-embeddings) if needed. Then train the validation-selected configurations:

```bash
python -m downstream_tasks.run_selected therapeutic_targets
python -m downstream_tasks.run_selected corum
```

The settings are in [`configs/downstream/`](../configs/downstream/). To repeat model selection for one embedding directory:

```bash
bash scripts/run_downstream_tt.sh s2gae_att_k1_fixed_do04_uni5e6
bash scripts/run_downstream_corum.sh s2gae_att_k1_fixed_do04_uni5e6
```

All models use the dataset's saved protein cohort and folds: fold 0 is the test set; folds 1–5 rotate through validation. Checkpoints are selected by validation AUPRC. Predictions, metrics and training curves are saved under `<output_root>/downstream_tasks/`.

## Build datasets

**Therapeutic targets.** Positive proteins have phase III/IV, approval, or completed phase-II clinical evidence for a disease or its descendants. Negatives are approved human DrugBank targets without non-literature Open Targets associations to that disease. Both groups are restricted to the global interactome.

Download one [Open Targets release](https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/) and use it for all input tables:

```bash
bash downstream_tasks/data_processing/download_opentargets.sh 24.03
python -m downstream_tasks.data_processing.therapeutic_target_processing
```

Set `therapeutic_target_ot_release` and `therapeutic_target_ot_root` to match the download. Release `26.03` is also supported. The DrugBank input is `therapeutic_target_drugbank_targets`.

**CORUM.** Human complexes with 10–300 interactome proteins, after removing highly overlapping complexes. Set `corum_raw_json` to the CORUM 4.1 snapshot, then run:

```bash
python -m downstream_tasks.data_processing.corum_processing
```

The builders write CSVs under `<output_root>/downstream_tasks/data/`. Set `therapeutic_target_dataset_dir` or `corum_dataset_dir` to the new folder. If building all datasets, prepare their shared cohorts and folds:

```bash
python -m downstream_tasks.prepare_splits --inference-model s2gae_att_k1_fixed_do04_uni5e6
```

For one disease, build and prepare only that task, then train directly:

```bash
python -m downstream_tasks.data_processing.therapeutic_target_processing --diseases MONDO_0005180
python -m downstream_tasks.prepare_splits --task therapeutic_target_mondo_0005180 --inference-model s2gae_att_k1_fixed_do04_uni5e6
python -m downstream_tasks.run --task therapeutic_target_mondo_0005180 --inference-model s2gae_att_k1_fixed_do04_uni5e6 --model abmil_hc_cell_ext_embed_gated_8_pdl --pdl-pmax 0.5
```

For a new disease, add its Open Targets ID to `THERAPEUTIC_TARGET_IDS` in [`config.py`](config.py) and replace the IDs above. Existing partitions are reused; use a separate dataset folder when changing labels or splits.
