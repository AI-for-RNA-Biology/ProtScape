# Downstream tasks

The downstream pipeline trains linear, ABMIL and ABMIL-PDL models for CORUM complex prediction and therapeutic-target prediction.

## Training inputs

Set these paths in `configs/paths.yaml`:

- `inference_root`: pretrained embedding directories.
- `corum_dataset_dir`: processed CORUM tables.
- `therapeutic_target_dataset_dir`: processed disease label tables.
- `protein_sequences`: table used to generate ESM-2 features when needed.
- `esm2_embeddings` and `prostt5_embeddings`: sequence embeddings used by the sequence and late-fusion models.
- `output_root`: destination for checkpoints, predictions and metrics.

Each `<inference-model>` is a directory below `inference_root` containing `protein_embeddings.pt` and `cell_embeddings.pt`. When using newly generated embeddings, set `inference_root` to `<output_root>/inference`.

The CORUM and therapeutic-target shell scripts generate the configured ESM-2 embedding file first if it is absent.

The released processed labels are the default inputs. `corum_dataset_dir` must contain `corum_memberships_filtered.csv`; `therapeutic_target_dataset_dir` must contain the 15 `therapeutic_target_<DISEASE_ID>.csv` tables.

## Rebuilding the labels

Released CORUM and therapeutic-target tables can be used directly. To rebuild them, set:

| Config key | Required input |
|---|---|
| `corum_raw_json` | Frozen `corum_humanComplexes.json` snapshot from [CORUM](https://mips.helmholtz-muenchen.de/corum/download) |
| `therapeutic_target_evidence_dir` | [Open Targets Platform 24.03 ChEMBL evidence](https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/24.03/output/etl/json/evidence/sourceId=chembl/) |
| `therapeutic_target_drugbank_targets` | Frozen approved-drug target table, `all_approved_oct2022.csv`, from [DrugBank](https://go.drugbank.com/) |
| `global_ppi` | The same two-column HGNC-symbol interactome used for pretraining |

The companion release contains the processed benchmark tables, the CORUM snapshot and the frozen DrugBank target table. Download the Open Targets evidence above only when rebuilding the labels.

After setting these paths, rebuild both datasets from the repository root:

```bash
python -m downstream_tasks.data_processing.corum_processing
python -m downstream_tasks.data_processing.therapeutic_target_processing
```

By default, the rebuilt labels are written below `<output_root>/downstream_tasks/data/`. Update `corum_dataset_dir` and `therapeutic_target_dataset_dir` to those generated directories before training.

## Training

List the available model configurations:

```bash
python -m downstream_tasks.run --list-models
```

Run CORUM once for each pretrained embedding directory:

```bash
bash scripts/run_downstream_corum.sh <inference-model>

# Example for <inference_root>/protscape_main/
bash scripts/run_downstream_corum.sh protscape_main
```

Run all 15 therapeutic-target disease areas:

```bash
bash scripts/run_downstream_tt.sh <inference-model>

# Example for <inference_root>/protscape_main/
bash scripts/run_downstream_tt.sh protscape_main
```

The scripts evaluate the sequence-only linear baselines, contextual linear models, ABMIL models across dropout values 0, 0.2, 0.4 and 0.6, and ABMIL-PDL models across `pmax` values 0.2--0.7. Model selection uses validation AUPRC.

All outputs are written below `<output_root>/downstream_tasks/`. Each run contains the five fold checkpoints, held-out predictions, training histories and summary metrics. Task-level summaries are generated automatically after each sweep.

The companion release also provides the selected retraining parameters in `models/downstream/corum/selected_hyperparameters.csv` and `models/downstream/therapeutic_targets/selected_hyperparameters.csv`.
