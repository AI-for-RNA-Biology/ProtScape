# Downstream tasks

The downstream pipeline trains linear, ABMIL and ABMIL-PDL models for CORUM complex prediction and therapeutic-target prediction.

## Training inputs

Set these paths in `configs/paths.yaml`:

- `inference_root`: pretrained embedding directories.
- `corum_dataset_dir`: processed CORUM tables.
- `therapeutic_target_dataset_dir`: processed disease label tables.
- `esm2_embeddings` and `prostt5_embeddings`: sequence embeddings used by the sequence and late-fusion models.
- `output_root`: destination for checkpoints, predictions and metrics.

Each `<inference-model>` is a directory below `inference_root` containing `protein_embeddings.pt` and `cell_embeddings.pt`. When using newly generated embeddings, set `inference_root` to `<output_root>/inference`.

The released processed labels are the canonical paper inputs. `corum_dataset_dir` must contain `corum_memberships_filtered.csv`; `therapeutic_target_dataset_dir` must contain the 15 `therapeutic_target_<DISEASE_ID>.csv` tables.

## Rebuilding the labels

Raw inputs are configured separately from the processed training tables:

| Config key | Required input |
|---|---|
| `corum_raw_json` | Frozen `corum_humanComplexes.json` snapshot from [CORUM](https://mips.helmholtz-muenchen.de/corum/download) |
| `therapeutic_target_evidence_dir` | Complete Open Targets Platform 24.03 ChEMBL evidence directory as line-delimited `.json` or Parquet files |
| `therapeutic_target_drugbank_targets` | October 2022 approved-drug target table, `all_approved_oct2022.csv` |
| `global_ppi` | The same two-column HGNC-symbol interactome used for pretraining |

The DrugBank table must contain `Species`, `Gene Name` and `GenAtlas ID`. It is an access-controlled input that must be obtained under an appropriate [DrugBank licence](https://go.drugbank.com/releases) and supplied locally; this repository does not download it. The Open Targets 24.03 data remain available from the [Open Targets archive](https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/24.03/).

After setting these paths, rebuild both datasets from the repository root:

```bash
python -m downstream_tasks.data_processing.corum_processing
python -m downstream_tasks.data_processing.therapeutic_target_processing
```

By default, the rebuilt labels are written below `<output_root>/downstream_tasks/data/`. Update `corum_dataset_dir` and `therapeutic_target_dataset_dir` to those generated directories before training.

The therapeutic-target rebuild uses the frozen Open Targets 24.03 ChEMBL evidence but queries the current Open Targets API for disease descendants and negative-set exclusions, UniProt and Ensembl for identifier mapping, and EBI OLS only when `--descendants-source efo` is selected. A later rebuild can therefore differ from the released snapshot.

Therapeutic-target positives have phase 3 or later evidence, or completed phase 2 evidence, for the root disease or its descendants. Negatives are approved-human DrugBank targets without a non-literature Open Targets association for the root disease. Both classes are restricted to the global-PPI proteins, and positives are excluded from negatives.

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
