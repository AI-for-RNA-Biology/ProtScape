# Downstream tasks

The downstream pipeline trains linear, ABMIL and ABMIL-PDL models for CORUM
complex prediction, therapeutic-target prediction, protein localization, and
pathway prediction.

## Training inputs

Set these paths in `configs/paths.yaml`:

- `inference_root`: pretrained embedding directories.
- `corum_dataset_dir`: processed CORUM tables.
- `therapeutic_target_dataset_dir`: processed disease label tables.
- `esm2_embeddings` and `prostt5_embeddings`: sequence embeddings used by the sequence and late-fusion models.
- `output_root`: destination for checkpoints, predictions and metrics.

Each `<inference-model>` is a directory below `inference_root` containing `protein_embeddings.pt` and `cell_embeddings.pt`. When using newly generated embeddings, set `inference_root` to `<output_root>/inference`.

The CORUM and therapeutic-target shell scripts generate the configured ESM-2 and ProstT5 embedding files first if they are absent.

`corum_dataset_dir` must contain `corum_memberships_filtered.csv`; `therapeutic_target_dataset_dir` must contain the 15 `therapeutic_target_<DISEASE_ID>.csv` tables.

## Rebuilding the labels

To rebuild the processed labels, set:

| Config key | Required input |
|---|---|
| `corum_raw_json` | Frozen `corum_humanComplexes.json` snapshot from [CORUM](https://mips.helmholtz-muenchen.de/corum/download) |
| `therapeutic_target_evidence_dir` | [Open Targets Platform 24.03 ChEMBL evidence](https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/24.03/output/etl/json/evidence/sourceId=chembl/) |
| `therapeutic_target_drugbank_targets` | Frozen approved-drug target table, `all_approved_oct2022.csv`, from [DrugBank](https://go.drugbank.com/) |
| `global_ppi` | The same two-column HGNC-symbol interactome used for pretraining |

Therapeutic-target reconstruction additionally uses a frozen Open Targets
release. Both the 24.03 JSON layout (`diseases`, `searchTarget`,
`associationByDatatypeIndirect`) and the official Parquet layout used by newer
releases (`disease/disease.parquet`, `target/*.parquet`,
`association_by_datatype_indirect/*.parquet`) are supported. The indirect table
is used because disease associations include evidence propagated from ontology
descendants. Reconstruction does not query live APIs and does not force the
class balances reported in the paper.

After setting these paths, rebuild both datasets from the repository root:

```bash
python -m downstream_tasks.data_processing.corum_processing
python -m downstream_tasks.data_processing.therapeutic_target_processing \
  --evidence-dir /path/to/opentargets_24_03_chembl \
  --evidence-release-name 24.03 \
  --static-release-dir /path/to/opentargets_26_03 \
  --static-release-name 26.03 \
  --association-scope indirect \
  --force
```

On CSCS, the same frozen rebuild for both tasks is packaged as:

```bash
bash scripts/cscs/rebuild_downstream_labels.sh
```

Its input and output roots can be overridden with `PROTSCAPE_DATA` and
`PROTSCAPE_DOWNSTREAM_DATA`; an alternate frozen Open Targets snapshot can be
selected with `PROTSCAPE_OT_STATIC_RELEASE` and
`PROTSCAPE_OT_RELEASE_NAME`; the evidence release can be recorded with
`PROTSCAPE_OT_EVIDENCE_RELEASE`. The association table defaults to `indirect`
and can be overridden with `PROTSCAPE_OT_ASSOCIATION_SCOPE`.

Therapeutic-target labels are written to the configured
`therapeutic_target_dataset_dir`, or to `--output-dir` when supplied. The
directory includes a deterministic `therapeutic_target_manifest.json` recording
the ChEMBL evidence release, association release and raw input paths separately.
Pass `--force` to rebuild existing tables. Update `corum_dataset_dir` and
`therapeutic_target_dataset_dir` to the generated directories before training.

## Training

List the available model configurations:

```bash
python -m downstream_tasks.run --list-models
```

Run CORUM once for each pretrained embedding directory:

```bash
bash scripts/run_downstream_corum.sh <inference-model>

# Example using the released ProtScape embeddings
bash scripts/run_downstream_corum.sh s2gae_att_k1_fixed_do04_uni5e6
```

Run all 15 therapeutic-target disease areas:

```bash
bash scripts/run_downstream_tt.sh <inference-model>

# Example using the released ProtScape embeddings
bash scripts/run_downstream_tt.sh s2gae_att_k1_fixed_do04_uni5e6
```

The scripts evaluate the sequence-only linear baselines, contextual linear models, ABMIL models across dropout values 0, 0.2, 0.4 and 0.6, and ABMIL-PDL models across `pmax` values 0.2--0.7. Model selection uses validation AUPRC.

Protein localization and pathway tasks require an explicit frozen long-form
membership CSV through `--task-csv`; no mutable default dataset path is used.

All outputs are written below `<output_root>/downstream_tasks/`. Each run contains the five fold checkpoints, held-out predictions, training histories and summary metrics. Task-level summaries are generated automatically after each sweep.

### Context-free global-S2GAE baseline

The global baseline has one frozen vector per protein, so only linear probes are
meaningful; ABMIL over a single vector is identical to using that vector
directly. Export the validation-selected encoder and submit the downstream
matrix with the reviewed local commit:

```bash
commit="$(git rev-parse HEAD)"
export_job="$(sbatch --parsable \
  --export="ALL,PROTSCAPE_GIT_COMMIT=${commit}" \
  scripts/cscs/export_global_s2gae_embeddings.sbatch)"
sbatch --dependency="afterok:${export_job}" \
  --export="ALL,PROTSCAPE_GIT_COMMIT=${commit}" \
  scripts/cscs/run_global_s2gae_downstream.sbatch
```

Each four-GPU node runs four tasks concurrently. Every task trains the same
three probes on one shared six-fold split: global-S2GAE, global-S2GAE + ESM2,
and ESM2 alone. Fold 0 is held out for testing; folds 1--5 rotate as validation.
The fixed paper LR settings are AdamW with learning rate and weight decay
`1e-4`, batch size 512, at most 300 epochs and patience 50. Aggregate completed
runs with:

```bash
python -m downstream_tasks.aggregate_global_s2gae_results \
  --output-root /users/aloistho/projects/outputs/global_s2gae_grid500/downstream_tasks \
  --inference-model global_s2gae_grid500_full_reference
```

## Selected configurations

The validation-selected settings used in the paper are stored in:

- `configs/downstream/corum_selected_hyperparameters.csv`
- `configs/downstream/therapeutic_target_selected_hyperparameters.csv`

Retrain every distinct selected configuration with:

```bash
python -m downstream_tasks.run_selected corum
python -m downstream_tasks.run_selected therapeutic_targets
```

These runs are consumed directly by the analysis scripts. The full sweep scripts above remain available for repeating model selection from scratch.
