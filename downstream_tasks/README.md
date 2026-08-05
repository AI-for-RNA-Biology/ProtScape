# Downstream tasks

This directory contains the shared training pipeline used for CORUM complex prediction and therapeutic-target prediction. Data, pretrained embeddings and output roots are read from `configs/paths.yaml`.

```bash
python -m downstream_tasks.run --list-models
```

```bash
bash scripts/run_downstream_corum.sh \
  <inference-model>

bash scripts/run_downstream_tt.sh \
  <inference-model>
```

The TT script loops over the 15 therapeutic-target disease areas reported in the paper; the CORUM script runs only CORUM. Both run the seven model types, with sequence-only LR evaluated separately using ESM2 and ProstT5: two contextual LR variants, two ABMIL variants and two ABMIL-PDL variants. Plain ABMIL is evaluated at dropout 0, 0.2, 0.4 and 0.6; PDL is evaluated at `pmax` 0.2--0.7 with dropout 0. Selection is by validation AUPRC.

Run either script once for each pretrained embedding folder being compared. The task IDs are listed explicitly in the TT script and `config.py`. All generated checkpoints, predictions and metrics go below `<output_root>/downstream_tasks/`.
