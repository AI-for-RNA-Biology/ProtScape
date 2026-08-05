#!/usr/bin/env bash

set -euo pipefail

if (( $# < 1 )); then
    echo "Usage: bash scripts/run_downstream_corum.sh <inference-model> [extra arguments]"
    exit 2
fi

INFERENCE_MODEL="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DROPOUT_VALUES=(0 0.2 0.4 0.6)
PDL_PMAX_VALUES=(0.2 0.3 0.4 0.5 0.6 0.7)

# Sequence-only linear readouts.
python -m downstream_tasks.run \
    --inference-model "${INFERENCE_MODEL}" --task corum \
    --model lr_ext_embed --embedding-source esm --dataset-mode bulk \
    --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
    --seed 42 --train-selection-metric auprc "$@"

python -m downstream_tasks.run \
    --inference-model "${INFERENCE_MODEL}" --task corum \
    --model lr_ext_embed --embedding-source prostt5 --dataset-mode bulk \
    --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
    --seed 42 --train-selection-metric auprc "$@"

# Mean-pooled contextual linear readouts.
python -m downstream_tasks.run \
    --inference-model "${INFERENCE_MODEL}" --task corum \
    --model lr_hc_cell --embedding-source esm --dataset-mode bulk \
    --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
    --seed 42 --train-selection-metric auprc "$@"

python -m downstream_tasks.run \
    --inference-model "${INFERENCE_MODEL}" --task corum \
    --model lr_hc_cell_ext_embed --embedding-source esm --dataset-mode bulk \
    --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
    --seed 42 --train-selection-metric auprc "$@"

# Context-only and late-fusion ABMIL readouts.
for DROPOUT in "${DROPOUT_VALUES[@]}"; do
    python -m downstream_tasks.run \
        --inference-model "${INFERENCE_MODEL}" --task corum \
        --model abmil_hc_cell_gated_8 --embedding-source esm --dataset-mode bulk \
        --att-dim 256 --dropout "${DROPOUT}" \
        --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
        --seed 42 --train-selection-metric auprc "$@"

    python -m downstream_tasks.run \
        --inference-model "${INFERENCE_MODEL}" --task corum \
        --model abmil_hc_cell_ext_embed_gated_8 --embedding-source esm --dataset-mode bulk \
        --att-dim 256 --dropout "${DROPOUT}" \
        --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
        --seed 42 --train-selection-metric auprc "$@"
done

# Context-only and late-fusion ABMIL-PDL readouts.
for PDL_PMAX in "${PDL_PMAX_VALUES[@]}"; do
    python -m downstream_tasks.run \
        --inference-model "${INFERENCE_MODEL}" --task corum \
        --model abmil_hc_cell_gated_8_pdl --embedding-source esm --dataset-mode bulk \
        --att-dim 256 --dropout 0 --pdl-pmax "${PDL_PMAX}" \
        --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
        --seed 42 --train-selection-metric auprc "$@"

    python -m downstream_tasks.run \
        --inference-model "${INFERENCE_MODEL}" --task corum \
        --model abmil_hc_cell_ext_embed_gated_8_pdl --embedding-source esm --dataset-mode bulk \
        --att-dim 256 --dropout 0 --pdl-pmax "${PDL_PMAX}" \
        --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
        --seed 42 --train-selection-metric auprc "$@"
done
