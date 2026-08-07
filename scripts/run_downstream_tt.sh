#!/usr/bin/env bash

set -euo pipefail

if (( $# < 1 )); then
    echo "Usage: bash scripts/run_downstream_tt.sh <inference-model> [extra arguments]"
    exit 2
fi

INFERENCE_MODEL="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m pretraining.generate_esm2_embeddings

THERAPEUTIC_TASKS=(
    therapeutic_target_efo_0003767
    therapeutic_target_efo_0000685
    therapeutic_target_efo_0000305
    therapeutic_target_efo_1001207
    therapeutic_target_efo_0000571
    therapeutic_target_efo_0000676
    therapeutic_target_efo_0001361
    therapeutic_target_mondo_0005148
    therapeutic_target_mondo_0007915
    therapeutic_target_efo_0000274
    therapeutic_target_mondo_0004979
    therapeutic_target_efo_0000341
    therapeutic_target_efo_1001249
    therapeutic_target_efo_0003884
    therapeutic_target_mondo_0005180
)
DROPOUT_VALUES=(0 0.2 0.4 0.6)
PDL_PMAX_VALUES=(0.2 0.3 0.4 0.5 0.6 0.7)

for TASK in "${THERAPEUTIC_TASKS[@]}"; do
    # Sequence-only linear readouts.
    python -m downstream_tasks.run \
        --inference-model "${INFERENCE_MODEL}" --task "${TASK}" \
        --model lr_ext_embed --embedding-source esm --dataset-mode bulk \
        --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
        --seed 42 --train-selection-metric auprc "$@"

    python -m downstream_tasks.run \
        --inference-model "${INFERENCE_MODEL}" --task "${TASK}" \
        --model lr_ext_embed --embedding-source prostt5 --dataset-mode bulk \
        --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
        --seed 42 --train-selection-metric auprc "$@"

    # Mean-pooled contextual linear readouts.
    python -m downstream_tasks.run \
        --inference-model "${INFERENCE_MODEL}" --task "${TASK}" \
        --model lr_hc_cell --embedding-source esm --dataset-mode bulk \
        --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
        --seed 42 --train-selection-metric auprc "$@"

    python -m downstream_tasks.run \
        --inference-model "${INFERENCE_MODEL}" --task "${TASK}" \
        --model lr_hc_cell_ext_embed --embedding-source esm --dataset-mode bulk \
        --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
        --seed 42 --train-selection-metric auprc "$@"

    # Context-only and late-fusion ABMIL readouts.
    for DROPOUT in "${DROPOUT_VALUES[@]}"; do
        python -m downstream_tasks.run \
            --inference-model "${INFERENCE_MODEL}" --task "${TASK}" \
            --model abmil_hc_cell_gated_8 --embedding-source esm --dataset-mode bulk \
            --att-dim 256 --dropout "${DROPOUT}" \
            --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
            --seed 42 --train-selection-metric auprc "$@"

        python -m downstream_tasks.run \
            --inference-model "${INFERENCE_MODEL}" --task "${TASK}" \
            --model abmil_hc_cell_ext_embed_gated_8 --embedding-source esm --dataset-mode bulk \
            --att-dim 256 --dropout "${DROPOUT}" \
            --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
            --seed 42 --train-selection-metric auprc "$@"
    done

    # Context-only and late-fusion ABMIL-PDL readouts.
    for PDL_PMAX in "${PDL_PMAX_VALUES[@]}"; do
        python -m downstream_tasks.run \
            --inference-model "${INFERENCE_MODEL}" --task "${TASK}" \
            --model abmil_hc_cell_gated_8_pdl --embedding-source esm --dataset-mode bulk \
            --att-dim 256 --dropout 0 --pdl-pmax "${PDL_PMAX}" \
            --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
            --seed 42 --train-selection-metric auprc "$@"

        python -m downstream_tasks.run \
            --inference-model "${INFERENCE_MODEL}" --task "${TASK}" \
            --model abmil_hc_cell_ext_embed_gated_8_pdl --embedding-source esm --dataset-mode bulk \
            --att-dim 256 --dropout 0 --pdl-pmax "${PDL_PMAX}" \
            --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
            --seed 42 --train-selection-metric auprc "$@"
    done

    python -m downstream_tasks.run --task "${TASK}" --aggregate-only
done
