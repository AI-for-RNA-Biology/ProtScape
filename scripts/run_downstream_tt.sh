#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if (( $# < 1 )); then
    echo "Usage: bash scripts/run_downstream_tt.sh <inference-model> [training options]" >&2
    exit 2
fi
INFERENCE_MODEL="$1"
shift
TRAINING_OPTIONS=("$@")

python -m pretraining.generate_esm2_embeddings
python -m pretraining.generate_prostt5_embeddings

TASKS=(
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

# Keep optimization and data splits the same for every readout.
train_model() {
    local model="$1"
    shift
    python -m downstream_tasks.run \
        --inference-model "$INFERENCE_MODEL" --task "$TASK" --model "$model" \
        --embedding-source esm --dataset-mode bulk \
        --lr 0.0001 --weight-decay 0.0001 --batch-size 512 \
        --seed 42 --train-selection-metric auprc "$@" "${TRAINING_OPTIONS[@]}"
}

for TASK in "${TASKS[@]}"; do
    train_model lr_ext_embed
    train_model lr_ext_embed --embedding-source prostt5
    train_model lr_hc_cell
    train_model lr_hc_cell_ext_embed

    for dropout in 0 0.2 0.4 0.6; do
        for model in abmil_hc_cell_gated_8 abmil_hc_cell_ext_embed_gated_8; do
            train_model "$model" --att-dim 256 --dropout "$dropout"
        done
    done
    for pmax in 0.2 0.3 0.4 0.5 0.6 0.7; do
        for model in abmil_hc_cell_gated_8_pdl abmil_hc_cell_ext_embed_gated_8_pdl; do
            train_model "$model" --att-dim 256 --dropout 0 --pdl-pmax "$pmax"
        done
    done
    python -m downstream_tasks.run --task "$TASK" --aggregate-only
done
