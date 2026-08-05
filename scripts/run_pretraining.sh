#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Main ProtScape model used in the paper.
python -m pretraining.train_protscape \
    ACM_RandomWalk concat 512 0.4 3 1 attention ESM2 0 1 512 2 0 \
    --dataset-mode bulk \
    --epochs 300 \
    --checkpointing true \
    --k-negatives 1 \
    --use-metagraph true \
    --loader graphsaint \
    --split-mode global \
    --lr 0.01 \
    --batch-size 64 \
    --weighted-ppi-loss false \
    --ppi-loss bce \
    --s2gae-mask-ratio 0.5 \
    --s2gae-mask-type dm \
    --s2gae-decoder-type cross_layer \
    --s2gae-decoder-dropout 0 \
    --s2gae-loss-weight 1 \
    --metagraph-loss-weight 1 \
    --uniformity-enabled true \
    --uniformity-lambda-reg 0.00005 \
    --uniformity-t 2 \
    --uniformity-dim 0 \
    --seed 0 \
    --wandb-mode disabled \
    "$@"
