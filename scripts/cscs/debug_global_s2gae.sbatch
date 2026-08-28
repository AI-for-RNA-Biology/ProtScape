#!/usr/bin/env bash
#SBATCH --job-name=protscape-global-debug
#SBATCH --account=a0204
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/aloistho/protscape/logs/%x_%j.out

set -euo pipefail

REPO_ROOT="${PROTSCAPE_ROOT:-/users/aloistho/projects/ProtScape}"
CONDA_ENV="${PROTSCAPE_CONDA_ENV:-/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae}"
PYTHON="${CONDA_ENV}/bin/python"
DATA_ROOT="${PROTSCAPE_DATA:-/iopsstor/scratch/cscs/aloistho/protscape/release-data}"
OUTPUT_ROOT="${PROTSCAPE_DEBUG_OUTPUT:-/iopsstor/scratch/cscs/aloistho/protscape/debug-runs}"

test -x "${PYTHON}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

hidden_dims=(512 128 256 512)
num_layers=(3 3 3 5)
pids=()
for index in 0 1 2 3; do
    srun --exclusive --exact --nodes=1 --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
        --mem=100G \
        --gpus-per-task=1 --gpu-bind=single:1 \
        "${PYTHON}" -m pretraining.train_global_s2gae \
        --networks-dir "${DATA_ROOT}/networks_bulk" \
        --esm2-embeddings "${DATA_ROOT}/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk" \
        --output-root "${OUTPUT_ROOT}" \
        --run-name "global_s2gae_v3_debug_${SLURM_JOB_ID}_${index}" \
        --experiment-role sensitivity \
        --hidden-dim "${hidden_dims[index]}" \
        --num-layers "${num_layers[index]}" \
        --dropout 0.4 \
        --epochs 1 \
        --mask-type dm \
        --wandb-mode disabled \
        --device cuda &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
