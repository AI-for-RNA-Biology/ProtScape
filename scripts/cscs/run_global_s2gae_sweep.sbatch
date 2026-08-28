#!/usr/bin/env bash
#SBATCH --job-name=protscape-global-sweep
#SBATCH --account=a0204
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/aloistho/protscape/logs/%x_%A_%a.out

set -euo pipefail

REPO_ROOT="${PROTSCAPE_ROOT:-/users/aloistho/projects/ProtScape}"
CONDA_ENV="${PROTSCAPE_CONDA_ENV:-/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae}"
PYTHON="${CONDA_ENV}/bin/python"
DATA_ROOT="${PROTSCAPE_DATA:-/iopsstor/scratch/cscs/aloistho/protscape/release-data}"
OUTPUT_ROOT="${PROTSCAPE_OUTPUT:-/capstor/scratch/cscs/aloistho/protscape/global-s2gae/runs}"
SWEEP_CONFIG="${PROTSCAPE_SWEEP:-${REPO_ROOT}/configs/global_s2gae_sweep.yaml}"
LOG_ROOT="${PROTSCAPE_LOGS:-/iopsstor/scratch/cscs/aloistho/protscape/logs}"
N_RUNS="${PROTSCAPE_N_RUNS:?submit through submit_global_s2gae_sweep.sh}"
RUNS_PER_NODE=4

test -x "${PYTHON}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}"
if [[ "$(git rev-parse HEAD)" != "${PROTSCAPE_GIT_COMMIT:?missing code version}" ]] \
    || ! git diff --quiet || ! git diff --cached --quiet \
    || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "The submitted reviewed Git state changed before job start." >&2
    exit 1
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export WANDB_CACHE_DIR="/iopsstor/scratch/cscs/aloistho/protscape/wandb-cache"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${WANDB_CACHE_DIR}"

pids=()
for offset in 0 1 2 3; do
    index=$((SLURM_ARRAY_TASK_ID * RUNS_PER_NODE + offset))
    if (( index >= N_RUNS )); then
        continue
    fi
    srun --exclusive --exact --nodes=1 --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
        --mem=100G \
        --gpus-per-task=1 --gpu-bind=single:1 \
        "${PYTHON}" -m pretraining.run_global_s2gae_sweep \
        --config "${SWEEP_CONFIG}" \
        --index "${index}" \
        --networks-dir "${DATA_ROOT}/networks_bulk" \
        --esm2-embeddings "${DATA_ROOT}/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk" \
        --output-root "${OUTPUT_ROOT}" \
        --wandb-mode online \
        --device cuda \
        >"${LOG_ROOT}/sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_run${index}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
