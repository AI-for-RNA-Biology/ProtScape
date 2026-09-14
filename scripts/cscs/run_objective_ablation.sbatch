#!/usr/bin/env bash
#SBATCH --job-name=protscape-objectives
#SBATCH --account=a0204
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=450G
#SBATCH --time=12:00:00

set -euo pipefail
stage=$1
root=$2
cd "${root}/source"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export WANDB_MODE=disabled WANDB_DISABLED=true
unset PYTHONHOME
export PYTHONPATH="${root}/artifacts/python-deps"
python_bin=/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae/bin/python
pids=()
for arm in neither cci_only tissue_only both; do
    worker_stage=${stage}
    if [[ "${arm}" == both && "${stage}" != recover ]]; then
        if [[ "${stage}" == evaluate ]]; then
            continue
        fi
        worker_stage=evaluate
    fi
    srun --exclusive --nodes=1 --ntasks=1 --gpus-per-task=1 --gpus-per-node=1 \
        --cpus-per-task=16 --mem=100G --gpu-bind=single:1 \
        "${python_bin}" -m scripts.cscs.run_objective_ablation "${worker_stage}" "${root}" --arm "${arm}" \
        >"${root}/logs/${stage}_${arm}_${SLURM_JOB_ID}_worker.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
if [[ "${status}" == 0 && ( "${stage}" == evaluate || "${stage}" == recover ) ]]; then
    "${python_bin}" -m scripts.cscs.run_objective_ablation summarize "${root}"
fi
exit "${status}"
