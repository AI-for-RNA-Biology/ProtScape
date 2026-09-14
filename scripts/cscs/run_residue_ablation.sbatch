#!/usr/bin/env bash
#SBATCH --account=a0204
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=450G
#SBATCH --time=12:00:00

set -euo pipefail
root=$1
stage=$2
allocation=$3
cd "${root}/source"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export WANDB_MODE=disabled WANDB_DISABLED=true
export TORCH_HOME="${root}/torch"
export PYTHONPATH="${root}/python-deps"
# Leave 20 minutes to close children, finalize manifests and submit the next job.
export RESIDUE_DEADLINE="$(( $(date +%s) + 42000 ))"
python_bin=/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae/bin/python
pids=()
for rank in 0 1 2 3; do
    srun --exclusive --nodes=1 --ntasks=1 --gpus-per-task=1 --gpus-per-node=1 \
        --cpus-per-task=16 --mem=100G --gpu-bind=single:1 \
        "${python_bin}" -m scripts.cscs.run_residue_ablation worker "${root}" "${stage}" \
        >"${root}/logs/${stage}_${SLURM_JOB_ID}_${rank}.worker.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
if [[ "${status}" == 0 ]]; then
    "${python_bin}" -m scripts.cscs.run_residue_ablation finish "${root}" "${stage}" "${allocation}"
fi
exit "${status}"
