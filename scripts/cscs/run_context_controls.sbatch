#!/usr/bin/env bash
#SBATCH --job-name=context-value
#SBATCH --account=a0204
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=450G
#SBATCH --time=01:00:00
#SBATCH --output=/users/aloistho/projects/outputs/context_value_ablation/logs/job_%j.out

set -euo pipefail
cd /users/aloistho/projects/ProtScape_dev
python_bin=/iopsstor/scratch/cscs/aloistho/protscape/conda-protscape-downstream/bin/python
output=/users/aloistho/projects/outputs/context_value_ablation
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
test -f "$output/inputs.json"
test -z "$(git status --porcelain)"
pids=()
for lane in 0 1 2 3; do
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=16 \
        --gpus-per-task=1 --gpu-bind=single:1 --mem=100G \
        "$python_bin" -m downstream_tasks.run_context_controls worker \
        --output "$output" --lane "$lane" \
        >"$output/logs/worker_${SLURM_JOB_ID}_${lane}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit "$status"; fi
"$python_bin" -m downstream_tasks.run_context_controls summarize --output "$output"
