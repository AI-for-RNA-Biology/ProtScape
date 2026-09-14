#!/usr/bin/env bash
#SBATCH --job-name=context-rewiring
#SBATCH --account=a0204
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=450G
#SBATCH --time=01:00:00
#SBATCH --exclude=nid007611

set -euo pipefail
root=$1
panel=${2:-shared_edges}
cd "${root}/source"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
python_bin=/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae/bin/python
if [[ "${panel}" == all_unobserved && ! -f "${root}/BANK_COMPLETE.json" ]]; then
    "${python_bin}" -m pretraining.evaluate_context_rewiring prepare-all "${root}"
fi
pids=()
for shard in 0 1 2 3; do
    srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
        --cpus-per-task=16 --mem=100G --gpu-bind=single:1 \
        "${python_bin}" -m pretraining.evaluate_context_rewiring evaluate "${root}" --shard "${shard}" --panel "${panel}" \
        >"${root}/logs/shard_${shard}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if (( status != 0 )); then exit "${status}"; fi
if [[ "${panel}" == all_unobserved ]]; then
    pids=()
    for shard in 0 1 2 3; do
        srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
            --cpus-per-task=16 --mem=100G \
            "${python_bin}" -m pretraining.evaluate_context_rewiring summarize-all "${root}" --shard "${shard}" \
            >"${root}/logs/summary_${shard}.log" 2>&1 &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
    if (( status != 0 )); then exit "${status}"; fi
    "${python_bin}" -m pretraining.evaluate_context_rewiring finish-all "${root}"
else
    "${python_bin}" -m pretraining.evaluate_context_rewiring summarize "${root}"
fi
if [[ "${panel}" == unobserved || "${panel}" == all_unobserved ]]; then
    "${python_bin}" -m pretraining.evaluate_context_rewiring report "${root}" \
        --reference-root /users/aloistho/projects/outputs/context_edge_intersections
fi
