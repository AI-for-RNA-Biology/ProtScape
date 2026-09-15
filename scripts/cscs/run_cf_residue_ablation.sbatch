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
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset WANDB_DISABLED WANDB_MODE
export PYTHONPATH="${root}/python-deps"
export PROTSCAPE_GIT_COMMIT
PROTSCAPE_GIT_COMMIT=$(/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_commit"])' "${root}/inputs.json")
residue_node_cache=$(mktemp -d /dev/shm/cf-residues.XXXXXX)
cleanup_residues() {
    if [[ "${residue_node_cache}" == /dev/shm/cf-residues.* && -d "${residue_node_cache}" ]]; then
        rm -f -- "${residue_node_cache}/residues.npy" "${residue_node_cache}/manifest.json"
        rmdir -- "${residue_node_cache}"
    fi
}
trap cleanup_residues EXIT
if [[ "${stage}" != downstream ]]; then
    cp --reflink=never "${root}/residues/residues.npy" "${residue_node_cache}/residues.npy"
    cp "${root}/residues/manifest.json" "${residue_node_cache}/manifest.json"
    export PROTSCAPE_RESIDUE_CACHE="${residue_node_cache}"
fi
export RESIDUE_DEADLINE="$(( $(date +%s) + 42000 ))"
python_bin=/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae/bin/python
pids=()
for rank in 0 1 2 3; do
    srun --exclusive --nodes=1 --ntasks=1 --gpus-per-task=1 --gpus-per-node=1 \
        --cpus-per-task=16 --mem=100G --gpu-bind=single:1 \
        "${python_bin}" -m scripts.cscs.run_cf_residue_ablation gpu_worker "${root}" "${stage}" \
        >"${root}/logs/${stage}_${SLURM_JOB_ID}_${rank}.worker.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if [[ "${status}" == 0 ]]; then
    "${python_bin}" -m scripts.cscs.run_cf_residue_ablation finish "${root}" "${stage}" "${allocation}"
fi
exit "${status}"
