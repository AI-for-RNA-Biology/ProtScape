#!/usr/bin/env bash
#SBATCH --job-name=protscape-global-eval
#SBATCH --account=a0204
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/aloistho/protscape/logs/%x_%j.out

set -euo pipefail

REPO_ROOT="${PROTSCAPE_ROOT:-/users/aloistho/projects/ProtScape}"
CONDA_ENV="${PROTSCAPE_CONDA_ENV:-/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae}"
PYTHON="${CONDA_ENV}/bin/python"
DATA_ROOT="${PROTSCAPE_DATA:-/iopsstor/scratch/cscs/aloistho/protscape/release-data}"
RUNS_ROOT="${PROTSCAPE_OUTPUT:-/capstor/scratch/cscs/aloistho/protscape/global-s2gae/runs}"
EVAL_DIR="${PROTSCAPE_EVAL:-/capstor/scratch/cscs/aloistho/protscape/global-s2gae/evaluation}"
SWEEP_CONFIG="${PROTSCAPE_SWEEP:-${REPO_ROOT}/configs/global_s2gae_sweep.yaml}"
LOG_ROOT="${PROTSCAPE_LOGS:-/iopsstor/scratch/cscs/aloistho/protscape/logs}"

test -x "${PYTHON}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}"
if ! git diff --quiet || ! git diff --cached --quiet \
    || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "Refusing to evaluate from a dirty worktree." >&2
    exit 1
fi
export PROTSCAPE_GIT_COMMIT="$(git rev-parse HEAD)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "${EVAL_DIR}")" "${LOG_ROOT}"
exec 9>"${EVAL_DIR}.lock"
if ! flock --nonblock 9; then
    echo "Another job is already evaluating ${EVAL_DIR}." >&2
    exit 1
fi

if [[ -f "${EVAL_DIR}/completed.json" ]]; then
    "${PYTHON}" - "${EVAL_DIR}" "${PROTSCAPE_GIT_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
expected_commit = sys.argv[2]
with (output_dir / "completed.json").open(encoding="utf-8") as handle:
    completion = json.load(handle)
with (output_dir / "summary.json").open(encoding="utf-8") as handle:
    summary = json.load(handle)
if not completion.get("test_evaluated"):
    raise SystemExit("Existing evaluation is not marked complete.")
if completion.get("git_commit") != expected_commit:
    raise SystemExit("Existing evaluation used another Git commit.")
if summary.get("git_commit") != expected_commit:
    raise SystemExit("Existing summary used another Git commit.")
PY
    echo "Evaluation already complete: ${EVAL_DIR}"
    exit 0
fi

srun --exact --nodes=1 --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --gpus-per-task=1 --gpu-bind=single:1 \
    "${PYTHON}" -m pretraining.evaluate_global_s2gae \
    --runs-root "${RUNS_ROOT}" \
    --sweep-config "${SWEEP_CONFIG}" \
    --networks-dir "${DATA_ROOT}/networks_bulk" \
    --esm2-embeddings "${DATA_ROOT}/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk" \
    --output-dir "${EVAL_DIR}" \
    --k-values 1,10,50,100,500 \
    --device cuda \
    >"${LOG_ROOT}/evaluation_${SLURM_JOB_ID}.log" 2>&1
