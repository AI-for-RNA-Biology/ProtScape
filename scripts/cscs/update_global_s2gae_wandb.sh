#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${PROTSCAPE_CONDA_ENV:-/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae}"
PYTHON="${CONDA_ENV}/bin/python"
REPO_ROOT="${PROTSCAPE_ROOT:-/users/aloistho/projects/ProtScape}"
DATA_ROOT="${PROTSCAPE_DATA:-/iopsstor/scratch/cscs/aloistho/protscape/release-data}"
RUNS_ROOT="${PROTSCAPE_OUTPUT:-/capstor/scratch/cscs/aloistho/protscape/global-s2gae/runs}"
EVAL_DIR="${PROTSCAPE_EVAL:-/capstor/scratch/cscs/aloistho/protscape/global-s2gae/evaluation}"
SWEEP_CONFIG="${PROTSCAPE_SWEEP:-${REPO_ROOT}/configs/global_s2gae_sweep.yaml}"

test -x "${PYTHON}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
cd "${REPO_ROOT}"
if ! git diff --quiet || ! git diff --cached --quiet \
    || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "Refusing to update W&B from a dirty worktree." >&2
    exit 1
fi
if [[ ! -f "${EVAL_DIR}/completed.json" || ! -f "${EVAL_DIR}/summary.json" ]]; then
    echo "The complete local evaluation is required before updating W&B." >&2
    exit 1
fi
export PROTSCAPE_GIT_COMMIT="$(git rev-parse HEAD)"
"${PYTHON}" \
    -m pretraining.evaluate_global_s2gae \
    --runs-root "${RUNS_ROOT}" \
    --sweep-config "${SWEEP_CONFIG}" \
    --networks-dir "${DATA_ROOT}/networks_bulk" \
    --esm2-embeddings "${DATA_ROOT}/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk" \
    --output-dir "${EVAL_DIR}" \
    --device cpu \
    --update-wandb
