#!/usr/bin/env bash
#SBATCH --job-name=protscape-global-export
#SBATCH --account=a0204
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=32
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=01:00:00
#SBATCH --output=/iopsstor/scratch/cscs/aloistho/protscape/logs/%x_%j.out

set -euo pipefail

REPO_ROOT="${PROTSCAPE_ROOT:-/users/aloistho/projects/ProtScape}"
CONDA_ENV="${PROTSCAPE_CONDA_ENV:-/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae}"
PYTHON="${CONDA_ENV}/bin/python"
DATA_ROOT="${PROTSCAPE_DATA:-/iopsstor/scratch/cscs/aloistho/protscape/release-data}"
EVAL_DIR="${PROTSCAPE_EVAL:-/capstor/scratch/cscs/aloistho/protscape/global-s2gae/evaluation}"
INFERENCE_ROOT="${PROTSCAPE_INFERENCE:-/capstor/scratch/cscs/aloistho/protscape/global-s2gae/inference}"
INFERENCE_MODEL="${PROTSCAPE_INFERENCE_MODEL:-global_s2gae_full_reference}"
OUTPUT_DIR="${INFERENCE_ROOT}/${INFERENCE_MODEL}"
LOG_ROOT="${PROTSCAPE_LOGS:-/iopsstor/scratch/cscs/aloistho/protscape/logs}"
ESM2_EMBEDDINGS="${DATA_ROOT}/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk"
NETWORKS_DIR="${DATA_ROOT}/networks_bulk"
EXPECTED_COMMIT="${PROTSCAPE_GIT_COMMIT:?submit with the reviewed Git commit}"

test -x "${PYTHON}"
test -f "${EVAL_DIR}/summary.json"
test -f "${EVAL_DIR}/completed.json"
test -f "${NETWORKS_DIR}/global_ppi_edgelist.txt"
test -f "${NETWORKS_DIR}/count_edge_dict.pkl"
test -f "${ESM2_EMBEDDINGS}"

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export WANDB_MODE=disabled

cd "${REPO_ROOT}"
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]] \
    || ! git diff --quiet || ! git diff --cached --quiet \
    || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "The submitted reviewed Git state changed before job start." >&2
    exit 1
fi

CHECKPOINT="$("${PYTHON}" - "${EVAL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

evaluation = Path(sys.argv[1])
with (evaluation / "summary.json").open(encoding="utf-8") as handle:
    summary = json.load(handle)
with (evaluation / "completed.json").open(encoding="utf-8") as handle:
    completion = json.load(handle)
if not completion.get("test_evaluated"):
    raise SystemExit("The selected-checkpoint evaluation is not complete.")
if completion.get("checkpoint") != summary.get("checkpoint"):
    raise SystemExit("Evaluation completion and summary select different checkpoints.")
checkpoint = Path(summary["checkpoint"])
if not checkpoint.is_file():
    raise SystemExit(f"Selected checkpoint not found: {checkpoint}")
print(checkpoint)
PY
)"
test -f "${CHECKPOINT}"

mkdir -p "${INFERENCE_ROOT}" "${LOG_ROOT}"
exec 9>"${OUTPUT_DIR}.lock"
if ! flock --nonblock 9; then
    echo "Another job is exporting ${OUTPUT_DIR}." >&2
    exit 1
fi

srun --exclusive --exact --nodes=1 --ntasks=1 \
    --gpus-per-task=1 --cpus-per-gpu=32 --mem=100G \
    --gpu-bind=single:1 \
    "${PYTHON}" -m pretraining.export_global_s2gae_embeddings \
    "${CHECKPOINT}" \
    --networks-dir "${NETWORKS_DIR}" \
    --esm2-embeddings "${ESM2_EMBEDDINGS}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    >"${LOG_ROOT}/global_export_${SLURM_JOB_ID}.log" 2>&1

echo "Global embeddings: ${OUTPUT_DIR}"
