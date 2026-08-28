#!/usr/bin/env bash
#SBATCH --job-name=protscape-global-downstream
#SBATCH --account=a0204
#SBATCH --partition=normal
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/aloistho/protscape/logs/%x_%A_%a.out

set -euo pipefail

REPO_ROOT="${PROTSCAPE_ROOT:-/users/aloistho/projects/ProtScape}"
CONDA_ENV="${PROTSCAPE_CONDA_ENV:-/iopsstor/scratch/cscs/aloistho/protscape/conda-protscape-downstream}"
PYTHON="${CONDA_ENV}/bin/python"
DATA_ROOT="${PROTSCAPE_DATA:-/iopsstor/scratch/cscs/aloistho/protscape/release-data}"
DOWNSTREAM_DATA_ROOT="${PROTSCAPE_DOWNSTREAM_DATA:-/iopsstor/scratch/cscs/aloistho/protscape/downstream-data}"
INFERENCE_ROOT="${PROTSCAPE_INFERENCE:-/capstor/scratch/cscs/aloistho/protscape/global-s2gae/inference}"
INFERENCE_MODEL="${PROTSCAPE_INFERENCE_MODEL:-global_s2gae_full_reference}"
GLOBAL_INFERENCE="${INFERENCE_ROOT}/${INFERENCE_MODEL}"
OUTPUT_ROOT="${PROTSCAPE_DOWNSTREAM_OUTPUT:-/capstor/scratch/cscs/aloistho/protscape/global-s2gae/downstream_tasks}"
LOG_ROOT="${PROTSCAPE_LOGS:-/iopsstor/scratch/cscs/aloistho/protscape/logs}"
ESM2_EMBEDDINGS="${DATA_ROOT}/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk"
PPI_EDGELISTS="${DATA_ROOT}/networks_bulk/ppi_edgelists"
EXPECTED_COMMIT="${PROTSCAPE_GIT_COMMIT:?submit with the reviewed Git commit}"
PACK_SIZE=4

TASKS=(
    corum
    therapeutic_target_efo_0003767
    therapeutic_target_efo_0000685
    therapeutic_target_efo_0000305
    therapeutic_target_efo_1001207
    therapeutic_target_efo_0000571
    therapeutic_target_efo_0000676
    therapeutic_target_efo_0001361
    therapeutic_target_mondo_0005148
    therapeutic_target_mondo_0007915
    therapeutic_target_efo_0000274
    therapeutic_target_mondo_0004979
    therapeutic_target_efo_0000341
    therapeutic_target_efo_1001249
    therapeutic_target_efo_0003884
    therapeutic_target_mondo_0005180
)
LABELS=(
    "${DOWNSTREAM_DATA_ROOT}/processed/corum/corum_memberships_filtered.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_EFO_0003767.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_EFO_0000685.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_EFO_0000305.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_EFO_1001207.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_EFO_0000571.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_EFO_0000676.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_EFO_0001361.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_MONDO_0005148.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_MONDO_0007915.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_EFO_0000274.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_MONDO_0004979.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_EFO_0000341.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_EFO_1001249.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_EFO_0003884.csv"
    "${DOWNSTREAM_DATA_ROOT}/processed/therapeutic_target/therapeutic_target_MONDO_0005180.csv"
)

test -x "${PYTHON}"
test -f "${GLOBAL_INFERENCE}/protein_embeddings.pt"
test -f "${GLOBAL_INFERENCE}/manifest.json"
test -f "${ESM2_EMBEDDINGS}"
test -d "${PPI_EDGELISTS}"
if (( ${#TASKS[@]} != 16 || ${#LABELS[@]} != ${#TASKS[@]} )); then
    echo "The packed task manifest must contain exactly 16 aligned units." >&2
    exit 1
fi
if [[ "$(find "${PPI_EDGELISTS}" -maxdepth 1 -type f -name '*.txt' | wc -l)" -ne 207 ]]; then
    echo "Expected exactly 207 released Cell-PPI edgelists." >&2
    exit 1
fi
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export WANDB_MODE=disabled

cd "${REPO_ROOT}"
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]] \
    || ! git diff --quiet || ! git diff --cached --quiet \
    || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "The submitted reviewed Git state changed before job start." >&2
    exit 1
fi

"${PYTHON}" - "${GLOBAL_INFERENCE}" "${EXPECTED_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

import torch

from pretraining.export_global_s2gae_embeddings import (
    _sha256_file,
    _validate_existing_export,
)

inference = Path(sys.argv[1])
expected_commit = sys.argv[2]
with (inference / "manifest.json").open(encoding="utf-8") as handle:
    manifest = json.load(handle)
expected = {
    "format_version": 1,
    "model_type": "global_s2gae",
    "embedding_scope": "global",
    "embedding_topology": "full_reference",
    "representation": "encoder_jk_concat",
    "downstream_only": True,
    "ppi_test_compatible": False,
}
mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
if mismatched:
    raise SystemExit("Invalid global export metadata: " + ", ".join(mismatched))
if manifest.get("export_git_commit") != expected_commit:
    raise SystemExit("The global export was not created by the submitted Git commit.")
checkpoint = Path(manifest["checkpoint"])
if not checkpoint.is_file():
    raise SystemExit(f"Global export checkpoint not found: {checkpoint}")
checkpoint_payload = torch.load(
    checkpoint,
    map_location="cpu",
    mmap=True,
    weights_only=True,
)
_validate_existing_export(
    inference,
    manifest,
    _sha256_file(checkpoint),
    checkpoint_payload,
)
PY

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

run_unit() {
    local index="$1"
    local task="${TASKS[index]}"
    local label="${LABELS[index]}"
    local lock_path="${OUTPUT_ROOT}/.${task}.lock"
    local log_path="${LOG_ROOT}/downstream_${SLURM_ARRAY_JOB_ID}_${task}.log"

    exec {lock_fd}>"${lock_path}"
    if ! flock --nonblock "${lock_fd}"; then
        echo "Another job is running downstream task ${task}." >&2
        return 1
    fi

    srun --exclusive --exact --nodes=1 --ntasks=1 \
        --gpus-per-task=1 --cpus-per-gpu=16 --mem=100G \
        --gpu-bind=single:1 \
        "${PYTHON}" -m downstream_tasks.run \
        --inference-model "${INFERENCE_MODEL}" \
        --global-inference "${GLOBAL_INFERENCE}" \
        --esm2-embeddings "${ESM2_EMBEDDINGS}" \
        --output-root "${OUTPUT_ROOT}" \
        --context-ppi-edgelists "${PPI_EDGELISTS}" \
        --task "${task}" \
        --task-csv "${label}" \
        --model lr_global,lr_global_ext_embed,lr_ext_embed \
        --embedding-source esm \
        --dataset-mode bulk \
        --force \
        --lr 0.0001 \
        --weight-decay 0.0001 \
        --batch-size 512 \
        --seed 42 \
        --train-selection-metric auprc \
        >"${log_path}" 2>&1
}

first_index=$((SLURM_ARRAY_TASK_ID * PACK_SIZE))
if (( first_index < 0 || first_index + PACK_SIZE > ${#TASKS[@]} )); then
    echo "Invalid array index ${SLURM_ARRAY_TASK_ID}; expected 0-3." >&2
    exit 1
fi

pids=()
indices=()
for offset in 0 1 2 3; do
    index=$((first_index + offset))
    run_unit "${index}" &
    pids+=("$!")
    indices+=("${index}")
done

status=0
for position in "${!pids[@]}"; do
    if ! wait "${pids[position]}"; then
        echo "Downstream task failed: ${TASKS[indices[position]]}" >&2
        status=1
    fi
done
exit "${status}"
