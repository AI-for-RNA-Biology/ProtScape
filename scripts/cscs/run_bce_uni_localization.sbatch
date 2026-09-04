#!/usr/bin/env bash
#SBATCH --job-name=protscape-bce-uni-loca
#SBATCH --account=a0204
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/aloistho/protscape/topology-baselines/logs/%x_%j.out

set -euo pipefail

REPO_ROOT="${PROTSCAPE_ROOT:-/users/aloistho/projects/ProtScape}"
CONDA_ENV="${PROTSCAPE_CONDA_ENV:-/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae}"
SOURCE_INFERENCE_ROOT="${PROTSCAPE_INFERENCE_ROOT:-/users/aloistho/projects/ProtScape_release/embeddings}"
SOURCE_ESM="${PROTSCAPE_ESM:-/iopsstor/scratch/cscs/aloistho/protscape/release-data/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk}"
LABELS="${PROTSCAPE_TASK_CSV:-/iopsstor/scratch/cscs/aloistho/protscape/topology-baselines/data/processed/hpa_v25_1_localization_per_location_confidence_min50.csv}"
OUTPUT_ROOT="${PROTSCAPE_OUTPUT_ROOT:-/capstor/scratch/cscs/aloistho/protscape/topology-baselines/results/protscape_downstream}"
LOG_ROOT="${PROTSCAPE_LOG_ROOT:-/iopsstor/scratch/cscs/aloistho/protscape/topology-baselines/logs}"
RUN_ID="${PROTSCAPE_RUN_ID:-bce_uni_hpa_v25_1_per_location_confidence_min50}"
RUN_TOPOLOGY="${PROTSCAPE_RUN_TOPOLOGY:-1}"
MODEL_RESULT_ROOT="${OUTPUT_ROOT}/protein_localization/${RUN_ID}"
TOPOLOGY_OUTPUT_ROOT="${PROTSCAPE_TOPOLOGY_OUTPUT_ROOT:-/capstor/scratch/cscs/aloistho/protscape/topology-baselines/results/topology/protein_localization/hpa_v25_1_per_location_confidence_min50}"
NETWORK_ROOT="${PROTSCAPE_NETWORK_ROOT:-/iopsstor/scratch/cscs/aloistho/protscape/release-data/networks_bulk}"
TOPOLOGY_OUTPUT="${TOPOLOGY_OUTPUT_ROOT}/global_binary_k_hop"
EXPECTED_FINGERPRINT="${PROTSCAPE_CODE_FINGERPRINT:?submit with a reviewed code fingerprint}"
EMBEDDING_MODEL="s2gae_att_k1_fixed_do04_uni5e6"
SOURCE_INFERENCE="${SOURCE_INFERENCE_ROOT}/${EMBEDDING_MODEL}"
PYTHON="${CONDA_ENV}/bin/python"

case "${RUN_TOPOLOGY}" in
    0|1) ;;
    *)
        echo "PROTSCAPE_RUN_TOPOLOGY must be 0 or 1, got: ${RUN_TOPOLOGY}" >&2
        exit 2
        ;;
esac

UNIT_IDS=(
    abmil_do0p0
    abmil_do0p2
    abmil_do0p4
    abmil_do0p6
    lr_esm
    lr_hc_cell
    lr_hc_cell_esm
)
UNIT_MODELS=(
    abmil_hc_cell_ext_embed_gated_8
    abmil_hc_cell_ext_embed_gated_8
    abmil_hc_cell_ext_embed_gated_8
    abmil_hc_cell_ext_embed_gated_8
    lr_ext_embed
    lr_hc_cell
    lr_hc_cell_ext_embed
)
UNIT_DROPOUTS=(0.0 0.2 0.4 0.6 0.0 0.0 0.0)
if (( ${#UNIT_IDS[@]} != 7 \
    || ${#UNIT_MODELS[@]} != ${#UNIT_IDS[@]} \
    || ${#UNIT_DROPOUTS[@]} != ${#UNIT_IDS[@]} )); then
    echo "The localization model-unit manifest is not aligned." >&2
    exit 1
fi

relevant_files() {
    git ls-files -co --exclude-standard -z -- \
        configs/paths.yaml \
        downstream_tasks/__init__.py \
        downstream_tasks/config.py \
        downstream_tasks/run.py \
        downstream_tasks/topology_baselines.py \
        downstream_tasks/data \
        downstream_tasks/models \
        downstream_tasks/training \
        downstream_tasks/utils \
        scripts/cscs/run_bce_uni_localization.sbatch
}

fingerprint_repo() {
    (
        cd "${REPO_ROOT}"
        relevant_files | sort -z | xargs -0 -r sha256sum | sha256sum | cut -d ' ' -f 1
    )
}

test -x "${PYTHON}"
test -f "${SOURCE_INFERENCE}/protein_embeddings.pt"
test -f "${SOURCE_INFERENCE}/cell_embeddings.pt"
test -f "${SOURCE_INFERENCE}/inference_config.yaml"
test -f "${SOURCE_ESM}"
test -f "${LABELS}"
test -f "${LABELS%.csv}.manifest.json"
if [[ "${RUN_TOPOLOGY}" == "1" ]]; then
    test -f "${NETWORK_ROOT}/global_ppi_edgelist.txt"
    if [[ -e "${TOPOLOGY_OUTPUT}" ]]; then
        echo "Topology output path already exists; refusing to overwrite." >&2
        exit 1
    fi
fi
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
if [[ "${RUN_TOPOLOGY}" == "1" ]]; then
    mkdir -p "${TOPOLOGY_OUTPUT_ROOT}"
fi

if [[ "$(fingerprint_repo)" != "${EXPECTED_FINGERPRINT}" ]]; then
    echo "The reviewed repository state changed after submission." >&2
    exit 1
fi
if [[ "$(nvidia-smi --list-gpus | wc -l)" -ne 4 ]]; then
    echo "Expected exactly four GPUs on the allocated Clariden node." >&2
    exit 1
fi

JOB_STAGE="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/protscape-loca-${SLURM_JOB_ID}.XXXXXX")"
cleanup_job_stage() {
    local attempt
    # A canceled Slurm step can briefly keep creating files while the batch
    # shell exits. Retry the exact mktemp path so cancellation does not leave a
    # node-local staging directory behind.
    for attempt in 1 2 3 4 5; do
        rm -rf -- "${JOB_STAGE}" 2>/dev/null || true
        if [[ ! -e "${JOB_STAGE}" ]]; then
            return
        fi
        sleep 1
    done
    echo "Warning: node-local staging cleanup did not finish: ${JOB_STAGE}" >&2
}
trap cleanup_job_stage EXIT
CODE_STAGE="${JOB_STAGE}/code"
INFERENCE_STAGE="${JOB_STAGE}/embeddings"
ESM_STAGE="${JOB_STAGE}/$(basename "${SOURCE_ESM}")"
mkdir -p "${CODE_STAGE}" "${INFERENCE_STAGE}/${EMBEDDING_MODEL}"

(
    cd "${REPO_ROOT}"
    relevant_files
) | tar -C "${REPO_ROOT}" --null --files-from=- -cf - \
    | tar -C "${CODE_STAGE}" -xf -
cp -a "${SOURCE_INFERENCE}/." "${INFERENCE_STAGE}/${EMBEDDING_MODEL}/"
cp -a "${SOURCE_ESM}" "${ESM_STAGE}"

export PROTSCAPE_CONDA_ENV="${CONDA_ENV}"
export PROTSCAPE_GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
export PROTSCAPE_CONTEXTUAL_EMBEDDING_SHA256="$(sha256sum "${INFERENCE_STAGE}/${EMBEDDING_MODEL}/protein_embeddings.pt" | cut -d ' ' -f 1)"
export PROTSCAPE_CONTEXTUAL_CONFIG_SHA256="$(sha256sum "${INFERENCE_STAGE}/${EMBEDDING_MODEL}/inference_config.yaml" | cut -d ' ' -f 1)"
export PROTSCAPE_CONTEXTUAL_INFERENCE_CONTRACT="bce_s2gae_mask0.5_uniformity_lambda5e-5_t2"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_DIR="${JOB_STAGE}/wandb-disabled"
export WANDB_CACHE_DIR="${JOB_STAGE}/wandb-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

"${PYTHON}" - "${INFERENCE_STAGE}/${EMBEDDING_MODEL}/inference_config.yaml" <<'PY'
import ast
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)
config = payload["config"]
s2gae = ast.literal_eval(config["s2gae_config"])
uniformity = ast.literal_eval(config["uniformity_config"])
expected = {
    "model_type": "hc",
    "dataset_mode": "bulk",
    "features_mode": "ESM2",
    "n_cells": 207,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"Inference contract mismatch: {key}")
if not s2gae.get("enabled") or s2gae.get("mask_ratio") != 0.5:
    raise SystemExit("Inference contract mismatch: S2GAE")
if (
    not uniformity.get("enabled")
    or uniformity.get("lambda_reg") != 5e-5
    or uniformity.get("t") != 2.0
):
    raise SystemExit("Inference contract mismatch: uniformity")
PY

SPLIT_WATCH_START="${JOB_STAGE}/model-dispatch-start"
MODEL_QUEUE_DONE="${JOB_STAGE}/model-queue.done"
TOPOLOGY_CONTROLLER_LOG="${LOG_ROOT}/bce_uni_loca_${SLURM_JOB_ID}_topology_controller.log"

validate_split_artifact() {
    local candidate="$1"
    (
        cd "${CODE_STAGE}"
        "${PYTHON}" - "${candidate}" "${LABELS}" <<'PY'
import sys
from pathlib import Path

from downstream_tasks.topology_baselines import load_split_data, load_task_data

split_path = Path(sys.argv[1])
task = load_task_data(Path(sys.argv[2]))
split = load_split_data(split_path, task)
print(split.fingerprint)
PY
    )
}

run_topology_after_split() {
    local candidate split_path="" validation_output
    local -a candidates=()

    echo "[INFO] waiting for a newly written, valid model split artifact"
    while [[ -z "${split_path}" ]]; do
        candidates=()
        if [[ -d "${MODEL_RESULT_ROOT}" ]]; then
            mapfile -t candidates < <(
                find "${MODEL_RESULT_ROOT}" -mindepth 2 -maxdepth 2 \
                    -type f -name split_indices.npz -newer "${SPLIT_WATCH_START}" \
                    | sort
            )
        fi
        for candidate in "${candidates[@]}"; do
            if validation_output="$(validate_split_artifact "${candidate}" 2>&1)"; then
                split_path="${candidate}"
                echo "[INFO] topology split=${split_path} fingerprint=${validation_output}"
                break
            fi
        done
        if [[ -z "${split_path}" && -e "${MODEL_QUEUE_DONE}" ]]; then
            echo "No valid current-run split artifact was produced; topology was not launched." >&2
            return 1
        fi
        if [[ -z "${split_path}" ]]; then
            sleep 2
        fi
    done

    cd "${CODE_STAGE}"
    echo "[INFO] running direct-neighbour and label-propagation baselines without a GPU"
    srun --exclusive --nodes=1 --ntasks=1 \
        --gres=none --cpus-per-task=4 --mem=8G \
        env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
            OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled WANDB_DISABLED=true \
        "${PYTHON}" -m downstream_tasks.topology_baselines \
        --task-csv "${LABELS}" \
        --split-indices "${split_path}" \
        --global-ppi "${NETWORK_ROOT}/global_ppi_edgelist.txt" \
        --output-dir "${TOPOLOGY_OUTPUT}" \
        --universes annotated,full \
        --propagation-normalizations random_walk,symmetric \
        --depths 1,2,3,4,5,6,7,8,9,10 \
        --save-predictions \
        >"${LOG_ROOT}/bce_uni_loca_${SLURM_JOB_ID}_topology.log" 2>&1
}

echo "[INFO] BCE-Uni localization sweep topology=${RUN_TOPOLOGY}"
TOPOLOGY_PID=""
if [[ "${RUN_TOPOLOGY}" == "1" ]]; then
    # The watcher consumes no Slurm resources until a fresh split is available.
    touch "${SPLIT_WATCH_START}"
    run_topology_after_split >"${TOPOLOGY_CONTROLLER_LOG}" 2>&1 &
    TOPOLOGY_PID="$!"
fi

# Submit every one-GPU step immediately. Slurm keeps at most four running and
# starts the next pending step as soon as a GPU becomes free.
MODEL_PIDS=()
status=0
for unit_index in "${!UNIT_IDS[@]}"; do
    unit_id="${UNIT_IDS[unit_index]}"
    model="${UNIT_MODELS[unit_index]}"
    dropout="${UNIT_DROPOUTS[unit_index]}"
    worker_log="${LOG_ROOT}/bce_uni_loca_${SLURM_JOB_ID}_${unit_id}.log"
    echo "[INFO] dispatch unit=${unit_id} model=${model} dropout=${dropout}"

    srun --exclusive --nodes=1 --ntasks=1 \
        --gpus-per-task=1 \
        --cpus-per-gpu=15 --mem=64G \
        --gpu-bind=single:1 \
        --chdir="${CODE_STAGE}" \
        env OMP_NUM_THREADS=15 MKL_NUM_THREADS=15 OPENBLAS_NUM_THREADS=15 \
            WANDB_MODE=disabled WANDB_DISABLED=true \
        "${PYTHON}" -m downstream_tasks.run \
            --inference-model "${RUN_ID}" \
            --embedding-inference-model "${EMBEDDING_MODEL}" \
            --inference-root "${INFERENCE_STAGE}" \
            --esm2-embeddings "${ESM_STAGE}" \
            --output-root "${OUTPUT_ROOT}" \
            --task protein_localization \
            --task-csv "${LABELS}" \
            --model "${model}" \
            --embedding-source esm \
            --dataset-mode bulk \
            --dropout "${dropout}" \
            --lr 0.0001 \
            --weight-decay 0.0001 \
            --batch-size 512 \
            --seed 42 \
            --train-selection-metric auprc \
            >"${worker_log}" 2>&1 &
    MODEL_PIDS+=("$!")
done

for unit_index in "${!MODEL_PIDS[@]}"; do
    if wait "${MODEL_PIDS[unit_index]}"; then
        echo "[OK] completed unit=${UNIT_IDS[unit_index]}"
    else
        echo "Model unit failed: ${UNIT_IDS[unit_index]}" >&2
        status=1
    fi
done

if [[ "${RUN_TOPOLOGY}" == "1" ]]; then
    touch "${MODEL_QUEUE_DONE}"
    if wait "${TOPOLOGY_PID}"; then
        echo "[OK] completed both topology baselines"
    else
        topology_status=$?
        echo "Topology controller failed (exit ${topology_status}); see ${TOPOLOGY_CONTROLLER_LOG}." >&2
        status=1
    fi
fi
exit "${status}"
