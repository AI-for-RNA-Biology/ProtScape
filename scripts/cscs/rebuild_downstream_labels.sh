#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${PROTSCAPE_ROOT:-/users/aloistho/projects/ProtScape}"
CONDA_ENV="${PROTSCAPE_CONDA_ENV:-/iopsstor/scratch/cscs/aloistho/protscape/conda-protscape-downstream}"
PYTHON="${CONDA_ENV}/bin/python"
DATA_ROOT="${PROTSCAPE_DATA:-/iopsstor/scratch/cscs/aloistho/protscape/release-data}"
DOWNSTREAM_DATA_ROOT="${PROTSCAPE_DOWNSTREAM_DATA:-/iopsstor/scratch/cscs/aloistho/protscape/downstream-data}"
RAW_ROOT="${DOWNSTREAM_DATA_ROOT}/raw"
PROCESSED_ROOT="${DOWNSTREAM_DATA_ROOT}/processed"

GLOBAL_PPI="${PROTSCAPE_GLOBAL_PPI:-${DATA_ROOT}/networks_bulk/global_ppi_edgelist.txt}"
CORUM_JSON="${PROTSCAPE_CORUM_JSON:-${RAW_ROOT}/corum_v4_1/humanComplexes.json}"
DRUGBANK_TARGETS="${PROTSCAPE_DRUGBANK_TARGETS:-${RAW_ROOT}/drugbank_oct2022/all_approved_oct2022.csv}"
OT_EVIDENCE="${PROTSCAPE_OT_EVIDENCE:-${RAW_ROOT}/opentargets_24_03_chembl}"
OT_EVIDENCE_RELEASE="${PROTSCAPE_OT_EVIDENCE_RELEASE:-24.03}"
OT_STATIC_RELEASE="${PROTSCAPE_OT_STATIC_RELEASE:-${RAW_ROOT}/opentargets_26_03}"
OT_RELEASE_NAME="${PROTSCAPE_OT_RELEASE_NAME:-26.03}"
OT_ASSOCIATION_SCOPE="${PROTSCAPE_OT_ASSOCIATION_SCOPE:-indirect}"
LABEL_THREADS="${PROTSCAPE_LABEL_THREADS:-1}"

test -x "${PYTHON}"
test -f "${GLOBAL_PPI}"
test -f "${CORUM_JSON}"
test -f "${DRUGBANK_TARGETS}"
test -d "${OT_EVIDENCE}"
test -d "${OT_STATIC_RELEASE}"

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${LABEL_THREADS}"
export MKL_NUM_THREADS="${LABEL_THREADS}"
export OPENBLAS_NUM_THREADS="${LABEL_THREADS}"
export NUMEXPR_NUM_THREADS="${LABEL_THREADS}"

cd "${REPO_ROOT}"

"${PYTHON}" -m downstream_tasks.data_processing.corum_processing \
    --corum-json "${CORUM_JSON}" \
    --global-ppi-path "${GLOBAL_PPI}" \
    --output-dir "${PROCESSED_ROOT}/corum"

"${PYTHON}" -m downstream_tasks.data_processing.therapeutic_target_processing \
    --static-release-dir "${OT_STATIC_RELEASE}" \
    --static-release-name "${OT_RELEASE_NAME}" \
    --association-scope "${OT_ASSOCIATION_SCOPE}" \
    --drugbank-targets "${DRUGBANK_TARGETS}" \
    --evidence-dir "${OT_EVIDENCE}" \
    --evidence-release-name "${OT_EVIDENCE_RELEASE}" \
    --evidence-format json \
    --global-ppi-path "${GLOBAL_PPI}" \
    --output-dir "${PROCESSED_ROOT}/therapeutic_target" \
    --force

printf 'Rebuilt downstream labels under %s\n' "${PROCESSED_ROOT}"
