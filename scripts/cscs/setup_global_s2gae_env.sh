#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${PROTSCAPE_ROOT:-/users/aloistho/projects/ProtScape}"
CONDA_ROOT="${PROTSCAPE_CONDA_ROOT:-/iopsstor/scratch/cscs/aloistho/protscape/miniconda3}"
CONDA_ENV="${PROTSCAPE_CONDA_ENV:-/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae}"
PIP_CACHE_DIR="${PROTSCAPE_PIP_CACHE:-/iopsstor/scratch/cscs/aloistho/protscape/pip-cache}"
INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-py312_26.7.1-0-Linux-aarch64.sh"
INSTALLER_SHA256="391edcfaef0e70b7047834a86e48f5686eb0fe1e7612a086ca73e0084ffea45f"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "This environment is pinned for Clariden aarch64 nodes." >&2
    exit 1
fi
if [[ "${CONDA_ROOT}" == "${CONDA_ENV}" ]]; then
    echo "The project environment must be separate from the Miniconda base." >&2
    exit 1
fi

mkdir -p "$(dirname "${CONDA_ROOT}")" "${PIP_CACHE_DIR}"
exec 9>"${CONDA_ROOT}.setup.lock"
flock 9

if [[ -e "${CONDA_ROOT}" && ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    echo "Incomplete Miniconda installation: ${CONDA_ROOT}" >&2
    exit 1
fi
if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    installer="$(mktemp "${TMPDIR:-/tmp}/protscape-miniconda.XXXXXX")"
    trap 'rm -f -- "${installer}"' EXIT
    curl --fail --location --silent --show-error \
        "${INSTALLER_URL}" --output "${installer}"
    printf '%s  %s\n' "${INSTALLER_SHA256}" "${installer}" | sha256sum --check
    bash "${installer}" -b -p "${CONDA_ROOT}"
    rm -f -- "${installer}"
    trap - EXIT
fi

if [[ -e "${CONDA_ENV}" && ! -x "${CONDA_ENV}/bin/python" ]]; then
    echo "Incomplete Conda environment: ${CONDA_ENV}" >&2
    exit 1
fi
if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    "${CONDA_ROOT}/bin/conda" create --yes \
        --override-channels --channel conda-forge \
        --prefix "${CONDA_ENV}" python=3.12.14 pip
fi

PYTHON="${CONDA_ENV}/bin/python"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

"${PYTHON}" -m pip install --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cu129 \
    'torch==2.9.1+cu129'
"${PYTHON}" -m pip install --only-binary=:all: \
    -r "${REPO_ROOT}/requirements-global-s2gae.txt"
pip_check="$("${PYTHON}" -m pip check 2>&1)" || {
    # NVIDIA's aarch64 cuSPARSELt wheel uses an unrecognized internal sbsa tag.
    if [[ "${pip_check}" != \
        "nvidia-cusparselt-cu12 0.7.1 is not supported on this platform" ]]; then
        printf '%s\n' "${pip_check}" >&2
        exit 1
    fi
}
"${PYTHON}" - <<'PY'
import platform
import sys

import torch
import torch_geometric
import wandb

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Expected Python 3.12, found {sys.version.split()[0]}")
if platform.machine() != "aarch64":
    raise SystemExit(f"Expected aarch64, found {platform.machine()}")
if torch.__version__ != "2.9.1+cu129" or torch.version.cuda != "12.9":
    raise SystemExit(
        f"Expected torch 2.9.1+cu129/CUDA 12.9, found "
        f"{torch.__version__}/{torch.version.cuda}"
    )
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_geometric={torch_geometric.__version__}")
print(f"wandb={wandb.__version__}")
PY
