#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${PROTSCAPE_ROOT:-/users/aloistho/projects/ProtScape}"
CONDA_ENV="${PROTSCAPE_CONDA_ENV:-/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae}"
SBATCH_SCRIPT="${REPO_ROOT}/scripts/cscs/run_bce_uni_localization.sbatch"
PYTHON="${CONDA_ENV}/bin/python"

cd "${REPO_ROOT}"
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled WANDB_DISABLED=true \
    "${PYTHON}" -m pytest -q -p no:cacheprovider \
    tests/test_functional_annotation_processing.py \
    tests/test_multilabel_task_loader.py \
    tests/test_topology_baselines.py

fingerprint="$({
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
        scripts/cscs/run_bce_uni_localization.sbatch \
        | sort -z \
        | xargs -0 -r sha256sum
} | sha256sum | cut -d ' ' -f 1)"

echo "Submitting reviewed code fingerprint ${fingerprint}"
sbatch --export="ALL,PROTSCAPE_CODE_FINGERPRINT=${fingerprint}" "${SBATCH_SCRIPT}"
