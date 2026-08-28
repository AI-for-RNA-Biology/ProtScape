#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${PROTSCAPE_CONDA_ENV:-/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae}"
PYTHON="${CONDA_ENV}/bin/python"
REPO_ROOT="${PROTSCAPE_ROOT:-/users/aloistho/projects/ProtScape}"
DATA_ROOT="${PROTSCAPE_DATA:-/iopsstor/scratch/cscs/aloistho/protscape/release-data}"
SWEEP_CONFIG="${PROTSCAPE_SWEEP:-${REPO_ROOT}/configs/global_s2gae_sweep.yaml}"
LOG_ROOT="${PROTSCAPE_LOGS:-/iopsstor/scratch/cscs/aloistho/protscape/logs}"

test -x "${PYTHON}"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "${LOG_ROOT}"
test -f "${DATA_ROOT}/networks_bulk/global_ppi_edgelist.txt"
test -f "${DATA_ROOT}/networks_bulk/count_edge_dict.pkl"
test -f "${DATA_ROOT}/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk"

cd "${REPO_ROOT}"
if ! git diff --quiet || ! git diff --cached --quiet \
    || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "Refusing to submit from a dirty worktree; commit the reviewed code first." >&2
    exit 1
fi
if [[ -n "$(squeue --noheader --user "${USER}" --name protscape-global-sweep)" ]]; then
    echo "A ProtScape global sweep is already queued or running." >&2
    exit 1
fi

RUN_COUNT="$("${PYTHON}" - "${SWEEP_CONFIG}" <<'PY'
import sys
from pathlib import Path

from pretraining.run_global_s2gae_sweep import load_sweep

_, configurations = load_sweep(Path(sys.argv[1]))
print(len(configurations))
PY
)"
if (( RUN_COUNT < 1 )); then
    echo "Sweep configuration contains no runs." >&2
    exit 1
fi
LAST_PACK=$(((RUN_COUNT - 1) / 4))
GIT_COMMIT="$(git rev-parse HEAD)"

"${PYTHON}" - <<'PY'
import wandb

if not wandb.login(timeout=30, verify=True):
    raise SystemExit("W&B credentials are not configured.")
api = wandb.Api()
next(iter(api.runs("cedricvincentcuaz/pinnacle", per_page=1)), None)
print("W&B access verified for cedricvincentcuaz/pinnacle")
PY

sbatch --parsable \
    --array="0-${LAST_PACK}" \
    --export="ALL,PROTSCAPE_N_RUNS=${RUN_COUNT},PROTSCAPE_GIT_COMMIT=${GIT_COMMIT}" \
    scripts/cscs/run_global_s2gae_sweep.sbatch
