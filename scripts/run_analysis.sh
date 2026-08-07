#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m exploration.analysis.data_bulk_statistics
python -m exploration.analysis.pretraining_evaluation
python -m exploration.analysis.consensus_analysis
python -m exploration.analysis.corum_analysis
python -m exploration.analysis.therapeutic_target_analysis
python -m exploration.analysis.parkinson_target_analysis
