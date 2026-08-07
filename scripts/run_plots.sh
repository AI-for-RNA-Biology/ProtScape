#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m exploration.plotting.plot_supplementary_figure_1
python -m exploration.plotting.plot_figure_2
python -m exploration.plotting.plot_supplementary_figure_2
python -m exploration.plotting.plot_figure_3
python -m exploration.plotting.plot_figure_5
python -m exploration.plotting.plot_supplementary_figure_5
