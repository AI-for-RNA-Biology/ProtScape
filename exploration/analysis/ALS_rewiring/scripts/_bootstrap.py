"""Adds the ProtScape repository root to sys.path so
`from exploration.analysis.ALS_rewiring import ...` works when a stage
script is run directly (`python 03_select_high_confidence_edges.py`)
rather than via `python -m`. Every stage script starts with:

    import _bootstrap  # noqa: F401
"""
import sys
from pathlib import Path

# scripts/ -> ALS_rewiring/ -> analysis/ -> exploration/ -> ProtScape/ (repo root)
REPO_ROOT = Path(__file__).resolve().parents[4]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
