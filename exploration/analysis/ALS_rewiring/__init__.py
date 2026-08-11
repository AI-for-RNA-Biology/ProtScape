"""ProtScape ALS PPI-module analysis pipeline.

This package is the single source of truth for every function used by the
`scripts/` stage runners. It's the same logic that was originally developed
interactively in `raph_analysis_protscape_ALS_python.ipynb` - that notebook
remains available for exploration, but every function it defined now lives
here so it isn't duplicated between the notebook and the reproducible
pipeline.

See `scripts/README.md` for how to run the pipeline (as a whole, or one
stage at a time from cached intermediate results).
"""
import matplotlib

# Editable text in exported PDFs (matches the notebook's plotting setup).
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
