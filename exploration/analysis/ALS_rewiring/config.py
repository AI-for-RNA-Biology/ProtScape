"""Central configuration for the ProtScape ALS_rewiring PPI-module pipeline.

Every stage script imports its paths/constants from here, so there is exactly
one place to change e.g. `DATA_DIR` or `HIGH_CONF_THRESHOLD` for the whole
pipeline. Paths themselves come from `configs/paths.yaml` (the
`als_rewiring_*` keys), following the same `downstream_tasks.config.PATHS`
pattern used by the rest of ProtScape's `exploration/analysis/` modules.
"""
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from downstream_tasks.config import PATHS

DATA_DIR = Path(PATHS["als_rewiring_data_root"]).expanduser()
D22_CSV = DATA_DIR / "subset_d22_pergenotype_presentp.csv"
D35_CSV = DATA_DIR / "subset_d35_pergenotype_presentp.csv"

# Final, human-facing results (figures/tables) live directly under the
# shared ProtScape `output_root/analysis/ALS_rewiring` directory, matching
# the ANALYSIS_DIR convention used by the other exploration/analysis/*
# modules (no extra nesting).
ANALYSIS_DIR = Path(PATHS["output_root"]) / "analysis" / "ALS_rewiring"
OUT_DIR = ANALYSIS_DIR
# Intermediate results (parquet/pickle/json) live in their own subdirectory so
# they're easy to tell apart from final figures/tables in OUT_DIR, and easy to
# wipe/regenerate independently (`rm -r INTERMEDIATE_DIR`) without touching
# anything you'd actually want to keep.
INTERMEDIATE_DIR = OUT_DIR / "intermediate"

HIGH_CONF_THRESHOLD = 0.95

# GO term annotation table used for the pathway embedding computed at the end
# of stage 05 and after each hierarchical-recluster split step (see
# pathway_embedding.py).
GO_TERMS_CSV = Path(PATHS["als_rewiring_go_terms_csv"]).expanduser()

# Leiden resolution sweep (see stage 05). No published reference sweep for this
# kind of network, so candidates are all in the "order 1" range: {0.25*x for x in 1..8}.
RESOLUTION_SWEEP = np.round(0.25 * np.arange(1, 9), 2)
N_STABILITY_RUNS = 10  # leidenalg (igraph/C-backed) is fast, so more seeds are cheap
LEIDEN_N_ITERATIONS = -1  # -1 = run each local-moving phase to convergence (leidenalg default)
TRADEOFF_MODULARITY_WEIGHT = 0.5  # combined_score weighting: modularity vs. ARI partition stability

SCORE_CATEGORIES = [
    "ct_d22_nuc", "ct_d22_cyto", "vcp_d22_nuc", "vcp_d22_cyto",
    "ct_d35_nuc", "ct_d35_cyto", "vcp_d35_nuc", "vcp_d35_cyto",
]

CONDITION_COLS = ["d22", "d35", "ct", "vcp", "ct_d22", "vcp_22", "ct_35", "vcp_35"]

# Section 6 modules-of-interest cutoffs (see condition_enrichment.modules_of_interest).
MOI_FE_CUTOFF = 1.2
MOI_PADJ_CUTOFF = 0.01

# --- Core pipeline stage 05b: hierarchical re-clustering
# (hierarchical_recluster.py, scripts/05b_hierarchical_recluster.py) ---
# Always runs as part of the pipeline, right after stage 05: it peels off
# stage 05's least intra-connected modules one at a time and re-clusters
# just their induced subgraphs, then makes the resulting partition
# INTERMEDIATE_DIR's canonical one - stages 06-10 run on it, not stage 05's
# original partition.
#
# recursive_refine_by_density always processes every original stage-05
# module (least- to most-connected) and records global modularity /
# mean intra-density after each split - this flag only picks which
# resulting partition becomes canonical:
#   True  - the partition AT THE ELBOW of the mean intra-density curve
#           (point of diminishing returns from further splitting)
#   False - the FULLY refined partition, after every original module that
#           could be split has been
HIERARCHICAL_RECLUSTER_STOP_AT_ELBOW = True

# --- scripts/neighborhood_by_context.py + neighborhood_new_gene_expression.py ---
# Default gene/CSV these run for automatically as part of the pipeline (still
# overridable per-invocation via --gene/--csv for ad hoc exploration of a
# different gene or CSV export).
NEIGHBORHOOD_GENE = "ALS2CL"
NEIGHBORHOOD_EDGES_CSV = Path(PATHS["als_rewiring_neighborhood_edges_csv"]).expanduser()


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
