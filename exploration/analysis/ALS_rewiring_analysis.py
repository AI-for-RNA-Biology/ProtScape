"""Run ALS rewiring analysis stages, stopping at the first failure.

    python -m exploration.analysis.ALS_rewiring_analysis
    python -m exploration.analysis.ALS_rewiring_analysis --from 05
    python -m exploration.analysis.ALS_rewiring_analysis --only 07 09

Stage 0 creates the D22/D35 subsets when absent. Stage 05b saves the initial
partition, then replaces it with the refined partition used by stages 06–10.
The final neighbourhood analyses require a separate neighbourhood-edge CSV.
See `ALS_rewiring/README.md` for inputs and configuration.
"""
import argparse
import subprocess
import sys
from pathlib import Path

STAGES = [
    "0_subset_d22.sh",
    "0_subset_d35.sh",
    "01_load_merge_data.py",
    "02_validate_s2gae_scores.py",
    "03_select_high_confidence_edges.py",
    "04_build_giant_graph.py",
    "05_cluster_leiden.py",
    "05b_hierarchical_recluster.py",
    "06_module_enrichment.py",
    "07_condition_enrichment.py",
    "08_string_enrichment.py",
    "09_visualize_giant_network.py",
    "10_edge_components.py",
    "neighborhood_by_context.py",
    "neighborhood_new_gene_expression.py",
]

SCRIPTS_DIR = Path(__file__).resolve().parent / "ALS_rewiring" / "scripts"


def stage_id(name: str) -> str:
    return name.split("_", 1)[0]


def stage_command(stage: str) -> list:
    path = str(SCRIPTS_DIR / stage)
    return ["bash", path] if stage.endswith(".sh") else [sys.executable, path]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="from_stage", default=None, metavar="NN",
                         help="Start from this stage id (e.g. 05) and run to the end.")
    parser.add_argument("--only", nargs="+", default=None, metavar="NN",
                         help="Run only these stage ids.")
    parser.add_argument("--skip", nargs="+", default=None, metavar="NN",
                         help="Run every stage except these ids.")
    args = parser.parse_args()

    stages = STAGES
    if args.from_stage:
        stages = [s for s in stages if stage_id(s) >= args.from_stage]
    if args.only:
        stages = [s for s in stages if stage_id(s) in args.only]
    if args.skip:
        stages = [s for s in stages if stage_id(s) not in args.skip]

    if not stages:
        print("No stages selected - check --from/--only/--skip.")
        sys.exit(1)

    print("Running stages:", ", ".join(stages))
    for stage in stages:
        print(f"\n{'=' * 60}\n{stage}\n{'=' * 60}")
        result = subprocess.run(stage_command(stage))
        if result.returncode != 0:
            print(f"\nStage {stage} failed (exit code {result.returncode}) - stopping.")
            sys.exit(result.returncode)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
