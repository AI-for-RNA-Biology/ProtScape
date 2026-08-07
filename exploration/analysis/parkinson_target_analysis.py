#!/usr/bin/env python3
"""Compute Parkinson target-discovery tables and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from downstream_tasks.config import PATHS
from exploration.analysis.parkinson_inference import (
    MODEL_LABELS,
    run_global_inference,
)
from exploration.analysis.parkinson_modules import (
    build_leiden_tables,
    build_module_reactome_enrichment,
    build_module_summary,
)
from exploration.analysis.parkinson_string import (
    build_main_string_enrichment,
    build_string_mapping,
    build_string_network,
    read_string_annotations,
)


ANALYSIS_DIR = Path(PATHS["output_root"]) / "analysis/parkinson_target_analysis"
EXTERNAL_SUPPORT_SNAPSHOT = Path(PATHS["parkinson_external_support"])
PROBABILITY_THRESHOLD = 0.5

SYNAPTIC_GROUPS = [
    (
        "Kainate receptors",
        "GRIK1–GRIK5",
        ["GRIK1", "GRIK2", "GRIK3", "GRIK4", "GRIK5"],
    ),
    (
        "Neuroligins",
        "NLGN1–NLGN4X",
        ["NLGN1", "NLGN2", "NLGN3", "NLGN4X"],
    ),
    (
        "DLGAP scaffolds",
        "DLGAP1–DLGAP4",
        ["DLGAP1", "DLGAP2", "DLGAP3", "DLGAP4"],
    ),
]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required Parkinson analysis input: {path}")
    return path


def build_candidates(scores: pd.DataFrame) -> pd.DataFrame:
    candidates = scores[
        scores["benchmark_split"].eq("label_excluded")
        & scores["mean_probability"].ge(PROBABILITY_THRESHOLD)
    ].copy()
    candidates = candidates.sort_values(
        ["model", "mean_probability", "protein"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    candidates["candidate_rank"] = candidates.groupby("model").cumcount() + 1
    return candidates


def build_recovery_tables(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = []
    summaries = []
    for model, model_scores in scores.groupby("model", sort=False):
        keep = model_scores["benchmark_split"].eq("label_excluded") | (
            model_scores["benchmark_split"].eq("permanent_test")
            & model_scores["benchmark_label"].eq(1)
        )
        ranked = (
            model_scores[keep]
            .sort_values(
                ["mean_probability", "protein"],
                ascending=[False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        ranked["rank"] = ranked.index + 1
        ranked["is_permanent_test_positive"] = (
            ranked["benchmark_split"].eq("permanent_test")
            & ranked["benchmark_label"].eq(1)
        )
        ranked["test_positives_recovered"] = ranked[
            "is_permanent_test_positive"
        ].cumsum()
        n_positives = int(ranked["is_permanent_test_positive"].sum())
        ranked["test_positive_recall_percent"] = (
            100 * ranked["test_positives_recovered"] / n_positives
        )
        ranked["other_proteins_screened"] = (
            ranked["rank"] - ranked["test_positives_recovered"]
        )
        curves.append(ranked)
        full = ranked[
            ranked["test_positives_recovered"].eq(n_positives)
        ].iloc[0]
        summaries.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_screening_proteins": len(ranked),
                "n_permanent_test_positives": n_positives,
                "n_label_excluded_proteins": int(
                    ranked["benchmark_split"].eq("label_excluded").sum()
                ),
                "full_recovery_rank": int(full["rank"]),
                "other_proteins_screened_before_full_recovery": int(
                    full["other_proteins_screened"]
                ),
                "score_at_full_recovery": full["mean_probability"],
            }
        )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summaries)


def build_external_support(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot = pd.read_csv(require_file(EXTERNAL_SUPPORT_SNAPSHOT))
    snapshot["protein"] = snapshot["protein"].str.upper()
    if snapshot["protein"].duplicated().any():
        raise ValueError("External-support snapshot has duplicate proteins")
    candidate_genes = set(candidates["protein"])
    if not candidate_genes.issubset(snapshot["protein"]):
        missing = sorted(candidate_genes - set(snapshot["protein"]))
        raise ValueError(
            f"External-support input lacks {len(missing)} rebuilt candidates"
        )
    displayed_flags = [
        "current_opentargets_parkinson_association_non_literature_only",
        "approved_human_drugbank_target_any_indication",
    ]
    annotation_flags = [
        *displayed_flags,
        "other_opentargets_disease_association",
    ]
    annotations = snapshot[["protein", *annotation_flags, "query_utc"]].copy()
    joined = candidates.merge(
        annotations, on="protein", how="left", validate="many_to_one"
    )
    summary_rows = []
    for model, rows in joined.groupby("model", sort=False):
        for flag in displayed_flags:
            count = int(rows[flag].astype(bool).sum())
            summary_rows.append(
                {
                    "support_flag": flag,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "supported_candidates": count,
                    "candidate_count": len(rows),
                    "support_percent": 100 * count / len(rows),
                    "nomination_probability_threshold": PROBABILITY_THRESHOLD,
                }
            )
    return joined, pd.DataFrame(summary_rows)


def build_synaptic_completion(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_excluded = scores[scores["benchmark_split"].eq("label_excluded")]
    member_rows = []
    wide_rows = []
    model_summaries: dict[tuple[str, str], dict] = {}
    nomination_counts = {}
    for model, rows in label_excluded.groupby("model", sort=False):
        ranked = (
            rows.sort_values(
                ["mean_probability", "protein"],
                ascending=[False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        ranked["discovery_rank"] = ranked.index + 1
        nomination_counts[model] = int(
            ranked["mean_probability"].ge(PROBABILITY_THRESHOLD).sum()
        )
        for group_order, (group, member_label, members) in enumerate(
            SYNAPTIC_GROUPS
        ):
            selected = ranked[ranked["protein"].isin(members)].copy()
            if set(selected["protein"]) != set(members):
                raise ValueError(f"Missing one or more {group} members for {model}")
            completion = int(selected["discovery_rank"].max())
            completion_member = selected.loc[
                selected["discovery_rank"].idxmax(), "protein"
            ]
            model_summaries[(group, model)] = {
                "rank": completion,
                "member": completion_member,
                "all_selected": bool(
                    selected["mean_probability"]
                    .ge(PROBABILITY_THRESHOLD)
                    .all()
                ),
            }
            for member_order, member in enumerate(members):
                row = selected[selected["protein"].eq(member)].iloc[0]
                member_rows.append(
                    {
                        "group": group,
                        "group_order": group_order,
                        "member_label": member_label,
                        "member": member,
                        "member_order": member_order,
                        "model": model,
                        "mean_probability": row["mean_probability"],
                        "discovery_rank": int(row["discovery_rank"]),
                        "group_completion_rank": completion,
                        "is_completion_member": member == completion_member,
                    }
                )
    for group_order, (group, member_label, members) in enumerate(SYNAPTIC_GROUPS):
        prot = model_summaries[(group, "protscape")]
        pinn = model_summaries[(group, "pinnacle")]
        wide_rows.append(
            {
                "group": group,
                "group_order": group_order,
                "member_label": member_label,
                "members": ";".join(members),
                "n_members": len(members),
                "protscape_completion_rank": prot["rank"],
                "protscape_completion_member": prot["member"],
                "pinnacle_completion_rank": pinn["rank"],
                "pinnacle_completion_member": pinn["member"],
                "protscape_all_members_probability_ge_0p5": prot[
                    "all_selected"
                ],
                "pinnacle_all_members_probability_ge_0p5": pinn[
                    "all_selected"
                ],
                "protscape_nomination_count_at_0p5": nomination_counts[
                    "protscape"
                ],
                "pinnacle_nomination_count_at_0p5": nomination_counts[
                    "pinnacle"
                ],
            }
        )
    return pd.DataFrame(member_rows), pd.DataFrame(wide_rows)


def build_statistics(
    membership: pd.DataFrame,
    candidates: pd.DataFrame,
    network_nodes: pd.DataFrame,
    edges: pd.DataFrame,
    leiden_nodes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split, split_rows in membership.groupby(
        "benchmark_split", dropna=False
    ):
        rows.append(
            {
                "section": "global_inference_cohort",
                "group": split,
                "statistic": "proteins",
                "value": len(split_rows),
            }
        )
        for label in [1, 0]:
            rows.append(
                {
                    "section": "global_inference_cohort",
                    "group": split,
                    "statistic": f"label_{label}_proteins",
                    "value": int(
                        split_rows["benchmark_label"].eq(label).sum()
                    ),
                }
            )
    for model, model_rows in candidates.groupby("model"):
        rows.append(
            {
                "section": "candidate_sets",
                "group": model,
                "statistic": "probability_ge_0p5",
                "value": len(model_rows),
            }
        )
    rows.extend(
        [
            {
                "section": "string_network",
                "group": "protscape_known_plus_candidates",
                "statistic": "submitted_nodes",
                "value": len(network_nodes),
            },
            {
                "section": "string_network",
                "group": "protscape_known_plus_candidates",
                "statistic": "connected_nodes",
                "value": len(leiden_nodes),
            },
            {
                "section": "string_network",
                "group": "protscape_known_plus_candidates",
                "statistic": "experimental_edges",
                "value": len(edges),
            },
            {
                "section": "string_network",
                "group": "protscape_known_plus_candidates",
                "statistic": "leiden_clusters",
                "value": leiden_nodes["leiden_cluster"].nunique(),
            },
        ]
    )
    return pd.DataFrame(rows)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    scores, membership = run_global_inference()
    candidates = build_candidates(scores)
    recovery_curve, recovery_summary = build_recovery_tables(scores)
    support_rows, support_summary = build_external_support(candidates)
    synaptic_members, synaptic_summary = build_synaptic_completion(scores)

    mapping = build_string_mapping(sorted(membership["protein"].unique()))
    annotations = read_string_annotations(
        set(mapping.loc[mapping["mapped"], "string_id"])
    )
    selected_terms, main_enrichment = build_main_string_enrichment(
        membership, candidates, mapping, annotations
    )
    network_nodes, edges = build_string_network(
        membership, candidates, mapping
    )
    graph, leiden_nodes, leiden_selection, leiden_stability = (
        build_leiden_tables(
            network_nodes, edges, membership, candidates, support_rows
        )
    )
    reactome_all, reactome_display = build_module_reactome_enrichment(
        graph, leiden_nodes, annotations
    )
    module_summary = build_module_summary(leiden_nodes)
    statistics = build_statistics(
        membership, candidates, network_nodes, edges, leiden_nodes
    )

    outputs = {
        "parkinson_dataset_statistics.csv": statistics,
        "parkinson_global_predictions.csv.gz": scores,
        "parkinson_candidate_set.csv": candidates,
        "parkinson_candidate_recovery_curve.csv": recovery_curve,
        "parkinson_candidate_recovery_summary.csv": recovery_summary,
        "parkinson_candidate_external_support_rows.csv": support_rows,
        "parkinson_candidate_external_support.csv": support_summary,
        "parkinson_synaptic_group_members.csv": synaptic_members,
        "parkinson_synaptic_group_completion.csv": synaptic_summary,
        "parkinson_string_mapping.csv": mapping,
        "parkinson_string_selected_terms.csv": selected_terms,
        "parkinson_string_main_enrichment.csv": main_enrichment,
        "parkinson_string_network_all_nodes.csv": network_nodes,
        "parkinson_string_network_edges.csv": edges,
        "parkinson_leiden_nodes.csv": leiden_nodes,
        "parkinson_leiden_selection.csv": leiden_selection,
        "parkinson_leiden_stability.csv": leiden_stability,
        "parkinson_reactome_all_tests.csv.gz": reactome_all,
        "parkinson_reactome_module_enrichment.csv": reactome_display,
        "parkinson_leiden_module_summary.csv": module_summary,
    }
    for filename, table in outputs.items():
        table.to_csv(ANALYSIS_DIR / filename, index=False)

    from exploration.plotting.parkinson_target_analysis_plots import plot_all

    plot_all(ANALYSIS_DIR, ANALYSIS_DIR)
    print(f"Wrote Parkinson analysis CSVs and plots to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
