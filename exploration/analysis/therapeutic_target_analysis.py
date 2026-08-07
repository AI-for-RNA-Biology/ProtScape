#!/usr/bin/env python3
"""Compute therapeutic-target results and LRP analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


from downstream_tasks.config import PATHS
from exploration.analysis.therapeutic_target.checkpoint_lrp import (
    generate_lrp,
    recompute_performance,
)


CHECKPOINT_ROOT = Path(PATHS["downstream_checkpoint_root"]) / "therapeutic_targets"
THERAPEUTIC_TARGET_DATASET_DIR = Path(PATHS["therapeutic_target_dataset_dir"])
CELL_METADATA_PATH = Path(PATHS["celltype_class_mapping"])
ANALYSIS_DIR = Path(PATHS["output_root"]) / "analysis/therapeutic_target_analysis"


TASKS = [
    ("therapeutic_target_mondo_0005180", "MONDO_0005180", "Parkinson disease"),
    ("therapeutic_target_efo_0000305", "EFO_0000305", "Breast carcinoma"),
    ("therapeutic_target_efo_1001207", "EFO_1001207", "Systolic heart failure"),
    ("therapeutic_target_efo_0000571", "EFO_0000571", "Lung adenocarcinoma"),
    ("therapeutic_target_efo_0003767", "EFO_0003767", "Inflammatory bowel disease"),
    ("therapeutic_target_efo_0000676", "EFO_0000676", "Psoriasis"),
    ("therapeutic_target_mondo_0007915", "MONDO_0007915", "Systemic lupus erythematosus"),
    ("therapeutic_target_efo_1001249", "EFO_1001249", "NASH"),
    ("therapeutic_target_mondo_0004979", "MONDO_0004979", "Asthma"),
    ("therapeutic_target_mondo_0005148", "MONDO_0005148", "Type 2 diabetes"),
    ("therapeutic_target_efo_0003884", "EFO_0003884", "Chronic kidney disease"),
    ("therapeutic_target_efo_0000685", "EFO_0000685", "Rheumatoid arthritis"),
    ("therapeutic_target_efo_0000341", "EFO_0000341", "COPD"),
    ("therapeutic_target_efo_0001361", "EFO_0001361", "Pulmonary arterial hypertension"),
    ("therapeutic_target_efo_0000274", "EFO_0000274", "Atopic eczema"),
]
TASK_TO_DISEASE = {task: disease for task, _, disease in TASKS}
PERFORMANCE_DISEASE_LABELS = {
    task: disease.title() for task, _, disease in TASKS
}
PERFORMANCE_DISEASE_LABELS["therapeutic_target_mondo_0005148"] = (
    "Type 2 Diabetes Mellitus"
)
PERFORMANCE_DISEASE_LABELS["therapeutic_target_efo_1001249"] = "NASH"
PERFORMANCE_DISEASE_LABELS["therapeutic_target_efo_0000341"] = "COPD"

MODEL_ORDER = [
    "pinnacle_random_fixed",
    "pinnacle_esm_fixed",
    "pinnacle_esm2_acm",
    "gae_att_fixed_do06",
    "s2gae_att_k1_fixed_do04_uni",
]
PDL_READOUT = "abmil8_pdl_id2_dropout"

LRP_MODELS = {
    "Pinnacle": ("pinnacle_random_fixed", "abmil8"),
    "Pinnacle-ESM2 (GAT)": ("pinnacle_esm_fixed", "abmil8"),
    "Pinnacle-ESM2 (ACM)": ("pinnacle_esm2_acm", "abmil8"),
    "ProtScape-GAE": ("gae_att_fixed_do06", "abmil8"),
    "ProtScape": ("s2gae_att_k1_fixed_do04_uni", PDL_READOUT),
}

CELL_CLASSES = [
    "Immune",
    "Stromal",
    "Muscle",
    "Epithelial",
    "Endocrine",
    "Blood",
    "Secretory",
    "Pigment",
    "Glial",
    "Neuronal",
    "Stromal-Epithelial",
    "Endothelial",
]
EPSILON = 1e-12
RAW_LRP_COLUMNS = [
    "task",
    "fold",
    "gene",
    "label",
    "prob",
    "cell_label",
    "context_evidence",
    "esm_evidence",
]
ABSOLUTE_LRP_COLUMNS = [
    "fold",
    "gene",
    "class_name",
    "label",
    "cell_label",
    "context_evidence",
    "esm_evidence",
    "context_abs_evidence",
    "esm_abs_evidence",
]


def dataset_statistics() -> pd.DataFrame:
    rows = []
    for task, disease_id, disease in TASKS:
        path = THERAPEUTIC_TARGET_DATASET_DIR / f"therapeutic_target_{disease_id}.csv"
        labels = pd.read_csv(path)
        counts = labels["label"].value_counts()
        positives = int(counts.get(1, 0))
        negatives = int(counts.get(0, 0))
        rows.append(
            {
                "task": task,
                "task_type": "therapeutic_target_prediction",
                "disease_id": disease_id,
                "disease": disease,
                "positive_label_source": "Open Targets phase >=3 or completed phase 2 evidence over root and descendants",
                "negative_label_source": "approved-human DrugBank targets without a non-literature Open Targets association for the root query",
                "label_file": path.name,
                "positive": positives,
                "negative": negatives,
                "total": len(labels),
                "positive_to_negative_ratio": positives / negatives,
                "positive_fraction": positives / len(labels),
            }
        )
    return pd.DataFrame(rows)


def mean_performance(performance: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "task",
        "readout_key",
        "readout_label",
        "inference_key",
        "inference_label",
    ]
    task_means = performance.melt(
        id_vars=identifiers,
        value_vars=["auprc", "f1"],
        var_name="metric",
        value_name="score",
    ).groupby([*identifiers, "metric"], as_index=False)["score"].mean()
    summary = task_means.groupby(
        [
            "metric",
            "readout_key",
            "readout_label",
            "inference_key",
            "inference_label",
        ],
        as_index=False,
    ).agg(
        mean_over_tasks=("score", "mean"),
        sem_over_tasks=("score", "sem"),
        std_over_tasks=("score", "std"),
        n_tasks=("task", "nunique"),
    )
    summary["score_percent"] = 100.0 * summary["mean_over_tasks"]
    summary["sem_percent"] = 100.0 * summary["sem_over_tasks"]
    return summary


def significance(p_value: float) -> str:
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return ""


def disease_model_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, _, _ in TASKS:
        task_rows = performance.loc[performance["task"].eq(task)]
        means = {}
        folds = {}
        for model_key in ["lr_esm", *MODEL_ORDER]:
            readout = "lr_esm" if model_key == "lr_esm" else PDL_READOUT
            values = task_rows.loc[
                task_rows["inference_key"].eq(model_key)
                & task_rows["readout_key"].eq(readout)
            ].sort_values("fold")["auprc"].to_numpy()
            means[model_key] = float(values.mean())
            folds[model_key] = values

        p_pinnacle = float(
            ttest_rel(
                folds["s2gae_att_k1_fixed_do04_uni"],
                folds["pinnacle_random_fixed"],
            ).pvalue
        )
        p_esm2 = float(
            ttest_rel(
                folds["s2gae_att_k1_fixed_do04_uni"],
                folds["lr_esm"],
            ).pvalue
        )
        row = {
            "task": task,
            "disease": PERFORMANCE_DISEASE_LABELS[task],
            **means,
        }
        row["delta_vs_pinnacle_percentage_points"] = 100.0 * (
            means["s2gae_att_k1_fixed_do04_uni"]
            - means["pinnacle_random_fixed"]
        )
        row["delta_vs_esm2_percentage_points"] = 100.0 * (
            means["s2gae_att_k1_fixed_do04_uni"] - means["lr_esm"]
        )
        row["p_vs_pinnacle"] = p_pinnacle
        row["p_vs_esm2"] = p_esm2
        row["stars_vs_pinnacle"] = significance(p_pinnacle)
        row["stars_vs_esm2"] = significance(p_esm2)
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(
        "s2gae_att_k1_fixed_do04_uni", ascending=False
    ).reset_index(drop=True)
    comparison["protscape_performance_rank"] = np.arange(1, len(comparison) + 1)
    return comparison


def lrp_path(inference_key: str, task: str) -> Path:
    return ANALYSIS_DIR / f"lrp__{inference_key}__{task}.csv.gz"


def generate_checkpoint_lrp() -> None:
    for _, (inference_key, readout_key) in LRP_MODELS.items():
        for task, _, _ in TASKS:
            run_dir = CHECKPOINT_ROOT / task / f"{inference_key}__{readout_key}"
            generate_lrp(
                run_dir,
                lrp_path(inference_key, task),
            )


def normalize_context(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def load_cell_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(CELL_METADATA_PATH).copy()
    if metadata["edgelist"].duplicated().any():
        raise ValueError("Cellular-context metadata contains duplicate edgelists")
    metadata["cell_key"] = metadata["edgelist"].map(normalize_context)
    if metadata["cell_key"].duplicated().any():
        raise ValueError("Normalized cellular-context keys are not unique")
    unexpected = set(metadata["cell_type_class"]) - set(CELL_CLASSES)
    if unexpected:
        raise ValueError(f"Unexpected broad cell classes: {sorted(unexpected)}")

    def display_name(row: pd.Series) -> str:
        name = str(row["canonical_name"]).strip()
        if str(row["has_condition"]).strip().lower() != "true":
            return name
        prefix = f"{row['base_cl_id']}_"
        suffix = str(row["edgelist"])
        if suffix.startswith(prefix):
            suffix = suffix[len(prefix):]
        return f"{name} · " + " · ".join(suffix.split("_"))

    metadata["display_cell_label"] = metadata.apply(display_name, axis=1)
    return metadata.set_index("cell_key")


def add_cell_metadata(values: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    values = values.copy()
    values["cell_key"] = values["cell_label"].map(normalize_context)
    values["cell_class"] = values["cell_key"].map(metadata["cell_type_class"])
    values["display_cell_label"] = values["cell_key"].map(metadata["display_cell_label"])
    if values[["cell_class", "display_cell_label"]].isna().any().any():
        missing = sorted(values.loc[values["cell_class"].isna(), "cell_label"].unique())
        raise ValueError(f"Unmapped cellular contexts: {missing[:5]}")
    return values


def read_positive_lrp(path: Path) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(path, usecols=RAW_LRP_COLUMNS, chunksize=200_000):
        chunk["gene"] = chunk["gene"].astype(str).str.upper()
        keep = chunk["label"].eq(1)
        if keep.any():
            frames.append(chunk.loc[keep].copy())
    if not frames:
        return pd.DataFrame(columns=RAW_LRP_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def calculate_cell_class_metrics(values: pd.DataFrame) -> pd.DataFrame:
    bags = ["task", "disease", "gene", "fold"]
    if not values.groupby([*bags, "cell_label"]).size().eq(1).all():
        raise RuntimeError("Duplicate contexts within a model-target bag")
    esm_range = values.groupby(bags)["esm_evidence"].agg(lambda x: x.max() - x.min())
    if esm_range.gt(1e-10).any():
        raise RuntimeError("Sequence relevance changes within a model-target bag")
    scored = values.assign(
        positive_context=values["context_evidence"].clip(lower=0.0),
        absolute_context=values["context_evidence"].abs(),
    )
    bag_values = scored.groupby(bags, as_index=False).agg(
        probability=("prob", "first"),
        esm_relevance=("esm_evidence", "first"),
        n_total_contexts=("cell_label", "nunique"),
        positive_context_total=("positive_context", "sum"),
        absolute_context_total=("absolute_context", "sum"),
    )
    observed = scored.groupby([*bags, "cell_class"], as_index=False).agg(
        n_class_contexts=("cell_label", "nunique"),
        positive_class_relevance=("positive_context", "sum"),
        signed_class_relevance=("context_evidence", "sum"),
    )
    metrics = bag_values.merge(pd.DataFrame({"cell_class": CELL_CLASSES}), how="cross")
    metrics = metrics.merge(observed, on=[*bags, "cell_class"], how="left")
    metrics[["n_class_contexts", "positive_class_relevance", "signed_class_relevance"]] = metrics[
        ["n_class_contexts", "positive_class_relevance", "signed_class_relevance"]
    ].fillna(0.0)
    positive_denominator = (
        metrics["positive_context_total"] + metrics["esm_relevance"].clip(lower=0.0)
    )
    absolute_denominator = metrics["absolute_context_total"] + metrics["esm_relevance"].abs()
    metrics["positive_context_profile_defined"] = (
        metrics["positive_context_total"].gt(EPSILON)
        & positive_denominator.gt(EPSILON)
    )
    metrics["q_positive"] = np.where(
        metrics["positive_context_profile_defined"],
        metrics["positive_context_total"] / positive_denominator,
        np.nan,
    )
    metrics["q_absolute"] = metrics["absolute_context_total"] / absolute_denominator
    metrics["positive_class_contribution"] = np.where(
        metrics["positive_context_profile_defined"],
        metrics["positive_class_relevance"] / positive_denominator,
        np.nan,
    )
    metrics["mean_positive_contribution_per_observed_context"] = np.where(
        metrics["n_class_contexts"].gt(0),
        metrics["positive_class_contribution"] / metrics["n_class_contexts"],
        0.0,
    )
    metrics["signed_class_contribution"] = (
        metrics["signed_class_relevance"] / absolute_denominator
    )
    metrics["mean_signed_contribution_per_observed_context"] = np.where(
        metrics["n_class_contexts"].gt(0),
        metrics["signed_class_contribution"] / metrics["n_class_contexts"],
        0.0,
    )
    return metrics


def prot_scape_lrp_tables(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for task, _, disease in TASKS:
        values = read_positive_lrp(
            lrp_path("s2gae_att_k1_fixed_do04_uni", task)
        )
        values["disease"] = disease
        frames.append(values)
    data = add_cell_metadata(pd.concat(frames, ignore_index=True), metadata)
    if not data.groupby(["task", "gene"])["fold"].nunique().eq(5).all():
        raise RuntimeError("Every permanent-test positive must have five explanations")
    metrics = calculate_cell_class_metrics(data)
    bag_columns = ["task", "disease", "gene", "fold"]
    bag = metrics.drop_duplicates(bag_columns)[
        [*bag_columns, "probability", "positive_context_profile_defined", "q_positive", "q_absolute", "n_total_contexts"]
    ]
    cohort = bag.groupby(["task", "disease", "gene"], as_index=False).agg(
        n_models=("fold", "nunique"),
        probability_mean=("probability", "mean"),
        probability_min=("probability", "min"),
        probability_max=("probability", "max"),
        n_models_with_positive_context_profile=("positive_context_profile_defined", "sum"),
        q_positive_mean=("q_positive", "mean"),
        q_absolute_mean=("q_absolute", "mean"),
        n_available_contexts=("n_total_contexts", "first"),
    )
    cohort["included_recovered_target"] = (
        cohort["probability_mean"].ge(0.5)
        & cohort["n_models_with_positive_context_profile"].eq(cohort["n_models"])
    )
    included = cohort.loc[cohort["included_recovered_target"], ["task", "gene"]]
    retained = metrics.merge(included, on=["task", "gene"], how="inner")
    target = retained.groupby(["task", "disease", "gene", "cell_class"], as_index=False).agg(
        mean_positive_contribution_per_observed_context=("mean_positive_contribution_per_observed_context", "mean"),
        signed_class_contribution=("signed_class_contribution", "mean"),
        mean_signed_contribution_per_observed_context=("mean_signed_contribution_per_observed_context", "mean"),
    )
    disease = target.groupby(["task", "disease", "cell_class"], as_index=False).agg(
        mean_signed_contribution_per_observed_context=("mean_signed_contribution_per_observed_context", "mean"),
        n_contributing_targets=("gene", "nunique"),
    )
    order = [disease for _, _, disease in TASKS]
    matrix = 100.0 * disease.pivot(
        index="disease", columns="cell_class", values="mean_signed_contribution_per_observed_context"
    ).reindex(index=order, columns=CELL_CLASSES)
    if matrix.isna().any().any():
        raise RuntimeError("Incomplete disease-by-cell-class attribution matrix")
    matrix = matrix.rename_axis(index="disease", columns="cell_class").reset_index()
    return data, cohort, matrix


def disease_top_contexts(
    data: pd.DataFrame,
    cohort: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    included = cohort.loc[cohort["included_recovered_target"]]
    values = data.merge(included[["task", "gene"]], on=["task", "gene"], how="inner")
    bags = ["task", "disease", "gene", "fold"]
    values["positive_context"] = values["context_evidence"].clip(lower=0.0)
    values["positive_context_total"] = values.groupby(bags)["positive_context"].transform("sum")
    values["positive_denominator"] = (
        values["positive_context_total"] + values["esm_evidence"].clip(lower=0.0)
    )
    values["overall_positive_contribution_pct"] = 100.0 * (
        values["positive_context"] / values["positive_denominator"]
    )
    contexts = metadata.reset_index()[
        ["cell_key", "display_cell_label", "cell_type_class"]
    ].rename(columns={"cell_type_class": "cell_class"})
    dense = values[bags].drop_duplicates().merge(contexts, how="cross")
    dense = dense.merge(
        values[[*bags, "cell_key", "overall_positive_contribution_pct"]],
        on=[*bags, "cell_key"],
        how="left",
        indicator=True,
    )
    dense["context_observed"] = dense["_merge"].eq("both")
    dense["overall_positive_contribution_pct"] = dense[
        "overall_positive_contribution_pct"
    ].fillna(0.0)
    target = dense.groupby(
        ["task", "disease", "gene", "cell_key", "display_cell_label", "cell_class"],
        as_index=False,
    ).agg(
        mean_over_models_pct=("overall_positive_contribution_pct", "mean"),
        context_observed_for_target=("context_observed", "any"),
        n_models=("fold", "nunique"),
    )
    disease = target.groupby(
        ["task", "disease", "cell_key", "display_cell_label", "cell_class"],
        as_index=False,
    ).agg(
        mean_positive_contribution_pct=("mean_over_models_pct", "mean"),
        n_contributing_targets=("gene", "nunique"),
        n_targets_with_context=("context_observed_for_target", "sum"),
    )
    disease["target_availability_fraction"] = (
        disease["n_targets_with_context"] / disease["n_contributing_targets"]
    )
    disease = disease.sort_values(
        ["task", "mean_positive_contribution_pct", "display_cell_label"],
        ascending=[True, False, True],
    )
    disease["rank"] = disease.groupby("task").cumcount() + 1
    top = disease.loc[disease["rank"].le(5)].copy()
    shared_max = max(0.5, np.ceil(top["mean_positive_contribution_pct"].max() * 1.08 * 2.0) / 2.0)
    top["shared_x_axis_max_pct"] = shared_max
    return top


FOCAL_TARGETS = [
    ("therapeutic_target_mondo_0005180", "Parkinson disease", "HTR1A", "Buspirone"),
    ("therapeutic_target_efo_0000571", "Lung adenocarcinoma", "ERBB3", "Patritumab deruxtecan"),
]


def focal_target_contexts(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_tables = []
    summaries = []
    maxima = []
    for task, disease, gene, clinical_anchor in FOCAL_TARGETS:
        values = data.loc[data["task"].eq(task) & data["gene"].eq(gene)].copy()
        if values.empty or values["fold"].nunique() != 5:
            raise RuntimeError(f"Expected five explanations for {task}/{gene}")
        model_tables = []
        model_summaries = []
        for fold, frame in values.groupby("fold", sort=True):
            positive = frame["context_evidence"].clip(lower=0.0)
            positive_total = float(positive.sum())
            esm_value = float(frame["esm_evidence"].iloc[0])
            denominator = positive_total + max(esm_value, 0.0)
            absolute_total = float(frame["context_evidence"].abs().sum())
            absolute_denominator = absolute_total + abs(esm_value)
            frame["overall_positive_contribution_pct"] = 100.0 * positive / denominator
            frame["model"] = int(fold) + 1
            model_tables.append(frame[["model", "cell_label", "display_cell_label", "cell_class", "overall_positive_contribution_pct"]])
            model_summaries.append(
                {
                    "probability": float(frame["prob"].iloc[0]),
                    "q_positive": positive_total / denominator,
                    "q_absolute": absolute_total / absolute_denominator,
                }
            )
        by_model = pd.concat(model_tables, ignore_index=True)
        summary = by_model.groupby(
            ["cell_label", "display_cell_label", "cell_class"], as_index=False
        ).agg(
            mean_overall_positive_contribution_pct=("overall_positive_contribution_pct", "mean"),
            sd_overall_positive_contribution_pct=("overall_positive_contribution_pct", "std"),
            n_models=("model", "nunique"),
        ).sort_values(
            ["mean_overall_positive_contribution_pct", "display_cell_label"],
            ascending=[False, True],
        )
        summary["rank"] = np.arange(1, len(summary) + 1)
        top = summary.head(5).copy()
        for column, value in {
            "task": task,
            "disease": disease,
            "gene": gene,
            "clinical_anchor": clinical_anchor,
        }.items():
            top.insert(0, column, value)
        top_tables.append(top)
        model_summary = pd.DataFrame(model_summaries)
        summaries.append(
            {
                "task": task,
                "disease": disease,
                "gene": gene,
                "clinical_anchor": clinical_anchor,
                "mean_probability": model_summary["probability"].mean(),
                "mean_q_positive": model_summary["q_positive"].mean(),
                "mean_q_absolute": model_summary["q_absolute"].mean(),
                "n_models": len(model_summary),
            }
        )
        maxima.append(float(by_model["overall_positive_contribution_pct"].max()))
    top = pd.concat(top_tables, ignore_index=True)
    top["shared_x_axis_max_pct"] = max(0.5, np.ceil(max(maxima) * 2.0) / 2.0)
    return top, pd.DataFrame(summaries)


def read_absolute_fractions(path: Path, task: str) -> pd.DataFrame:
    values = pd.read_csv(path, usecols=ABSOLUTE_LRP_COLUMNS)
    values = values.loc[values["label"].eq(1)].copy()
    values["gene"] = values["gene"].astype(str).str.upper()
    values["context_positive_evidence"] = values["context_evidence"].clip(lower=0)
    values["esm_positive_evidence"] = values["esm_evidence"].clip(lower=0)
    values["net_absolute_context_relevance"] = values["context_evidence"].abs()
    bags = values.groupby(["fold", "gene", "class_name"], as_index=False).agg(
        context_absolute_relevance=("context_abs_evidence", "sum"),
        sequence_absolute_relevance=("esm_abs_evidence", "first"),
        context_positive_relevance=("context_positive_evidence", "sum"),
        sequence_positive_relevance=("esm_positive_evidence", "first"),
        net_context_absolute_relevance=("net_absolute_context_relevance", "sum"),
        net_sequence_relevance=("esm_evidence", "first"),
        n_context_rows=("cell_label", "size"),
        n_contexts=("cell_label", "nunique"),
    )
    if not bags["n_context_rows"].eq(bags["n_contexts"]).all():
        raise RuntimeError(f"Duplicate contexts within a bag in {path}")
    absolute_denominator = bags["context_absolute_relevance"] + bags["sequence_absolute_relevance"]
    positive_denominator = bags["context_positive_relevance"] + bags["sequence_positive_relevance"]
    net_denominator = bags["net_context_absolute_relevance"] + bags["net_sequence_relevance"].abs()
    bags["q_absolute_context"] = bags["context_absolute_relevance"] / absolute_denominator
    positive_denominator = positive_denominator.where(positive_denominator.gt(EPSILON))
    bags["q_positive_context"] = (
        bags["context_positive_relevance"] / positive_denominator
    ).fillna(0.0)
    bags["q_positive_sequence"] = (
        bags["sequence_positive_relevance"] / positive_denominator
    ).fillna(0.0)
    bags["q_net_absolute_context"] = bags["net_context_absolute_relevance"] / net_denominator
    bags["task"] = task
    return bags


def absolute_contextual_relevance() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_targets = {}
    for model, (inference_key, _) in LRP_MODELS.items():
        frames = [
            read_absolute_fractions(lrp_path(inference_key, task), task)
            for task, _, _ in TASKS
        ]
        bags = pd.concat(frames, ignore_index=True)
        if not bags.groupby(["task", "gene"])["fold"].nunique().eq(5).all():
            raise RuntimeError(f"Incomplete fold explanations for {model}")
        model_targets[model] = bags.groupby(["task", "gene"], as_index=False).agg(
            q_absolute_context=("q_absolute_context", "mean"),
            q_positive_context=("q_positive_context", "mean"),
            q_positive_sequence=("q_positive_sequence", "mean"),
            q_net_absolute_context=("q_net_absolute_context", "mean"),
            n_models=("fold", "nunique"),
        )
    reference = set(map(tuple, model_targets["Pinnacle"][["task", "gene"]].to_numpy()))
    if any(set(map(tuple, table[["task", "gene"]].to_numpy())) != reference for table in model_targets.values()):
        raise RuntimeError("Absolute-relevance models do not share the same target cohort")
    targets = pd.concat(
        [table.assign(model=model) for model, table in model_targets.items()],
        ignore_index=True,
    )
    targets["disease"] = targets["task"].map(TASK_TO_DISEASE)
    diseases = targets.groupby(["model", "task", "disease"], as_index=False, sort=False).agg(
        q_absolute_context=("q_absolute_context", "mean"),
        q_positive_context=("q_positive_context", "mean"),
        q_positive_sequence=("q_positive_sequence", "mean"),
        q_net_absolute_context=("q_net_absolute_context", "mean"),
        n_targets=("gene", "nunique"),
    )
    summary = diseases.groupby("model", as_index=False, sort=False).agg(
        n_diseases=("disease", "nunique"),
        mean_q_absolute_context=("q_absolute_context", "mean"),
        median_q_absolute_context=("q_absolute_context", "median"),
        standard_deviation=("q_absolute_context", "std"),
        mean_q_positive_context=("q_positive_context", "mean"),
        mean_q_positive_sequence=("q_positive_sequence", "mean"),
        mean_q_net_absolute_context=("q_net_absolute_context", "mean"),
    )
    return targets, diseases, summary


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    statistics = dataset_statistics()
    statistics.to_csv(ANALYSIS_DIR / "dataset_statistics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model,
                "inference_key": inference_key,
                "readout_key": readout_key,
            }
            for model, (inference_key, readout_key) in LRP_MODELS.items()
        ]
    ).to_csv(ANALYSIS_DIR / "lrp_model_readouts.csv", index=False)

    performance = recompute_performance(CHECKPOINT_ROOT)
    performance.to_csv(ANALYSIS_DIR / "held_out_performance.csv", index=False)
    mean_performance(performance).to_csv(
        ANALYSIS_DIR / "mean_performance.csv", index=False
    )
    disease_model_comparison(performance).to_csv(
        ANALYSIS_DIR / "disease_model_comparison.csv", index=False
    )

    generate_checkpoint_lrp()
    metadata = load_cell_metadata()
    lrp_data, cohort, signed_matrix = prot_scape_lrp_tables(metadata)
    cohort.to_csv(ANALYSIS_DIR / "target_recovery_cohort.csv", index=False)
    signed_matrix.to_csv(
        ANALYSIS_DIR / "cell_class_signed_contribution_percent.csv", index=False
    )
    disease_top_contexts(lrp_data, cohort, metadata).to_csv(
        ANALYSIS_DIR / "disease_top_contexts.csv", index=False
    )
    focal_top, focal_summary = focal_target_contexts(lrp_data)
    focal_top.to_csv(ANALYSIS_DIR / "focal_target_top_contexts.csv", index=False)
    focal_summary.to_csv(ANALYSIS_DIR / "focal_target_summary.csv", index=False)

    targets, diseases, relevance_summary = absolute_contextual_relevance()
    targets.to_csv(
        ANALYSIS_DIR / "absolute_contextual_relevance_targets.csv", index=False
    )
    diseases.to_csv(
        ANALYSIS_DIR / "absolute_contextual_relevance_diseases.csv", index=False
    )
    relevance_summary.to_csv(
        ANALYSIS_DIR / "absolute_contextual_relevance_summary.csv", index=False
    )

    from exploration.plotting.therapeutic_target_analysis_plots import plot_all

    plot_all(ANALYSIS_DIR, ANALYSIS_DIR)
    print(f"Wrote therapeutic-target analysis to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
