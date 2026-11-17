"""Therapeutic-target contextual attribution tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from downstream_tasks.config import PATHS
from downstream_tasks.run_selected import find_selected_run
from exploration.analysis.therapeutic_target.benchmark import (
    PDL_READOUT,
    TASKS,
    TASK_TO_DISEASE,
)
from exploration.analysis.therapeutic_target.checkpoint_lrp import generate_lrp


CELL_METADATA_PATH = Path(PATHS["celltype_class_mapping"])
ANALYSIS_DIR = Path(PATHS["output_root"]) / "analysis/therapeutic_target_analysis"


LRP_MODELS = {
    "Pinnacle": ("pinnacle_random_fixed", "abmil8"),
    "Pinnacle-ESM2 (GATv2)": ("pinnacle_esm_fixed", "abmil8"),
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
    "context_abs_evidence",
    "esm_abs_evidence",
]


def lrp_path(inference_key: str, task: str) -> Path:
    """Locate the compressed per-context explanations for one model and disease."""
    return ANALYSIS_DIR / f"lrp__{inference_key}__{task}.csv.gz"


def generate_checkpoint_lrp() -> None:
    """Explain the selected model for each disease."""
    for inference_key, readout_key in LRP_MODELS.values():
        for task, _, _ in TASKS:
            _, run_dir = find_selected_run(
                "therapeutic_targets",
                task=task,
                inference_key=inference_key,
                readout_key=readout_key,
            )
            generate_lrp(
                run_dir,
                lrp_path(inference_key, task),
            )


def normalize_context(value: object) -> str:
    """Normalize context labels before matching them to cell metadata."""
    return str(value).strip().lower().replace(" ", "_")


def load_cell_metadata() -> pd.DataFrame:
    """Match context identifiers to broad cell classes and readable labels."""
    metadata = pd.read_csv(CELL_METADATA_PATH).copy()
    metadata["cell_key"] = metadata["edgelist"].map(normalize_context)
    if metadata["cell_key"].duplicated().any():
        raise ValueError("Normalized cellular-context keys are not unique")
    unexpected = set(metadata["cell_type_class"]) - set(CELL_CLASSES)
    if unexpected:
        raise ValueError(f"Unexpected broad cell classes: {sorted(unexpected)}")

    display_names = []
    for _, row in metadata.iterrows():
        name = str(row["canonical_name"]).strip()
        if str(row["has_condition"]).strip().lower() == "true":
            suffix = str(row["edgelist"]).removeprefix(f"{row['base_cl_id']}_")
            name += " · " + " · ".join(suffix.split("_"))
        display_names.append(name)
    metadata["display_cell_label"] = display_names
    return metadata.set_index("cell_key")


def add_cell_metadata(values: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach broad cell classes and readable names to context-level relevance rows."""
    values = values.copy()
    values["cell_key"] = values["cell_label"].map(normalize_context)
    values["cell_class"] = values["cell_key"].map(metadata["cell_type_class"])
    values["display_cell_label"] = values["cell_key"].map(metadata["display_cell_label"])
    if values[["cell_class", "display_cell_label"]].isna().any().any():
        missing = sorted(values.loc[values["cell_class"].isna(), "cell_label"].unique())
        raise ValueError(f"Unmapped cellular contexts: {missing[:5]}")
    return values


def read_positive_lrp(path: Path) -> pd.DataFrame:
    """Read explanations for known targets without loading the full CSV at once."""
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
    """Normalize relevance within each gene/fold, then summarize by cell class."""
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
    # Include absent classes with zero relevance so diseases remain comparable.
    metrics = bag_values.merge(pd.DataFrame({"cell_class": CELL_CLASSES}), how="cross")
    metrics = metrics.merge(observed, on=[*bags, "cell_class"], how="left")
    class_columns = ["n_class_contexts", "positive_class_relevance", "signed_class_relevance"]
    metrics[class_columns] = metrics[class_columns].fillna(0.0)
    positive_denominator = (
        metrics["positive_context_total"] + metrics["esm_relevance"].clip(lower=0.0)
    )
    absolute_denominator = metrics["absolute_context_total"] + metrics["esm_relevance"].abs()
    metrics["positive_context_profile_defined"] = (
        metrics["positive_context_total"].gt(EPSILON)
        & positive_denominator.gt(EPSILON)
    )
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
    # Signed profiles use the absolute net evidence of each context and sequence.
    metrics["signed_class_contribution"] = (
        metrics["signed_class_relevance"] / absolute_denominator
    )
    metrics["mean_signed_contribution_per_observed_context"] = np.where(
        metrics["n_class_contexts"].gt(0),
        metrics["signed_class_contribution"] / metrics["n_class_contexts"],
        0.0,
    )
    return metrics


def protscape_attribution_tables(
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Build the recovered-target cohort and disease-by-cell-class heatmaps."""
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
        [*bag_columns, "probability", "positive_context_profile_defined", "n_total_contexts"]
    ]
    cohort = bag.groupby(["task", "disease", "gene"], as_index=False).agg(
        n_models=("fold", "nunique"),
        probability_mean=("probability", "mean"),
        n_models_with_positive_context_profile=("positive_context_profile_defined", "sum"),
        n_available_contexts=("n_total_contexts", "first"),
    )
    # Use ensemble-positive targets with defined positive relevance in every fold.
    cohort["included_recovered_target"] = (
        cohort["probability_mean"].ge(0.5)
        & cohort["n_models_with_positive_context_profile"].eq(cohort["n_models"])
    )
    included = cohort.loc[cohort["included_recovered_target"], ["task", "gene"]]
    retained = metrics.merge(included, on=["task", "gene"], how="inner")
    target = retained.groupby(["task", "disease", "gene", "cell_class"], as_index=False).agg(
        mean_positive_contribution_per_observed_context=("mean_positive_contribution_per_observed_context", "mean"),
        mean_signed_contribution_per_observed_context=("mean_signed_contribution_per_observed_context", "mean"),
    )
    disease = target.groupby(["task", "disease", "cell_class"], as_index=False).agg(
        mean_positive_contribution_per_observed_context=(
            "mean_positive_contribution_per_observed_context",
            "mean",
        ),
        mean_signed_contribution_per_observed_context=(
            "mean_signed_contribution_per_observed_context",
            "mean",
        ),
        n_contributing_targets=("gene", "nunique"),
    )
    strongest = target.sort_values(
        [
            "task",
            "disease",
            "cell_class",
            "mean_positive_contribution_per_observed_context",
            "gene",
        ],
        ascending=[True, True, True, False, True],
    ).drop_duplicates(["task", "disease", "cell_class"])

    order = [disease for _, _, disease in TASKS]

    def disease_cell_matrix(values: pd.DataFrame, metric: str, scale: float = 1.0) -> pd.DataFrame:
        """Pivot relevance into the fixed disease-by-cell-class order used by the heatmaps."""
        result = scale * values.pivot(
            index="disease",
            columns="cell_class",
            values=metric,
        ).reindex(index=order, columns=CELL_CLASSES)
        if result.shape != (len(TASKS), len(CELL_CLASSES)) or result.isna().any().any():
            raise RuntimeError(f"Incomplete disease matrix for {metric}")
        return result.rename_axis(
            index="disease",
            columns="cell_class",
        ).reset_index()

    positive_matrix = disease_cell_matrix(
        disease,
        "mean_positive_contribution_per_observed_context",
    )
    signed_matrix = disease_cell_matrix(
        disease,
        "mean_signed_contribution_per_observed_context",
        scale=100.0,
    )
    max_positive_matrix = disease_cell_matrix(
        strongest,
        "mean_positive_contribution_per_observed_context",
    )
    return (
        data,
        cohort,
        positive_matrix,
        signed_matrix,
        max_positive_matrix,
        strongest,
    )


def disease_top_contexts(
    data: pd.DataFrame,
    cohort: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Rank contexts by their mean positive contribution across recovered targets."""
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
    # An unobserved context contributes zero when averaging over targets.
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


def focal_target_contexts(data: pd.DataFrame) -> pd.DataFrame:
    """Select the five strongest contexts for the two illustrated targets."""
    top_tables = []
    maxima = []
    for task, disease, gene, clinical_anchor in FOCAL_TARGETS:
        values = data.loc[data["task"].eq(task) & data["gene"].eq(gene)].copy()
        if values.empty:
            print(f"Skipping {task}/{gene}: no held-out contextual explanations.")
            continue
        if values["fold"].nunique() != 5:
            raise RuntimeError(f"Expected five explanations for {task}/{gene}")
        model_tables = []
        for fold, frame in values.groupby("fold", sort=True):
            positive = frame["context_evidence"].clip(lower=0.0)
            positive_total = float(positive.sum())
            esm_value = float(frame["esm_evidence"].iloc[0])
            denominator = positive_total + max(esm_value, 0.0)
            frame["overall_positive_contribution_pct"] = 100.0 * positive / denominator
            frame["model"] = int(fold) + 1
            model_tables.append(frame[["model", "cell_label", "display_cell_label", "cell_class", "overall_positive_contribution_pct"]])
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
        maxima.append(float(by_model["overall_positive_contribution_pct"].max()))
    if not top_tables:
        return pd.DataFrame(columns=["task", "disease", "gene", "shared_x_axis_max_pct"])
    top = pd.concat(top_tables, ignore_index=True)
    top["shared_x_axis_max_pct"] = max(0.5, np.ceil(max(maxima) * 2.0) / 2.0)
    return top


def read_absolute_fractions(path: Path, task: str) -> pd.DataFrame:
    """Measure contextual |LRP| as a fraction of total feature-level |LRP|."""
    values = pd.read_csv(path, usecols=ABSOLUTE_LRP_COLUMNS)
    values = values.loc[values["label"].eq(1)].copy()
    values["gene"] = values["gene"].astype(str).str.upper()
    bags = values.groupby(["fold", "gene", "class_name"], as_index=False).agg(
        context_absolute_relevance=("context_abs_evidence", "sum"),
        sequence_absolute_relevance=("esm_abs_evidence", "first"),
        n_context_rows=("cell_label", "size"),
        n_contexts=("cell_label", "nunique"),
    )
    if not bags["n_context_rows"].eq(bags["n_contexts"]).all():
        raise RuntimeError(f"Duplicate contexts within a bag in {path}")
    absolute_denominator = bags["context_absolute_relevance"] + bags["sequence_absolute_relevance"]
    bags["q_absolute_context"] = bags["context_absolute_relevance"] / absolute_denominator
    bags["task"] = task
    return bags


def absolute_contextual_relevance() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Average contextual relevance over folds, targets and then diseases."""
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
            n_models=("fold", "nunique"),
        )
    reference = set(map(tuple, model_targets["Pinnacle"][["task", "gene"]].to_numpy()))
    for model, table in model_targets.items():
        cohort = set(map(tuple, table[["task", "gene"]].to_numpy()))
        if cohort != reference:
            raise RuntimeError(f"{model} uses a different target cohort for relevance")
    targets = pd.concat(
        [table.assign(model=model) for model, table in model_targets.items()],
        ignore_index=True,
    )
    targets["disease"] = targets["task"].map(TASK_TO_DISEASE)
    diseases = targets.groupby(["model", "task", "disease"], as_index=False, sort=False).agg(
        q_absolute_context=("q_absolute_context", "mean"),
        n_targets=("gene", "nunique"),
    )
    summary = diseases.groupby("model", as_index=False, sort=False).agg(
        n_diseases=("disease", "nunique"),
        mean_q_absolute_context=("q_absolute_context", "mean"),
        median_q_absolute_context=("q_absolute_context", "median"),
        standard_deviation=("q_absolute_context", "std"),
    )
    return targets, diseases, summary
