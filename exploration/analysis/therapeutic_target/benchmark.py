"""Therapeutic-target benchmark and performance tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from downstream_tasks.config import PATHS


THERAPEUTIC_TARGET_DATASET_DIR = Path(PATHS["therapeutic_target_dataset_dir"])


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
