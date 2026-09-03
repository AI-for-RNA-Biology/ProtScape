"""Build tables from the pretraining evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exploration.analysis.pretraining_model_specs import (
    CORE_MODEL_ORDER,
    K_NEGATIVES,
    LOSS_MODEL_ORDER,
    MODELS,
    NEGATIVE_BANK_SIZE,
    POOLING_MODEL_ORDER,
    TABLE2_MODEL_ORDER,
)


CONTEXT_FREE_MODEL_KEY = "context_free_protscape"


def add_context_free_curve(
    table: pd.DataFrame,
    metrics_path: str | Path,
    metric: str,
) -> pd.DataFrame:
    """Append context-free scores measured with the shared Cell-PPI protocol."""
    metrics_path = Path(metrics_path)
    metrics = pd.read_csv(metrics_path)
    required = {"scope", "k_negatives", metric}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            f"{metrics_path} is missing required columns: {sorted(missing)}"
        )

    scopes = set(metrics["scope"].dropna().astype(str))
    if scopes != {"cell_ppi_macro"}:
        raise ValueError(
            f"{metrics_path} has scope={sorted(scopes)}; expected cell_ppi_macro"
        )

    expected_k = {1, 10, 50, 100, 500}
    k_values = pd.to_numeric(metrics["k_negatives"], errors="raise")
    if not np.equal(k_values, k_values.astype(int)).all():
        raise ValueError(f"{metrics_path}:k_negatives must contain integers.")
    k_values = k_values.astype(int)
    if k_values.duplicated().any():
        duplicates = sorted(set(k_values[k_values.duplicated(keep=False)]))
        raise ValueError(f"{metrics_path} has duplicate k rows: {duplicates}")
    observed_k = set(k_values)
    if observed_k != expected_k:
        raise ValueError(
            f"{metrics_path} has k={sorted(observed_k)}; expected {sorted(expected_k)}"
        )

    values = pd.to_numeric(metrics[metric], errors="raise")
    if not values.between(0.0, 1.0).all():
        raise ValueError(f"{metrics_path}:{metric} must contain values in [0, 1].")

    context_free = pd.DataFrame(
        {
            "model_key": CONTEXT_FREE_MODEL_KEY,
            "k_negatives": k_values,
            "pos_neg_ratio": k_values.map(lambda k: f"1:{k}"),
            "score_percent": 100.0 * values,
        }
    ).sort_values("k_negatives")
    return pd.concat([table, context_free], ignore_index=True, sort=False)


def add_context_metadata(table: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path)
    frequent = mapping["cell_type_class"].value_counts()
    frequent = set(frequent[frequent > 10].index)
    mapping = mapping.copy()
    mapping["plot_cell_type_class"] = mapping["cell_type_class"].where(
        mapping["cell_type_class"].isin(frequent), "Other"
    )
    columns = [
        "edgelist",
        "cl_id",
        "canonical_name",
        "cell_type_class",
        "plot_cell_type_class",
        "primary_dataset",
    ]
    return table.merge(mapping[columns], on="edgelist", how="left")


def robust_rows(model_key, metrics_by_k) -> list[dict]:
    rows = []
    for k in sorted(metrics_by_k):
        metrics = metrics_by_k[k]
        rows.append(
            {
                "model_key": model_key,
                "k_negatives": k,
                "pos_neg_ratio": f"1:{k}",
                "test_roc_ppi": metrics["roc"],
                "test_ap_ppi": metrics["ap"],
                "test_acc_ppi": metrics["acc"],
                "test_f1_ppi": metrics["f1"],
                "n_cells": metrics["n_cells"],
                "n_pos": metrics["n_pos"],
                "n_neg": metrics["n_neg"],
            }
        )
    return rows


def curve_table(robust: pd.DataFrame, order, metric, *, chance=False) -> pd.DataFrame:
    table = robust[robust["model_key"].isin(order)][
        ["model_key", "k_negatives", "pos_neg_ratio", metric]
    ].copy()
    table["model_key"] = pd.Categorical(table["model_key"], order, ordered=True)
    table = table.sort_values(["model_key", "k_negatives"]).reset_index(drop=True)
    table["model_key"] = table["model_key"].astype(object)
    table["score_percent"] = 100.0 * table[metric]
    if chance:
        baseline = pd.DataFrame(
            {
                "model_key": "chance_auprc",
                "k_negatives": K_NEGATIVES,
                "pos_neg_ratio": [f"1:{k}" for k in K_NEGATIVES],
                metric: np.nan,
                "score_percent": [100.0 / (1.0 + k) for k in K_NEGATIVES],
            }
        )
        table = pd.concat([table, baseline], ignore_index=True)
    return table


def full_metrics_table(robust: pd.DataFrame, cci: dict) -> pd.DataFrame:
    """Build the complete balanced PPI/CCI metrics table."""
    by_model = robust.set_index(["model_key", "k_negatives"])
    rows = []
    for key in TABLE2_MODEL_ORDER:
        ppi = by_model.loc[(key, 1)]
        cci_metrics = cci[key]
        is_pinnacle = MODELS[key]["kind"] == "pinnacle"
        rows.append(
            {
                "model_key": key,
                "model": MODELS[key]["short_name"],
                "encoder_pooling": MODELS[key]["encoder_pooling"],
                "uniformity_enabled": MODELS[key]["uniformity"],
                "ppi_auprc": ppi["test_ap_ppi"],
                "ppi_macro_f1": ppi["test_f1_ppi"],
                "ppi_accuracy": ppi["test_acc_ppi"],
                "ppi_auroc": ppi["test_roc_ppi"],
                "cci_auprc": cci_metrics["ap"],
                "cci_macro_f1": cci_metrics["f1"],
                "cci_accuracy": cci_metrics["acc"],
                "cci_auroc": cci_metrics["roc"],
                "ppi_n_contexts": ppi["n_cells"],
                "ppi_n_pos": ppi["n_pos"],
                "ppi_n_neg": ppi["n_neg"],
                "cci_n_pos": cci_metrics["n_pos"],
                "cci_n_neg": cci_metrics["n_neg"],
                "encoder_dtype": "float16" if is_pinnacle else "float32",
                "evaluation_device": "CUDA",
                "ppi_protocol": "held_out_per_context_seeded_500_bank_macro_across_contexts",
                "ppi_negative_sampling": "structured_target_corruption",
                "ppi_negative_seed": "stable_cell_id",
                "ppi_negative_bank_size": NEGATIVE_BANK_SIZE,
                "cci_protocol": (
                    "in_sample_reconstruction"
                    if is_pinnacle
                    else "held_out_1to1_train_only_message_passing"
                ),
                "cci_scope": "cell_cell",
            }
        )
    return pd.DataFrame(rows)


def build_output_tables(robust, metagraph, cci, parameters, contextwise):
    robust = pd.DataFrame(robust)
    by_model = robust.set_index(["model_key", "k_negatives"])

    metagraph_rows = []
    for key in CORE_MODEL_ORDER:
        is_pinnacle = MODELS[key]["kind"] == "pinnacle"
        for metric in ("ap", "f1"):
            ppi_score = by_model.loc[(key, 1), f"test_{metric}_ppi"]
            meta_score = metagraph[key][metric]
            metagraph_rows.append(
                {
                    "model_key": key,
                    "metric": metric,
                    "protein_score": ppi_score,
                    "metagraph_score": meta_score,
                    "protein_percent": 100.0 * ppi_score,
                    "metagraph_percent": 100.0 * meta_score,
                    "higher_level_scope": (
                        "full_metagraph" if is_pinnacle else "cell_cell"
                    ),
                    "higher_level_protocol": (
                        "in_sample_reconstruction"
                        if is_pinnacle
                        else "held_out_1to1_train_only_message_passing"
                    ),
                }
            )

    pooling_rows = []
    for key in POOLING_MODEL_ORDER:
        for metric, label in [("ap", "AUPRC"), ("f1", "F1")]:
            pooling_rows.extend(
                [
                    {
                        "model_key": key,
                        "metric": f"PPI - {label}",
                        "score": 100.0 * by_model.loc[(key, 1), f"test_{metric}_ppi"],
                        "higher_level_scope": "not_applicable",
                        "higher_level_protocol": "not_applicable",
                    },
                    {
                        "model_key": key,
                        "metric": f"Metagraph - {label}",
                        "score": 100.0 * metagraph[key][metric],
                        "higher_level_scope": "cell_cell",
                        "higher_level_protocol": "held_out_1to1_train_only_message_passing",
                    },
                ]
            )

    contextwise = pd.DataFrame(contextwise)
    metagraph_table = pd.DataFrame(metagraph_rows)
    outputs = {
        "robust_ppi_auprc.csv": curve_table(
            robust, CORE_MODEL_ORDER, "test_ap_ppi", chance=True
        ),
        "robust_ppi_f1.csv": curve_table(
            robust, CORE_MODEL_ORDER, "test_f1_ppi"
        ),
        "loss_sensitivity_auprc.csv": curve_table(
            robust, LOSS_MODEL_ORDER, "test_ap_ppi", chance=True
        ),
        "loss_sensitivity_f1.csv": curve_table(
            robust, LOSS_MODEL_ORDER, "test_f1_ppi"
        ),
        "metagraph_auprc.csv": metagraph_table[
            metagraph_table["metric"] == "ap"
        ].reset_index(drop=True),
        "metagraph_metrics.csv": metagraph_table,
        "parameter_counts.csv": pd.DataFrame(
            [
                {
                    "model_key": key,
                    "model": MODELS[key]["name"],
                    "parameter_count": parameters[key],
                }
                for key in CORE_MODEL_ORDER
            ]
        ),
        "contextwise_ppi_auprc.csv": contextwise[contextwise["metric"] == "ap"],
        "contextwise_ppi_f1.csv": contextwise[contextwise["metric"] == "f1"],
        "pooling_sensitivity.csv": pd.DataFrame(pooling_rows),
        "pretraining_full_metrics.csv": full_metrics_table(robust, cci),
    }
    return outputs
