"""Validate and summarize the fixed global-S2GAE linear-probe experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import THERAPEUTIC_TARGET_IDS
from .data.global_split import split_fingerprint
from .data.task_loaders import TherapeuticTargetLoader


MODEL_KEYS = ("lr_global", "lr_global_ext_embed", "lr_ext_embed")
TASKS = ("corum",) + tuple(
    f"therapeutic_target_{disease_id.lower()}"
    for disease_id in THERAPEUTIC_TARGET_IDS
)
METRICS = ("test_auroc_macro_mean", "test_auprc_macro_mean", "test_f1_macro_mean")
PAPER_REFERENCES = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "downstream"
    / "paper_lr_references.csv"
)
PAPER_CLASS_BALANCES = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "downstream"
    / "paper_tt_class_balances.csv"
)
KNOWN_PAPER_COHORT_MISMATCHES = {
    "therapeutic_target_efo_0000305",
    "therapeutic_target_mondo_0007915",
}
COMMON_PROVENANCE_FIELDS = (
    "gene_universe",
    "inference_name",
    "embedding_inference_name",
    "inference_dir",
    "sequence_embedding_path",
    "esm2_embedding_path",
    "global_embedding_path",
    "global_embedding_manifest",
    "global_embedding_manifest_sha256",
    "global_embedding_sha256",
    "global_checkpoint_sha256",
    "global_training_git_commit",
    "global_export_git_commit",
    "global_embedding_topology",
    "global_embedding_representation",
    "context_ppi_edgelists",
    "context_presence_count",
    "context_presence_fingerprint",
    "downstream_git_commit",
)
TASK_PROVENANCE_FIELDS = (
    "task_csv_path",
    "task_csv_sha256",
    "task_dataset_manifest",
    "task_dataset_manifest_sha256",
    "task_dataset_reconstruction",
    "open_targets_release",
    "open_targets_association_scope",
)
TASK_COHORT_FIELDS = (
    "n_shared_samples",
    "n_label_classes",
    "n_positive",
    "n_negative",
)
EXPECTED_VALUES: dict[str, Any] = {
    "embedding_source": "esm",
    "dataset_mode": "bulk",
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "seed": 42,
    "train_selection_metric": "auprc",
    "use_class_weights": True,
    "n_folds": 6,
    "n_cv_folds": 5,
    "test_fold": 0,
    "epochs": 300,
    "patience": 50,
    "split_protocol": "fold_0_test_remaining_folds_rotate_validation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--inference-model",
        default="global_s2gae_grid500_full_reference",
    )
    parser.add_argument("--summary-dir", type=Path)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _load_split_artifact(path: Path) -> tuple[np.ndarray, str, str, str]:
    with np.load(path, allow_pickle=False) as arrays:
        required = {
            "genes",
            "split_fingerprint",
            "context_presence_fingerprint",
            "split_stratification",
        }
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise ValueError(f"Missing split metadata in {path}: {', '.join(missing)}")
        genes = np.asarray(arrays["genes"]).astype(str)
        if genes.ndim != 1 or not len(genes):
            raise ValueError(f"Invalid gene ordering in {path}.")
        upper = np.char.upper(genes)
        if len(set(upper.tolist())) != len(upper):
            raise ValueError(f"Split genes are not unique after uppercasing in {path}.")

        fold_numbers = sorted(
            int(match.group(1))
            for key in arrays.files
            if (match := re.fullmatch(r"fold_(\d+)", key))
        )
        if fold_numbers != list(range(int(EXPECTED_VALUES["n_folds"]))):
            raise ValueError(f"Expected folds 0-5 in {path}, found {fold_numbers}.")
        folds = [
            np.asarray(arrays[f"fold_{index}"], dtype=np.int64)
            for index in fold_numbers
        ]
        assigned = np.concatenate(folds)
        if len(assigned) != len(genes) or not np.array_equal(
            np.sort(assigned), np.arange(len(genes), dtype=np.int64)
        ):
            raise ValueError(f"Split folds do not partition the gene cohort in {path}.")
        embedded = _text(np.asarray(arrays["split_fingerprint"]).item())
        context_fingerprint = _text(
            np.asarray(arrays["context_presence_fingerprint"]).item()
        )
        stratification = _text(
            np.asarray(arrays["split_stratification"]).item()
        )

    computed = split_fingerprint(genes.tolist(), folds)
    if embedded != computed:
        raise ValueError(
            f"Embedded split fingerprint does not match split contents in {path}."
        )
    return upper, computed, context_fingerprint, stratification


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return str(actual).strip().lower() in {"true", "1"} if expected else str(
            actual
        ).strip().lower() in {"false", "0"}
    if isinstance(expected, float):
        try:
            return bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            return False
    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _load_result(path: Path, task: str, model_key: str, inference_model: str) -> dict:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(frame)}.")
    row = frame.iloc[0].to_dict()
    expected = {
        **EXPECTED_VALUES,
        "task": task,
        "base_model_key": model_key,
        "inference_name": inference_model,
    }
    required = (
        *expected,
        *METRICS,
        *COMMON_PROVENANCE_FIELDS,
        *TASK_PROVENANCE_FIELDS,
        *TASK_COHORT_FIELDS,
        "n_samples",
        "shared_split_fingerprint",
        "split_stratification",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
    mismatched = [
        key for key, value in expected.items() if not _matches(row[key], value)
    ]
    if mismatched:
        raise ValueError(f"Unexpected protocol in {path}: {', '.join(mismatched)}")
    nonfinite = [metric for metric in METRICS if not np.isfinite(float(row[metric]))]
    if nonfinite:
        raise ValueError(f"Non-finite test metrics in {path}: {', '.join(nonfinite)}")
    return row


def _require_common_provenance(
    frame: pd.DataFrame,
    fields: tuple[str, ...],
    scope: str,
) -> None:
    mismatched = [
        field
        for field in fields
        if len({_text(value) for value in frame[field].tolist()}) != 1
    ]
    if mismatched:
        raise ValueError(
            f"Results do not share {scope} provenance: {', '.join(mismatched)}"
        )


def _verify_task_input(row: dict) -> Path:
    path_text = _text(row["task_csv_path"])
    if not path_text:
        raise ValueError("Result has no task_csv_path provenance.")
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256_file(path) != _text(row["task_csv_sha256"]):
        raise ValueError(f"Task input checksum differs from results.csv: {path}")
    return path


def _verify_tt_manifest(frame: pd.DataFrame) -> None:
    _require_common_provenance(
        frame,
        (
            "task_dataset_manifest",
            "task_dataset_manifest_sha256",
            "task_dataset_reconstruction",
            "open_targets_release",
            "open_targets_association_scope",
        ),
        "therapeutic-target dataset",
    )
    row = frame.iloc[0]
    manifest_text = _text(row["task_dataset_manifest"])
    if not manifest_text:
        raise ValueError("Therapeutic-target results have no dataset manifest.")
    manifest = Path(manifest_text).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if _sha256_file(manifest) != _text(row["task_dataset_manifest_sha256"]):
        raise ValueError(
            f"Therapeutic-target manifest checksum differs from results.csv: {manifest}"
        )


def _global_export_provenance(row: pd.Series) -> dict[str, Any]:
    embedding = Path(_text(row["global_embedding_path"])).expanduser().resolve()
    esm2 = Path(_text(row["esm2_embedding_path"])).expanduser().resolve()
    contexts = Path(_text(row["context_ppi_edgelists"])).expanduser().resolve()
    for path in (embedding, esm2):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not contexts.is_dir():
        raise FileNotFoundError(contexts)

    manifest_path = embedding.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "model_type": "global_s2gae",
        "embedding_scope": "global",
        "embedding_topology": "full_reference",
        "representation": "encoder_jk_concat",
        "downstream_only": True,
        "ppi_test_compatible": False,
    }
    missing = sorted(
        {
            "checkpoint_sha256",
            "embedding_sha256",
            "protein_names_sha256",
            "graph_fingerprint",
            "feature_fingerprint",
            "split_fingerprint",
            "training_git_commit",
            "export_git_commit",
        }.difference(manifest)
    )
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    if missing or mismatched:
        details = [*(f"missing {key}" for key in missing), *mismatched]
        raise ValueError(
            f"Invalid global export manifest {manifest_path}: {', '.join(details)}"
        )
    recorded = {
        "global_embedding_manifest": str(manifest_path),
        "global_embedding_manifest_sha256": _sha256_file(manifest_path),
        "global_embedding_sha256": manifest["embedding_sha256"],
        "global_checkpoint_sha256": manifest["checkpoint_sha256"],
        "global_training_git_commit": manifest["training_git_commit"],
        "global_export_git_commit": manifest["export_git_commit"],
        "global_embedding_topology": manifest["embedding_topology"],
        "global_embedding_representation": manifest["representation"],
    }
    recorded_mismatches = [
        key for key, value in recorded.items() if _text(row[key]) != _text(value)
    ]
    if recorded_mismatches:
        raise ValueError(
            "Global export provenance differs from results.csv: "
            + ", ".join(recorded_mismatches)
        )
    downstream_commit = _text(row["downstream_git_commit"])
    if (
        not downstream_commit
        or downstream_commit == "uncommitted"
        or downstream_commit != _text(manifest["export_git_commit"])
    ):
        raise ValueError(
            "Downstream results and the frozen export do not share one reviewed commit."
        )
    return {
        **recorded,
        **{
            f"global_export_{key}": manifest[key]
            for key in (
                "checkpoint_sha256",
                "embedding_sha256",
                "protein_names_sha256",
                "graph_fingerprint",
                "feature_fingerprint",
                "split_fingerprint",
                "training_git_commit",
                "export_git_commit",
            )
        },
        "esm2_embedding_sha256": _sha256_file(esm2),
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _cohort_comparison(
    task_genes: dict[str, np.ndarray],
    task_inputs: dict[str, Path],
    all_results: pd.DataFrame,
) -> pd.DataFrame:
    references = pd.read_csv(PAPER_CLASS_BALANCES)
    expected_tasks = set(TASKS[1:])
    if set(references["task"]) != expected_tasks or len(references) != len(
        expected_tasks
    ):
        raise ValueError("Paper therapeutic-target class balances are incomplete.")

    observed = []
    for task in TASKS[1:]:
        genes, labels, _ = TherapeuticTargetLoader(task_inputs[task]).load()
        label_by_gene = {
            gene: int(label)
            for gene, label in zip(genes, labels[:, 0], strict=True)
        }
        cohort = task_genes[task].tolist()
        missing = sorted(set(cohort).difference(label_by_gene))
        if missing:
            raise ValueError(
                f"Split cohort has genes absent from {task_inputs[task]}: "
                + ", ".join(missing[:5])
            )
        cohort_labels = np.asarray([label_by_gene[gene] for gene in cohort])
        positives = int(cohort_labels.sum())
        result = all_results[
            (all_results["task"] == task)
            & (all_results["base_model_key"] == MODEL_KEYS[0])
        ].iloc[0]
        recorded = (int(result["n_positive"]), int(result["n_negative"]))
        calculated = (positives, int(len(cohort_labels) - positives))
        if recorded != calculated:
            raise ValueError(
                f"Recorded class balance differs from the split cohort for {task}."
            )
        observed.append(
            {
                "task": task,
                "current_positive": positives,
                "current_negative": int(len(cohort_labels) - positives),
                "current_total": int(len(cohort_labels)),
            }
        )

    comparison = references.merge(
        pd.DataFrame(observed),
        on="task",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    comparison["paper_total"] = (
        comparison["paper_positive"] + comparison["paper_negative"]
    )
    comparison["positive_delta"] = (
        comparison["current_positive"] - comparison["paper_positive"]
    )
    comparison["negative_delta"] = (
        comparison["current_negative"] - comparison["paper_negative"]
    )
    comparison["total_delta"] = (
        comparison["current_total"] - comparison["paper_total"]
    )
    comparison["cohort_match"] = (
        (comparison["positive_delta"] == 0)
        & (comparison["negative_delta"] == 0)
    )
    comparison["comparison_basis"] = np.where(
        comparison["cohort_match"],
        "paper_class_counts_match",
        "descriptive_non_like_for_like",
    )
    unexpected = set(comparison.loc[~comparison["cohort_match"], "task"]).difference(
        KNOWN_PAPER_COHORT_MISMATCHES
    )
    if unexpected:
        raise ValueError(
            "Unexpected paper/shared-cohort mismatch: " + ", ".join(sorted(unexpected))
        )
    return comparison


def _same_cohort_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("corum", "therapeutic_target_mean"):
        baseline = summary[
            (summary["scope"] == scope)
            & (summary["model_key"] == "lr_ext_embed")
        ].iloc[0]
        for model_key in ("lr_global", "lr_global_ext_embed"):
            observed = summary[
                (summary["scope"] == scope)
                & (summary["model_key"] == model_key)
            ].iloc[0]
            rows.append(
                {
                    "scope": scope,
                    "model_key": model_key,
                    "baseline_model_key": "lr_ext_embed",
                    "n_tasks": int(observed["n_tasks"]),
                    "our_auprc_pct": observed[
                        "test_auprc_macro_mean_percent"
                    ],
                    "baseline_auprc_pct": baseline[
                        "test_auprc_macro_mean_percent"
                    ],
                    "auprc_delta_pp": observed[
                        "test_auprc_macro_mean_percent"
                    ]
                    - baseline["test_auprc_macro_mean_percent"],
                    "our_f1_pct": observed["test_f1_macro_mean_percent"],
                    "baseline_f1_pct": baseline["test_f1_macro_mean_percent"],
                    "f1_delta_pp": observed["test_f1_macro_mean_percent"]
                    - baseline["test_f1_macro_mean_percent"],
                    "comparison_priority": "primary",
                    "comparison_basis": "same_cohort_esm2",
                    "is_like_for_like": True,
                }
            )
    frame = pd.DataFrame(rows)
    numeric = [
        "our_auprc_pct",
        "baseline_auprc_pct",
        "auprc_delta_pp",
        "our_f1_pct",
        "baseline_f1_pct",
        "f1_delta_pp",
    ]
    frame[numeric] = frame[numeric].round(6)
    return frame


def _paper_comparison(
    summary: pd.DataFrame,
    cohort_comparison: pd.DataFrame,
) -> pd.DataFrame:
    references = pd.read_csv(PAPER_REFERENCES)
    observed = summary[
        [
            "scope",
            "model_key",
            "n_tasks",
            "test_auprc_macro_mean_percent",
            "test_f1_macro_mean_percent",
        ]
    ].rename(
        columns={
            "model_key": "our_model_key",
            "test_auprc_macro_mean_percent": "our_auprc_pct",
            "test_f1_macro_mean_percent": "our_f1_pct",
        }
    )
    comparison = references.merge(
        observed,
        on=["scope", "our_model_key"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if comparison[["our_auprc_pct", "our_f1_pct"]].isna().any().any():
        raise ValueError("Paper reference has no matching aggregated result.")
    comparison["auprc_delta_pp"] = (
        comparison["our_auprc_pct"] - comparison["paper_auprc_pct"]
    )
    comparison["f1_delta_pp"] = (
        comparison["our_f1_pct"] - comparison["paper_f1_pct"]
    )
    n_matching = int(cohort_comparison["cohort_match"].sum())
    n_cohorts = len(cohort_comparison)
    comparison["comparison_priority"] = "descriptive"
    comparison["comparison_basis"] = np.where(
        comparison["scope"] == "therapeutic_target_mean",
        "descriptive_non_like_for_like",
        "descriptive_paper_reference_cohort_not_audited",
    )
    comparison["is_like_for_like"] = False
    comparison["paper_cohort_tasks_matching"] = np.where(
        comparison["scope"] == "therapeutic_target_mean", n_matching, np.nan
    )
    comparison["paper_cohort_tasks_total"] = np.where(
        comparison["scope"] == "therapeutic_target_mean", n_cohorts, np.nan
    )
    comparison[["our_auprc_pct", "our_f1_pct", "auprc_delta_pp", "f1_delta_pp"]] = (
        comparison[
            ["our_auprc_pct", "our_f1_pct", "auprc_delta_pp", "f1_delta_pp"]
        ].round(6)
    )
    return comparison


def aggregate(output_root: Path, inference_model: str, summary_dir: Path) -> None:
    rows = []
    split_fingerprints: dict[str, str] = {}
    source_results: dict[str, str] = {}
    task_genes: dict[str, np.ndarray] = {}
    task_inputs: dict[str, Path] = {}

    for task in TASKS:
        task_dir = output_root / task / inference_model
        candidates = sorted(task_dir.glob("*/results.csv"))
        by_model: dict[str, list[Path]] = {model: [] for model in MODEL_KEYS}
        for path in candidates:
            frame = pd.read_csv(path, usecols=lambda column: column == "base_model_key")
            if len(frame) == 1 and frame.iloc[0].get("base_model_key") in by_model:
                by_model[str(frame.iloc[0]["base_model_key"])].append(path)

        task_rows = []
        task_splits = set()
        task_gene_orders = []
        for model_key in MODEL_KEYS:
            paths = by_model[model_key]
            if len(paths) != 1:
                raise FileNotFoundError(
                    f"Expected one {model_key} result for {task}, found {len(paths)}."
                )
            result_path = paths[0]
            split_path = result_path.parent / "split_indices.npz"
            if not split_path.is_file():
                raise FileNotFoundError(split_path)
            genes, split_hash, context_fingerprint, stratification = (
                _load_split_artifact(split_path)
            )
            task_splits.add(split_hash)
            row = _load_result(result_path, task, model_key, inference_model)
            if _text(row["shared_split_fingerprint"]) != split_hash:
                raise ValueError(
                    f"results.csv split fingerprint differs from {split_path}."
                )
            if _text(row["context_presence_fingerprint"]) != context_fingerprint:
                raise ValueError(
                    f"Context-presence fingerprint differs from {split_path}."
                )
            if _text(row["split_stratification"]) != stratification:
                raise ValueError(
                    f"Split-stratification method differs from {split_path}."
                )
            if int(row["n_samples"]) != len(genes):
                raise ValueError(f"Sample count differs from {split_path}.")
            if int(row["n_shared_samples"]) != len(genes):
                raise ValueError(f"Shared sample count differs from {split_path}.")
            row["results_path"] = str(result_path)
            row["split_fingerprint_verified"] = split_hash
            rows.append(row)
            task_rows.append(row)
            task_gene_orders.append(genes)
            source_results[str(result_path)] = _sha256_file(result_path)
        if len(task_splits) != 1:
            raise ValueError(f"The three probes do not share one split for {task}.")
        if any(
            not np.array_equal(task_gene_orders[0], genes)
            for genes in task_gene_orders[1:]
        ):
            raise ValueError(f"The three probes do not share one cohort for {task}.")
        task_frame = pd.DataFrame(task_rows)
        _require_common_provenance(
            task_frame,
            (
                *COMMON_PROVENANCE_FIELDS,
                *TASK_PROVENANCE_FIELDS,
                *TASK_COHORT_FIELDS,
                "split_stratification",
            ),
            f"{task} input",
        )
        task_inputs[task] = _verify_task_input(task_rows[0])
        task_genes[task] = task_gene_orders[0]
        split_fingerprints[task] = task_splits.pop()

    all_results = pd.DataFrame(rows)
    order = {task: index for index, task in enumerate(TASKS)}
    model_order = {model: index for index, model in enumerate(MODEL_KEYS)}
    all_results["_task_order"] = all_results["task"].map(order)
    all_results["_model_order"] = all_results["base_model_key"].map(model_order)
    all_results = all_results.sort_values(["_task_order", "_model_order"]).drop(
        columns=["_task_order", "_model_order"]
    )
    _require_common_provenance(
        all_results,
        COMMON_PROVENANCE_FIELDS,
        "cross-task global input",
    )
    tt_results = all_results[all_results["task"] != "corum"]
    _verify_tt_manifest(tt_results)
    export_provenance = _global_export_provenance(all_results.iloc[0])
    for key, value in export_provenance.items():
        all_results[key] = value

    cohort_comparison = _cohort_comparison(task_genes, task_inputs, all_results)

    summary_rows = []
    scopes = {
        "corum": ("corum",),
        "therapeutic_target_mean": TASKS[1:],
    }
    for scope, tasks in scopes.items():
        for model_key in MODEL_KEYS:
            selected = all_results[
                all_results["task"].isin(tasks)
                & (all_results["base_model_key"] == model_key)
            ]
            summary: dict[str, Any] = {
                "scope": scope,
                "model_key": model_key,
                "n_tasks": len(selected),
            }
            for metric in METRICS:
                values = selected[metric].astype(float).to_numpy()
                summary[metric] = float(values.mean())
                summary[f"{metric}_percent"] = float(100.0 * values.mean())
                summary[f"{metric}_task_sd"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
                summary[f"{metric}_task_sem"] = (
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0
                )
            summary_rows.append(summary)
    summary = pd.DataFrame(summary_rows)

    summary_dir.mkdir(parents=True, exist_ok=True)
    all_results_path = summary_dir / "all_results.csv"
    summary_path = summary_dir / "summary.csv"
    paper_comparison_path = summary_dir / "paper_comparison.csv"
    same_cohort_path = summary_dir / "same_cohort_comparison.csv"
    cohort_comparison_path = summary_dir / "cohort_comparison.csv"
    _atomic_csv(all_results, all_results_path)
    _atomic_csv(summary, summary_path)
    _atomic_csv(
        _paper_comparison(summary, cohort_comparison), paper_comparison_path
    )
    _atomic_csv(_same_cohort_comparison(summary), same_cohort_path)
    _atomic_csv(cohort_comparison, cohort_comparison_path)
    completion = {
        "format_version": 3,
        "inference_model": inference_model,
        "model_keys": list(MODEL_KEYS),
        "tasks": list(TASKS),
        "n_results": len(all_results),
        "split_fingerprints": split_fingerprints,
        "source_results_sha256": source_results,
        "task_inputs_sha256": {
            task: _sha256_file(path) for task, path in task_inputs.items()
        },
        "global_export_provenance": export_provenance,
        "downstream_git_commit": _text(all_results.iloc[0]["downstream_git_commit"]),
        "all_results_sha256": _sha256_file(all_results_path),
        "summary_sha256": _sha256_file(summary_path),
        "paper_references_sha256": _sha256_file(PAPER_REFERENCES),
        "paper_class_balances_sha256": _sha256_file(PAPER_CLASS_BALANCES),
        "paper_comparison_sha256": _sha256_file(paper_comparison_path),
        "same_cohort_comparison_sha256": _sha256_file(same_cohort_path),
        "cohort_comparison_sha256": _sha256_file(cohort_comparison_path),
    }
    _atomic_json(completion, summary_dir / "completed.json")
    print(summary.to_string(index=False))
    print(f"Validated results: {summary_dir}")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    summary_dir = (
        args.summary_dir.expanduser().resolve()
        if args.summary_dir is not None
        else output_root / "global_s2gae_summary"
    )
    aggregate(output_root, args.inference_model, summary_dir)


if __name__ == "__main__":
    main()
