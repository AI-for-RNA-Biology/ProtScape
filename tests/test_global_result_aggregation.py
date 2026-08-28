import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream_tasks.aggregate_global_s2gae_results import (
    EXPECTED_VALUES,
    KNOWN_PAPER_COHORT_MISMATCHES,
    MODEL_KEYS,
    PAPER_CLASS_BALANCES,
    TASKS,
    _load_split_artifact,
    aggregate,
)
from downstream_tasks.data.global_split import split_fingerprint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_split(
    path: Path,
    genes: list[str],
    context_fingerprint: str,
    stratification: str,
) -> str:
    folds = [part.astype(np.int64) for part in np.array_split(np.arange(len(genes)), 6)]
    fingerprint = split_fingerprint(genes, folds)
    np.savez_compressed(
        path,
        genes=np.asarray(genes, dtype=str),
        **{f"fold_{index}": fold for index, fold in enumerate(folds)},
        split_fingerprint=np.asarray(fingerprint, dtype=str),
        context_presence_fingerprint=np.asarray(context_fingerprint, dtype=str),
        split_stratification=np.asarray(stratification, dtype=str),
    )
    return fingerprint


def _write_fixture(tmp_path: Path) -> tuple[Path, str]:
    output_root = tmp_path / "results"
    inference_model = "global_s2gae_full_reference"
    export_dir = tmp_path / "export" / inference_model
    export_dir.mkdir(parents=True)
    global_embedding = export_dir / "protein_embeddings.pt"
    global_embedding.write_bytes(b"frozen global embedding")
    manifest = {
        "model_type": "global_s2gae",
        "embedding_scope": "global",
        "embedding_topology": "full_reference",
        "representation": "encoder_jk_concat",
        "downstream_only": True,
        "ppi_test_compatible": False,
        "checkpoint_sha256": "checkpoint-sha",
        "embedding_sha256": "embedding-sha",
        "protein_names_sha256": "names-sha",
        "graph_fingerprint": "graph-fingerprint",
        "feature_fingerprint": "feature-fingerprint",
        "split_fingerprint": "pretraining-split-fingerprint",
        "training_git_commit": "training-commit",
        "export_git_commit": "reviewed-commit",
    }
    export_manifest = export_dir / "manifest.json"
    export_manifest.write_text(json.dumps(manifest))
    esm2 = tmp_path / "esm2.plk"
    esm2.write_bytes(b"frozen esm2 embedding")
    contexts = tmp_path / "ppi_edgelists"
    contexts.mkdir()

    dataset_dir = tmp_path / "labels"
    dataset_dir.mkdir()
    tt_manifest = dataset_dir / "therapeutic_target_manifest.json"
    tt_manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "dataset": "therapeutic_target",
                "reconstruction": "official_static_release",
                "open_targets_release": "26.03",
                "association_scope": "indirect",
            }
        )
    )
    paper_balances = pd.read_csv(PAPER_CLASS_BALANCES).set_index("task")
    metric_by_model = {
        "lr_global": (0.70, 0.60, 0.50),
        "lr_global_ext_embed": (0.75, 0.65, 0.55),
        "lr_ext_embed": (0.65, 0.55, 0.45),
    }
    context_fingerprint = "shared-context-fingerprint"

    for task in TASKS:
        if task == "corum":
            stratification = "task_label_multilabel_fallback"
            genes = [f"CORUM_{index:04d}" for index in range(12)]
            task_csv = dataset_dir / "corum.csv"
            pd.DataFrame(
                {"protein": genes, "complex_id": ["complex"] * len(genes)}
            ).to_csv(task_csv, index=False)
            dataset_provenance = {
                "task_dataset_manifest": "",
                "task_dataset_manifest_sha256": "",
                "task_dataset_reconstruction": "",
                "open_targets_release": "",
                "open_targets_association_scope": "",
            }
            cohort_provenance = {
                "n_shared_samples": len(genes),
                "n_label_classes": 1,
                "n_positive": "",
                "n_negative": "",
            }
        else:
            stratification = "task_label_and_context_cluster"
            paper = paper_balances.loc[task]
            positives = int(paper["paper_positive"])
            negatives = int(paper["paper_negative"])
            if task == "therapeutic_target_efo_0000305":
                negatives += 1
            elif task == "therapeutic_target_mondo_0007915":
                positives -= 1
            prefix = task.removeprefix("therapeutic_target_").upper()
            genes = [f"{prefix}_P{index:04d}" for index in range(positives)] + [
                f"{prefix}_N{index:04d}" for index in range(negatives)
            ]
            task_csv = dataset_dir / f"{task}.csv"
            pd.DataFrame(
                {
                    "protein": genes,
                    "label": [1] * positives + [0] * negatives,
                }
            ).to_csv(task_csv, index=False)
            dataset_provenance = {
                "task_dataset_manifest": str(tt_manifest.resolve()),
                "task_dataset_manifest_sha256": _sha256(tt_manifest),
                "task_dataset_reconstruction": "official_static_release",
                "open_targets_release": "26.03",
                "open_targets_association_scope": "indirect",
            }
            cohort_provenance = {
                "n_shared_samples": len(genes),
                "n_label_classes": 1,
                "n_positive": positives,
                "n_negative": negatives,
            }

        for model_key in MODEL_KEYS:
            output_dir = output_root / task / inference_model / model_key
            output_dir.mkdir(parents=True)
            fingerprint = _write_split(
                output_dir / "split_indices.npz",
                genes,
                context_fingerprint,
                stratification,
            )
            auroc, auprc, f1 = metric_by_model[model_key]
            row = {
                **EXPECTED_VALUES,
                "test_auroc_macro_mean": auroc,
                "test_auprc_macro_mean": auprc,
                "test_f1_macro_mean": f1,
                "task": task,
                "task_csv_path": str(task_csv.resolve()),
                "task_csv_sha256": _sha256(task_csv),
                "base_model_key": model_key,
                "inference_name": inference_model,
                "embedding_inference_name": inference_model,
                "inference_dir": inference_model,
                "sequence_embedding_path": esm2.name,
                "esm2_embedding_path": str(esm2.resolve()),
                "global_embedding_path": str(global_embedding.resolve()),
                "global_embedding_manifest": str(export_manifest.resolve()),
                "global_embedding_manifest_sha256": _sha256(export_manifest),
                "global_embedding_sha256": manifest["embedding_sha256"],
                "global_checkpoint_sha256": manifest["checkpoint_sha256"],
                "global_training_git_commit": manifest["training_git_commit"],
                "global_export_git_commit": manifest["export_git_commit"],
                "global_embedding_topology": manifest["embedding_topology"],
                "global_embedding_representation": manifest["representation"],
                "context_ppi_edgelists": str(contexts.resolve()),
                "context_presence_count": 207,
                "context_presence_fingerprint": context_fingerprint,
                "gene_universe": "task_global_esm2_cellppi",
                "downstream_git_commit": "reviewed-commit",
                "n_samples": len(genes),
                "shared_split_fingerprint": fingerprint,
                "split_stratification": stratification,
                **dataset_provenance,
                **cohort_provenance,
            }
            pd.DataFrame([row]).to_csv(output_dir / "results.csv", index=False)

    return output_root, inference_model


def test_global_result_aggregation_reports_primary_same_cohort_comparison(tmp_path):
    output_root, inference_model = _write_fixture(tmp_path)
    summary_dir = tmp_path / "summary"

    aggregate(output_root, inference_model, summary_dir)

    all_results = pd.read_csv(summary_dir / "all_results.csv")
    summary = pd.read_csv(summary_dir / "summary.csv")
    paper = pd.read_csv(summary_dir / "paper_comparison.csv")
    same_cohort = pd.read_csv(summary_dir / "same_cohort_comparison.csv")
    cohorts = pd.read_csv(summary_dir / "cohort_comparison.csv")
    completion = json.loads((summary_dir / "completed.json").read_text())

    assert len(all_results) == len(TASKS) * len(MODEL_KEYS) == 48
    assert len(summary) == 2 * len(MODEL_KEYS) == 6
    assert set(summary["n_tasks"]) == {1, 15}
    assert all_results["global_embedding_manifest_sha256"].nunique() == 1
    assert (
        all_results["split_fingerprint_verified"]
        == all_results["shared_split_fingerprint"]
    ).all()

    assert len(same_cohort) == 4
    assert set(same_cohort["comparison_priority"]) == {"primary"}
    assert set(same_cohort["comparison_basis"]) == {"same_cohort_esm2"}
    assert same_cohort["is_like_for_like"].all()
    np.testing.assert_allclose(
        same_cohort.loc[
            same_cohort["model_key"] == "lr_global", "auprc_delta_pp"
        ],
        5.0,
    )
    np.testing.assert_allclose(
        same_cohort.loc[
            same_cohort["model_key"] == "lr_global_ext_embed", "auprc_delta_pp"
        ],
        10.0,
    )

    mismatches = set(cohorts.loc[~cohorts["cohort_match"], "task"])
    assert mismatches == KNOWN_PAPER_COHORT_MISMATCHES
    asthma = cohorts[
        cohorts["task"] == "therapeutic_target_mondo_0004979"
    ].iloc[0]
    assert (asthma["paper_positive"], asthma["paper_negative"]) == (112, 1523)
    assert asthma["cohort_match"]
    tt_paper = paper[paper["scope"] == "therapeutic_target_mean"]
    assert set(tt_paper["comparison_priority"]) == {"descriptive"}
    assert set(tt_paper["comparison_basis"]) == {
        "descriptive_non_like_for_like"
    }
    assert not tt_paper["is_like_for_like"].any()
    assert set(tt_paper["paper_cohort_tasks_matching"]) == {13.0}
    assert set(tt_paper["paper_cohort_tasks_total"]) == {15.0}

    assert completion["n_results"] == 48
    assert completion["format_version"] == 3
    assert "global_export_provenance" in completion
    assert "paper_class_balances_sha256" in completion
    assert "same_cohort_comparison_sha256" in completion
    assert "cohort_comparison_sha256" in completion
    assert set(completion["split_fingerprints"]) == set(TASKS)


def test_split_artifact_rejects_an_incorrect_embedded_fingerprint(tmp_path):
    path = tmp_path / "split_indices.npz"
    genes = [f"G{index}" for index in range(12)]
    _write_split(path, genes, "contexts", "task_label_multilabel_fallback")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["split_fingerprint"] = np.asarray("incorrect", dtype=str)
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="Embedded split fingerprint"):
        _load_split_artifact(path)


def test_aggregation_rejects_results_split_fingerprint_disagreement(tmp_path):
    output_root, inference_model = _write_fixture(tmp_path)
    result_path = (
        output_root
        / "corum"
        / inference_model
        / "lr_global"
        / "results.csv"
    )
    result = pd.read_csv(result_path)
    result.loc[0, "shared_split_fingerprint"] = "incorrect"
    result.to_csv(result_path, index=False)

    with pytest.raises(ValueError, match="results.csv split fingerprint"):
        aggregate(output_root, inference_model, tmp_path / "summary")


def test_aggregation_rejects_cross_task_global_export_disagreement(tmp_path):
    output_root, inference_model = _write_fixture(tmp_path)
    task = "therapeutic_target_efo_0003767"
    for model_key in MODEL_KEYS:
        result_path = output_root / task / inference_model / model_key / "results.csv"
        result = pd.read_csv(result_path)
        result.loc[0, "global_embedding_path"] = str(
            (tmp_path / "another_export" / "protein_embeddings.pt").resolve()
        )
        result.to_csv(result_path, index=False)

    with pytest.raises(ValueError, match="cross-task global input provenance"):
        aggregate(output_root, inference_model, tmp_path / "summary")
