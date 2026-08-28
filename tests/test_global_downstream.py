import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import downstream_tasks.run as downstream_run
from downstream_tasks.data.global_split import (
    ContextPresence,
    load_context_presence,
    restrict_global_probe_universe,
    split_fingerprint,
)
from downstream_tasks.data.task_loaders import TherapeuticTargetLoader
from downstream_tasks.training.cv_utils import SplitPlan
from downstream_tasks.training.trainer import Trainer


def test_load_context_presence_indexes_released_edgelists(tmp_path):
    (tmp_path / "context_b.txt").write_text("gene3 gene2\ngene2 gene1\n")
    (tmp_path / "context_a.txt").write_text("gene1 gene4\n")

    presence = load_context_presence(tmp_path, expected_context_count=2)

    assert presence.context_names == ("context_a", "context_b")
    assert presence.gene_to_contexts == {
        "GENE1": ("context_a", "context_b"),
        "GENE4": ("context_a",),
        "GENE2": ("context_b",),
        "GENE3": ("context_b",),
    }
    assert len(presence.fingerprint) == 64



def test_load_context_presence_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No .txt Cell-PPI edgelists"):
        load_context_presence(tmp_path)


def test_load_context_presence_requires_released_context_count(tmp_path):
    (tmp_path / "context.txt").write_text("gene1 gene2\n")
    with pytest.raises(ValueError, match="Expected 207"):
        load_context_presence(tmp_path, expected_context_count=207)


def test_reset_random_seed_restarts_each_model_variant():
    downstream_run._reset_random_seed(42)
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)

    np.random.random(10)
    torch.rand(10)
    downstream_run._reset_random_seed(42)

    np.testing.assert_array_equal(np.random.random(4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0.0, atol=0.0)


def test_tt_reconstruction_provenance_is_carried_from_manifest(tmp_path):
    labels = tmp_path / "therapeutic_target_EFO_TEST.csv"
    labels.write_text("protein,label\nA,1\nB,0\n")
    manifest = {
        "format_version": 1,
        "dataset": "therapeutic_target",
        "reconstruction": "frozen_open_targets",
        "open_targets_release": "26.03",
        "association_scope": "indirect",
    }
    (tmp_path / "therapeutic_target_manifest.json").write_text(
        json.dumps(manifest) + "\n"
    )

    provenance = downstream_run._load_task_dataset_provenance(
        "therapeutic_target_efo_test",
        labels,
    )

    assert provenance["reconstruction"] == "frozen_open_targets"
    assert provenance["open_targets_release"] == "26.03"
    assert provenance["association_scope"] == "indirect"
    assert len(provenance["manifest_sha256"]) == 64


def test_global_embedding_provenance_is_bound_to_manifest(tmp_path):
    embedding_path = tmp_path / "protein_embeddings.pt"
    embedding_path.write_bytes(b"payload")
    manifest = {
        "format_version": 1,
        "model_type": "global_s2gae",
        "embedding_scope": "global",
        "embedding_topology": "full_reference",
        "representation": "encoder_jk_concat",
        "embedding_sha256": "embedding-sha",
        "checkpoint_sha256": "checkpoint-sha",
        "training_git_commit": "training-commit",
        "export_git_commit": "export-commit",
        "ppi_test_compatible": False,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n")

    provenance = downstream_run._load_global_embedding_provenance(
        tmp_path,
        embedding_path,
    )

    assert provenance["embedding_path"] == str(embedding_path.resolve())
    assert provenance["embedding_sha256"] == "embedding-sha"
    assert provenance["checkpoint_sha256"] == "checkpoint-sha"
    assert provenance["export_git_commit"] == "export-commit"
    assert len(provenance["manifest_sha256"]) == 64


def test_global_embedding_provenance_rejects_test_compatible_export(tmp_path):
    embedding_path = tmp_path / "protein_embeddings.pt"
    embedding_path.write_bytes(b"payload")
    manifest = {
        "format_version": 1,
        "model_type": "global_s2gae",
        "embedding_scope": "global",
        "embedding_topology": "full_reference",
        "representation": "encoder_jk_concat",
        "embedding_sha256": "embedding-sha",
        "checkpoint_sha256": "checkpoint-sha",
        "training_git_commit": "training-commit",
        "export_git_commit": "export-commit",
        "ppi_test_compatible": True,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="ppi_test_compatible"):
        downstream_run._load_global_embedding_provenance(tmp_path, embedding_path)


@pytest.mark.parametrize(
    "rows, message",
    [
        ("protein,label\nA,2\nB,0\n", "must be binary 0/1"),
        ("protein,label\nA,1\nA,0\nB,0\n", "conflicting labels"),
        ("protein,label\nA,1\nB,1\n", "must contain both classes"),
    ],
)
def test_tt_loader_rejects_invalid_label_data(tmp_path, rows, message):
    labels = tmp_path / "therapeutic_target.csv"
    labels.write_text(rows)
    with pytest.raises(ValueError, match=message):
        TherapeuticTargetLoader(labels).load()


def test_global_probe_universe_is_one_ordered_four_way_intersection():
    presence = ContextPresence(
        gene_to_contexts={
            "A": ("c1",),
            "B": ("c1", "c2"),
            "C": ("c2",),
        },
        context_names=("c1", "c2"),
        fingerprint="context-fingerprint",
    )
    labels = np.arange(8).reshape(4, 2)

    genes, shared_labels, contexts = restrict_global_probe_universe(
        ["b", "a", "c", "d"],
        labels,
        global_genes={"A", "B", "D"},
        sequence_genes={"A", "B", "C"},
        context_presence=presence,
    )

    assert genes == ["B", "A"]
    np.testing.assert_array_equal(shared_labels, labels[[0, 1]])
    assert contexts == [["c1", "c2"], ["c1"]]


def test_global_shared_split_passes_contexts_only_to_split_builder(monkeypatch):
    presence = ContextPresence(
        gene_to_contexts={
            "A": ("c1",),
            "B": ("c1", "c2"),
            "C": ("c2",),
        },
        context_names=("c1", "c2"),
        fingerprint="context-fingerprint",
    )

    class FakeEmbeddingLoader:
        def load_esm(self):
            return {gene: np.ones(2) for gene in ("A", "B", "C")}

        def load_global(self):
            return {gene: np.ones(3) for gene in ("A", "B", "D")}

    sentinel = object()
    captured = {}

    def fake_build_cv_splits(Y, **kwargs):
        captured["Y"] = Y.copy()
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(downstream_run, "build_cv_splits", fake_build_cv_splits)
    labels = np.arange(8).reshape(4, 2)

    genes, shared_labels, plan, contexts = downstream_run._build_global_shared_split(
        genes=["B", "A", "C", "D"],
        Y=labels,
        embedding_loader=FakeEmbeddingLoader(),
        context_presence=presence,
        seed=42,
        n_splits=2,
    )

    assert genes == ["B", "A"]
    np.testing.assert_array_equal(shared_labels, labels[[0, 1]])
    assert plan is sentinel
    assert contexts == [["c1", "c2"], ["c1"]]
    assert captured["cell_ids_per_bag"] == contexts
    assert captured["use_context_split"] is True
    assert captured["seed"] == 42
    assert captured["n_splits"] == 2


def test_three_lr_variants_keep_gene_order_and_expected_feature_blocks(tmp_path):
    class FakeEmbeddingLoader:
        def load_esm(self):
            return {
                "A": np.array([10.0, 11.0, 12.0], dtype=np.float32),
                "B": np.array([20.0, 21.0, 22.0], dtype=np.float32),
            }

        def load_global(self):
            return {
                "A": np.array([1.0, 2.0], dtype=np.float32),
                "B": np.array([3.0, 4.0], dtype=np.float32),
            }

    trainer = Trainer(
        config=SimpleNamespace(),
        embedding_loader=FakeEmbeddingLoader(),
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    sequence = trainer._build_features(
        downstream_run.MODEL_VARIANTS["lr_ext_embed"], ["B", "A"]
    )
    global_only = trainer._build_features(
        downstream_run.MODEL_VARIANTS["lr_global"], ["B", "A"]
    )
    combined = trainer._build_features(
        downstream_run.MODEL_VARIANTS["lr_global_ext_embed"], ["B", "A"]
    )

    assert sequence["genes"] == ["B", "A"]
    assert global_only["genes"] == sequence["genes"]
    assert combined["genes"] == sequence["genes"]
    np.testing.assert_array_equal(sequence["X"], sequence["esm"])
    np.testing.assert_array_equal(global_only["X"], global_only["global"])
    np.testing.assert_array_equal(
        combined["X"],
        np.concatenate([combined["global"], combined["esm"]], axis=1),
    )


def test_global_split_artifact_records_exact_shared_context_matrix(tmp_path):
    presence = ContextPresence(
        gene_to_contexts={
            "A": ("c1",),
            "B": ("c1", "c2"),
        },
        context_names=("c1", "c2"),
        fingerprint="context-fingerprint",
    )
    plan = SplitPlan(
        folds=[np.array([0]), np.array([1])],
        test_fold_idx=0,
        cv_folds=[np.array([1])],
        label_clusters=np.array([0, 1]),
        n_cv_folds=1,
        stratification_method="task_label_and_context_cluster",
    )

    fingerprint = downstream_run._save_split_artifacts(
        tmp_path,
        ["B", "A"],
        plan,
        seed=42,
        context_presence=presence,
        gene_contexts=[["c1", "c2"], ["c1"]],
    )

    assert fingerprint == split_fingerprint(["B", "A"], plan.folds)
    with np.load(tmp_path / "split_indices.npz", allow_pickle=False) as saved:
        assert saved["genes"].tolist() == ["B", "A"]
        assert saved["context_names"].tolist() == ["c1", "c2"]
        np.testing.assert_array_equal(
            saved["context_presence"],
            np.array([[True, True], [True, False]]),
        )
        assert saved["split_fingerprint"].item() == fingerprint
        assert (
            saved["split_stratification"].item()
            == "task_label_and_context_cluster"
        )

    metadata = json.loads((tmp_path / "split_metadata.json").read_text())
    assert metadata["n_contexts"] == 2
    assert metadata["seed"] == 42
    assert metadata["split_fingerprint"] == fingerprint
    assert metadata["split_stratification"] == "task_label_and_context_cluster"
