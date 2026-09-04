from pathlib import Path

import numpy as np
import pytest

from downstream_tasks.config import (
    EXPLICIT_CSV_TASKS,
    TaskConfig,
    load_config,
    resolve_task_csv,
)
from downstream_tasks.data.task_loaders import (
    MultiLabelMembershipLoader,
    get_task_loader,
)


def test_multilabel_loader_uses_label_id_as_stable_class_identity(tmp_path):
    labels_path = tmp_path / "localization.csv"
    labels_path.write_text(
        "protein,label,label_id,source\n"
        " gene_b ,Nucleus,GO:0005634,GOCC\n"
        "gene_a,Cytosol,GO:0005829,GOCC\n"
        "GENE_A,Nucleus,GO:0005634,GOCC\n"
        "gene_b,Nucleus,GO:0005634,curated\n",
        encoding="utf-8",
    )

    genes, labels, class_names = MultiLabelMembershipLoader(labels_path).load()

    assert genes == ["GENE_A", "GENE_B"]
    assert class_names == ["GO:0005634", "GO:0005829"]
    np.testing.assert_array_equal(
        labels,
        np.asarray([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    assert labels.dtype == np.float32


def test_multilabel_loader_falls_back_to_label_names(tmp_path):
    labels_path = tmp_path / "pathways.csv"
    labels_path.write_text(
        "label,protein\nBeta pathway,B\nAlpha pathway,A\nBeta pathway,A\n",
        encoding="utf-8",
    )

    genes, labels, class_names = MultiLabelMembershipLoader(labels_path).load()

    assert genes == ["A", "B"]
    assert class_names == ["Alpha pathway", "Beta pathway"]
    np.testing.assert_array_equal(labels, [[1.0, 1.0], [0.0, 1.0]])


@pytest.mark.parametrize(
    "contents, message",
    [
        ("protein,source\nA,GOCC\n", "missing required column"),
        ("protein,label,evidence\nA,Nucleus,IDA\n", "unexpected column"),
        ("protein,label,label_id\nA,Nucleus,\n", "missing values"),
        (
            "protein,label,label_id\nA,Nucleus,GO:1\nB,Nucleus,GO:2\n",
            "one-to-one mapping",
        ),
        (
            "protein,label,label_id\nA,Nucleus,GO:1\nB,Cytosol,GO:1\n",
            "one-to-one mapping",
        ),
    ],
)
def test_multilabel_loader_rejects_noncanonical_tables(tmp_path, contents, message):
    labels_path = tmp_path / "memberships.csv"
    labels_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        MultiLabelMembershipLoader(labels_path).load()


@pytest.mark.parametrize("task", sorted(EXPLICIT_CSV_TASKS))
def test_explicit_multilabel_tasks_share_membership_loader(tmp_path, task):
    loader = get_task_loader(task, tmp_path / "labels.csv")
    assert isinstance(loader, MultiLabelMembershipLoader)


def test_explicit_multilabel_tasks_require_task_csv():
    configured = {
        "corum": TaskConfig(label_csv=Path("configured.csv"), name="CORUM")
    }

    with pytest.raises(ValueError, match="requires --task-csv"):
        resolve_task_csv(configured, "protein_localization", None)
    with pytest.raises(ValueError, match="requires --task-csv"):
        resolve_task_csv(configured, "pathway", None)

    supplied = Path("frozen-memberships.csv")
    assert resolve_task_csv(configured, "protein_localization", supplied) == supplied
    assert resolve_task_csv(configured, "pathway", supplied) == supplied


def test_explicit_multilabel_tasks_have_no_configured_default_paths():
    config = load_config(inference_model="unused-in-test")
    assert EXPLICIT_CSV_TASKS.isdisjoint(config.tasks)


def test_task_csv_resolution_preserves_existing_task_behavior():
    configured_path = Path("configured.csv")
    override_path = Path("override.csv")
    configured = {
        "corum": TaskConfig(label_csv=configured_path, name="CORUM")
    }

    assert resolve_task_csv(configured, "corum", None) == configured_path
    assert resolve_task_csv(configured, "corum", override_path) == override_path
    with pytest.raises(ValueError, match="Unknown task"):
        resolve_task_csv(configured, "not_a_task", override_path)
