"""Checks for the official frozen-input TT builder and released partitions."""
import json
import numpy as np
import pandas as pd
import pytest

from downstream_tasks.data_processing import therapeutic_target_processing as tt
from downstream_tasks.data.partitions import load_task_partition


def test_official_clinical_evidence_criteria():
    assert tt.clinically_relevant(3, "terminated")
    assert tt.clinically_relevant(2, "completed")
    assert not tt.clinically_relevant(2, "recruiting")
    assert not tt.clinically_relevant(None, "completed")


def test_official_builder_excludes_associations_and_positive_negative_overlap(tmp_path):
    evidence = tmp_path / "evidence.json"
    records = [
        {"diseaseId": "CHILD", "targetId": "ENSG1", "clinicalPhase": 3},
        {"diseaseId": "ROOT", "targetId": "ENSG2", "clinicalPhase": 2, "clinicalStatus": "recruiting"},
    ]
    evidence.write_text("".join(json.dumps(row) + "\n" for row in records))
    result = tt.build_dataset(
        disease="ROOT", descendants={"ROOT", "CHILD"}, associated_targets={"B"},
        target_symbols={"ENSG1": "A", "ENSG2": "B"}, evidence_files=[evidence],
        evidence_format="json", ppi_genes={"A", "B", "C"},
        druggable_targets={"A", "B", "C"}, output_dir=tmp_path / "labels",
        processed_dir=tmp_path / "processed", min_proteins_per_label=1,
    )
    labels = pd.read_csv(tmp_path / "labels/therapeutic_target_ROOT.csv")
    assert dict(zip(labels.protein, labels.label)) == {"A": 1, "C": 0}
    assert result["clinical_evidence_release"] == "24.03"


def test_official_target_mapping_uses_only_unambiguous_hgnc_alias(tmp_path):
    records = [
        {"id": "ENSG1", "approvedSymbol": "NEW", "obsoleteSymbols": [{"label": "OLD", "source": "HGNC"}]},
        {"id": "ENSG2", "approvedSymbol": "OTHER", "obsoleteSymbols": [{"label": "UNKNOWN", "source": "OTHER"}]},
    ]
    path = tmp_path / "targets.json"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    symbols = tt.load_target_symbols(path, "json", {"OLD", "UNKNOWN"})
    assert symbols == {"ENSG1": "OLD", "ENSG2": "OTHER"}


def test_official_evidence_reader_rejects_corrupt_json(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text('{"diseaseId":')
    with pytest.raises(ValueError):
        tt.collect_evidence([path], "json", {"ROOT"})


def test_released_partition_is_loaded_without_regenerating_folds(tmp_path, monkeypatch):
    labels = tmp_path / "therapeutic_target_TEST.csv"
    labels.write_text("protein,label\nA,1\nB,0\nC,1\nD,0\nE,1\nF,0\n")
    genes, truth, classes = tt_module_loader(labels)
    folds = [np.array([i]) for i in [3, 5, 1, 4, 0, 2]]
    directory = tmp_path / "splits"
    directory.mkdir()
    path = directory / "therapeutic_target_test.npz"
    np.savez(path, genes=genes, labels=truth, class_names=classes,
             **{f"fold_{i}": fold for i, fold in enumerate(folds)})
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved partitions must not be regenerated")
    monkeypatch.setattr("downstream_tasks.data.partitions.build_cv_splits", forbidden)
    loaded_genes, loaded_truth, _, plan = load_task_partition("therapeutic_target_test", labels)
    assert loaded_genes == genes
    np.testing.assert_array_equal(loaded_truth, truth)
    for got, expected in zip(plan.folds, folds):
        np.testing.assert_array_equal(got, expected)
    labels.write_text(labels.read_text().replace("A,1", "A,0"))
    with pytest.raises(ValueError, match="Labels differ"):
        load_task_partition("therapeutic_target_test", labels)


def tt_module_loader(path):
    from downstream_tasks.data.task_loaders import TherapeuticTargetLoader
    return TherapeuticTargetLoader(path).load()
