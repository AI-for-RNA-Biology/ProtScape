from pathlib import Path

import pandas as pd

from downstream_tasks.data.task_loaders import MultiLabelMembershipLoader
from downstream_tasks.data_processing.deeploc_processing import process_deeploc


def _write_table(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_deeploc_processing_filters_labels_and_unannotated_proteins(tmp_path):
    test_csv = tmp_path / "test.csv"
    train_csv = tmp_path / "train.csv"
    output_csv = tmp_path / "localization.csv"
    _write_table(
        test_csv,
        [
            {"hgnc_symbol": "a", "Nucleus": 1, "Peroxisome": 0},
            {"hgnc_symbol": "b", "Nucleus": 0, "Peroxisome": 1},
        ],
    )
    _write_table(
        train_csv,
        [
            {"hgnc_symbol": "a", "Nucleus": 0, "Peroxisome": 1},
            {"hgnc_symbol": "c", "Nucleus": 1, "Peroxisome": 0},
        ],
    )

    manifest = process_deeploc(
        test_csv=test_csv,
        train_csv=train_csv,
        output_csv=output_csv,
        min_positive_count=2,
    )
    genes, labels, class_names = MultiLabelMembershipLoader(output_csv).load()

    assert genes == ["A", "C"]
    assert labels.tolist() == [[1.0], [1.0]]
    assert class_names == ["DeepLoc2:Nucleus"]
    assert manifest["parameters"]["retained_labels"] == ["Nucleus"]
    assert manifest["statistics"]["mapped_unique_proteins"] == 3
    assert manifest["statistics"]["retained_proteins"] == 2
