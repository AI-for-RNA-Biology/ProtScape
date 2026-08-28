import json

import pandas as pd
import pytest

from downstream_tasks.data_processing import therapeutic_target_processing as tt


def _write_static_dataset(root, name, records):
    dataset = root / name
    dataset.mkdir(parents=True)
    (dataset / "_SUCCESS").touch()
    with (dataset / "part-00000.json").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _static_release(tmp_path):
    root = tmp_path / "opentargets_24_03"
    _write_static_dataset(
        root,
        "diseases",
        [{"id": "EFO_ROOT", "descendants": ["EFO_CHILD"]}],
    )
    _write_static_dataset(
        root,
        "searchTarget",
        [
            {"id": "ENSG1", "name": "GENEA", "entity": "target"},
            {"id": "ENSG2", "name": "GENEB", "entity": "target"},
            {"id": "ENSG3", "name": "GENEC", "entity": "target"},
            {"id": "ENSG4", "name": "GENED", "entity": "target"},
        ],
    )
    _write_static_dataset(
        root,
        "associationByDatatypeDirect",
        [
            {
                "diseaseId": "EFO_ROOT",
                "targetId": "ENSG1",
                "datatypeId": "literature",
                "score": 1.0,
            },
            {
                "diseaseId": "EFO_ROOT",
                "targetId": "ENSG2",
                "datatypeId": "genetic_association",
                "score": 0.4,
            },
            {
                "diseaseId": "EFO_CHILD",
                "targetId": "ENSG3",
                "datatypeId": "known_drug",
                "score": 1.0,
            },
        ],
    )
    _write_static_dataset(
        root,
        "associationByDatatypeIndirect",
        [
            {
                "diseaseId": "EFO_ROOT",
                "targetId": "ENSG2",
                "datatypeId": "genetic_association",
                "score": 0.4,
            },
            {
                "diseaseId": "EFO_ROOT",
                "targetId": "ENSG3",
                "datatypeId": "known_drug",
                "score": 1.0,
            },
        ],
    )
    return root


def _parquet_release(tmp_path):
    pytest.importorskip("pyarrow")
    root = tmp_path / "opentargets_26_03"
    (root / "disease").mkdir(parents=True)
    pd.DataFrame(
        [
            {"id": "EFO_ROOT", "descendants": ["EFO_CHILD"]},
            {"id": "EFO_OTHER", "descendants": []},
        ]
    ).to_parquet(root / "disease" / "disease.parquet", index=False)

    (root / "target").mkdir()
    (root / "target" / "_SUCCESS").touch()
    pd.DataFrame(
        [
            {"id": "ENSG1", "approvedSymbol": "GENEA"},
            {"id": "ENSG2", "approvedSymbol": "GeneB"},
        ]
    ).to_parquet(root / "target" / "part-00000.parquet", index=False)
    pd.DataFrame(
        [
            {"id": "ENSG3", "approvedSymbol": "GENEC"},
            {"id": "ENSG4", "approvedSymbol": "GENED"},
        ]
    ).to_parquet(root / "target" / "part-00001.parquet", index=False)

    (root / "association_by_datatype_direct").mkdir()
    (root / "association_by_datatype_direct" / "_SUCCESS").touch()
    pd.DataFrame(
        [
            {
                "diseaseId": "EFO_ROOT",
                "targetId": "ENSG1",
                "datatypeId": "literature",
                "score": 1.0,
            },
            {
                "diseaseId": "EFO_ROOT",
                "targetId": "ENSG2",
                "datatypeId": "genetic_association",
                "score": 0.4,
            },
            {
                "diseaseId": "EFO_OTHER",
                "targetId": "ENSG4",
                "datatypeId": "known_drug",
                "score": 1.0,
            },
        ]
    ).to_parquet(
        root / "association_by_datatype_direct" / "part-00000.parquet",
        index=False,
    )

    (root / "association_by_datatype_indirect").mkdir()
    (root / "association_by_datatype_indirect" / "_SUCCESS").touch()
    pd.DataFrame(
        [
            {
                "diseaseId": "EFO_ROOT",
                "targetId": "ENSG2",
                "aggregationValue": "genetic_association",
                "associationScore": 0.4,
            },
            {
                "diseaseId": "EFO_ROOT",
                "targetId": "ENSG3",
                "aggregationValue": "known_drug",
                "associationScore": 1.0,
            },
            {
                "diseaseId": "EFO_ROOT",
                "targetId": "ENSG4",
                "aggregationValue": "known_drug",
                "associationScore": 0.0,
            },
        ]
    ).to_parquet(
        root / "association_by_datatype_indirect" / "part-00000.parquet",
        index=False,
    )
    return root


def test_static_release_supports_direct_and_indirect_associations(tmp_path):
    root = _static_release(tmp_path)

    direct = tt.load_static_open_targets_release(root, ["EFO_ROOT"], "direct")
    indirect = tt.load_static_open_targets_release(root, ["EFO_ROOT"], "indirect")

    assert direct.release_name == "opentargets_24_03"
    assert direct.storage_format == "json"
    assert direct.descendants["EFO_ROOT"] == {"EFO_ROOT", "EFO_CHILD"}
    assert direct.associated_targets["EFO_ROOT"] == {"GENEB"}
    assert indirect.associated_targets["EFO_ROOT"] == {"GENEB", "GENEC"}


def test_parquet_release_supports_2603_schema_and_legacy_column_aliases(tmp_path):
    root = _parquet_release(tmp_path)

    direct = tt.load_static_open_targets_release(root, ["EFO_ROOT"], "direct")
    indirect = tt.load_static_open_targets_release(root, ["EFO_ROOT"], "indirect")

    assert direct.storage_format == "parquet"
    assert direct.disease_dataset == "disease"
    assert direct.target_dataset == "target"
    assert direct.association_dataset == "association_by_datatype_direct"
    assert direct.descendants["EFO_ROOT"] == {"EFO_ROOT", "EFO_CHILD"}
    assert direct.associated_targets["EFO_ROOT"] == {"GENEB"}
    assert indirect.associated_targets["EFO_ROOT"] == {"GENEB", "GENEC"}


def test_parquet_release_rejects_incomplete_association_schema(tmp_path):
    root = _parquet_release(tmp_path)
    broken = root / "association_by_datatype_indirect" / "part-00000.parquet"
    pd.DataFrame(
        [
            {
                "diseaseId": "EFO_ROOT",
                "targetId": "ENSG1",
                "aggregationValue": "known_drug",
            }
        ]
    ).to_parquet(broken, index=False)

    with pytest.raises(ValueError, match="aggregationValue/associationScore"):
        tt.load_static_open_targets_release(root, ["EFO_ROOT"], "indirect")


def test_parquet_release_rejects_missing_success_marker(tmp_path):
    root = _parquet_release(tmp_path)
    (root / "association_by_datatype_indirect" / "_SUCCESS").unlink()

    with pytest.raises(FileNotFoundError, match="missing _SUCCESS"):
        tt.load_static_open_targets_release(root, ["EFO_ROOT"], "indirect")


@pytest.mark.parametrize(
    "artifact",
    [
        "therapeutic_target_EFO_ROOT.csv",
        "therapeutic_target_summary.csv",
        "therapeutic_target_manifest.json",
    ],
)
def test_static_reconstruction_requires_force_for_existing_outputs(
    tmp_path,
    artifact,
):
    output = tmp_path / "output"
    output.mkdir()
    (output / artifact).touch()

    with pytest.raises(FileExistsError, match="--force"):
        tt._require_fresh_static_outputs(output, ["EFO_ROOT"], force=False)

    tt._require_fresh_static_outputs(output, ["EFO_ROOT"], force=True)


def test_static_manifest_distinguishes_evidence_and_association_releases(tmp_path):
    release = tt.load_static_open_targets_release(
        _parquet_release(tmp_path),
        ["EFO_ROOT"],
        "indirect",
        release_name="26.03",
    )

    provenance = tt._static_manifest_provenance(release, "24.03")

    assert provenance["open_targets_release"] == "26.03"
    assert provenance["open_targets_association_release"] == "26.03"
    assert provenance["open_targets_evidence_release"] == "24.03"


def test_static_build_never_uses_live_mapping_or_association_apis(tmp_path, monkeypatch):
    root = _static_release(tmp_path)
    release = tt.load_static_open_targets_release(root, ["EFO_ROOT"], "direct")
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    evidence = [
        {
            "diseaseId": "EFO_CHILD",
            "targetId": "ENSG1",
            "targetFromSourceId": "P1",
            "clinicalPhase": 3,
            "clinicalStatus": "Completed",
        },
        {
            "diseaseId": "EFO_ROOT",
            "targetId": "ENSG4",
            "targetFromSourceId": "P4",
            "clinicalPhase": 2,
            "clinicalStatus": "Completed",
        },
        {
            "diseaseId": "EFO_ROOT",
            "targetId": "ENSG3",
            "targetFromSourceId": "P3",
            "clinicalPhase": 2,
            "clinicalStatus": "Recruiting",
        },
    ]
    evidence_file = evidence_dir / "part-00000.json"
    with evidence_file.open("w", encoding="utf-8") as handle:
        for record in evidence:
            handle.write(json.dumps(record) + "\n")

    def fail_live(*args, **kwargs):
        raise AssertionError("static mode called a live API helper")

    monkeypatch.setattr(tt, "get_disease_descendants_ot_api", fail_live)
    monkeypatch.setattr(tt, "get_disease_descendants_efo", fail_live)
    monkeypatch.setattr(tt, "get_ot_associated_targets", fail_live)
    monkeypatch.setattr(tt, "map_uniprot_to_gene_names", fail_live)
    monkeypatch.setattr(tt, "map_ensembl_to_gene_symbols", fail_live)

    output = tmp_path / "output"
    result = tt.build_dataset_for_disease(
        disease="EFO_ROOT",
        evidence_files=[evidence_file],
        evidence_format="json",
        descendants_source="ot",
        ppi_genes={"GENEA", "GENEB", "GENEC", "GENED", "GENEE"},
        druggable_targets={"GENEA", "GENEB", "GENEC", "GENED", "GENEE"},
        out_dir=output,
        processed_dir=output / "processed",
        min_proteins_per_label=1,
        static_release=release,
    )

    rows = (output / "therapeutic_target_EFO_ROOT.csv").read_text()
    assert result["open_targets_mode"] == "static_direct"
    assert result["n_positive"] == 2
    assert result["n_negative"] == 2
    assert "GENEA,1" in rows
    assert "GENED,1" in rows
    assert "GENEC,0" in rows
    assert "GENEE,0" in rows

    with pytest.raises(FileExistsError, match="--force"):
        tt.build_dataset_for_disease(
            disease="EFO_ROOT",
            evidence_files=[],
            evidence_format="json",
            descendants_source="none",
            ppi_genes=set(),
            druggable_targets=set(),
            out_dir=output,
            processed_dir=output / "processed",
            static_release=release,
        )


def test_static_release_rejects_missing_success_marker(tmp_path):
    dataset = tmp_path / "diseases"
    dataset.mkdir()
    (dataset / "part-00000.json").write_text("{}\n", encoding="utf-8")

    try:
        tt._static_jsonl_files(tmp_path, "diseases")
    except FileNotFoundError as exc:
        assert "missing _SUCCESS" in str(exc)
    else:
        raise AssertionError("partial static release was accepted")


def test_evidence_reader_rejects_corrupt_json(tmp_path):
    evidence = tmp_path / "part-00000.json"
    evidence.write_text('{"diseaseId":"EFO_ROOT"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in evidence file"):
        tt.collect_clinically_relevant_evidence(
            [evidence],
            "json",
            {"EFO_ROOT"},
        )
