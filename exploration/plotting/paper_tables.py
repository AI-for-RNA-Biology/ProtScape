"""Export the five manuscript result tables as numeric CSV files."""

import json
import statistics as st
from pathlib import Path

import pandas as pd
import yaml


DISEASE_MODELS = {
    "lr_esm", "lr_prostt5", "pinnacle_random_fixed", "pinnacle_esm_fixed",
    "pinnacle_esm2_acm", "gae_att_fixed_do06", "s2gae_att_k1_fixed_do04_uni",
    "s2gae_phuber_uni", "s2gae_l1_uni",
}


def summarize(values):
    return {"auprc_mean": st.mean(values), "auprc_sd": st.stdev(values)}


def write_scores(rows, path):
    table = pd.DataFrame(rows).copy()
    table[["auprc_mean", "auprc_sd"]] *= 100
    for column in ("inference_label", "readout_label"):
        table[column] = table[column].str.replace("\n", "; ", regex=False)
    table.rename(columns={"auprc_mean": "auprc_percent",
                          "auprc_sd": "sample_sd_percent"}).to_csv(path, index=False)


def write_tables(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)

    pretraining = pd.read_csv(source / "pretraining_evaluation/pretraining_full_metrics.csv")
    metrics = [column for column in pretraining if column.endswith(
        ("_auprc", "_macro_f1", "_f1", "_accuracy", "_auroc"))]
    pretraining[metrics] *= 100
    pretraining.rename(columns={column: column + "_percent" for column in metrics}).to_csv(
        output / "table_pretraining.csv", index=False
    )

    datasets = pd.read_csv(source / "therapeutic_target_analysis/dataset_statistics.csv")
    datasets[["disease_id", "disease", "cohort_positive", "cohort_negative",
              "positive_label_source", "negative_label_source"]].to_csv(
        output / "table_therapeutic_datasets.csv", index=False
    )

    corum = pd.concat([
        pd.read_csv(source / "corum_analysis" / f"corum_{name}_performance.csv")
        for name in ("main", "loss", "ablation")
    ], ignore_index=True)
    corum = corum.loc[corum.metric.eq("auprc"), [
        "inference_key", "inference_label", "readout_key", "readout_label",
        "fold_scores", "n_test_samples",
    ]].drop_duplicates()
    if corum.duplicated(["inference_key", "readout_key"]).any():
        raise ValueError("Conflicting CORUM results for the same model/readout.")
    corum_rows = []
    for row in corum.to_dict("records"):
        values = json.loads(row.pop("fold_scores"))
        if len(values) != 5:
            raise ValueError("Each CORUM result must contain five model scores.")
        corum_rows.append(dict(row, **summarize(values), n_models=5, sd_basis="fold_models"))
    write_scores(corum_rows, output / "table_corum.csv")

    folds = pd.concat([
        pd.read_csv(source / "therapeutic_target_analysis" / name)
        for name in ("held_out_performance.csv", "ablation_fold_performance.csv")
    ], ignore_index=True)
    disease_rows = []
    for (model, readout, task), group in folds.groupby(
        ["inference_key", "readout_key", "task"], sort=False
    ):
        if sorted(group.fold) != list(range(5)):
            raise ValueError(f"Expected five distinct fold scores: {model}/{readout}/{task}")
        disease_rows.append(dict(
            inference_key=model, inference_label=group.inference_label.iloc[0],
            readout_key=readout, readout_label=group.readout_label.iloc[0],
            task=task, disease=group.disease.iloc[0], n_models=5, sd_basis="fold_models",
            **summarize(group.auprc.tolist()),
        ))
    diseases = pd.DataFrame(disease_rows)
    readout_rows = []
    for (model, readout), group in diseases.groupby(["inference_key", "readout_key"], sort=False):
        if len(group) != 15:
            raise ValueError(f"Expected 15 diseases: {model}/{readout}")
        readout_rows.append(dict(
            inference_key=model, inference_label=group.inference_label.iloc[0],
            readout_key=readout, readout_label=group.readout_label.iloc[0],
            n_models=5, n_diseases=15, sd_basis="disease_means",
            **summarize(group.auprc_mean.tolist()),
        ))
    readouts = pd.DataFrame(readout_rows)
    write_scores(readouts, output / "table_therapeutic_readouts.csv")

    def primary(table):
        return table.inference_key.isin(DISEASE_MODELS) & (
            table.readout_key.isin(["lr_esm", "lr_prostt5", "abmil8_pdl_id2_dropout"])
        )

    selected = diseases.loc[primary(diseases)].copy()
    means = readouts.loc[primary(readouts)].assign(task="", disease="Mean")
    write_scores(pd.concat([means, selected], ignore_index=True),
                 output / "table_therapeutic_diseases.csv")


def main():
    with (Path(__file__).resolve().parents[2] / "configs/paths.yaml").open() as handle:
        paths = yaml.safe_load(handle)
    write_tables(Path(paths["paper_source_data"]), Path(paths["output_root"]) / "paper_tables")


if __name__ == "__main__":
    main()
