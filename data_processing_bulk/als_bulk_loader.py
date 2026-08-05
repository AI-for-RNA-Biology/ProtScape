"""Load Kallisto counts and metadata for the ALS bulk RNA-seq cohorts."""

import logging
import os
from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd


logger = logging.getLogger(__name__)


def _motor_neuron_samples(metadata: pd.DataFrame, cl_id: str) -> Dict[str, dict]:
    samples = {}
    for _, row in metadata.iterrows():
        run_id = row["Run"]
        genotype = row["genotype"]
        fraction = row["Fraction"]
        days = int(row["days_of_differentiation"])

        if genotype == "Wildtype":
            condition = "CTRL"
        elif "R155C" in genotype:
            condition = "R155C"
        elif "R191Q" in genotype:
            condition = "R191Q"
        else:
            condition = "UNKNOWN"

        compartment = "nuc" if fraction == "Nuclear" else "cyto"
        cell_type_id = f"{cl_id}_{condition}_{compartment}_d{days}"
        samples[run_id] = {
            "cell_type_id": cell_type_id,
            "cell_type_name": f"Motor neuron ({condition}, {compartment}, day {days})",
            "cl_id": cl_id,
            "condition": condition,
            "compartment": compartment,
            "timepoint": days,
            "dataset": "als_motor_neuron",
        }
    return samples


def _astrocyte_samples(metadata: pd.DataFrame, cl_id: str) -> Dict[str, dict]:
    samples = {}
    for _, row in metadata.iterrows():
        run_id = row["Run"]
        genotype = row.get("genotype")
        rna_fraction = row.get("rna_fraction")
        if pd.isna(genotype) or pd.isna(rna_fraction):
            logger.warning("Skipping %s because genotype or RNA fraction is missing", run_id)
            continue

        cell_line = str(row.get("cell_line", "unknown"))
        if "wild_type" in genotype or "ctrl" in cell_line.lower():
            condition = "CTRL"
        elif "R155C" in genotype:
            condition = "R155C"
        elif "R191Q" in genotype:
            condition = "R191Q"
        else:
            condition = "UNKNOWN"

        if "nuclear" in rna_fraction:
            compartment = "nuc"
        elif "cytoplasmic" in rna_fraction:
            compartment = "cyto"
        elif "whole" in rna_fraction:
            compartment = "whole"
        else:
            compartment = "unknown"

        cell_type_id = f"{cl_id}_{condition}_{compartment}"
        samples[run_id] = {
            "cell_type_id": cell_type_id,
            "cell_type_name": f"Astrocyte ({condition}, {compartment})",
            "cl_id": cl_id,
            "condition": condition,
            "compartment": compartment,
            "timepoint": None,
            "dataset": "als_astrocyte",
        }
    return samples


def _load_gene_counts(kallisto_dir: str, run_id: str) -> pd.Series:
    abundance_path = os.path.join(kallisto_dir, run_id, "abundance.tsv")
    if not os.path.exists(abundance_path):
        raise FileNotFoundError(abundance_path)

    abundance = pd.read_csv(abundance_path, sep="\t", usecols=["target_id", "est_counts"])
    target_parts = abundance["target_id"].str.split("|")
    abundance["gene_id"] = target_parts.str[1].str.split(".").str[0]
    abundance = abundance.dropna(subset=["gene_id"])
    return abundance.groupby("gene_id")["est_counts"].sum()


def load_als_bulk(
    *,
    kallisto_dir: str,
    metadata_path: str,
    dataset: str,
    cl_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return replicate-averaged gene counts and one metadata row per ALS context."""
    metadata = pd.read_csv(metadata_path)
    metadata.columns = metadata.columns.str.strip()

    if dataset == "motor_neuron":
        samples = _motor_neuron_samples(metadata, cl_id)
    elif dataset == "astrocyte":
        samples = _astrocyte_samples(metadata, cl_id)
    else:
        raise ValueError(f"Unknown ALS dataset: {dataset}")

    counts_by_context = defaultdict(list)
    runs_by_context = defaultdict(list)
    failed_runs = []
    all_genes = set()

    for run_id, sample in samples.items():
        try:
            counts = _load_gene_counts(kallisto_dir, run_id)
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.warning("Skipping %s: %s", run_id, exc)
            failed_runs.append(run_id)
            continue

        context = sample["cell_type_id"]
        counts_by_context[context].append(counts)
        runs_by_context[context].append(run_id)
        all_genes.update(counts.index)

    if not counts_by_context:
        raise ValueError("No Kallisto samples were loaded")
    if failed_runs:
        logger.warning("Failed to load %d runs", len(failed_runs))

    genes = sorted(all_genes)
    averaged_counts = {}
    metadata_rows = []
    for context, replicate_counts in counts_by_context.items():
        aligned = [counts.reindex(genes, fill_value=0) for counts in replicate_counts]
        averaged_counts[context] = pd.concat(aligned, axis=1).mean(axis=1)

        representative = next(sample for sample in samples.values() if sample["cell_type_id"] == context)
        run_ids = runs_by_context[context]
        metadata_rows.append(
            {
                **representative,
                "n_replicates": len(run_ids),
                "replicate_run_ids": ",".join(run_ids),
            }
        )
        logger.info("%s: averaged %d replicate(s)", context, len(run_ids))

    counts_df = pd.DataFrame(averaged_counts).fillna(0)
    metadata_df = pd.DataFrame(metadata_rows)
    logger.info(
        "Loaded %s: %d genes, %d contexts, %d runs",
        dataset,
        counts_df.shape[0],
        counts_df.shape[1],
        sum(metadata_df["n_replicates"]),
    )
    return counts_df, metadata_df
