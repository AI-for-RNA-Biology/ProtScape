#!/usr/bin/env python3
"""Run the complete Parkinson target-discovery analysis.

The analysis performs five-fold global inference for ProtScape and PINNACLE,
builds the probability-threshold candidate sets, scans STRING v12, runs
weighted Leiden and computes the Reactome and STRING enrichments.

Expected runtime is dominated by ten global ABMIL inference passes and the
STRING detailed-links scan. Use one CUDA GPU; the paper run used batch size 64.
The Open Targets/DrugBank annotation snapshot is a dated input dataset because
a live API query would not reproduce the paper snapshot.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom
from torch.utils.data import DataLoader, Dataset


from downstream_tasks.config import (
    DEFAULT_ESM_EMBEDDINGS,
    DEFAULT_INFERENCE_ROOT,
    DEFAULT_THERAPEUTIC_TARGET_DATASET_DIR,
    PATHS,
    get_hc_embedding_paths,
)
from downstream_tasks.data.datasets import collate_abmil
from downstream_tasks.data.loaders import EmbeddingLoader
from downstream_tasks.data.task_loaders import get_task_loader
from downstream_tasks.models.abmil import ABMIL_LateFusion
from downstream_tasks.models.registry import MODEL_VARIANTS
from downstream_tasks.training.cv_utils import (
    build_cv_splits,
    get_cv_train_val_indices,
    get_test_indices,
)
from exploration.plotting.parkinson_target_plots import plot_all


ANALYSIS_DIR = Path(PATHS["output_root"]) / "analysis/parkinson_target_analysis"
CHECKPOINT_ROOT = Path(PATHS["downstream_checkpoint_root"]) / "parkinson"
TASK_CSV = DEFAULT_THERAPEUTIC_TARGET_DATASET_DIR / "therapeutic_target_MONDO_0005180.csv"
ESM_PATH = DEFAULT_ESM_EMBEDDINGS

STRING_INFO = Path(PATHS["string_protein_info"])
STRING_ALIASES = Path(PATHS["string_protein_aliases"])
STRING_TERMS = Path(PATHS["string_enrichment_terms"])
STRING_LINKS = Path(PATHS["string_links_detailed"])
EXTERNAL_SUPPORT_SNAPSHOT = Path(PATHS["parkinson_external_support"])

TASK = "therapeutic_target_mondo_0005180"
MODEL_KEY = "abmil_hc_cell_ext_embed_gated_8_pdl"
PROBABILITY_THRESHOLD = 0.5
DEVICE = "cuda"
BATCH_SIZE = 64
SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    checkpoint_dir: str
    embedding_run: str


# These are the exact downstream checkpoints used for the paper panels.
MODEL_SPECS = {
    "protscape": ModelSpec(
        checkpoint_dir="protscape",
        embedding_run="s2gae_att_k1_fixed_do04_uni5e6_paper_20260510",
    ),
    "pinnacle": ModelSpec(
        checkpoint_dir="pinnacle",
        embedding_run=(
            "PINNACLE_model_globalsplit_random_symmetric_PPI-True_"
            "GATv2_H64_lambda001_drop02_out32__ep150_bulk"
        ),
    ),
}

MODEL_LABELS = {"protscape": "ProtScape", "pinnacle": "Pinnacle"}
SYNAPTIC_GROUPS = [
    ("Kainate receptors", "GRIK1–GRIK5", ["GRIK1", "GRIK2", "GRIK3", "GRIK4", "GRIK5"]),
    ("Neuroligins", "NLGN1–NLGN4X", ["NLGN1", "NLGN2", "NLGN3", "NLGN4X"]),
    ("DLGAP scaffolds", "DLGAP1–DLGAP4", ["DLGAP1", "DLGAP2", "DLGAP3", "DLGAP4"]),
]

ALIAS_PRIORITY = (
    "KEGG_NAME",
    "UniProt_GN_Name",
    "Ensembl_HGNC_alias_symbol",
    "Ensembl_HGNC_symbol",
)
STRING_CATEGORIES = {
    "Biological Process (Gene Ontology)": ("Process", "GO biological process"),
    "Molecular Function (Gene Ontology)": ("Function", "GO molecular function"),
    "Protein Domains (Pfam)": ("Pfam", "Pfam"),
    "Reactome Pathways": ("RCTM", "Reactome"),
}
MODULE_ANCHORS = ["DRD2", "GABRB3", "GRIN1", "GRM5", "SCN2A", "POLE"]
MODULE_LABELS = {
    1: "Class-A GPCR and monoaminergic receptors",
    2: "GABA/cholinergic ligand-gated receptors",
    3: "Ionotropic glutamate/NMDA receptor signalling",
    4: "Metabotropic glutamate and Class-C GPCR signalling",
    5: "Voltage-gated ion channels and excitability",
    6: "DNA replication/repair",
}
LEIDEN_RESOLUTIONS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
LEIDEN_SEEDS = tuple(range(10))


@dataclass
class InferenceData:
    genes: list[str]
    bags: list[np.ndarray]
    cell_ids: list[list[str]]
    esm: np.ndarray
    task_genes: list[str]
    task_labels: np.ndarray
    task_global_indices: np.ndarray
    split: object


class NormalizedBagDataset(Dataset):
    def __init__(
        self,
        bags: list[np.ndarray],
        esm: np.ndarray,
        bag_mean: np.ndarray,
        bag_std: np.ndarray,
        esm_mean: np.ndarray,
        esm_std: np.ndarray,
    ):
        self.bags = bags
        self.esm = esm
        self.bag_mean = bag_mean.astype(np.float32, copy=False)
        self.bag_std = bag_std.astype(np.float32, copy=False)
        self.esm_mean = esm_mean.astype(np.float32, copy=False)
        self.esm_std = esm_std.astype(np.float32, copy=False)
        self.labels = torch.zeros((len(bags), 1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, index: int):
        bag = ((self.bags[index] - self.bag_mean) / self.bag_std).astype(np.float32)
        esm = ((self.esm[index] - self.esm_mean) / self.esm_std).astype(np.float32)
        return {"ctx": torch.from_numpy(bag), "esm": torch.from_numpy(esm)}, self.labels[index]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required Parkinson analysis input: {path}")
    return path


def build_model(
    config: pd.Series, ctx_dim: int, esm_dim: int, device: torch.device
):
    variant = MODEL_VARIANTS[MODEL_KEY]
    return ABMIL_LateFusion(
        num_classes=1,
        ctx_dim=ctx_dim,
        esm_dim=esm_dim,
        num_heads=int(config["num_heads"]),
        att_hidden=int(config["att_hidden_dim"]),
        dropout=float(config["dropout"]),
        att_dropout=float(config["att_dropout"]),
        attention_type=variant.attention_type,
        classifier_type=variant.classifier_type,
        mlp_hidden_dim=int(config["mil_mlp_dim"]),
        use_pdl=variant.use_pdl,
        pdl_proj_mode=str(config["pdl_proj_mode"]),
        pdl_proj_layers=int(config["pdl_proj_layers"]),
    ).to(device)


def load_inference_data(spec: ModelSpec) -> InferenceData:
    task_genes, labels, _ = get_task_loader(TASK, require_file(TASK_CSV)).load()
    task_genes = [gene.upper() for gene in task_genes]
    if len(task_genes) != len(set(task_genes)):
        raise RuntimeError("The Parkinson benchmark contains duplicate proteins")

    embedding_paths = get_hc_embedding_paths(
        require_file(DEFAULT_INFERENCE_ROOT / spec.embedding_run / "protein_embeddings.pt").parent
    )
    loader = EmbeddingLoader(
        require_file(ESM_PATH),
        require_file(embedding_paths["protein_embed"]),
        require_file(embedding_paths["cell_embed"]),
    )
    esm_by_gene = loader.load_esm()
    task_means, task_cells = loader.load_hc_with_cell_mean(set(task_genes))
    bags_by_gene, cells_by_gene = loader.load_hc_with_cell(set(esm_by_gene))

    genes = sorted(set(esm_by_gene) & set(bags_by_gene))
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    bags = [np.stack(bags_by_gene[gene]).astype(np.float32, copy=False) for gene in genes]
    cell_ids = [cells_by_gene[gene] for gene in genes]
    esm = np.stack([esm_by_gene[gene] for gene in genes]).astype(np.float32, copy=False)
    usable_task_genes = [
        gene for gene in task_genes if gene in gene_to_index and gene in task_means
    ]
    task_row = {gene: index for index, gene in enumerate(task_genes)}
    task_labels = labels[[task_row[gene] for gene in usable_task_genes]]
    task_global_indices = np.asarray([gene_to_index[gene] for gene in usable_task_genes])
    split = build_cv_splits(
        task_labels,
        cell_ids_per_bag=[task_cells[gene] for gene in usable_task_genes],
        seed=SEED,
        n_splits=6,
    )
    if split.n_cv_folds != 5:
        raise RuntimeError(f"Expected five fold models, found {split.n_cv_folds}")
    return InferenceData(
        genes, bags, cell_ids, esm, usable_task_genes,
        task_labels, task_global_indices, split,
    )


def score_ensemble(
    data: InferenceData,
    config: pd.Series,
    run_dir: Path,
    device: torch.device,
) -> np.ndarray:
    fold_probabilities = []
    for fold in range(data.split.n_cv_folds):
        train_indices, _, _ = get_cv_train_val_indices(data.split, fold)
        train_global = data.task_global_indices[train_indices]
        train_contexts = np.concatenate([data.bags[index] for index in train_global])
        bag_mean = train_contexts.mean(axis=0)
        bag_std = np.maximum(train_contexts.std(axis=0), 1e-8)
        esm_mean = data.esm[train_global].mean(axis=0)
        esm_std = np.maximum(data.esm[train_global].std(axis=0), 1e-8)
        batches = DataLoader(
            NormalizedBagDataset(
                data.bags, data.esm, bag_mean, bag_std, esm_mean, esm_std
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_abmil,
        )
        model = build_model(config, data.bags[0].shape[1], data.esm.shape[1], device)
        checkpoint = require_file(run_dir / f"fold_{fold}.pt")
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        probabilities = []
        with torch.no_grad():
            for features, _ in batches:
                features = {name: value.to(device) for name, value in features.items()}
                with torch.amp.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    probabilities.append(
                        torch.sigmoid(model(features)).cpu().numpy()[:, 0]
                    )
        fold_probabilities.append(np.concatenate(probabilities).astype(np.float32))
        print(f"Completed global inference fold {fold + 1}/5", flush=True)
    return np.stack(fold_probabilities)


def cohort_membership(data: InferenceData) -> pd.DataFrame:
    test_indices = get_test_indices(data.split)
    label_by_gene = dict(zip(data.task_genes, data.task_labels[:, 0].astype(int)))
    test_genes = {data.task_genes[index] for index in test_indices}
    membership = pd.DataFrame({"protein": data.genes})
    membership["benchmark_label"] = membership["protein"].map(label_by_gene).astype("Int64")
    membership["benchmark_split"] = "label_excluded"
    membership.loc[membership["benchmark_label"].notna(), "benchmark_split"] = "train_validation"
    membership.loc[membership["protein"].isin(test_genes), "benchmark_split"] = "permanent_test"
    return membership


def run_global_inference() -> tuple[pd.DataFrame, pd.DataFrame]:
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Parkinson global inference requires a CUDA GPU")
    device = torch.device(DEVICE)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    score_tables = []
    shared_membership = None
    for model_name, spec in MODEL_SPECS.items():
        run_dir = CHECKPOINT_ROOT / spec.checkpoint_dir
        config = pd.read_csv(require_file(run_dir / "model_config.csv")).iloc[0]
        data = load_inference_data(spec)
        fold_scores = score_ensemble(data, config, run_dir, device)
        membership = cohort_membership(data)
        if shared_membership is None:
            shared_membership = membership
        else:
            pd.testing.assert_frame_equal(shared_membership, membership)

        scores = pd.DataFrame(
            {
                "protein": data.genes,
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "mean_probability": fold_scores.mean(axis=0),
                "std_probability": fold_scores.std(axis=0),
                "n_contexts": [len(ids) for ids in data.cell_ids],
            }
        )
        for fold, values in enumerate(fold_scores):
            scores[f"fold_{fold}_probability"] = values
        score_tables.append(scores)
        del data, fold_scores
        torch.cuda.empty_cache()

    assert shared_membership is not None
    scores = pd.concat(score_tables, ignore_index=True).merge(
        shared_membership, on="protein", how="left", validate="many_to_one"
    )
    return scores, shared_membership


def build_candidates(scores: pd.DataFrame) -> pd.DataFrame:
    candidates = scores[
        scores["benchmark_split"].eq("label_excluded")
        & scores["mean_probability"].ge(PROBABILITY_THRESHOLD)
    ].copy()
    candidates = candidates.sort_values(
        ["model", "mean_probability", "protein"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    candidates["candidate_rank"] = candidates.groupby("model").cumcount() + 1
    return candidates


def build_recovery_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = []
    summaries = []
    for model, model_scores in scores.groupby("model", sort=False):
        keep = model_scores["benchmark_split"].eq("label_excluded") | (
            model_scores["benchmark_split"].eq("permanent_test")
            & model_scores["benchmark_label"].eq(1)
        )
        ranked = model_scores[keep].sort_values(
            ["mean_probability", "protein"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        ranked["is_permanent_test_positive"] = (
            ranked["benchmark_split"].eq("permanent_test")
            & ranked["benchmark_label"].eq(1)
        )
        ranked["test_positives_recovered"] = ranked[
            "is_permanent_test_positive"
        ].cumsum()
        n_positives = int(ranked["is_permanent_test_positive"].sum())
        ranked["test_positive_recall_percent"] = (
            100 * ranked["test_positives_recovered"] / n_positives
        )
        ranked["other_proteins_screened"] = (
            ranked["rank"] - ranked["test_positives_recovered"]
        )
        curves.append(ranked)
        full = ranked[ranked["test_positives_recovered"].eq(n_positives)].iloc[0]
        summaries.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_screening_proteins": len(ranked),
                "n_permanent_test_positives": n_positives,
                "n_label_excluded_proteins": int(
                    ranked["benchmark_split"].eq("label_excluded").sum()
                ),
                "full_recovery_rank": int(full["rank"]),
                "other_proteins_screened_before_full_recovery": int(
                    full["other_proteins_screened"]
                ),
                "score_at_full_recovery": full["mean_probability"],
            }
        )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summaries)


def build_external_support(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join the dated raw Open Targets/DrugBank snapshot to rebuilt candidates."""
    snapshot = pd.read_csv(require_file(EXTERNAL_SUPPORT_SNAPSHOT))
    snapshot["protein"] = snapshot["protein"].str.upper()
    if snapshot["protein"].duplicated().any():
        raise ValueError("External-support snapshot has duplicate proteins")
    candidate_genes = set(candidates["protein"])
    if not candidate_genes.issubset(snapshot["protein"]):
        missing = sorted(candidate_genes - set(snapshot["protein"]))
        raise ValueError(
            f"External-support input lacks {len(missing)} rebuilt candidates"
        )
    flags = [
        "current_opentargets_parkinson_association_non_literature_only",
        "approved_human_drugbank_target_any_indication",
        "other_opentargets_disease_association",
    ]
    annotations = snapshot[["protein", *flags, "query_utc"]].copy()
    joined = candidates.merge(
        annotations, on="protein", how="left", validate="many_to_one"
    )
    summary_rows = []
    for model, rows in joined.groupby("model", sort=False):
        for flag in flags:
            count = int(rows[flag].astype(bool).sum())
            summary_rows.append(
                {
                    "support_flag": flag,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "supported_candidates": count,
                    "candidate_count": len(rows),
                    "support_percent": 100 * count / len(rows),
                    "nomination_probability_threshold": PROBABILITY_THRESHOLD,
                }
            )
    return joined, pd.DataFrame(summary_rows)


def build_synaptic_completion(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_excluded = scores[scores["benchmark_split"].eq("label_excluded")]
    member_rows = []
    wide_rows = []
    model_summaries: dict[tuple[str, str], dict] = {}
    nomination_counts = {}
    for model, rows in label_excluded.groupby("model", sort=False):
        ranked = rows.sort_values(
            ["mean_probability", "protein"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ranked["discovery_rank"] = ranked.index + 1
        nomination_counts[model] = int(
            ranked["mean_probability"].ge(PROBABILITY_THRESHOLD).sum()
        )
        for group_order, (group, member_label, members) in enumerate(SYNAPTIC_GROUPS):
            selected = ranked[ranked["protein"].isin(members)].copy()
            if set(selected["protein"]) != set(members):
                raise ValueError(f"Missing one or more {group} members for {model}")
            completion = int(selected["discovery_rank"].max())
            completion_member = selected.loc[
                selected["discovery_rank"].idxmax(), "protein"
            ]
            model_summaries[(group, model)] = {
                "rank": completion,
                "member": completion_member,
                "all_selected": bool(
                    selected["mean_probability"].ge(PROBABILITY_THRESHOLD).all()
                ),
            }
            for member_order, member in enumerate(members):
                row = selected[selected["protein"].eq(member)].iloc[0]
                member_rows.append(
                    {
                        "group": group,
                        "group_order": group_order,
                        "member_label": member_label,
                        "member": member,
                        "member_order": member_order,
                        "model": model,
                        "mean_probability": row["mean_probability"],
                        "discovery_rank": int(row["discovery_rank"]),
                        "group_completion_rank": completion,
                        "is_completion_member": member == completion_member,
                    }
                )
    for group_order, (group, member_label, members) in enumerate(SYNAPTIC_GROUPS):
        prot = model_summaries[(group, "protscape")]
        pinn = model_summaries[(group, "pinnacle")]
        wide_rows.append(
            {
                "group": group,
                "group_order": group_order,
                "member_label": member_label,
                "members": ";".join(members),
                "n_members": len(members),
                "protscape_completion_rank": prot["rank"],
                "protscape_completion_member": prot["member"],
                "pinnacle_completion_rank": pinn["rank"],
                "pinnacle_completion_member": pinn["member"],
                "protscape_all_members_probability_ge_0p5": prot["all_selected"],
                "pinnacle_all_members_probability_ge_0p5": pinn["all_selected"],
                "protscape_nomination_count_at_0p5": nomination_counts["protscape"],
                "pinnacle_nomination_count_at_0p5": nomination_counts["pinnacle"],
            }
        )
    return pd.DataFrame(member_rows), pd.DataFrame(wide_rows)


def benjamini_hochberg(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    order = np.argsort(array)
    adjusted = array[order] * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def build_string_mapping(genes: list[str]) -> pd.DataFrame:
    info = pd.read_csv(require_file(STRING_INFO), sep="\t").rename(
        columns={"#string_protein_id": "string_id"}
    )
    preferred = (
        info.assign(symbol=info["preferred_name"].str.upper())
        .groupby("symbol")["string_id"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    chosen = {}
    unresolved = set(genes)
    for gene in genes:
        hits = preferred.get(gene, [])
        if len(hits) == 1:
            chosen[gene] = (hits[0], "preferred_name")
            unresolved.remove(gene)

    alias_hits = {
        gene: {source: set() for source in ALIAS_PRIORITY} for gene in unresolved
    }
    with gzip.open(require_file(STRING_ALIASES), "rt") as handle:
        next(handle)
        for line in handle:
            string_id, alias, source = line.rstrip("\n").split("\t", 2)
            symbol = alias.upper()
            if symbol in alias_hits and source in alias_hits[symbol]:
                alias_hits[symbol][source].add(string_id)

    rows = []
    for gene in genes:
        string_id, method = chosen.get(gene, ("", "none"))
        if not string_id:
            for source in ALIAS_PRIORITY:
                hits = alias_hits[gene][source]
                if len(hits) == 1:
                    string_id, method = next(iter(hits)), source
                    break
        rows.append(
            {
                "protein": gene,
                "string_id": string_id,
                "mapping_method": method,
                "mapped": bool(string_id),
            }
        )
    return pd.DataFrame(rows)


def read_string_annotations(allowed_ids: set[str]) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(require_file(STRING_TERMS), sep="\t", chunksize=500_000):
        chunk = chunk[
            chunk["#string_protein_id"].isin(allowed_ids)
            & chunk["category"].isin(STRING_CATEGORIES)
        ].copy()
        if chunk.empty:
            continue
        chunk[["category_code", "source"]] = (
            chunk["category"].map(STRING_CATEGORIES).apply(pd.Series)
        )
        parts.append(
            chunk.rename(columns={"#string_protein_id": "string_id"})[
                ["string_id", "category_code", "source", "term", "description"]
            ]
        )
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def enrichment_table(
    annotations: pd.DataFrame,
    string_to_gene: dict[str, str],
    set_name: str,
    query_ids: set[str],
    background_ids: set[str],
) -> pd.DataFrame:
    annotations = annotations[annotations["string_id"].isin(background_ids)]
    keys = ["category_code", "source", "term", "description"]
    result = (
        annotations.groupby(keys, as_index=False)["string_id"].nunique()
        .rename(columns={"string_id": "term_size"})
    )
    result = result[result["term_size"].between(10, 500)].copy()
    hits = (
        annotations[annotations["string_id"].isin(query_ids)]
        .groupby(keys)["string_id"]
        .agg(lambda values: sorted(set(values)))
    )
    result["hit_string_ids"] = [
        hits.get(tuple(row), [])
        for row in result[keys].itertuples(index=False, name=None)
    ]
    result["gene_count"] = result["hit_string_ids"].map(len)
    result["p_value"] = hypergeom.sf(
        result["gene_count"] - 1,
        len(background_ids),
        result["term_size"],
        len(query_ids),
    )
    result["fdr"] = result.groupby("source")["p_value"].transform(
        benjamini_hochberg
    )
    result["fold_enrichment"] = (
        result["gene_count"] / len(query_ids)
    ) / (result["term_size"] / len(background_ids))
    result["log2_fold_enrichment"] = np.nan
    positive = result["fold_enrichment"].gt(0)
    result.loc[positive, "log2_fold_enrichment"] = np.log2(
        result.loc[positive, "fold_enrichment"]
    )
    result["minus_log10_fdr"] = -np.log10(result["fdr"].clip(lower=1e-300))
    result["set"] = set_name
    result["category"] = result["category_code"]
    result["number_of_genes"] = result["gene_count"]
    result["hit_gene_symbols"] = result["hit_string_ids"].map(
        lambda ids: ";".join(sorted(string_to_gene[string_id] for string_id in ids))
    )
    return result


def build_main_string_enrichment(
    membership: pd.DataFrame,
    candidates: pd.DataFrame,
    mapping: pd.DataFrame,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_to_string = mapping[mapping["mapped"]].set_index("protein")["string_id"].to_dict()
    string_to_gene = {string_id: gene for gene, string_id in gene_to_string.items()}
    labelled = set(
        membership.loc[~membership["benchmark_split"].eq("label_excluded"), "protein"]
    )
    label_excluded = set(
        membership.loc[membership["benchmark_split"].eq("label_excluded"), "protein"]
    )
    positives = set(membership.loc[membership["benchmark_label"].eq(1), "protein"])
    protscape_candidates = set(
        candidates.loc[candidates["model"].eq("protscape"), "protein"]
    )
    specifications = [
        ("known_targets", positives, labelled),
        ("protscape_candidates", protscape_candidates, label_excluded),
    ]
    tables = []
    for set_name, query_genes, background_genes in specifications:
        tables.append(
            enrichment_table(
                annotations,
                string_to_gene,
                set_name,
                {gene_to_string[g] for g in query_genes if g in gene_to_string},
                {gene_to_string[g] for g in background_genes if g in gene_to_string},
            )
        )
    enrichment = pd.concat(tables, ignore_index=True)

    known = enrichment[
        enrichment["set"].eq("known_targets")
        & enrichment["category"].isin(["Process", "Function", "Pfam"])
    ]
    selected = []
    for category_order, category in enumerate(["Process", "Function", "Pfam"]):
        rows = known[known["category"].eq(category)].sort_values(
            ["fdr", "fold_enrichment", "term"], ascending=[True, False, True]
        ).head(3).copy()
        rows["category_order"] = category_order
        rows["term_order"] = np.arange(len(rows))
        selected.append(rows)
    selected = pd.concat(selected, ignore_index=True)[
        ["category", "term", "description", "category_order", "term_order"]
    ]
    display = selected.merge(
        enrichment[enrichment["set"].isin(["known_targets", "protscape_candidates"])],
        on=["category", "term", "description"],
        how="left",
    )
    return selected, display


def build_string_network(
    membership: pd.DataFrame,
    candidates: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    known = set(membership.loc[membership["benchmark_label"].eq(1), "protein"])
    novel = set(candidates.loc[candidates["model"].eq("protscape"), "protein"])
    submitted = known | novel
    gene_to_string = mapping[mapping["mapped"]].set_index("protein")["string_id"].to_dict()
    string_to_gene = {string_id: gene for gene, string_id in gene_to_string.items()}
    ids = {gene_to_string[gene] for gene in submitted if gene in gene_to_string}
    parts = []
    usecols = ["protein1", "protein2", "experimental", "combined_score"]
    for chunk in pd.read_csv(
        require_file(STRING_LINKS), sep=r"\s+", usecols=usecols, chunksize=500_000
    ):
        selected = chunk[
            chunk["experimental"].gt(0)
            & chunk["protein1"].isin(ids)
            & chunk["protein2"].isin(ids)
        ].copy()
        if selected.empty:
            continue
        first = selected["protein1"].map(string_to_gene)
        second = selected["protein2"].map(string_to_gene)
        selected["protein_a"] = [min(a, b) for a, b in zip(first, second)]
        selected["protein_b"] = [max(a, b) for a, b in zip(first, second)]
        selected["experimental_score"] = selected["experimental"] / 1000
        selected["combined_score"] = selected["combined_score"] / 1000
        parts.append(
            selected[
                ["protein_a", "protein_b", "experimental_score", "combined_score"]
            ]
        )
    edges = pd.concat(parts, ignore_index=True).sort_values(
        ["experimental_score", "combined_score"], ascending=False
    ).drop_duplicates(["protein_a", "protein_b"])
    degree = pd.concat([edges["protein_a"], edges["protein_b"]]).value_counts()
    nodes = pd.DataFrame({"protein": sorted(submitted)})
    nodes["node_role"] = np.where(
        nodes["protein"].isin(known), "benchmark_positive", "candidate"
    )
    nodes["string_id"] = nodes["protein"].map(gene_to_string)
    nodes["mapped"] = nodes["string_id"].notna()
    nodes["degree"] = nodes["protein"].map(degree).fillna(0).astype(int)
    nodes["connected"] = nodes["degree"].gt(0)
    return nodes, edges


def leiden_communities(
    graph: nx.Graph, resolution: float, seed: int
) -> list[set[str]]:
    import igraph as ig
    import leidenalg

    nodes = sorted(graph)
    node_index = {node: index for index, node in enumerate(nodes)}
    edges = list(graph.edges())
    igraph_graph = ig.Graph(
        n=len(nodes),
        edges=[(node_index[first], node_index[second]) for first, second in edges],
        directed=False,
    )
    igraph_graph.es["weight"] = [
        float(graph[first][second]["experimental_score"])
        for first, second in edges
    ]
    partition = leidenalg.find_partition(
        igraph_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=seed,
    )
    return [{nodes[index] for index in community} for community in partition]


def min_max_normalize(values: pd.Series) -> pd.Series:
    value_range = values.max() - values.min()
    if np.isclose(value_range, 0.0):
        return pd.Series(1.0, index=values.index)
    return (values - values.min()) / value_range


def select_leiden_partition(
    graph: nx.Graph,
) -> tuple[list[set[str]], pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import adjusted_rand_score

    nodes = sorted(graph)
    runs_by_resolution = {}
    resolution_rows = []
    stability_rows = []
    for resolution in LEIDEN_RESOLUTIONS:
        runs = []
        for seed in LEIDEN_SEEDS:
            communities = leiden_communities(graph, resolution, seed)
            labels = {
                node: community_index
                for community_index, community in enumerate(communities)
                for node in community
            }
            runs.append(
                {
                    "resolution": resolution,
                    "seed": seed,
                    "communities": communities,
                    "labels": np.asarray([labels[node] for node in nodes]),
                    "modularity": nx.community.modularity(
                        graph, communities, weight="experimental_score", resolution=1.0
                    ),
                    "n_modules": len(communities),
                }
            )
        agreement = np.eye(len(runs))
        pairwise = []
        for first in range(len(runs)):
            for second in range(first + 1, len(runs)):
                ari = adjusted_rand_score(runs[first]["labels"], runs[second]["labels"])
                agreement[first, second] = agreement[second, first] = ari
                pairwise.append(ari)
        mean_ari = (agreement.sum(axis=1) - 1.0) / (len(runs) - 1)
        for run, value in zip(runs, mean_ari):
            run["mean_ari"] = float(value)
        runs_by_resolution[resolution] = runs
        module_counts = [run["n_modules"] for run in runs]
        resolution_rows.append(
            {
                "resolution": resolution,
                "mean_modularity": np.mean([run["modularity"] for run in runs]),
                "mean_pairwise_ari": np.mean(pairwise),
                "mean_n_modules": np.mean(module_counts),
                "minimum_n_modules": min(module_counts),
                "maximum_n_modules": max(module_counts),
                "eligible_more_than_one_module_in_every_run": min(module_counts) > 1,
            }
        )
    selection = pd.DataFrame(resolution_rows)
    eligible = selection[
        selection["eligible_more_than_one_module_in_every_run"]
    ].copy()
    eligible["mean_modularity_normalized"] = min_max_normalize(
        eligible["mean_modularity"]
    )
    eligible["mean_pairwise_ari_normalized"] = min_max_normalize(
        eligible["mean_pairwise_ari"]
    )
    eligible["combined_score"] = 0.5 * (
        eligible["mean_modularity_normalized"]
        + eligible["mean_pairwise_ari_normalized"]
    )
    selected_resolution = float(
        eligible.sort_values(
            ["combined_score", "mean_pairwise_ari", "mean_modularity", "resolution"],
            ascending=[False, False, False, True],
        ).iloc[0]["resolution"]
    )
    for column in [
        "mean_modularity_normalized",
        "mean_pairwise_ari_normalized",
        "combined_score",
    ]:
        selection[column] = selection["resolution"].map(
            eligible.set_index("resolution")[column]
        )
    selection["selected_resolution"] = selection["resolution"].eq(selected_resolution)

    selected_runs = runs_by_resolution[selected_resolution]
    best_ari = max(run["mean_ari"] for run in selected_runs)
    medoid_candidates = [
        index for index, run in enumerate(selected_runs)
        if np.isclose(run["mean_ari"], best_ari, rtol=0.0, atol=1e-12)
    ]
    medoid_index = max(
        medoid_candidates,
        key=lambda index: (selected_runs[index]["modularity"], -selected_runs[index]["seed"]),
    )
    for resolution, runs in runs_by_resolution.items():
        for index, run in enumerate(runs):
            stability_rows.append(
                {
                    "resolution": resolution,
                    "seed": run["seed"],
                    "modularity": run["modularity"],
                    "n_modules": run["n_modules"],
                    "mean_ari_to_other_runs_at_same_resolution": run["mean_ari"],
                    "selected_resolution": resolution == selected_resolution,
                    "selected_medoid_partition": (
                        resolution == selected_resolution and index == medoid_index
                    ),
                }
            )
    return selected_runs[medoid_index]["communities"], selection, pd.DataFrame(stability_rows)


def order_modules(communities: list[set[str]]) -> list[set[str]]:
    ordered = []
    for anchor in MODULE_ANCHORS:
        match = next((community for community in communities if anchor in community), None)
        if match is not None and match not in ordered:
            ordered.append(match)
    ordered.extend(
        sorted(
            (community for community in communities if community not in ordered),
            key=lambda community: (-len(community), min(community)),
        )
    )
    return ordered


def build_leiden_tables(
    network_nodes: pd.DataFrame,
    edges: pd.DataFrame,
    membership: pd.DataFrame,
    candidates: pd.DataFrame,
    support: pd.DataFrame,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    graph = nx.from_pandas_edgelist(
        edges,
        "protein_a",
        "protein_b",
        edge_attr=["experimental_score", "combined_score"],
    )
    connected = network_nodes[network_nodes["connected"]].copy()
    if set(graph) != set(connected["protein"]):
        raise ValueError("Connected STRING nodes and edge endpoints disagree")
    medoid, selection, stability = select_leiden_partition(graph)
    modules = order_modules(medoid)
    if len(modules) != len(MODULE_LABELS):
        raise RuntimeError(
            f"Expected six Leiden communities, found {len(modules)}"
        )
    cluster_by_protein = {
        protein: module_id
        for module_id, module in enumerate(modules, start=1)
        for protein in module
    }
    connected["leiden_cluster"] = connected["protein"].map(cluster_by_protein)
    connected["leiden_cluster_label"] = connected["leiden_cluster"].map(MODULE_LABELS)
    connected = connected.merge(
        membership[["protein", "benchmark_label", "benchmark_split"]],
        on="protein",
        how="left",
        validate="one_to_one",
    )
    prot_candidates = candidates[candidates["model"].eq("protscape")][
        ["protein", "candidate_rank", "mean_probability"]
    ].rename(
        columns={
            "candidate_rank": "protscape_candidate_rank",
            "mean_probability": "protscape_mean_probability",
        }
    )
    connected = connected.merge(
        prot_candidates, on="protein", how="left", validate="one_to_one"
    )
    support_columns = [
        "protein",
        "current_opentargets_parkinson_association_non_literature_only",
        "approved_human_drugbank_target_any_indication",
        "other_opentargets_disease_association",
    ]
    prot_support = support[support["model"].eq("protscape")][support_columns]
    connected = connected.merge(
        prot_support, on="protein", how="left", validate="one_to_one"
    )
    for column in support_columns[1:]:
        connected[column] = connected[column].fillna(False).astype(bool)
    return graph, connected, selection, stability


def build_module_reactome_enrichment(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    string_by_protein = nodes.set_index("protein")["string_id"].to_dict()
    protein_by_string = {string_id: protein for protein, string_id in string_by_protein.items()}
    network_ids = set(protein_by_string)
    reactome = annotations[
        annotations["source"].eq("Reactome")
        & annotations["string_id"].isin(network_ids)
    ].copy()
    reactome["protein"] = reactome["string_id"].map(protein_by_string)
    background = set(graph)
    background_size = len(background)
    rows = []
    for (term, description), term_rows in reactome.groupby(["term", "description"]):
        members = set(term_rows["protein"]) & background
        if not (3 <= len(members) <= 0.8 * background_size):
            continue
        for module_id in sorted(MODULE_LABELS):
            module = set(
                nodes.loc[nodes["leiden_cluster"].eq(module_id), "protein"]
            )
            hits = sorted(module & members)
            fold = (len(hits) / len(module)) / (len(members) / background_size)
            rows.append(
                {
                    "leiden_cluster": module_id,
                    "leiden_cluster_label": MODULE_LABELS[module_id],
                    "module_size": len(module),
                    "term": term,
                    "description": description,
                    "term_network_size": len(members),
                    "module_hits": len(hits),
                    "module_hit_genes": ";".join(hits),
                    "fold_enrichment": fold,
                    "p_value": hypergeom.sf(
                        len(hits) - 1,
                        background_size,
                        len(members),
                        len(module),
                    ),
                }
            )
    enrichment = pd.DataFrame(rows)
    enrichment["fdr"] = benjamini_hochberg(enrichment["p_value"])
    enrichment["minus_log10_fdr"] = -np.log10(
        enrichment["fdr"].clip(lower=1e-300)
    )
    enrichment["significant"] = (
        enrichment["fdr"].lt(0.05)
        & enrichment["module_hits"].ge(3)
        & enrichment["fold_enrichment"].gt(1)
    )
    displayed = (
        enrichment[enrichment["significant"]]
        .sort_values(
            ["leiden_cluster", "fdr", "fold_enrichment", "module_hits", "term"],
            ascending=[True, True, False, False, True],
        )
        .groupby("leiden_cluster", as_index=False)
        .head(5)
        .sort_values("leiden_cluster")
        .reset_index(drop=True)
    )
    if set(displayed["leiden_cluster"]) != set(MODULE_LABELS):
        raise RuntimeError("One or more Leiden clusters lack five displayable Reactome terms")
    return enrichment, displayed


def build_module_summary(nodes: pd.DataFrame) -> pd.DataFrame:
    summary = (
        nodes.assign(
            protscape_protein=nodes["node_role"].eq("candidate"),
            known_target_protein=nodes["node_role"].eq("benchmark_positive"),
        )
        .groupby("leiden_cluster", as_index=False)
        .agg(
            total_proteins=("protein", "size"),
            protscape_proteins=("protscape_protein", "sum"),
            known_target_proteins=("known_target_protein", "sum"),
        )
    )
    summary["leiden_cluster_label"] = summary["leiden_cluster"].map(MODULE_LABELS)
    summary["protscape_percent"] = (
        100 * summary["protscape_proteins"] / summary["total_proteins"]
    )
    return summary


def build_statistics(
    membership: pd.DataFrame,
    candidates: pd.DataFrame,
    network_nodes: pd.DataFrame,
    edges: pd.DataFrame,
    leiden_nodes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split, split_rows in membership.groupby("benchmark_split", dropna=False):
        rows.append(
            {
                "section": "global_inference_cohort",
                "group": split,
                "statistic": "proteins",
                "value": len(split_rows),
            }
        )
        for label in [1, 0]:
            rows.append(
                {
                    "section": "global_inference_cohort",
                    "group": split,
                    "statistic": f"label_{label}_proteins",
                    "value": int(split_rows["benchmark_label"].eq(label).sum()),
                }
            )
    for model, model_rows in candidates.groupby("model"):
        rows.append(
            {
                "section": "candidate_sets",
                "group": model,
                "statistic": "probability_ge_0p5",
                "value": len(model_rows),
            }
        )
    rows.extend(
        [
            {
                "section": "string_network",
                "group": "protscape_known_plus_candidates",
                "statistic": "submitted_nodes",
                "value": len(network_nodes),
            },
            {
                "section": "string_network",
                "group": "protscape_known_plus_candidates",
                "statistic": "connected_nodes",
                "value": len(leiden_nodes),
            },
            {
                "section": "string_network",
                "group": "protscape_known_plus_candidates",
                "statistic": "experimental_edges",
                "value": len(edges),
            },
            {
                "section": "string_network",
                "group": "protscape_known_plus_candidates",
                "statistic": "leiden_clusters",
                "value": leiden_nodes["leiden_cluster"].nunique(),
            },
        ]
    )
    return pd.DataFrame(rows)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    scores, membership = run_global_inference()
    candidates = build_candidates(scores)
    recovery_curve, recovery_summary = build_recovery_tables(scores)
    support_rows, support_summary = build_external_support(candidates)
    synaptic_members, synaptic_summary = build_synaptic_completion(scores)

    mapping = build_string_mapping(sorted(membership["protein"].unique()))
    annotations = read_string_annotations(
        set(mapping.loc[mapping["mapped"], "string_id"])
    )
    selected_terms, main_enrichment = build_main_string_enrichment(
        membership, candidates, mapping, annotations
    )
    network_nodes, edges = build_string_network(membership, candidates, mapping)
    graph, leiden_nodes, leiden_selection, leiden_stability = build_leiden_tables(
        network_nodes, edges, membership, candidates, support_rows
    )
    reactome_all, reactome_display = build_module_reactome_enrichment(
        graph, leiden_nodes, annotations
    )
    module_summary = build_module_summary(leiden_nodes)
    statistics = build_statistics(
        membership, candidates, network_nodes, edges, leiden_nodes
    )

    outputs = {
        "parkinson_dataset_statistics.csv": statistics,
        "parkinson_global_predictions.csv.gz": scores,
        "parkinson_candidate_set.csv": candidates,
        "parkinson_candidate_recovery_curve.csv": recovery_curve,
        "parkinson_candidate_recovery_summary.csv": recovery_summary,
        "parkinson_candidate_external_support_rows.csv": support_rows,
        "parkinson_candidate_external_support.csv": support_summary,
        "parkinson_synaptic_group_members.csv": synaptic_members,
        "parkinson_synaptic_group_completion.csv": synaptic_summary,
        "parkinson_string_mapping.csv": mapping,
        "parkinson_string_selected_terms.csv": selected_terms,
        "parkinson_string_main_enrichment.csv": main_enrichment,
        "parkinson_string_network_all_nodes.csv": network_nodes,
        "parkinson_string_network_edges.csv": edges,
        "parkinson_leiden_nodes.csv": leiden_nodes,
        "parkinson_leiden_selection.csv": leiden_selection,
        "parkinson_leiden_stability.csv": leiden_stability,
        "parkinson_reactome_all_tests.csv.gz": reactome_all,
        "parkinson_reactome_module_enrichment.csv": reactome_display,
        "parkinson_leiden_module_summary.csv": module_summary,
    }
    for filename, table in outputs.items():
        table.to_csv(ANALYSIS_DIR / filename, index=False)
    plot_all(ANALYSIS_DIR, ANALYSIS_DIR, ANALYSIS_DIR)
    print(f"Wrote Parkinson analysis CSVs and plots to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
