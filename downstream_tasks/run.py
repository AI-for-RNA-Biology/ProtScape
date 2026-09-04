#!/usr/bin/env python

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import torch

from .config import (
    DEFAULT_OUTPUT_ROOT,
    EXPLICIT_CSV_TASKS,
    get_hc_embedding_paths,
    load_config,
    resolve_task_csv,
)
from .data.global_split import (
    ContextPresence,
    load_context_presence,
    restrict_global_probe_universe,
    split_fingerprint,
)
from .data.loaders import EmbeddingLoader, load_pinnacle_paper_gene_universe
from .data.task_loaders import get_task_loader
from .models.registry import MODEL_VARIANTS, parse_model_argument
from .training.cv_utils import SplitPlan, build_cv_splits
from .training.trainer import Trainer
from .utils.io_utils import save_results_csv, save_task_results, setup_output_dirs


CANONICAL_PDL_IMPLEMENTATION = "canonical"
GLOBAL_LR_MODEL_KEYS = frozenset(
    {"lr_ext_embed", "lr_global", "lr_global_ext_embed"}
)
EXPECTED_CELL_PPI_CONTEXTS = 207


def _reset_random_seed(seed: int) -> None:
    """Restart every model variant from the configured random seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_task_dataset_provenance(task: str, task_csv: Path) -> dict:
    """Load optional frozen-label metadata adjacent to a TT label table."""
    if not task.startswith("therapeutic_target_"):
        return {}
    manifest_path = task_csv.parent / "therapeutic_target_manifest.json"
    if not manifest_path.is_file():
        return {}
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid therapeutic-target manifest: {manifest_path}")
    required = {
        "format_version",
        "dataset",
        "reconstruction",
        "open_targets_release",
        "association_scope",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"Therapeutic-target manifest is missing {', '.join(missing)}: "
            f"{manifest_path}"
        )
    if payload["dataset"] != "therapeutic_target":
        raise ValueError(f"Unexpected dataset in {manifest_path}: {payload['dataset']}")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "reconstruction": str(payload["reconstruction"]),
        "open_targets_release": str(payload["open_targets_release"]),
        "association_scope": str(payload["association_scope"]),
    }


def _load_global_embedding_provenance(
    inference_path: Path,
    embedding_path: Path,
) -> dict:
    """Bind downstream results to the exact frozen global embedding export."""
    manifest_path = inference_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Global embedding manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid global embedding manifest: {manifest_path}")

    required = {
        "format_version",
        "model_type",
        "embedding_scope",
        "embedding_topology",
        "representation",
        "embedding_sha256",
        "checkpoint_sha256",
        "training_git_commit",
        "export_git_commit",
        "ppi_test_compatible",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(
            f"Global embedding manifest is missing {', '.join(missing)}: "
            f"{manifest_path}"
        )
    expected = {
        "model_type": "global_s2gae",
        "embedding_scope": "global",
        "embedding_topology": "full_reference",
        "representation": "encoder_jk_concat",
        "ppi_test_compatible": False,
    }
    mismatched = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if mismatched:
        raise ValueError(
            "Unexpected global embedding provenance: " + ", ".join(mismatched)
        )

    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "embedding_path": str(embedding_path.resolve()),
        "embedding_sha256": str(manifest["embedding_sha256"]),
        "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
        "training_git_commit": str(manifest["training_git_commit"]),
        "export_git_commit": str(manifest["export_git_commit"]),
        "embedding_topology": str(manifest["embedding_topology"]),
        "representation": str(manifest["representation"]),
    }


def _fmt_hp_value(value):
    if isinstance(value, float):
        text = f"{value:.12g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _build_hp_suffix(variant, config) -> str:
    parts = [f"lr{_fmt_hp_value(config.lr)}"]
    if variant.model_type.value == "abmil":
        parts.append(f"att{_fmt_hp_value(config.att_hidden_dim)}")
        parts.append(f"do{_fmt_hp_value(config.dropout)}")
        if getattr(config, "att_dropout", 0.0) > 0:
            parts.append(f"attdo{_fmt_hp_value(config.att_dropout)}")
        if variant.use_aem and config.aem_lambda > 0:
            parts.append(f"aem{_fmt_hp_value(config.aem_lambda)}")
        if variant.use_pdl and config.pdl_pmax > 0:
            pdl_proj_mode = variant.resolve_pdl_proj_mode(config.pdl_proj_mode)
            parts.append(f"pdl{_fmt_hp_value(config.pdl_pmax)}")
            parts.append(f"pdlproj{pdl_proj_mode}")
            if pdl_proj_mode == "identity":
                parts.append(f"pdlprojl{_fmt_hp_value(config.pdl_proj_layers)}")
        if variant.classifier_type.value == "mlp":
            parts.append(f"mlp{_fmt_hp_value(config.mil_mlp_dim)}")
    parts.append(f"wd{_fmt_hp_value(config.weight_decay)}")
    parts.append(f"sel{config.train_selection_metric}")
    parts.append("cw1" if config.use_class_weights else "cw0")
    return "_".join(parts)


def _add_prediction_labels(
    predictions_path: Path,
    genes: List[str],
    class_names: List[str],
) -> None:
    """Store the held-out gene and class order alongside the prediction arrays."""
    if not predictions_path.exists():
        return

    with np.load(predictions_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}

    test_idx = arrays["test_idx"].astype(np.int64)
    arrays["test_genes"] = np.asarray([genes[index] for index in test_idx], dtype=str)
    arrays["class_names"] = np.asarray(class_names, dtype=str)

    np.savez_compressed(predictions_path, **arrays)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-model")
    parser.add_argument(
        "--embedding-inference-model",
        default=None,
        help=(
            "Optional frozen embedding snapshot. The --inference-model value remains "
            "the downstream run identifier and output folder name."
        ),
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=None,
        help="Optional root containing the frozen inference embedding directory.",
    )
    parser.add_argument(
        "--cell-embedding-file",
        default="cell_embeddings.pt",
        help="Cell representation file within the inference directory.",
    )
    parser.add_argument(
        "--global-inference",
        type=Path,
        default=None,
        help=(
            "Global S2GAE export directory, or its protein_embeddings.pt file. "
            "Enables the shared context-free LR protocol."
        ),
    )
    parser.add_argument(
        "--esm2-embeddings",
        type=Path,
        default=None,
        help="Explicit ESM2 embedding pickle used by all global LR comparisons.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Downstream output root override (does not modify configs/paths.yaml).",
    )
    parser.add_argument(
        "--context-ppi-edgelists",
        type=Path,
        default=None,
        help=(
            "Directory containing the 207 released Cell-PPI .txt edgelists. "
            "Protein presence is used only to define and stratify shared folds."
        ),
    )
    parser.add_argument(
        "--output-model-key",
        default=None,
        help="Optional output folder name for a selected configuration.",
    )
    parser.add_argument("--scope", default="")
    parser.add_argument("--disease", default="")
    parser.add_argument("--inference-key", default="")
    parser.add_argument("--inference-label", default="")
    parser.add_argument("--readout-key", default="")
    parser.add_argument("--readout-label", default="")
    parser.add_argument(
        "--task",
        help=(
            "corum, protein_localization, pathway, or one of the 15 "
            "therapeutic_target_<disease_id> tasks."
        ),
    )
    parser.add_argument(
        "--task-csv",
        type=Path,
        default=None,
        help="Optional label CSV override for a versioned or corrected task snapshot.",
    )
    parser.add_argument("--model")
    parser.add_argument("--embedding-source", default="esm", choices=["esm", "prostt5"])
    parser.add_argument(
        "--dataset-mode",
        default="bulk",
        choices=["bulk", "legacy"],
        help=(
            "bulk: current task ∩ sequence ∩ HC universe; legacy: additionally "
            "restrict to the original PINNACLE gene set."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--att-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pdl-pmax", type=float, default=None)
    parser.add_argument("--train-selection-metric", type=str, default="auprc", choices=["auprc", "f1"])
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild <task>/full_results.csv from completed run-level results.",
    )
    args = parser.parse_args()
    if args.list_models:
        return args

    required = ["task"]
    if not args.aggregate_only:
        required.extend(["inference_model", "model"])
    missing = [f"--{name.replace('_', '-')}" for name in required if not getattr(args, name)]
    if missing:
        parser.error(f"required arguments: {', '.join(missing)}")
    return args


def list_models():
    print("\nAvailable model variants:\n")
    for key, variant in MODEL_VARIANTS.items():
        print(f"  {key:40s} {variant.name}")


def _build_shared_split(
    genes: List[str],
    Y: np.ndarray,
    embedding_loader: EmbeddingLoader,
    seed: int,
    n_splits: int,
    gene_universe: Optional[Set[str]] = None,
    gene_universe_name: str = "gene universe",
) -> Tuple[List[str], np.ndarray, SplitPlan]:
    genes_upper = [g.upper() for g in genes]
    task_gene_set = set(genes_upper)

    seq_genes = set(embedding_loader.load_esm().keys())
    hc_means, hc_cells = embedding_loader.load_hc_mean(task_gene_set)
    shared_genes_set = task_gene_set & seq_genes & set(hc_means.keys())
    universe_parts = ["task", "seq", "HC"]
    if gene_universe is not None:
        shared_genes_set &= {g.upper() for g in gene_universe}
        universe_parts.append(gene_universe_name)

    keep_idx = [i for i, g in enumerate(genes_upper) if g in shared_genes_set]
    if len(keep_idx) < n_splits:
        raise ValueError(
            f"Shared intersection has too few genes ({len(keep_idx)}) for {n_splits}-fold CV."
        )

    shared_genes = [genes_upper[i] for i in keep_idx]
    Y_shared = Y[keep_idx]
    shared_cell_ids = [hc_cells.get(g, ["unknown"]) for g in shared_genes]
    split_plan = build_cv_splits(
        Y_shared,
        cell_ids_per_bag=shared_cell_ids,
        use_context_split=True,
        k_label=None,
        n_splits=n_splits,
        seed=seed,
    )
    universe_label = " ∩ ".join(universe_parts)
    print(f"[INFO] Shared intersection ({universe_label}): {len(shared_genes)}/{len(genes)} genes")
    return shared_genes, Y_shared, split_plan


def _resolve_global_embedding_path(path: Path) -> Tuple[Path, Path]:
    """Resolve a global export file and its containing inference directory."""
    path = Path(path).expanduser()
    if path.is_dir():
        embedding_path = path / "protein_embeddings.pt"
        inference_path = path
    elif path.is_file():
        embedding_path = path
        inference_path = path.parent
    else:
        raise FileNotFoundError(f"Global inference path not found: {path}")
    if not embedding_path.is_file():
        raise FileNotFoundError(
            f"Global protein embedding export not found: {embedding_path}"
        )
    return embedding_path, inference_path


def _build_global_shared_split(
    genes: List[str],
    Y: np.ndarray,
    embedding_loader: EmbeddingLoader,
    context_presence: ContextPresence,
    seed: int,
    n_splits: int,
) -> Tuple[List[str], np.ndarray, SplitPlan, List[List[str]]]:
    """Build one split over task/global/ESM2/Cell-PPI shared proteins."""
    sequence_genes = embedding_loader.load_esm().keys()
    global_genes = embedding_loader.load_global().keys()
    shared_genes, Y_shared, shared_contexts = restrict_global_probe_universe(
        genes,
        Y,
        global_genes=global_genes,
        sequence_genes=sequence_genes,
        context_presence=context_presence,
    )
    if len(shared_genes) < n_splits:
        raise ValueError(
            "Global shared intersection has too few genes "
            f"({len(shared_genes)}) for {n_splits}-fold CV."
        )

    split_plan = build_cv_splits(
        Y_shared,
        cell_ids_per_bag=shared_contexts,
        use_context_split=True,
        k_label=None,
        n_splits=n_splits,
        seed=seed,
    )
    print(
        "[INFO] Shared intersection (task ∩ global ∩ ESM2 ∩ Cell-PPI): "
        f"{len(shared_genes)}/{len(genes)} genes"
    )
    return shared_genes, Y_shared, split_plan, shared_contexts


def _save_split_artifacts(
    output_dir: Path,
    genes: List[str],
    split_plan: SplitPlan,
    *,
    seed: int,
    context_presence: Optional[ContextPresence] = None,
    gene_contexts: Optional[List[List[str]]] = None,
) -> str:
    """Save the common folds and, for global probes, split-only context metadata."""
    fingerprint = split_fingerprint(genes, split_plan.folds)
    split_arrays = {
        "genes": np.asarray(genes, dtype=str),
        "split_stratification": np.asarray(
            split_plan.stratification_method, dtype=str
        ),
        **{
            f"fold_{fold}": np.asarray(indices, dtype=np.int64)
            for fold, indices in enumerate(split_plan.folds)
        },
    }
    if context_presence is not None:
        if gene_contexts is None or len(gene_contexts) != len(genes):
            raise ValueError("Global split metadata is not aligned with shared genes.")
        context_to_index = {
            name: index for index, name in enumerate(context_presence.context_names)
        }
        context_matrix = np.zeros(
            (len(genes), len(context_presence.context_names)), dtype=np.bool_
        )
        for gene_index, contexts in enumerate(gene_contexts):
            for context in contexts:
                context_matrix[gene_index, context_to_index[context]] = True
        split_arrays.update(
            {
                "context_names": np.asarray(context_presence.context_names, dtype=str),
                "context_presence": context_matrix,
                "context_presence_fingerprint": np.asarray(
                    context_presence.fingerprint, dtype=str
                ),
                "split_fingerprint": np.asarray(fingerprint, dtype=str),
            }
        )

    np.savez_compressed(output_dir / "split_indices.npz", **split_arrays)
    np.save(
        output_dir / "test_idx.npy",
        np.asarray(split_plan.folds[split_plan.test_fold_idx], dtype=np.int64),
    )
    if context_presence is not None:
        metadata = {
            "version": 1,
            "gene_universe": "task_global_esm2_cellppi",
            "n_genes": len(genes),
            "n_contexts": len(context_presence.context_names),
            "context_presence_fingerprint": context_presence.fingerprint,
            "split_fingerprint": fingerprint,
            "seed": int(seed),
            "test_fold": int(split_plan.test_fold_idx),
            "split_protocol": "fold_0_test_remaining_folds_rotate_validation",
            "split_stratification": split_plan.stratification_method,
        }
        (output_dir / "split_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return fingerprint


def main():
    args = parse_args()

    if args.list_models:
        list_models()
        return 0

    if args.aggregate_only:
        output_root = (
            args.output_root.expanduser()
            if args.output_root is not None
            else DEFAULT_OUTPUT_ROOT
        )
        agg_path = save_task_results(output_root, args.task)
        if agg_path is None:
            print("[WARN] No results found to aggregate")
            return 1
        print(f"[OK] Task results saved to: {agg_path}")
        return 0

    is_global_protocol = args.global_inference is not None
    if is_global_protocol and args.embedding_inference_model is not None:
        print(
            "[ERROR] --embedding-inference-model cannot be combined with "
            "--global-inference; use --inference-model as the run identifier."
        )
        return 1

    embedding_inference_model = args.embedding_inference_model or args.inference_model
    config = load_config(
        inference_model=embedding_inference_model,
        embedding_source=args.embedding_source,
        dataset_mode=args.dataset_mode,
    )
    if args.task not in config.tasks and args.task not in EXPLICIT_CSV_TASKS:
        available = ", ".join(sorted(set(config.tasks) | EXPLICIT_CSV_TASKS))
        print(f"[ERROR] Unknown task: {args.task}")
        print(f"[INFO] Available tasks: {available}")
        return 1

    if args.output_root is not None:
        config.output_root = args.output_root.expanduser()
    if args.inference_root is not None:
        config.inference_root = args.inference_root.expanduser()
    if args.esm2_embeddings is not None:
        config.embeddings.esm = args.esm2_embeddings.expanduser()

    if is_global_protocol:
        if config.dataset_mode != "bulk":
            print("[ERROR] Global LR evaluation requires --dataset-mode bulk.")
            return 1
        if config.embedding_source != "esm":
            print("[ERROR] Global LR evaluation requires --embedding-source esm.")
            return 1
        missing_global_args = []
        if args.esm2_embeddings is None:
            missing_global_args.append("--esm2-embeddings")
        if args.context_ppi_edgelists is None:
            missing_global_args.append("--context-ppi-edgelists")
        if missing_global_args:
            print(
                "[ERROR] Global LR evaluation requires explicit released-data paths: "
                + ", ".join(missing_global_args)
            )
            return 1
        if not config.embeddings.esm.is_file():
            print(f"[ERROR] ESM2 embedding file not found: {config.embeddings.esm}")
            return 1

    _reset_random_seed(args.seed)

    config.seed = args.seed
    config.train_selection_metric = args.train_selection_metric
    if args.att_dim is not None:
        config.att_hidden_dim = args.att_dim
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.lr is not None:
        config.lr = args.lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.pdl_pmax is not None:
        config.pdl_pmax = args.pdl_pmax
    if args.no_class_weight:
        config.use_class_weights = False

    is_pinnacle_paper = (
        not is_global_protocol and embedding_inference_model == "pinnacle_paper"
    )
    if is_pinnacle_paper and config.dataset_mode != "legacy":
        print("[INFO] inference-model=pinnacle_paper implies --dataset-mode legacy")
        config.dataset_mode = "legacy"

    hc_protein_labels_path = None
    hc_cell_labels_path = None
    global_embedding_provenance = {}
    inference_path = config.get_inference_path()
    if is_global_protocol:
        try:
            global_embedding_path, inference_path = _resolve_global_embedding_path(
                args.global_inference
            )
            global_embedding_provenance = _load_global_embedding_provenance(
                inference_path,
                global_embedding_path,
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            print(f"[ERROR] {error}")
            return 1
        hc_paths = {
            "protein_embed": global_embedding_path,
            "cell_embed": None,
            "protein_labels": None,
            "cell_labels": None,
        }
    elif is_pinnacle_paper:
        inference_path = config.embeddings.pinnacle_paper_protein.parent
        required_paths = [
            config.embeddings.pinnacle_paper_protein,
            config.embeddings.pinnacle_paper_labels,
        ]
        if config.embeddings.pinnacle_paper_cell.exists():
            required_paths.append(config.embeddings.pinnacle_paper_cell_labels)
        missing_paths = [p for p in required_paths if not p.exists()]
        if missing_paths:
            print("[ERROR] Missing PINNACLE embedding files:")
            for path in missing_paths:
                print(f"  {path}")
            return 1
        hc_paths = {
            "protein_embed": config.embeddings.pinnacle_paper_protein,
            "cell_embed": (
                config.embeddings.pinnacle_paper_cell
                if config.embeddings.pinnacle_paper_cell.exists()
                else None
            ),
            "protein_labels": config.embeddings.pinnacle_paper_labels,
            "cell_labels": config.embeddings.pinnacle_paper_cell_labels,
        }
        hc_protein_labels_path = config.embeddings.pinnacle_paper_labels
        hc_cell_labels_path = (
            config.embeddings.pinnacle_paper_cell_labels
            if hc_paths["cell_embed"] is not None
            else None
        )
    elif not inference_path.exists():
        print(f"[ERROR] Inference model not found: {inference_path}")
        return 1
    else:
        hc_paths = get_hc_embedding_paths(
            inference_path,
            cell_embedding_file=args.cell_embedding_file,
        )
        if hc_paths["protein_embed"] is None:
            print(f"[ERROR] No HC protein embeddings found in {inference_path}")
            return 1

    try:
        model_keys = parse_model_argument(args.model)
    except ValueError as e:
        print(f"[ERROR] {e}")
        list_models()
        return 1

    if args.output_model_key is not None and len(model_keys) != 1:
        print("[ERROR] --output-model-key can only be used with one model.")
        return 1
    if is_global_protocol:
        invalid_model_keys = [
            model_key
            for model_key in model_keys
            if model_key not in GLOBAL_LR_MODEL_KEYS
        ]
        if invalid_model_keys:
            print(
                "[ERROR] --global-inference supports only: "
                + ", ".join(sorted(GLOBAL_LR_MODEL_KEYS))
                + ". Invalid: "
                + ", ".join(invalid_model_keys)
            )
            return 1
    else:
        global_model_keys = [
            model_key
            for model_key in model_keys
            if "global" in MODEL_VARIANTS[model_key].embedding_sources
        ]
        if global_model_keys:
            print(
                "[ERROR] Models using global embeddings require --global-inference: "
                + ", ".join(global_model_keys)
            )
            return 1

    try:
        task_csv = resolve_task_csv(config.tasks, args.task, args.task_csv)
    except ValueError as error:
        print(f"[ERROR] {error}")
        return 1
    if not task_csv.exists():
        print(f"[ERROR] Label CSV not found: {task_csv}")
        return 1
    task_csv = task_csv.expanduser().resolve()
    try:
        task_dataset_provenance = _load_task_dataset_provenance(
            args.task,
            task_csv,
        )
    except (json.JSONDecodeError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 1
    task_csv_sha256 = _sha256_file(task_csv)

    task_loader = get_task_loader(args.task, task_csv)
    genes, Y, class_names = task_loader.load()

    embedding_loader = EmbeddingLoader(
        esm_path=config.embeddings.esm,
        hc_protein_path=hc_paths["protein_embed"],
        hc_cell_path=hc_paths["cell_embed"],
        hc_protein_labels_path=hc_protein_labels_path,
        hc_cell_labels_path=hc_cell_labels_path,
    )

    legacy_gene_universe = None
    if not is_global_protocol and config.dataset_mode == "legacy":
        if not config.embeddings.pinnacle_paper_labels.exists():
            print(
                "[ERROR] PINNACLE labels not found: "
                f"{config.embeddings.pinnacle_paper_labels}"
            )
            return 1
        legacy_gene_universe = load_pinnacle_paper_gene_universe(
            config.embeddings.pinnacle_paper_labels
        )
        print(f"[INFO] Legacy gene universe (pinnacle_paper): {len(legacy_gene_universe)} genes")

    context_presence = None
    shared_contexts = None
    if is_global_protocol:
        try:
            context_presence = load_context_presence(
                args.context_ppi_edgelists,
                expected_context_count=EXPECTED_CELL_PPI_CONTEXTS,
            )
            shared_genes, Y_shared, shared_split_plan, shared_contexts = (
                _build_global_shared_split(
                    genes=genes,
                    Y=Y,
                    embedding_loader=embedding_loader,
                    context_presence=context_presence,
                    seed=config.seed,
                    n_splits=config.n_folds,
                )
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"[ERROR] {error}")
            return 1
    else:
        shared_genes, Y_shared, shared_split_plan = _build_shared_split(
            genes=genes,
            Y=Y,
            embedding_loader=embedding_loader,
            seed=config.seed,
            n_splits=config.n_folds,
            gene_universe=legacy_gene_universe,
            gene_universe_name="pinnacle_paper",
        )
    embedding_loader.clear_cache()
    shared_split_fingerprint = split_fingerprint(
        shared_genes, shared_split_plan.folds
    )

    print(f"Inference model: {args.inference_model}")
    print(f"Embedding inference model: {embedding_inference_model}")
    print(f"Task: {args.task}")
    print(f"Models: {model_keys}")
    print(f"Embedding source: {config.embedding_source}")
    print(f"Dataset mode: {config.dataset_mode}")
    print(f"Train selection metric: {config.train_selection_metric}")
    print(f"Sequence embedding file: {config.embeddings.esm}")
    if is_global_protocol:
        print(f"Global embedding file: {hc_paths['protein_embed']}")
        print(f"Cell-PPI edgelists: {args.context_ppi_edgelists}")
        print(f"Shared split fingerprint: {shared_split_fingerprint}")

    for model_key in model_keys:
        _reset_random_seed(config.seed)
        variant = MODEL_VARIANTS[model_key]
        config.num_heads = variant.num_heads
        if variant.use_pdl and not 0.0 < float(config.pdl_pmax) <= 1.0:
            raise ValueError(
                f"{model_key} requires --pdl-pmax in (0, 1], got {config.pdl_pmax}."
            )

        hp_suffix = _build_hp_suffix(variant, config)
        dataset_suffix = "" if config.dataset_mode == "bulk" else f"__data_{config.dataset_mode}"
        output_model_key = args.output_model_key or (
            f"{model_key}__hp_{hp_suffix}__emb_{config.embedding_source}"
            f"{dataset_suffix}"
        )

        output_dir = setup_output_dirs(
            config.output_root,
            args.task,
            args.inference_model,
            output_model_key,
        )
        saved_split_fingerprint = _save_split_artifacts(
            output_dir,
            shared_genes,
            shared_split_plan,
            seed=config.seed,
            context_presence=context_presence,
            gene_contexts=shared_contexts,
        )
        if saved_split_fingerprint != shared_split_fingerprint:
            raise RuntimeError("Shared split changed between model variants.")

        trainer = Trainer(
            config=config,
            embedding_loader=embedding_loader,
            output_dir=output_dir,
            force=args.force,
        )

        result = trainer.train_and_evaluate(
            variant=variant,
            genes=shared_genes.copy(),
            Y=Y_shared.copy(),
            split_plan=shared_split_plan,
            strict_gene_universe=True,
        )

        _add_prediction_labels(
            output_dir / "test_predictions.npz",
            shared_genes,
            class_names,
        )

        result["task"] = args.task
        result["task_csv"] = task_csv.name
        result["task_csv_path"] = str(task_csv)
        result["task_csv_sha256"] = task_csv_sha256
        result["task_dataset_manifest"] = task_dataset_provenance.get(
            "manifest_path", ""
        )
        result["task_dataset_manifest_sha256"] = task_dataset_provenance.get(
            "manifest_sha256", ""
        )
        result["task_dataset_reconstruction"] = task_dataset_provenance.get(
            "reconstruction", ""
        )
        result["open_targets_release"] = task_dataset_provenance.get(
            "open_targets_release", ""
        )
        result["open_targets_association_scope"] = task_dataset_provenance.get(
            "association_scope", ""
        )
        result["model_key"] = model_key
        result["base_model_key"] = model_key
        result["output_model_key"] = output_model_key
        result["scope"] = args.scope
        result["disease"] = args.disease
        result["inference_key"] = args.inference_key or args.inference_model
        result["inference_label"] = args.inference_label or args.inference_model
        result["readout_key"] = args.readout_key or model_key
        result["readout_label"] = args.readout_label or variant.name
        result["embedding_source"] = config.embedding_source
        result["dataset_mode"] = config.dataset_mode
        if is_global_protocol:
            result["gene_universe"] = "task_global_esm2_cellppi"
        else:
            result["gene_universe"] = (
                "pinnacle_paper" if config.dataset_mode == "legacy" else "bulk_shared"
            )
        result["inference_name"] = args.inference_model
        result["embedding_inference_name"] = embedding_inference_model
        result["inference_dir"] = inference_path.name
        result["sequence_embedding_path"] = Path(config.embeddings.esm).name
        result["esm2_embedding_path"] = (
            str(Path(config.embeddings.esm).expanduser().resolve())
            if is_global_protocol
            else ""
        )
        result["hc_protein_embedding_path"] = (
            Path(hc_paths["protein_embed"]).name if hc_paths["protein_embed"] is not None else ""
        )
        result["hc_cell_embedding_path"] = (
            Path(hc_paths["cell_embed"]).name if hc_paths["cell_embed"] is not None else ""
        )
        result["global_embedding_path"] = (
            str(Path(hc_paths["protein_embed"]).resolve())
            if is_global_protocol
            else ""
        )
        result["global_embedding_manifest"] = global_embedding_provenance.get(
            "manifest_path", ""
        )
        result["global_embedding_manifest_sha256"] = global_embedding_provenance.get(
            "manifest_sha256", ""
        )
        result["global_embedding_sha256"] = global_embedding_provenance.get(
            "embedding_sha256", ""
        )
        result["global_checkpoint_sha256"] = global_embedding_provenance.get(
            "checkpoint_sha256", ""
        )
        result["global_training_git_commit"] = global_embedding_provenance.get(
            "training_git_commit", ""
        )
        result["global_export_git_commit"] = global_embedding_provenance.get(
            "export_git_commit", ""
        )
        result["global_embedding_topology"] = global_embedding_provenance.get(
            "embedding_topology", ""
        )
        result["global_embedding_representation"] = global_embedding_provenance.get(
            "representation", ""
        )
        result["context_ppi_edgelists"] = (
            str(Path(args.context_ppi_edgelists).expanduser().resolve())
            if is_global_protocol
            else ""
        )
        result["context_presence_count"] = (
            len(context_presence.context_names) if context_presence is not None else 0
        )
        result["context_presence_fingerprint"] = (
            context_presence.fingerprint if context_presence is not None else ""
        )
        result["shared_split_fingerprint"] = shared_split_fingerprint
        result["downstream_git_commit"] = os.environ.get(
            "PROTSCAPE_GIT_COMMIT", "uncommitted"
        )
        result["n_shared_samples"] = int(len(shared_genes))
        result["n_label_classes"] = int(Y_shared.shape[1])
        if Y_shared.shape[1] == 1:
            result["n_positive"] = int(Y_shared[:, 0].sum())
            result["n_negative"] = int(len(Y_shared) - Y_shared[:, 0].sum())
        else:
            result["n_positive"] = ""
            result["n_negative"] = ""
        result["folder name"] = args.inference_model
        result["lr"] = float(config.lr)
        result["weight_decay"] = float(config.weight_decay)
        result["dropout"] = float(config.dropout)
        result["att_dropout"] = float(config.att_dropout)
        result["att_hidden_dim"] = int(config.att_hidden_dim)
        result["num_heads"] = int(config.num_heads)
        result["mil_mlp_dim"] = int(config.mil_mlp_dim)
        result["ctx_proj_dim"] = 0
        result["aem_lambda"] = float(config.aem_lambda)
        result["pdl_pmax"] = float(config.pdl_pmax)
        result["pdl_impl_tag"] = (
            CANONICAL_PDL_IMPLEMENTATION
            if variant.use_pdl and config.pdl_pmax > 0
            else ""
        )
        result["model_schema_version"] = 2
        if variant.use_pdl and config.pdl_pmax > 0:
            pdl_mode = variant.resolve_pdl_proj_mode(config.pdl_proj_mode)
            result["pdl_layer_count"] = (
                int(config.pdl_proj_layers) if pdl_mode == "identity" else 1
            )
        else:
            result["pdl_layer_count"] = 0
        result["pdl_proj_mode"] = variant.resolve_pdl_proj_mode(config.pdl_proj_mode) if variant.use_pdl else ""
        result["pdl_proj_layers"] = config.pdl_proj_layers if variant.use_pdl and variant.resolve_pdl_proj_mode(config.pdl_proj_mode) == "identity" else 0
        result["esm_proj_dim"] = 0
        result["abmil_arch"] = "late_fusion" if "ext_embed" in variant.embedding_sources else "context_only"
        result["esm_conditioned_attention"] = False
        result["classifier_type"] = variant.classifier_type.value
        result["use_class_weights"] = bool(config.use_class_weights)
        result["train_selection_metric"] = config.train_selection_metric
        result["seed"] = int(config.seed)
        result["n_folds"] = int(config.n_folds)
        result["n_cv_folds"] = int(config.n_folds - 1)
        result["test_fold"] = 0
        result["split_protocol"] = "fold_0_test_remaining_folds_rotate_validation"
        result["split_stratification"] = (
            shared_split_plan.stratification_method
        )
        result["batch_size"] = int(config.batch_size)
        result["epochs"] = int(config.epochs)
        result["patience"] = int(config.patience)

        save_results_csv(result, output_dir)
        save_results_csv(result, output_dir, "model_config.csv")
        print(f"[OK] Finished {variant.name} -> {output_dir}")

    print(f"Results in: {config.output_root / args.task / args.inference_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
