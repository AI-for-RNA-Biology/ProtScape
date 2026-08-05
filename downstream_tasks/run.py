#!/usr/bin/env python

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import torch

from .config import get_hc_embedding_paths, load_config
from .data.loaders import EmbeddingLoader, load_pinnacle_paper_gene_universe
from .data.task_loaders import get_task_loader
from .models.registry import MODEL_VARIANTS, parse_model_argument
from .training.cv_utils import SplitPlan, build_cv_splits
from .training.trainer import Trainer
from .utils.io_utils import save_aggregated_results, save_results_csv, setup_output_dirs


CANONICAL_PDL_IMPLEMENTATION = "canonical"


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
        "--task",
        help="corum or one of the 15 therapeutic_target_<disease_id> paper tasks.",
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
            "restrict to the original PINNACLE paper gene set."
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
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        return args

    required = ["inference_model", "task"]
    if not args.aggregate_only:
        required.append("model")
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


def main():
    args = parse_args()

    if args.list_models:
        list_models()
        return 0

    embedding_inference_model = args.embedding_inference_model or args.inference_model
    config = load_config(
        inference_model=embedding_inference_model,
        embedding_source=args.embedding_source,
        dataset_mode=args.dataset_mode,
    )
    if args.task not in config.tasks:
        available = ", ".join(sorted(config.tasks.keys()))
        print(f"[ERROR] Unknown task: {args.task}")
        print(f"[INFO] Available tasks: {available}")
        return 1

    if args.aggregate_only:
        agg_path = save_aggregated_results(config.output_root, args.task, args.inference_model)
        if agg_path is None:
            print("[WARN] No results found to aggregate")
            return 1
        print(f"[OK] Aggregated results saved to: {agg_path}")
        return 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    is_pinnacle_paper = embedding_inference_model == "pinnacle_paper"
    if is_pinnacle_paper and config.dataset_mode != "legacy":
        print("[INFO] inference-model=pinnacle_paper implies --dataset-mode legacy")
        config.dataset_mode = "legacy"

    hc_protein_labels_path = None
    hc_cell_labels_path = None
    inference_path = config.get_inference_path()
    if is_pinnacle_paper:
        inference_path = config.embeddings.pinnacle_paper_protein.parent
        required_paths = [
            config.embeddings.pinnacle_paper_protein,
            config.embeddings.pinnacle_paper_labels,
        ]
        if config.embeddings.pinnacle_paper_cell.exists():
            required_paths.append(config.embeddings.pinnacle_paper_cell_labels)
        missing_paths = [p for p in required_paths if not p.exists()]
        if missing_paths:
            print("[ERROR] Missing PINNACLE paper embedding files:")
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
        hc_paths = get_hc_embedding_paths(inference_path)
        if hc_paths["protein_embed"] is None:
            print(f"[ERROR] No HC protein embeddings found in {inference_path}")
            return 1

    try:
        model_keys = parse_model_argument(args.model)
    except ValueError as e:
        print(f"[ERROR] {e}")
        list_models()
        return 1

    task_config = config.tasks[args.task]
    task_csv = args.task_csv if args.task_csv is not None else task_config.label_csv
    if not task_csv.exists():
        print(f"[ERROR] Label CSV not found: {task_csv}")
        return 1

    task_loader = get_task_loader(args.task, task_csv)
    genes, Y, _ = task_loader.load()

    embedding_loader = EmbeddingLoader(
        esm_path=config.embeddings.esm,
        hc_protein_path=hc_paths["protein_embed"],
        hc_cell_path=hc_paths["cell_embed"],
        hc_protein_labels_path=hc_protein_labels_path,
        hc_cell_labels_path=hc_cell_labels_path,
    )

    legacy_gene_universe = None
    if config.dataset_mode == "legacy":
        if not config.embeddings.pinnacle_paper_labels.exists():
            print(
                "[ERROR] PINNACLE paper labels not found: "
                f"{config.embeddings.pinnacle_paper_labels}"
            )
            return 1
        legacy_gene_universe = load_pinnacle_paper_gene_universe(
            config.embeddings.pinnacle_paper_labels
        )
        print(f"[INFO] Legacy gene universe (pinnacle_paper): {len(legacy_gene_universe)} genes")

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

    print(f"Inference model: {args.inference_model}")
    print(f"Embedding inference model: {embedding_inference_model}")
    print(f"Task: {args.task}")
    print(f"Models: {model_keys}")
    print(f"Embedding source: {config.embedding_source}")
    print(f"Dataset mode: {config.dataset_mode}")
    print(f"Train selection metric: {config.train_selection_metric}")
    print(f"Sequence embedding file: {config.embeddings.esm}")

    for model_key in model_keys:
        variant = MODEL_VARIANTS[model_key]
        config.num_heads = variant.num_heads
        if variant.use_pdl and not 0.0 < float(config.pdl_pmax) <= 1.0:
            raise ValueError(
                f"{model_key} requires --pdl-pmax in (0, 1], got {config.pdl_pmax}."
            )

        hp_suffix = _build_hp_suffix(variant, config)
        dataset_suffix = "" if config.dataset_mode == "bulk" else f"__data_{config.dataset_mode}"
        output_model_key = f"{model_key}__hp_{hp_suffix}__emb_{config.embedding_source}{dataset_suffix}"

        output_dir = setup_output_dirs(
            config.output_root,
            args.task,
            args.inference_model,
            output_model_key,
        )

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

        result["task"] = args.task
        result["task_csv"] = task_csv.name
        result["model_key"] = model_key
        result["base_model_key"] = model_key
        result["output_model_key"] = output_model_key
        result["embedding_source"] = config.embedding_source
        result["dataset_mode"] = config.dataset_mode
        result["gene_universe"] = (
            "pinnacle_paper" if config.dataset_mode == "legacy" else "bulk_shared"
        )
        result["inference_name"] = args.inference_model
        result["embedding_inference_name"] = embedding_inference_model
        result["inference_dir"] = inference_path.name
        result["sequence_embedding_path"] = Path(config.embeddings.esm).name
        result["hc_protein_embedding_path"] = (
            Path(hc_paths["protein_embed"]).name if hc_paths["protein_embed"] is not None else ""
        )
        result["hc_cell_embedding_path"] = (
            Path(hc_paths["cell_embed"]).name if hc_paths["cell_embed"] is not None else ""
        )
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

        save_results_csv(result, output_dir)
        print(f"[OK] Finished {variant.name} -> {output_dir}")

    print(f"Results in: {config.output_root / args.task / args.inference_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
