"""Prepare one shared cohort and partition for each downstream task."""

import argparse
from pathlib import Path

import numpy as np

from .config import get_hc_embedding_paths, load_config
from .data.loaders import EmbeddingLoader, load_pinnacle_paper_gene_universe
from .data.partitions import build_shared_split, load_task_partition, partition_path
from .data.task_loaders import get_task_loader


def prepare_task(task, label_csv, loader, seed=42, n_splits=6,
                 dataset_mode="bulk", gene_universe=None):
    path = partition_path(task, label_csv, dataset_mode)
    if path.exists():
        load_task_partition(task, label_csv, n_splits, dataset_mode)
        print(f"Using dataset partitions: {path}")
        return path
    genes, labels, classes = get_task_loader(task, label_csv).load()
    genes, labels, plan = build_shared_split(
        genes, labels, loader, seed, n_splits, gene_universe
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle, genes=np.asarray(genes, dtype=str), labels=labels,
            class_names=np.asarray(classes, dtype=str), seed=seed,
            **{f"fold_{i}": fold for i, fold in enumerate(plan.folds)},
        )
    print(f"Saved {len(genes)} proteins and {n_splits} folds: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-model", required=True)
    parser.add_argument("--task", default="all")
    parser.add_argument("--task-csv", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-mode", choices=["bulk", "legacy"], default="bulk")
    args = parser.parse_args()
    if args.task_csv is not None and args.task == "all":
        parser.error("--task-csv requires a single --task")
    config = load_config(args.inference_model, dataset_mode=args.dataset_mode)
    if args.inference_model == "pinnacle_paper":
        config.dataset_mode = "legacy"
        loader = EmbeddingLoader(
            config.embeddings.esm, config.embeddings.pinnacle_paper_protein,
            config.embeddings.pinnacle_paper_cell,
            config.embeddings.pinnacle_paper_labels,
            config.embeddings.pinnacle_paper_cell_labels,
        )
    else:
        paths = get_hc_embedding_paths(config.get_inference_path())
        loader = EmbeddingLoader(config.embeddings.esm, paths["protein_embed"], paths["cell_embed"])
    universe = (load_pinnacle_paper_gene_universe(config.embeddings.pinnacle_paper_labels)
                if config.dataset_mode == "legacy" else None)
    tasks = config.tasks if args.task == "all" else {args.task: config.tasks[args.task]}
    for task, specification in tasks.items():
        prepare_task(task, args.task_csv or specification.label_csv, loader, args.seed, config.n_folds,
                     config.dataset_mode, universe)


if __name__ == "__main__":
    main()
