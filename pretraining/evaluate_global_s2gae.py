"""Evaluate the selected context-free model on global and contextual PPI tests."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from .contextwise_ppi import (
    CONTEXTWISE_NEGATIVE_BANK_SIZE,
    aggregate_context_metrics,
    metrics_from_pos_neg,
    sample_structured_negatives,
)
from .data_handler.generate_input import read_data
from .global_s2gae import (
    GLOBAL_K_VALUES,
    GlobalS2GAE,
    _canonical_keys,
    evaluate_global_edges,
    load_global_ppi_data,
    protocol_metadata,
)
from .run_global_s2gae_sweep import load_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint", type=Path)
    checkpoint.add_argument("--runs-root", type=Path)
    parser.add_argument("--sweep-config", type=Path)
    parser.add_argument("--networks-dir", type=Path, required=True)
    parser.add_argument("--esm2-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--k-values", default="1,10,50,100,500")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--update-wandb", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _validate_checkpoint(path: Path, checkpoint: dict) -> None:
    if checkpoint.get("format_version") != 3:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if checkpoint.get("protocol") != protocol_metadata():
        raise ValueError(f"Protocol mismatch: {path}")
    if int(checkpoint["epoch"]) != int(checkpoint["best_epoch"]):
        raise ValueError(f"Best checkpoint epoch mismatch: {path}")


def select_checkpoint(args: argparse.Namespace) -> tuple[Path, dict, list[dict]]:
    if args.checkpoint is not None:
        path = args.checkpoint.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = load_checkpoint(path)
        _validate_checkpoint(path, checkpoint)
        return path, checkpoint, []

    if args.sweep_config is None:
        raise ValueError("--sweep-config is required with --runs-root.")
    sweep, expected_runs = load_sweep(args.sweep_config.expanduser())
    if sweep.get("selection_metric") != protocol_metadata()["selection_metric"]:
        raise ValueError("Sweep selection metric does not match the protocol.")

    runs_root = args.runs_root.expanduser().resolve()
    best: tuple[Path, dict] | None = None
    rows: list[dict] = []
    primary_fingerprints = set()
    for config in expected_runs:
        run_dir = runs_root / config["name"]
        path = run_dir / "best_model_state_dict.pt"
        completion_path = run_dir / "completed.json"
        if not path.is_file() or not completion_path.is_file():
            if config["experiment_role"] == "sweep":
                raise FileNotFoundError(f"Incomplete primary sweep run: {run_dir}")
            rows.append(
                {
                    "run_name": config["name"],
                    "experiment_role": config["experiment_role"],
                    "status": "incomplete",
                    "eligible_for_selection": False,
                    "hidden_dim": config["hidden_dim"],
                    "num_layers": config["num_layers"],
                    "dropout": config["dropout"],
                    "mask_type": config["mask_type"],
                    "seed": config["seed"],
                    "best_epoch": "",
                    "best_global_val_ap": "",
                    "parameter_count": "",
                    "completed_epochs": "",
                }
            )
            continue

        with completion_path.open(encoding="utf-8") as handle:
            completion = json.load(handle)
        checkpoint = load_checkpoint(path)
        _validate_checkpoint(path, checkpoint)
        if completion.get("protocol") != protocol_metadata():
            raise ValueError(f"Completion protocol mismatch: {completion_path}")
        if checkpoint["experiment_role"] != config["experiment_role"]:
            raise ValueError(f"Checkpoint experiment role mismatch: {path}")
        if completion["experiment_role"] != config["experiment_role"]:
            raise ValueError(f"Completion experiment role mismatch: {completion_path}")
        if int(completion["completed_epochs"]) != int(config["epochs"]):
            raise ValueError(f"Run did not complete all epochs: {completion_path}")
        if int(completion["last_epoch"]) != int(config["epochs"]) - 1:
            raise ValueError(f"Invalid completion epoch: {completion_path}")
        if completion["selection_metric"] != protocol_metadata()["selection_metric"]:
            raise ValueError(f"Invalid selection metric: {completion_path}")
        if float(completion["best_global_val_ap"]) != float(
            checkpoint["best_global_val_ap"]
        ):
            raise ValueError(f"Best validation metric mismatch: {run_dir}")
        if int(completion["best_epoch"]) != int(checkpoint["best_epoch"]):
            raise ValueError(f"Best epoch mismatch: {run_dir}")
        if config["experiment_role"] == "sweep" and (
            config["mask_type"] != "dm"
            or int(config["seed"]) != 0
            or int(config["split_seed"]) != 0
        ):
            raise ValueError(f"Primary sweep protocol changed: {config['name']}")

        for key in (
            "hidden_dim",
            "num_layers",
            "dropout",
            "decode_channels",
            "decoder_layers",
            "decoder_dropout",
        ):
            if checkpoint["model_config"][key] != config[key]:
                raise ValueError(f"Model parameter {key} mismatch: {path}")
        for key in (
            "epochs",
            "lr",
            "mask_ratio",
            "mask_type",
            "k_negatives",
            "seed",
            "split_seed",
        ):
            if checkpoint["training_config"][key] != config[key]:
                raise ValueError(f"Training parameter {key} mismatch: {path}")

        fingerprints = (
            checkpoint["git_commit"],
            checkpoint["graph_fingerprint"],
            checkpoint["feature_fingerprint"],
            checkpoint["split_fingerprint"],
        )
        fingerprint_names = (
            "git_commit",
            "graph_fingerprint",
            "feature_fingerprint",
            "split_fingerprint",
        )
        if any(
            completion[name] != value
            for name, value in zip(fingerprint_names, fingerprints)
        ):
            raise ValueError(f"Checkpoint/completion fingerprint mismatch: {run_dir}")
        if config["experiment_role"] == "sweep":
            primary_fingerprints.add(fingerprints)

        parameter_count = int(checkpoint["parameter_count"])
        if int(completion["parameter_count"]) != parameter_count:
            raise ValueError(f"Parameter count mismatch: {run_dir}")
        row = {
            "run_name": config["name"],
            "experiment_role": config["experiment_role"],
            "status": "complete",
            "eligible_for_selection": config["experiment_role"] == "sweep",
            "hidden_dim": config["hidden_dim"],
            "num_layers": config["num_layers"],
            "dropout": config["dropout"],
            "mask_type": config["mask_type"],
            "seed": config["seed"],
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_global_val_ap": float(checkpoint["best_global_val_ap"]),
            "parameter_count": parameter_count,
            "completed_epochs": int(completion["completed_epochs"]),
        }
        rows.append(row)
        if row["eligible_for_selection"] and (
            best is None
            or row["best_global_val_ap"] > float(best[1]["best_global_val_ap"])
        ):
            best = path, checkpoint

    if len(primary_fingerprints) != 1:
        raise ValueError(
            "Primary runs do not share code, graph, feature, and split fingerprints."
        )
    if best is None:
        raise ValueError("Sweep manifest has no selection-eligible run.")

    complete_primary = sorted(
        (
            row
            for row in rows
            if row["status"] == "complete" and row["eligible_for_selection"]
        ),
        key=lambda row: row["best_global_val_ap"],
        reverse=True,
    )
    ranks = {row["run_name"]: rank for rank, row in enumerate(complete_primary, 1)}
    rows.sort(
        key=lambda row: (
            row["status"] == "complete",
            float(row["best_global_val_ap"])
            if row["status"] == "complete"
            else float("-inf"),
        ),
        reverse=True,
    )
    for row in rows:
        row["validation_rank"] = ranks.get(row["run_name"], "")
        row["selected"] = row["run_name"] == best[0].parent.name
    return best[0], best[1], rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(metrics: dict[int, dict], scope: str) -> list[dict]:
    return [
        {
            "scope": scope,
            "k_negatives": k,
            "pos_neg_ratio": f"1:{k}",
            "chance_ap": 1.0 / (1.0 + k),
            **values,
        }
        for k, values in sorted(metrics.items())
    ]


def contextwise_protocol_metadata() -> dict:
    """Protocol shared with the contextual ProtScape/PINNACLE PPI curves."""
    return {
        "version": "context_ppi_macro_v1",
        "evaluation_unit": "directed_edge_context_occurrence",
        "symmetric_edge_representation": True,
        "negative_scope": "within_cell_ppi_nonedge",
        "negative_sampling": "structured_target_corruption",
        "negative_seed": "cell_id",
        "negative_bank_size": CONTEXTWISE_NEGATIVE_BANK_SIZE,
        "aggregation": "unweighted_macro_across_contexts",
        "message_topology": "global_train_plus_validation",
        "reported_k_values": list(GLOBAL_K_VALUES),
    }


def load_context_graphs(
    networks_dir: Path,
    esm2_embeddings: Path,
    *,
    seed: int,
) -> tuple[dict, list[str], dict[int, str]]:
    """Load the exact Cell-PPI targets used by the contextual evaluator."""
    networks_dir = Path(networks_dir).expanduser()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        loaded = read_data(
            networks_dir / "global_ppi_edgelist.txt",
            networks_dir / "ppi_edgelists",
            networks_dir / "mg_edgelist.txt",
            get_CT_map=True,
            ppi_feat_dir=Path(esm2_embeddings).expanduser(),
            symmetric_ppi=True,
            dataset_mode="bulk",
            split_mode="global",
            count_edge_path=networks_dir / "count_edge_dict.pkl",
            weighted_ppi_loss=False,
            defer_ppi_features=True,
            seed=seed,
            verbose=False,
        )
    contexts, metagraph, _, celltype_map, _, _, _, _ = loaded
    protein_names = [str(name) for name in metagraph.global_protein_names]
    id_to_name = {int(cell_id): str(name) for name, cell_id in celltype_map.items()}
    return contexts, protein_names, id_to_name


def _score_context_edges(
    model: GlobalS2GAE,
    layer_embeddings: list[torch.Tensor],
    edge_index: torch.Tensor,
    device: torch.device,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    # The global decoder is symmetric and context-independent. Reuse scores for
    # duplicate/reversed pairs without changing the occurrence-level metrics.
    canonical = torch.stack(
        [
            torch.minimum(edge_index[0], edge_index[1]),
            torch.maximum(edge_index[0], edge_index[1]),
        ],
        dim=1,
    )
    unique_edges, inverse = torch.unique(
        canonical,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    unique_edges = unique_edges.t().contiguous()
    scores = []
    for start in range(0, unique_edges.size(1), chunk_size):
        edges = unique_edges[:, start : start + chunk_size].to(device)
        logits = model.score_edges(layer_embeddings, edges)
        scores.append(torch.sigmoid(logits).cpu().numpy())
    if not scores:
        return np.empty(0, dtype=np.float32)
    unique_scores = np.concatenate(scores)
    return unique_scores[inverse.numpy()]


@torch.no_grad()
def evaluate_contextwise_edges(
    model: GlobalS2GAE,
    data,
    contexts: dict,
    id_to_name: dict[int, str],
    *,
    k_values: list[int],
    device: torch.device,
) -> tuple[dict[int, dict], list[dict], dict, list[dict]]:
    """Score frozen global representations on the paper's Cell-PPI test banks."""
    values = sorted({int(k) for k in k_values})
    if values != list(GLOBAL_K_VALUES):
        raise ValueError(f"k_values must be exactly {GLOBAL_K_VALUES}.")
    if len(contexts) != int(data.source_context_count):
        raise ValueError("Context count differs from the global split source.")

    model.eval()
    _, layers = model.encode(
        data.features.to(device),
        data.train_val_edge_index.to(device),
    )
    metrics_by_k = {k: [] for k in values}
    context_rows = []
    pair_contexts: dict[int, set[int]] = {}
    directed_positive_occurrences = 0
    pair_context_occurrences = 0
    global_test_keys = torch.unique(
        _canonical_keys(data.test_edge_index, data.features.size(0))
    )

    for context_index, (cell_id, graph) in enumerate(contexts.items(), start=1):
        cell_id = int(cell_id)
        local_to_global = graph.feature_index.detach().cpu().long()
        positive_local = graph.edge_index[:, graph.test_mask].detach().cpu().long()
        if positive_local.size(1) == 0:
            raise ValueError(f"Cell-PPI {cell_id} has no test-positive edges.")
        positive_global = local_to_global[positive_local]
        positive_keys = _canonical_keys(positive_global, data.features.size(0))
        if not torch.isin(positive_keys, global_test_keys).all():
            raise ValueError(
                f"Cell-PPI {cell_id} contains a test pair outside the global test split."
            )
        unique_positive_keys = torch.unique(positive_keys)
        directed_positive_occurrences += int(positive_keys.numel())
        pair_context_occurrences += int(unique_positive_keys.numel())
        for key in unique_positive_keys.tolist():
            pair_contexts.setdefault(int(key), set()).add(cell_id)

        np.random.seed(cell_id)
        negative_local = sample_structured_negatives(
            positive_local,
            graph.edge_index.detach().cpu(),
            int(graph.num_nodes),
            CONTEXTWISE_NEGATIVE_BANK_SIZE,
        ).reshape(2, positive_local.size(1), CONTEXTWISE_NEGATIVE_BANK_SIZE)
        negative_global = local_to_global[negative_local]

        positive_scores = _score_context_edges(
            model,
            layers,
            positive_global,
            device,
        )
        negative_scores = _score_context_edges(
            model,
            layers,
            negative_global.reshape(2, -1),
            device,
        ).reshape(positive_local.size(1), CONTEXTWISE_NEGATIVE_BANK_SIZE)

        for k in values:
            metrics = metrics_from_pos_neg(
                positive_scores,
                negative_scores[:, :k].reshape(-1),
            )
            metrics_by_k[k].append(metrics)
            context_rows.append(
                {
                    "scope": "cell_ppi",
                    "cell_id": cell_id,
                    "edgelist": id_to_name[cell_id],
                    "k_negatives": k,
                    "pos_neg_ratio": f"1:{k}",
                    "chance_ap": 1.0 / (1.0 + k),
                    **metrics,
                }
            )
        if context_index % 25 == 0 or context_index == len(contexts):
            print(
                f"context-free model: scored {context_index}/{len(contexts)} "
                "Cell-PPIs",
                flush=True,
            )

    observed_keys = torch.tensor(sorted(pair_contexts), dtype=torch.long)
    if not torch.equal(observed_keys, torch.sort(global_test_keys).values):
        raise ValueError(
            "Context-specific targets do not cover the complete global test split."
        )

    num_nodes = int(data.features.size(0))
    multiplicity_rows = []
    for key, cell_ids in sorted(pair_contexts.items()):
        source_index, target_index = divmod(key, num_nodes)
        multiplicity_rows.append(
            {
                "global_source_index": source_index,
                "global_target_index": target_index,
                "source_protein": data.protein_names[source_index],
                "target_protein": data.protein_names[target_index],
                "n_contexts": len(cell_ids),
            }
        )
    multiplicities = np.asarray(
        [row["n_contexts"] for row in multiplicity_rows], dtype=np.int64
    )
    repeated = int((multiplicities > 1).sum())
    diagnostics = {
        "directed_positive_occurrences_scored": directed_positive_occurrences,
        "unique_pair_context_occurrences": pair_context_occurrences,
        "unique_global_test_pairs_observed": len(multiplicity_rows),
        "unique_global_test_pairs_total": int(global_test_keys.numel()),
        "pairs_observed_in_multiple_contexts": repeated,
        "fraction_observed_pairs_in_multiple_contexts": (
            float(repeated / multiplicities.size) if multiplicities.size else 0.0
        ),
        "mean_contexts_per_observed_pair": (
            float(multiplicities.mean()) if multiplicities.size else 0.0
        ),
        "median_contexts_per_observed_pair": (
            float(np.median(multiplicities)) if multiplicities.size else 0.0
        ),
        "max_contexts_per_observed_pair": (
            int(multiplicities.max()) if multiplicities.size else 0
        ),
    }
    return (
        aggregate_context_metrics(metrics_by_k),
        context_rows,
        diagnostics,
        multiplicity_rows,
    )


def build_model(checkpoint: dict, device: torch.device) -> GlobalS2GAE:
    config = dict(checkpoint["model_config"])
    input_dim = int(config.pop("input_dim"))
    model = GlobalS2GAE(input_dim, device=device, **config).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        checkpoint["parameter_count"]
    ):
        raise ValueError("Checkpoint parameter count does not match the model.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def update_wandb_summary(checkpoint: dict, summary: dict) -> None:
    import wandb

    tracking = checkpoint["wandb"]
    run = wandb.Api().run(
        f"{tracking['entity']}/{tracking['project']}/{tracking['run_id']}"
    )
    run.summary["test/selected_checkpoint"] = True
    run.summary["test/best_global_val_ap"] = summary["best_global_val_ap"]
    for k, values in summary["global_test"].items():
        for metric, value in values.items():
            run.summary[f"test/global_{metric}_k{k}"] = value
    for k, values in summary["context_ppi_macro_test"].items():
        for metric, value in values.items():
            run.summary[f"test/context_macro_{metric}_k{k}"] = value
    run.update()


def _validate_complete_summary(summary: dict) -> None:
    if summary.get("protocol") != protocol_metadata():
        raise ValueError("Existing evaluation protocol is incomplete or incompatible.")
    if sorted(int(k) for k in summary.get("global_test", {})) != list(
        GLOBAL_K_VALUES
    ):
        raise ValueError("Existing evaluation does not contain every required K value.")
    if sorted(int(k) for k in summary.get("context_ppi_macro_test", {})) != list(
        GLOBAL_K_VALUES
    ):
        raise ValueError(
            "Existing evaluation does not contain every context-specific K value."
        )
    if summary.get("context_ppi_protocol") != contextwise_protocol_metadata():
        raise ValueError("Existing context-specific evaluation protocol is incompatible.")
    required_multiplicity = {
        "directed_positive_occurrences_scored",
        "unique_pair_context_occurrences",
        "unique_global_test_pairs_observed",
        "unique_global_test_pairs_total",
        "pairs_observed_in_multiple_contexts",
        "fraction_observed_pairs_in_multiple_contexts",
        "mean_contexts_per_observed_pair",
        "median_contexts_per_observed_pair",
        "max_contexts_per_observed_pair",
    }
    if set(summary.get("context_ppi_multiplicity", {})) != required_multiplicity:
        raise ValueError("Existing context-specific multiplicity audit is incomplete.")


def main() -> None:
    args = parse_args()
    k_values = sorted({int(value) for value in args.k_values.split(",")})
    if k_values != list(GLOBAL_K_VALUES):
        raise ValueError(f"k-values must be exactly {GLOBAL_K_VALUES}.")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if (
        device.type == "cuda"
        and os.environ.get("SLURM_JOB_ID")
        and torch.cuda.device_count() != 1
    ):
        raise RuntimeError(
            f"The evaluation task must see one GPU, found {torch.cuda.device_count()}."
        )

    checkpoint_path, checkpoint, selection_rows = select_checkpoint(args)
    submitted_commit = os.environ.get("PROTSCAPE_GIT_COMMIT")
    if submitted_commit and checkpoint["git_commit"] != submitted_commit:
        raise ValueError("Evaluation code and selected training commit differ.")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent / "evaluation"
    )
    if output_dir.exists():
        if not args.update_wandb:
            raise FileExistsError(f"Evaluation path already exists: {output_dir}")
        summary_path = output_dir / "summary.json"
        completion_path = output_dir / "completed.json"
        if not summary_path.is_file() or not completion_path.is_file():
            raise FileExistsError(f"Evaluation path is incomplete: {output_dir}")
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary["checkpoint"] != str(checkpoint_path):
            raise ValueError("Existing evaluation used another checkpoint.")
        _validate_complete_summary(summary)
        update_wandb_summary(checkpoint, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    split_seed = int(checkpoint["training_config"]["split_seed"])
    data = load_global_ppi_data(
        args.networks_dir,
        args.esm2_embeddings,
        seed=split_seed,
        verbose=False,
    )
    checks = {
        "protein ordering": checkpoint["protein_names"] == data.protein_names,
        "global PPI": checkpoint["graph_fingerprint"] == data.graph_fingerprint,
        "ESM2 features": checkpoint["feature_fingerprint"]
        == data.feature_fingerprint,
        "edge split": checkpoint["split_fingerprint"] == data.split_fingerprint,
        "feature means": torch.equal(checkpoint["feature_mean"], data.feature_mean),
        "feature scales": torch.equal(checkpoint["feature_std"], data.feature_std),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Checkpoint/data mismatch: " + ", ".join(failed))

    model = build_model(checkpoint, device)
    data.features = data.features.to(device)
    test_metrics = evaluate_global_edges(
        model,
        data,
        split="test",
        k_values=k_values,
        device=device,
        seed=split_seed,
    )
    contexts, context_protein_names, id_to_name = load_context_graphs(
        args.networks_dir,
        args.esm2_embeddings,
        seed=split_seed,
    )
    if context_protein_names != data.protein_names:
        raise ValueError("Context and global loaders produced different protein ordering.")
    (
        context_test_metrics,
        context_rows,
        context_diagnostics,
        multiplicity_rows,
    ) = evaluate_contextwise_edges(
        model,
        data,
        contexts,
        id_to_name,
        k_values=k_values,
        device=device,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir.with_name(
        f".{output_dir.name}.incomplete-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    )
    work_dir.mkdir(parents=False, exist_ok=False)
    if selection_rows:
        write_rows(work_dir / "sweep_selection.csv", selection_rows)
        with (work_dir / "sweep_selection.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "selection_metric": protocol_metadata()["selection_metric"],
                    "selected_run": checkpoint_path.parent.name,
                    "selected_checkpoint": str(checkpoint_path),
                    "runs": selection_rows,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
    write_rows(
        work_dir / "global_test_metrics.csv",
        metric_rows(test_metrics, "global_unique_pairs"),
    )
    write_rows(
        work_dir / "context_ppi_macro_test_metrics.csv",
        metric_rows(context_test_metrics, "cell_ppi_macro"),
    )
    write_rows(work_dir / "context_ppi_per_cell_test_metrics.csv", context_rows)
    write_rows(work_dir / "context_ppi_pair_multiplicity.csv", multiplicity_rows)

    summary = {
        "checkpoint": str(checkpoint_path),
        "selected_epoch": int(checkpoint["epoch"]),
        "best_global_val_ap": float(checkpoint["best_global_val_ap"]),
        "parameter_count": int(checkpoint["parameter_count"]),
        "split_counts": data.split_counts,
        "source_context_count": data.source_context_count,
        "unique_edge_counts": {
            "reference": data.all_edge_index.size(1) // 2,
            "train": data.train_edge_index.size(1) // 2,
            "validation": data.val_edge_index.size(1),
            "test": data.test_edge_index.size(1),
        },
        "graph_fingerprint": data.graph_fingerprint,
        "feature_fingerprint": data.feature_fingerprint,
        "split_fingerprint": data.split_fingerprint,
        "evaluation_topology": "train_plus_validation",
        "evaluation_unit": "unique_undirected_global_pair",
        "mask_type": checkpoint["training_config"]["mask_type"],
        "protocol": checkpoint["protocol"],
        "context_ppi_protocol": contextwise_protocol_metadata(),
        "git_commit": checkpoint["git_commit"],
        "global_test": test_metrics,
        "context_ppi_macro_test": context_test_metrics,
        "context_ppi_multiplicity": context_diagnostics,
    }
    with (work_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (work_dir / "completed.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "protocol": checkpoint["protocol"],
                "test_evaluated": True,
                "context_ppi_test_evaluated": True,
                "git_commit": checkpoint["git_commit"],
                "graph_fingerprint": data.graph_fingerprint,
                "feature_fingerprint": data.feature_fingerprint,
                "split_fingerprint": data.split_fingerprint,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    os.replace(work_dir, output_dir)
    if args.update_wandb:
        update_wandb_summary(checkpoint, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
