"""Four fixed contextual ProtScape objective controls; no hyperparameter search."""
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

ARMS = {"neither": (0, 0), "cci_only": (1, 0), "tissue_only": (0, 1), "both": (1, 1)}
TASKS = [
    ("corum", "corum_dataset/corum_memberships_filtered.csv", 0.4, None),
    ("therapeutic_target_efo_0003767", "therapeutic_target_dataset/therapeutic_target_EFO_0003767.csv", 0, 0.4),
    ("therapeutic_target_efo_0000685", "therapeutic_target_dataset/therapeutic_target_EFO_0000685.csv", 0, 0.5),
    ("therapeutic_target_efo_0000305", "therapeutic_target_dataset/therapeutic_target_EFO_0000305.csv", 0, 0.4),
]
GNN_PYTHON = "/iopsstor/scratch/cscs/aloistho/protscape/conda-global-s2gae/bin/python"
DS_PYTHON = "/iopsstor/scratch/cscs/aloistho/protscape/conda-protscape-downstream/bin/python"
RELEASE = Path("/capstor/store/cscs/swissai/a0204/aloistho/ProtScape_release_22645081/Protscape_release_final")
ESM = Path("/capstor/store/cscs/swissai/a0204/aloistho/ProtScape_release/data/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk")
BASELINE = ESM.parents[2] / "models/pretraining/protscape_main_state_dict.pt"


def checkpoint_path(root, arm):
    return BASELINE if arm == "both" else root / "artifacts/pretraining" / arm / "best_model_state_dict.pt"


def evaluate_ppi(root, arm, saved):
    import hashlib
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score
    from pretraining.checkpoints import load_protscape_model
    from pretraining.compare_inference_topologies import (
        load_context_data, encode_protscape_local, prepare_decoder_input, score_decoder, split_masks,
    )
    from pretraining.contextwise_ppi import sample_structured_negatives
    graphs, features, _, names = load_context_data(saved, RELEASE / "data/networks_bulk", ESM)
    device = torch.device("cuda")
    model = load_protscape_model(saved, graphs, device=device).eval()
    digest, rows = hashlib.sha256(), []
    with torch.no_grad():
        for cell, graph in graphs.items():
            message_mask, query_mask = split_masks(graph, "test")
            positive = graph.edge_index[:, query_mask]
            np.random.seed(cell)  # Identical query bank across all four checkpoints.
            negative = sample_structured_negatives(positive, graph.edge_index, graph.num_nodes, 1)
            digest.update(str(cell).encode())
            digest.update(positive.numpy().tobytes())
            digest.update(negative.numpy().tobytes())
            layers, raw = encode_protscape_local(model, features, graph, cell, message_mask, device)
            decoder_input = prepare_decoder_input(model.s2gae_decoder, layers, raw)
            scores = score_decoder(model.s2gae_decoder, decoder_input, torch.cat([positive, negative], dim=1), device, 100000)
            labels = np.r_[np.ones(positive.shape[1]), np.zeros(negative.shape[1])]
            rows.append({"cell": names[cell], "auprc": float(average_precision_score(labels, scores))})
    with (root / f"ppi_{arm}.json").open("w") as handle:
        json.dump({"query_sha256": digest.hexdigest(), "auprc": float(np.mean([r["auprc"] for r in rows])), "contexts": rows}, handle, indent=2)


def run(command, log):
    print("Running:", " ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    if str(command[0]) == DS_PYTHON:
        environment.pop("PYTHONPATH", None)  # GraphSAINT extensions target the pretraining Python ABI.
    with log.open("a") as handle:
        subprocess.run(list(map(str, command)), stdout=handle, stderr=subprocess.STDOUT, check=True, env=environment)


def train(root, arm):
    if arm == "both":
        raise ValueError("Reuse the released full model; do not retrain it")
    cci, tissue = ARMS[arm]
    command = [
        GNN_PYTHON, "-m", "pretraining.train_protscape",
        "ACM_RandomWalk", "concat", "512", "0.4", "3", str(tissue),
        "attention", "ESM2", "0", "1", "512", "2", "0",
        "--dataset-mode", "bulk", "--epochs", "300", "--checkpointing", "true",
        "--k-negatives", "1", "--use-metagraph", "true", "--loader", "graphsaint",
        "--split-mode", "global", "--lr", "0.01", "--batch-size", "64",
        "--weighted-ppi-loss", "false", "--ppi-loss", "bce",
        "--s2gae-mask-ratio", "0.5", "--s2gae-mask-type", "dm",
        "--s2gae-decoder-type", "cross_layer", "--s2gae-decoder-dropout", "0",
        "--s2gae-loss-weight", "1", "--metagraph-loss-weight", str(cci),
        "--uniformity-enabled", "true", "--uniformity-lambda-reg", "0.00005",
        "--uniformity-t", "2", "--uniformity-dim", "0", "--seed", "0",
        "--checkpoint-selection", "ppi", "--wandb-mode", "disabled",
        "--eval-save-prefix", root / "artifacts" / "pretraining" / arm,
    ]
    run(command, root / "logs" / f"{arm}_train.log")


def evaluate(root, arm):
    import torch
    checkpoint = checkpoint_path(root, arm)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_selection = "ap_ppi_plus_ap_meta" if arm == "both" else "ap_ppi"
    assert saved["score_metric"] == expected_selection, "Unexpected checkpoint selection rule"
    cci, tissue = ARMS[arm]
    assert saved["config"]["metagraph_loss_weight"] == cci
    assert saved["config"]["reg_CTassignment"] == tissue
    if not (root / f"ppi_{arm}.json").exists():
        evaluate_ppi(root, arm, saved)
    inference_root = root / "artifacts" / "inference"
    if arm != "both" and not (inference_root / arm / "export_complete.json").exists():
        run([GNN_PYTHON, "-m", "pretraining.inference", checkpoint,
             "--output-dir", inference_root / arm, "--overwrite"], root / "logs" / f"{arm}_inference.log")
        (inference_root / arm / "export_complete.json").write_text(json.dumps({"checkpoint": str(checkpoint)}))
    for task, labels, dropout, pdl in TASKS:
        completed = list((root / "artifacts/downstream_tasks" / task / arm).glob("*/results.csv"))
        if completed:
            if len(completed) != 1:
                raise ValueError(f"Ambiguous results: {completed}")
            continue
        model = "abmil_hc_cell_gated_8" if pdl is None else "abmil_hc_cell_ext_embed_gated_8_pdl"
        command = [
            DS_PYTHON, "-m", "downstream_tasks.run", "--inference-model", arm,
            "--inference-root", inference_root, "--esm2-embeddings", ESM,
            "--output-root", root / "artifacts" / "downstream_tasks",
            "--task", task, "--task-csv", RELEASE / "data/downstream_tasks" / labels,
            "--model", model, "--embedding-source", "esm", "--dataset-mode", "bulk",
            "--dropout", str(dropout), "--lr", "0.0001", "--weight-decay", "0.0001",
            "--batch-size", "512", "--seed", "42", "--train-selection-metric", "auprc",
        ]
        if pdl is not None:
            command += ["--pdl-pmax", str(pdl)]
        run(command, root / "logs" / f"{arm}_{task}.log")


def recover(root, arm):
    # Portable checkpoints are exported only after the 300-epoch loop finishes.
    if arm != "both" and not checkpoint_path(root, arm).exists():
        train(root, arm)
    evaluate(root, arm)


def preflight(root):
    import torch
    import torch_sparse
    import torch_scatter
    from torch_geometric.data import Data
    from torch_geometric.loader import GraphSAINTEdgeSampler
    from pretraining.compare_inference_topologies import split_masks, encode_protscape_local, score_decoder
    from pretraining.contextwise_ppi import sample_structured_negatives
    graph = Data(x=torch.ones(4, 2), edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]))
    sample = next(iter(GraphSAINTEdgeSampler(graph, batch_size=4, num_steps=1)))
    assert sample.num_nodes > 0 and sample.num_edges > 0
    run([GNN_PYTHON, "-m", "pretraining.train_protscape", "--help"], root / "logs/preflight.log")
    (root / "SETUP_OK").write_text("GraphSAINT CPU sampling and contextual training imports passed.\\n")


def summarize(root):
    import numpy as np
    import torch
    rows, reference_splits, query_hash = [], {}, None
    for arm in ARMS:
        checkpoint = torch.load(checkpoint_path(root, arm), map_location="cpu", weights_only=False)
        ppi = json.loads((root / f"ppi_{arm}.json").read_text())
        query_hash = query_hash or ppi["query_sha256"]
        assert query_hash == ppi["query_sha256"], "PPI queries differ across arms"
        rows.append(dict(variant=arm, benchmark="PPI 1:1", auprc=100 * ppi["auprc"],
                         selected_epoch=checkpoint["epoch"] + 1))
        tt = []
        for task, *_ in TASKS:
            results = list((root / "artifacts/downstream_tasks" / task / arm).glob("*/results.csv"))
            if len(results) != 1:
                raise ValueError(f"Expected one result for {arm}/{task}: {results}")
            folder = results[0].parent
            with np.load(folder / "split_indices.npz", allow_pickle=False) as data:
                split = {k: data[k].copy() for k in data.files if k == "genes" or k.startswith("fold_")}
            if task not in reference_splits:
                reference_splits[task] = split
            assert split.keys() == reference_splits[task].keys()
            for key, value in split.items():
                np.testing.assert_array_equal(value, reference_splits[task][key], err_msg=f"Unpaired {arm}/{task}/{key}")
            with results[0].open() as handle:
                value = 100 * float(next(csv.DictReader(handle))["test_auprc_macro_mean"])
            rows.append(dict(variant=arm, benchmark=task, auprc=value, selected_epoch=checkpoint["epoch"] + 1))
            if task != "corum":
                tt.append(value)
        rows.append(dict(variant=arm, benchmark="TT mean (3 diseases)", auprc=float(np.mean(tt)),
                         selected_epoch=checkpoint["epoch"] + 1))
    for row in rows:
        row["checkpoint_selection"] = "PPI+CCI (released)" if row["variant"] == "both" else "PPI only"
    with (root / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "COMPLETE.json").write_text(json.dumps({"arms": list(ARMS), "tasks": [t[0] for t in TASKS]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "evaluate", "recover", "summarize", "preflight"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--arm", choices=list(ARMS))
    args = parser.parse_args()
    if args.stage in {"summarize", "preflight"}:
        globals()[args.stage](args.root)
    else:
        if args.arm is None:
            parser.error("--arm is required for workers")
        globals()[args.stage](args.root, args.arm)
