"""Generate protein and cell embeddings from a portable checkpoint."""

import argparse
import pickle
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from .checkpoints import load_pinnacle_model, load_protscape_model
from .data_handler.generate_input import get_metapaths, read_data
from .models.hierarchical_model import add_virtual_node
from .train.factored_batching import build_cci_edge_index
from .utils import construct_metapath


class _LazyPPIFeatures(dict):
    """Load one context's input features at a time during PINNACLE's first layer."""

    def __init__(self, ppi_data, global_features, device):
        super().__init__((cell_id, None) for cell_id in ppi_data)
        self.ppi_data = ppi_data
        self.global_features = global_features
        self.device = device

    def items(self):
        for cell_id in self:
            value = self[cell_id]
            if value is None:
                feature_index = self.ppi_data[cell_id].feature_index
                value = self.global_features[feature_index].to(self.device)
            yield cell_id, value


def parse_args():
    parser = argparse.ArgumentParser(description="Run ProtScape or adapted PINNACLE inference.")
    parser.add_argument("checkpoint", type=Path, help="Portable state-dictionary checkpoint")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def encode_proteins(model, ppi_data, device):
    """Encode one context at a time and return protein plus pooled cell embeddings."""
    protein_embeddings = {}
    pooled_cells = []

    with torch.no_grad():
        for cell_id, graph in tqdm(ppi_data.items(), desc="Encoding contexts"):
            local = graph.clone().to(device)
            pooling = model.cell_config["pooling"]
            uses_virtual_node = (
                pooling in {"vn", "learnedvn"}
                or model.protein_config.get("add_virtual_node")
            )
            if pooling == "vn" or model.protein_config.get("add_virtual_node"):
                local = add_virtual_node(
                    local, model.virtual_node_features, clone=False
                )
            elif pooling == "learnedvn":
                vn_id = model.celltype_to_vn_id[cell_id]
                local = add_virtual_node(
                    local, model.virtual_node_features[vn_id], clone=False
                )
            embedding = model.prot_encoder(
                {cell_id: local}, batching=False, return_layer_outputs=False
            )[cell_id]
            pooling_embedding = (
                embedding[:-1]
                if uses_virtual_node and pooling not in {"vn", "learnedvn"}
                else embedding
            )
            pooled = model.cell_encoder.get_cell_embedding_per_celltype(
                pooling_embedding
            )
            if uses_virtual_node:
                embedding = embedding[:-1]
            protein_embeddings[cell_id] = embedding.detach().cpu()
            pooled_cells.append(pooled.detach().cpu())
            del local, embedding, pooling_embedding

    return protein_embeddings, torch.stack(pooled_cells)


def encode_pinnacle(model, ppi_data, mg_data, tissue_neighbors, device, seed):
    """Run the complete adapted PINNACLE protein and metagraph forward pass."""
    ppi_metapaths, mg_metapaths = get_metapaths()
    ppi_x = _LazyPPIFeatures(
        ppi_data,
        mg_data.global_protein_features,
        device,
    )
    ppi_edges = {
        cell_id: {"total_edge_index": graph.edge_index.to(device)}
        for cell_id, graph in ppi_data.items()
    }
    ppi_adjacencies = {
        cell_id: [
            adjacency.to(device)
            for adjacency in construct_metapath(
                ppi_metapaths,
                graph.edge_index,
                graph.edge_attr,
                int(graph.num_nodes),
            )
        ]
        for cell_id, graph in ppi_data.items()
    }
    mg_edge_index = mg_data.edge_index.to(device)
    mg_edges = {"total_edge_index": mg_edge_index}
    mg_adjacencies = [
        adjacency.to(device)
        for adjacency in construct_metapath(
            mg_metapaths,
            mg_data.edge_index,
            mg_data.edge_attr,
            int(mg_data.num_nodes),
        )
    ]
    tissue_neighbors = {
        key: torch.as_tensor(value, device=device) for key, value in tissue_neighbors.items()
    }

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        protein_embeddings, metagraph_embeddings = model(
            ppi_x,
            mg_data.x.to(device),
            ppi_adjacencies,
            mg_adjacencies,
            ppi_edges,
            mg_edges,
            tissue_neighbors,
        )

    protein_embeddings = {
        cell_id: embedding.detach().cpu()
        for cell_id, embedding in protein_embeddings.items()
    }
    return protein_embeddings, metagraph_embeddings.detach().cpu()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "configs" / "paths.yaml").open("r", encoding="utf-8") as handle:
        paths = yaml.safe_load(handle)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(paths["output_root"]).expanduser()
            / "inference"
            / args.checkpoint.stem
        )
    output_dir = output_dir.expanduser()
    output_names = {
        "protein_embeddings.pt",
        "cell_embeddings.pt",
        "mappings.pkl",
        "tissue_predictions.pt",
        "cell_embeddings_before_pool.pt",
        "metagraph_embeddings.pt",
    }
    existing_outputs = [
        output_dir / filename
        for filename in output_names
        if (output_dir / filename).exists()
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"Inference output already exists in {output_dir}. "
            "Use --overwrite to replace it."
        )

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    config = checkpoint["config"]
    model_type = checkpoint.get("model_type")
    if model_type not in {"protscape", "pinnacle"}:
        raise ValueError("Expected a portable ProtScape or PINNACLE checkpoint.")

    incompatible_outputs = (
        [output_dir / "metagraph_embeddings.pt"]
        if model_type == "protscape"
        else [
            output_dir / "tissue_predictions.pt",
            output_dir / "cell_embeddings_before_pool.pt",
        ]
    )

    networks = Path(paths["networks_bulk"]).expanduser()
    features_mode = config.get("features_mode")
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    if features_mode == "ESM2":
        feature_path = Path(paths["esm2_embeddings"]).expanduser()
        feature_dim = None
    elif features_mode == "random":
        feature_path = None
        feature_dim = int(config["input_dim"])
    else:
        raise ValueError(f"Unsupported checkpoint protein features: {features_mode}")

    ppi_data, mg_data, edge_types, celltype_map, tissue_neighbors, ppi_layers, _, _ = read_data(
        networks / "global_ppi_edgelist.txt",
        networks / "ppi_edgelists",
        networks / "mg_edgelist.txt",
        feat_mat_dim=feature_dim,
        get_CT_map=True,
        ppi_feat_dir=feature_path,
        symmetric_ppi=config.get("symmetric_ppi", True),
        dataset_mode=config["dataset_mode"],
        split_mode=config["split_mode"],
        count_edge_path=networks / "count_edge_dict.pkl",
        weighted_ppi_loss=config.get("weighted_ppi_loss", False),
        defer_ppi_features=model_type == "pinnacle",
        verbose=False,
    )

    cell_ids = [int(cell_id) for cell_id in checkpoint.get("cell_ids", [])]
    if not cell_ids:
        raise ValueError("The checkpoint is missing the cell order required for inference.")
    if len(cell_ids) != len(set(cell_ids)) or set(cell_ids) != set(ppi_data):
        raise ValueError("Checkpoint cell IDs do not match the loaded PPI contexts.")
    ppi_data = {cell_id: ppi_data[cell_id] for cell_id in cell_ids}

    id_to_name = {cell_id: name for name, cell_id in celltype_map.items()}
    cell_names = [id_to_name[cell_id] for cell_id in cell_ids]
    if checkpoint.get("cell_names") != cell_names:
        raise ValueError("Checkpoint cell names do not match the loaded PPI contexts.")

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device)
    if model_type == "protscape":
        model = load_protscape_model(checkpoint, ppi_data, device=device)
        protein_embeddings, pooled_cells = encode_proteins(model, ppi_data, device)
    else:
        model = load_pinnacle_model(checkpoint, ppi_data, device=device)
        protein_embeddings, metagraph_embeddings = encode_pinnacle(
            model,
            ppi_data,
            mg_data,
            tissue_neighbors,
            device,
            seed=seed,
        )
    protein_names = {
        cell_id: list(ppi_layers[id_to_name[cell_id]].nodes()) for cell_id in cell_ids
    }
    for cell_id in cell_ids:
        if len(protein_names[cell_id]) != protein_embeddings[cell_id].shape[0]:
            raise ValueError(f"Protein names and embeddings do not align for cell {cell_id}.")

    if model_type == "protscape":
        with torch.no_grad():
            cells = pooled_cells.to(device)
            cell_memory = model.cell_encoder.cell_memory_layers
            if cell_memory is not None:
                cells = cell_memory(cells)

            # This is the pooled cell representation before CCI refinement.
            # Historically it was exported under the misleading name
            # ``cell_embeddings_before_pool.pt`` even though pooling had already
            # happened.
            pre_cci_cells = cells.detach().cpu()

            if model.use_metagraph:
                cci_edges = build_cci_edge_index(
                    mg_data, edge_types, cell_ids=cell_ids, device=device
                )
                model.set_cci_graph(cci_edges, mg_data.num_nodes)
                cells = model.apply_cci(cells, cell_ids)

            tissue_predictions = model.tissue_encoder(cells).detach().cpu()
            cells = cells.detach().cpu()
    else:
        cells = metagraph_embeddings[cell_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in incompatible_outputs:
            if path.exists():
                path.unlink()

    common = {
        "cell_ids": cell_ids,
        "cell_names": cell_names,
        "dataset_mode": config["dataset_mode"],
        "model_type": model_type,
        "checkpoint_epoch": checkpoint["epoch"],
    }
    torch.save(
        {
            "embeddings": protein_embeddings,
            "protein_names": protein_names,
            **common,
        },
        output_dir / "protein_embeddings.pt",
    )
    torch.save(
        {
            "embeddings": cells,
            "embed_dim": int(cells.shape[1]),
            "n_cells": len(cell_ids),
            "representation": (
                "post_cci" if model_type == "protscape" and model.use_metagraph
                else "final_metagraph_cell_nodes" if model_type == "pinnacle"
                else "pooled"
            ),
            **common,
        },
        output_dir / "cell_embeddings.pt",
    )
    if model_type == "protscape":
        torch.save(
            {
                "embeddings": pre_cci_cells,
                "embed_dim": int(pre_cci_cells.shape[1]),
                "n_cells": len(cell_ids),
                "representation": "pooled_before_cci",
                **common,
            },
            output_dir / "cell_embeddings_before_pool.pt",
        )
        torch.save(
            {
                "predictions": tissue_predictions,
                "n_tissues": int(tissue_predictions.shape[1]),
                **common,
            },
            output_dir / "tissue_predictions.pt",
        )
    else:
        torch.save(
            {
                "embeddings": metagraph_embeddings,
                "dataset_mode": config["dataset_mode"],
                "model_type": model_type,
                "checkpoint_epoch": checkpoint["epoch"],
            },
            output_dir / "metagraph_embeddings.pt",
        )
    with (output_dir / "mappings.pkl").open("wb") as handle:
        pickle.dump({"celltype_map": celltype_map, "id_to_name": id_to_name, **common}, handle)

    print(f"Saved {model_type} inference outputs to {output_dir}")


if __name__ == "__main__":
    main()
