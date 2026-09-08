"""Checkpoint loading helpers."""

import importlib
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from .models.hierarchical_model import hierarchical_model
from .models.pinnacle_model import Pinnacle


# Module aliases required to load trusted full-object checkpoints.
_LEGACY_MODULES = {
    "models": "pretraining.models",
    "models.attention": "pretraining.models.attention",
    "models.cell_modules": "pretraining.models.cell_modules",
    "models.custom_gnn": "pretraining.models.custom_gnn",
    "models.hierarchical_model": "pretraining.models.hierarchical_model",
    "models.pinnacle_conv": "pretraining.models.pinnacle_conv",
    "models.pinnacle_model": "pretraining.models.pinnacle_model",
    "models.protein_modules": "pretraining.models.protein_modules",
    "s2gae_utils": "pretraining.s2gae_utils",
}


def load_legacy_checkpoint(path, map_location="cpu", mmap=False):
    """Load a trusted full-object checkpoint."""
    previous_modules = {name: sys.modules.get(name) for name in _LEGACY_MODULES}
    try:
        for old_name, new_name in _LEGACY_MODULES.items():
            sys.modules[old_name] = importlib.import_module(new_name)
        return torch.load(
            Path(path),
            map_location=map_location,
            mmap=mmap,
            weights_only=False,
        )
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def model_state_dict(checkpoint):
    """Return a state dictionary from either legacy or portable checkpoints."""
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    model = checkpoint["model"]
    return model if isinstance(model, dict) else model.state_dict()


def save_portable_checkpoint(
    path,
    legacy_path,
    *,
    model_type,
    config,
    cell_ids,
    cell_names,
):
    """Save the selected legacy model as a portable state-dict checkpoint."""
    if model_type not in {"protscape", "pinnacle"}:
        raise ValueError(f"Unsupported model type: {model_type}")

    cell_ids = [int(cell_id) for cell_id in cell_ids]
    cell_names = [str(cell_name) for cell_name in cell_names]
    if len(cell_ids) != len(cell_names) or len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Cell IDs and names must define one unique name per context.")

    checkpoint = load_legacy_checkpoint(legacy_path, map_location="cpu")
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    portable = {
        "format_version": 1,
        "model_type": model_type,
        "epoch": int(checkpoint["epoch"]),
        "config": config,
        "model_state_dict": model_state_dict(checkpoint),
        "cell_ids": cell_ids,
        "cell_names": cell_names,
    }
    for key in ("score", "score_metric"):
        if key in checkpoint:
            portable[key] = checkpoint[key]

    if model_type == "protscape":
        model = checkpoint["model"]
        cell_memory = getattr(model.cell_encoder, "cell_memory_layers", None)
        if cell_memory is not None:
            portable["cell_memory_step"] = int(cell_memory.step)

    path = Path(path)
    torch.save(portable, path)
    print(f"Saved portable checkpoint: {path}")


def load_protscape_model(checkpoint, ppi_data, device="cpu"):
    """Rebuild ProtScape and load a portable state-dictionary checkpoint."""
    config = checkpoint["config"]
    model = hierarchical_model(
        config["hierarchical_mode"],
        config["protein_config"].copy(),
        config["cell_config"].copy(),
        DictConfig(config["tissue_config"]),
        ppi_data,
        device,
        s2gae_config=config.get("s2gae_config"),
        use_metagraph=config.get("use_metagraph", False),
        graph_saint_norm=config.get("graph_saint_norm", False),
        uniformity_dim=config.get("uniformity_config", {}).get("dim", 0),
    )
    model.load_state_dict(model_state_dict(checkpoint))

    cell_memory = getattr(model.cell_encoder, "cell_memory_layers", None)
    if cell_memory is not None and "cell_memory_step" in checkpoint:
        cell_memory.step = int(checkpoint["cell_memory_step"])

    return model.to(device).eval()


def load_pinnacle_model(checkpoint, ppi_data, device="cpu"):
    """Rebuild the adapted PINNACLE comparator from a portable checkpoint."""
    config = checkpoint["config"]
    if set(ppi_data) != set(range(len(ppi_data))):
        raise ValueError("PINNACLE expects contiguous cell IDs starting at zero.")
    model = Pinnacle(
        config["gnn_method"],
        config["input_dim"],
        config["hidden"],
        config["output"],
        config["num_ppi_relations"],
        config["num_mg_relations"],
        ppi_data,
        config["n_heads"],
        config["pc_att_channels"],
        config["dropout"],
        shared_ppi_gnn=config.get("shared_ppi_gnn", False),
        device=device,
    )
    model.load_state_dict(model_state_dict(checkpoint))
    return model.to(device).eval()
