"""Export the selected global-S2GAE encoder representation for downstream probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from .evaluate_global_s2gae import build_model
from .global_s2gae import load_global_ppi_data, protocol_metadata


EXPORT_FORMAT_VERSION = 1
EXPORT_TOPOLOGY = "full_reference"
EXPORT_REPRESENTATION = "encoder_jk_concat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen global-PPI ACM embeddings for downstream tasks."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--networks-dir", type=Path, required=True)
    parser.add_argument("--esm2-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _sha256_names(names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_checkpoint_data(checkpoint: dict, data) -> None:
    if checkpoint.get("format_version") != 3:
        raise ValueError("Expected a format-version 3 global-S2GAE checkpoint.")
    if checkpoint.get("model_type") != "global_s2gae":
        raise ValueError("Expected a global-S2GAE checkpoint.")
    if int(checkpoint.get("epoch", -1)) != int(checkpoint.get("best_epoch", -2)):
        raise ValueError("Checkpoint is not the validation-selected model state.")

    checks = {
        "protein ordering": checkpoint.get("protein_names") == data.protein_names,
        "global PPI": checkpoint.get("graph_fingerprint") == data.graph_fingerprint,
        "ESM2 features": checkpoint.get("feature_fingerprint")
        == data.feature_fingerprint,
        "edge split": checkpoint.get("split_fingerprint") == data.split_fingerprint,
        "feature means": torch.equal(checkpoint["feature_mean"], data.feature_mean),
        "feature scales": torch.equal(checkpoint["feature_std"], data.feature_std),
        "training protocol": checkpoint.get("protocol") == protocol_metadata(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Checkpoint/data mismatch: " + ", ".join(failed))


def _validate_existing_export(
    output_dir: Path,
    manifest: dict,
    checkpoint_sha256: str,
    checkpoint: dict,
) -> None:
    model_config = checkpoint["model_config"]
    expected = {
        "format_version": EXPORT_FORMAT_VERSION,
        "model_type": "global_s2gae",
        "embedding_scope": "global",
        "embedding_topology": EXPORT_TOPOLOGY,
        "representation": EXPORT_REPRESENTATION,
        "n_proteins": len(checkpoint["protein_names"]),
        "embedding_dim": int(model_config["hidden_dim"])
        * int(model_config["num_layers"]),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch_index": int(checkpoint["epoch"]),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "best_global_val_ap": float(checkpoint["best_global_val_ap"]),
        "training_git_commit": checkpoint["git_commit"],
        "graph_fingerprint": checkpoint["graph_fingerprint"],
        "feature_fingerprint": checkpoint["feature_fingerprint"],
        "split_fingerprint": checkpoint["split_fingerprint"],
        "protein_names_sha256": _sha256_names(checkpoint["protein_names"]),
        "downstream_only": True,
        "ppi_test_compatible": False,
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatched:
        raise FileExistsError(
            "Existing export has incompatible metadata: " + ", ".join(mismatched)
        )

    payload_path = output_dir / "protein_embeddings.pt"
    if not payload_path.is_file():
        raise FileExistsError(f"Incomplete export directory exists: {output_dir}")
    payload = torch.load(payload_path, map_location="cpu", mmap=True, weights_only=True)
    embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
    protein_names = payload.get("protein_names") if isinstance(payload, dict) else None
    if not isinstance(embeddings, torch.Tensor) or not isinstance(protein_names, list):
        raise FileExistsError(f"Invalid embedding payload: {payload_path}")
    expected_shape = (
        int(manifest.get("n_proteins", -1)),
        int(manifest.get("embedding_dim", -1)),
    )
    if (
        tuple(embeddings.shape) != expected_shape
        or len(protein_names) != expected_shape[0]
    ):
        raise FileExistsError(
            f"Existing export shape or protein mapping is invalid: {payload_path}"
        )
    if _sha256_tensor(embeddings) != manifest.get("embedding_sha256"):
        raise FileExistsError(f"Existing export checksum is invalid: {payload_path}")
    if _sha256_names(protein_names) != manifest.get("protein_names_sha256"):
        raise FileExistsError(
            f"Existing export protein mapping checksum is invalid: {payload_path}"
        )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise FileExistsError(
                f"Embedding payload metadata differs for {key}: {payload_path}"
            )


def export_embeddings(
    checkpoint_path: Path,
    networks_dir: Path,
    esm2_embeddings: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint_sha256 = _sha256_file(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    split_seed = int(checkpoint["training_config"]["split_seed"])
    data = load_global_ppi_data(
        networks_dir,
        esm2_embeddings,
        seed=split_seed,
        verbose=False,
    )
    _validate_checkpoint_data(checkpoint, data)

    manifest_path = output_dir / "manifest.json"
    if output_dir.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"Incomplete export directory exists: {output_dir}")
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        _validate_existing_export(
            output_dir,
            manifest,
            checkpoint_sha256,
            checkpoint,
        )
        print(f"Embedding export already complete: {output_dir}", flush=True)
        return manifest

    protein_names = [str(name) for name in checkpoint["protein_names"]]
    uppercase_names = [name.upper() for name in protein_names]
    if len(protein_names) != len(set(protein_names)):
        raise ValueError("Checkpoint protein names are not unique.")
    if len(uppercase_names) != len(set(uppercase_names)):
        raise ValueError("Uppercasing checkpoint protein names creates collisions.")

    model = build_model(checkpoint, device)
    with torch.no_grad():
        embeddings, _ = model.encode(
            data.features.to(device),
            data.all_edge_index.to(device),
        )
    embeddings = embeddings.detach().cpu().contiguous()
    expected_shape = (len(protein_names), model.embedding_dim)
    if tuple(embeddings.shape) != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape {tuple(embeddings.shape)}; expected {expected_shape}."
        )
    if not torch.isfinite(embeddings).all():
        raise ValueError("Exported embeddings contain non-finite values.")

    embedding_sha256 = _sha256_tensor(embeddings)
    export_commit = os.environ.get("PROTSCAPE_GIT_COMMIT", "uncommitted")
    common = {
        "format_version": EXPORT_FORMAT_VERSION,
        "model_type": "global_s2gae",
        "embedding_scope": "global",
        "embedding_topology": EXPORT_TOPOLOGY,
        "representation": EXPORT_REPRESENTATION,
        "n_proteins": len(protein_names),
        "embedding_dim": int(embeddings.shape[1]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch_index": int(checkpoint["epoch"]),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "best_global_val_ap": float(checkpoint["best_global_val_ap"]),
        "training_git_commit": checkpoint["git_commit"],
        "export_git_commit": export_commit,
        "graph_fingerprint": data.graph_fingerprint,
        "feature_fingerprint": data.feature_fingerprint,
        "split_fingerprint": data.split_fingerprint,
        "embedding_sha256": embedding_sha256,
        "protein_names_sha256": _sha256_names(protein_names),
        "reference_unique_edges": int(data.all_edge_index.size(1) // 2),
        "source_context_count": int(data.source_context_count),
        "downstream_only": True,
        "ppi_test_compatible": False,
    }
    payload = {
        "embeddings": embeddings,
        "protein_names": protein_names,
        **common,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir.with_name(
        f".{output_dir.name}.incomplete-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    )
    work_dir.mkdir(parents=False, exist_ok=False)
    torch.save(payload, work_dir / "protein_embeddings.pt")
    with (work_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(common, handle, indent=2, sort_keys=True)
    os.replace(work_dir, output_dir)
    print(json.dumps(common, indent=2, sort_keys=True), flush=True)
    return common


def main() -> None:
    args = parse_args()
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
            f"The export task must see exactly one GPU, found {torch.cuda.device_count()}."
        )
    export_embeddings(
        args.checkpoint,
        args.networks_dir,
        args.esm2_embeddings,
        args.output_dir,
        device,
    )


if __name__ == "__main__":
    main()
