"""Measure PCA entropy across contextual protein representations."""

import csv
import math
from pathlib import Path

import torch

from downstream_tasks.config import PATHS


EMBEDDING_ROOT = Path(PATHS["inference_root"])
MODEL_FOLDERS = {
    "ProtScape-GAE": "gae_att_fixed_do06_ep300",
    "ProtScape-GAE + uniformity": "gae_att_fixed_do06_uni5e6",
    "ProtScape-S2GAE": "s2gae_att_k1_fixed_do04",
    "ProtScape-S2GAE + uniformity": "s2gae_att_k1_fixed_do04_uni5e6",
}
OUTPUT_PATH = Path(PATHS["output_root"]) / "analysis/embedding_entropy.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pca_entropy(path):
    """Compute normalized eigenvalue entropy without stacking all context bags."""
    export = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    embeddings = export["embeddings"]
    dimension = next(iter(embeddings.values())).shape[1]

    total = torch.zeros(dimension, device=DEVICE)
    gram = torch.zeros((dimension, dimension), device=DEVICE)
    n_instances = 0

    with torch.inference_mode():
        for values in embeddings.values():
            values = values.to(DEVICE)
            total += values.sum(dim=0)
            gram += values.T @ values
            n_instances += values.shape[0]

    covariance = (gram - torch.outer(total, total) / n_instances) / (n_instances - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    variance = eigenvalues / eigenvalues.sum()
    variance = variance[variance > 0]
    entropy = -(variance * variance.log()).sum() / math.log(dimension)
    return n_instances, dimension, entropy.item()


def main():
    """Compute normalized embedding entropies for the configured model representations."""
    rows = []
    for label, folder in MODEL_FOLDERS.items():
        path = EMBEDDING_ROOT / folder / "protein_embeddings.pt"
        n_instances, dimension, entropy = pca_entropy(path)
        rows.append(
            {
                "model": label,
                "protein_context_instances": n_instances,
                "embedding_dimension": dimension,
                "normalized_pca_entropy": entropy,
            }
        )
        print(f"{label}: {entropy:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
