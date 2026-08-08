"""Generate the ESM-2 protein features used by ProtScape."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


PATHS_FILE = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
MODEL_NAME = "esm2_t36_3B_UR50D"
MODEL_LAYER = 33
EMBEDDING_DIM = 2560
CHUNK_SIZE = 1024


def validate_embeddings(path):
    embeddings = pd.read_pickle(path)
    required = {"gene_name", "ESM2-Embeddings"}
    missing = required - set(embeddings.columns)
    if missing:
        raise ValueError(f"ESM-2 embedding file is missing columns: {sorted(missing)}")

    for vector in embeddings["ESM2-Embeddings"]:
        vector = np.asarray(vector)
        if vector.shape != (EMBEDDING_DIM,) or not np.isfinite(vector).all():
            raise ValueError(
                f"ESM-2 embeddings must be finite {EMBEDDING_DIM}-dimensional vectors"
            )


def bos_embedding(sequence, model, batch_converter, device):
    with torch.no_grad():
        _, _, tokens = batch_converter([("protein", sequence)])
        output = model(
            tokens.to(device),
            repr_layers=[MODEL_LAYER],
            return_contacts=False,
        )
    return output["representations"][MODEL_LAYER][0, 0].float().cpu().numpy()


def embed_sequence(sequence, model, batch_converter, device):
    try:
        embedding = bos_embedding(sequence, model, batch_converter, device)
    except RuntimeError as error:
        if device.type != "cuda" or "out of memory" not in str(error).lower():
            raise
        torch.cuda.empty_cache()
        chunks = [
            sequence[start : start + CHUNK_SIZE]
            for start in range(0, len(sequence), CHUNK_SIZE)
        ]
        vectors = [
            bos_embedding(chunk, model, batch_converter, device)
            for chunk in chunks
        ]
        embedding = np.mean(vectors, axis=0)

    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.shape != (EMBEDDING_DIM,):
        raise RuntimeError(
            f"{MODEL_NAME} returned shape {embedding.shape}; "
            f"expected ({EMBEDDING_DIM},)"
        )
    return embedding


def main():
    with PATHS_FILE.open(encoding="utf-8") as handle:
        paths = yaml.safe_load(handle)

    sequences_path = Path(paths["protein_sequences"]).expanduser()
    output_path = Path(paths["esm2_embeddings"]).expanduser()

    if output_path.is_file():
        validate_embeddings(output_path)
        print(f"Using existing ESM-2 embeddings: {output_path}")
        return
    if not sequences_path.is_file():
        raise FileNotFoundError(f"Protein sequence table not found: {sequences_path}")

    import esm

    sequences = pd.read_csv(sequences_path)
    required = {"gene_name", "fasta_seq"}
    missing = required - set(sequences.columns)
    if missing:
        raise KeyError(f"Protein sequence table is missing columns: {sorted(missing)}")
    if sequences["fasta_seq"].isna().any():
        raise ValueError("Protein sequence table contains missing sequences")

    sequences["fasta_seq"] = sequences["fasta_seq"].astype(str).str.strip().str.upper()
    if sequences["fasta_seq"].eq("").any():
        raise ValueError("Protein sequence table contains empty sequences")
    sequences["seq_length"] = sequences["fasta_seq"].str.len()
    if not sequences["seq_length"].is_monotonic_increasing:
        sequences = sequences.sort_values("seq_length").reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} on {device}")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    sequences["ESM2-Embeddings"] = [
        embed_sequence(sequence, model, batch_converter, device)
        for sequence in tqdm(sequences["fasta_seq"], desc="ESM-2 embeddings")
    ]

    embeddings = sequences[["gene_name", "ESM2-Embeddings"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    embeddings.to_pickle(temporary_path)
    temporary_path.replace(output_path)
    validate_embeddings(output_path)
    print(f"Saved {len(sequences):,} protein embeddings to {output_path}")


if __name__ == "__main__":
    main()
