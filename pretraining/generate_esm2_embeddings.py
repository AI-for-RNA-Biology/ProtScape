"""Save BOS and mean-residue ESM-2 features for each protein.

ESM-2: Evolutionary-scale prediction of atomic-level protein structure with a
language model (Lin et al., 2023), https://doi.org/10.1126/science.ade2574.
"""

import argparse
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


def sequence_embeddings(sequence, model, batch_converter, device):
    """Read BOS and average the residues, excluding BOS and EOS tokens."""
    with torch.no_grad():
        _, _, tokens = batch_converter([("protein", sequence)])
        output = model(tokens.to(device), repr_layers=[MODEL_LAYER], return_contacts=False)
        representations = output["representations"][MODEL_LAYER][0].float()
        return {
            "bos": representations[0].cpu().numpy(),
            "mean": representations[1 : len(sequence) + 1].mean(dim=0).cpu().numpy(),
        }


def embed_sequence(sequence, model, batch_converter, device):
    """Embed the whole sequence; use 1,024-residue chunks if it exceeds GPU memory."""
    try:
        return sequence_embeddings(sequence, model, batch_converter, device)
    except RuntimeError as error:
        if device.type != "cuda" or "out of memory" not in str(error).lower():
            raise
        torch.cuda.empty_cache()

    chunks = [sequence[start : start + CHUNK_SIZE] for start in range(0, len(sequence), CHUNK_SIZE)]
    embeddings = [sequence_embeddings(chunk, model, batch_converter, device) for chunk in chunks]
    return {
        # Keep equal chunk weights for BOS; residue means need length weights.
        "bos": np.mean([embedding["bos"] for embedding in embeddings], axis=0),
        "mean": np.average(
            [embedding["mean"] for embedding in embeddings],
            axis=0,
            weights=[len(chunk) for chunk in chunks],
        ).astype(np.float32),
    }


def main():
    """Load the configured sequences and write the requested ESM-2 BOS/mean tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pooling", choices=["bos", "mean", "both"], default="bos")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    args = parser.parse_args()
    with PATHS_FILE.open(encoding="utf-8") as handle:
        paths = yaml.safe_load(handle)

    modes = ["bos", "mean"] if args.pooling == "both" else [args.pooling]
    output_paths = {
        mode: Path(paths["esm2_embeddings" if mode == "bos" else "esm2_mean_embeddings"]).expanduser()
        for mode in modes
    }
    for mode, path in output_paths.items():
        if path.is_file():
            print(f"Using ESM-2 {mode} embeddings: {path}")
    output_paths = {mode: path for mode, path in output_paths.items() if not path.is_file()}
    if not output_paths:
        return

    import esm

    sequences = pd.read_csv(Path(paths["protein_sequences"]).expanduser())
    sequences["fasta_seq"] = sequences["fasta_seq"].str.strip().str.upper()
    if sequences["fasta_seq"].isna().any() or sequences["fasta_seq"].eq("").any():
        raise ValueError("Protein sequence table contains missing or empty sequences")
    sequences["seq_length"] = sequences["fasta_seq"].str.len()
    if not sequences["seq_length"].is_monotonic_increasing:
        sequences = sequences.sort_values("seq_length").reset_index(drop=True)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    print(f"Loading {MODEL_NAME} on {device}")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    vectors = {mode: [] for mode in output_paths}
    for sequence in tqdm(sequences["fasta_seq"], desc="ESM-2 embeddings"):
        embeddings = embed_sequence(sequence, model, batch_converter, device)
        for mode in output_paths:
            vectors[mode].append(embeddings[mode])

    for mode, path in output_paths.items():
        embeddings = pd.DataFrame({"gene_name": sequences["gene_name"], "ESM2-Embeddings": vectors[mode]})
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        embeddings.to_pickle(temporary_path)
        temporary_path.replace(path)
        print(f"Saved {len(embeddings):,} ESM-2 {mode} embeddings to {path}")


if __name__ == "__main__":
    main()
