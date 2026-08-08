"""Generate the ProstT5 protein features used by ProtScape."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


PATHS_FILE = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
MODEL_NAME = "Rostlab/ProstT5"
MODEL_REVISION = "d7d097d5bf9a993ab8f68488b4681d6ca70db9e5"
EMBEDDING_DIM = 1024
BATCH_SIZE = 8
MAX_TOKENS_PER_BATCH = 12000
MAX_AA_FOR_DIRECT = 1500
CHUNK_SIZE = 1000

AMBIGUOUS_AA = re.compile(r"[UZOB]")
NON_STANDARD_AA = re.compile(r"[^A-Z]")


def validate_embeddings(path):
    embeddings = pd.read_pickle(path)
    required = {"gene_name", "ProstT5-Embeddings"}
    missing = required - set(embeddings.columns)
    if missing:
        raise ValueError(f"ProstT5 embedding file is missing columns: {sorted(missing)}")

    for vector in embeddings["ProstT5-Embeddings"]:
        vector = np.asarray(vector)
        if vector.shape != (EMBEDDING_DIM,) or not np.isfinite(vector).all():
            raise ValueError(
                f"ProstT5 embeddings must be finite {EMBEDDING_DIM}-dimensional vectors"
            )


def load_ppi_genes(path):
    genes = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                genes.update(parts[:2])
    return genes


def sanitize_sequence(sequence):
    if not isinstance(sequence, str):
        return ""
    sequence = str(sequence).strip().upper()
    sequence = AMBIGUOUS_AA.sub("X", sequence)
    return NON_STANDARD_AA.sub("X", sequence)


def model_input(sequence):
    return "<AA2fold> " + " ".join(sequence)


def build_batches(records):
    batch = []
    longest_sequence = 0
    for row_index, sequence in records:
        next_longest = max(longest_sequence, len(sequence))
        if batch and (
            len(batch) == BATCH_SIZE
            or next_longest * (len(batch) + 1) > MAX_TOKENS_PER_BATCH
        ):
            yield batch
            batch = []
            longest_sequence = 0
        batch.append((row_index, sequence))
        longest_sequence = max(longest_sequence, len(sequence))
    if batch:
        yield batch


def embed_batch(sequences, tokenizer, model, device):
    tokens = tokenizer(
        [model_input(sequence) for sequence in sequences],
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = model(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
        ).last_hidden_state

    return [
        hidden[index, 1 : 1 + len(sequence)]
        .float()
        .mean(dim=0)
        .cpu()
        .numpy()
        .astype(np.float32)
        for index, sequence in enumerate(sequences)
    ]


def embed_long_sequence(sequence, tokenizer, model, device):
    chunks = [
        sequence[start : start + CHUNK_SIZE]
        for start in range(0, len(sequence), CHUNK_SIZE)
    ]
    vectors = [
        embed_batch([chunk], tokenizer, model, device)[0]
        for chunk in chunks
    ]
    lengths = np.asarray(
        [len(chunk) for chunk in chunks], dtype=np.float32
    ).reshape(-1, 1)
    return ((np.stack(vectors) * lengths).sum(axis=0) / lengths.sum()).astype(np.float32)


def main():
    with PATHS_FILE.open(encoding="utf-8") as handle:
        paths = yaml.safe_load(handle)

    sequences_path = Path(paths["protein_sequences"]).expanduser()
    global_ppi_path = Path(paths["global_ppi"]).expanduser()
    output_path = Path(paths["prostt5_embeddings"]).expanduser()

    if output_path.is_file():
        validate_embeddings(output_path)
        print(f"Using existing ProstT5 embeddings: {output_path}")
        return
    for path in (sequences_path, global_ppi_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    from transformers import T5EncoderModel, T5Tokenizer

    sequences = pd.read_csv(sequences_path)
    required = {"gene_name", "fasta_seq"}
    missing = required - set(sequences.columns)
    if missing:
        raise KeyError(f"Protein sequence table is missing columns: {sorted(missing)}")

    sequences["seq_length"] = sequences["fasta_seq"].astype(str).str.len()
    if not sequences["seq_length"].is_monotonic_increasing:
        sequences = sequences.sort_values("seq_length").reset_index(drop=True)
    ppi_genes = load_ppi_genes(global_ppi_path)
    sequences = sequences[
        sequences["gene_name"].astype(str).isin(ppi_genes)
    ].copy()
    sequences["fasta_seq"] = sequences["fasta_seq"].apply(sanitize_sequence)
    sequences = sequences[sequences["fasta_seq"].str.len() > 0].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} on {device}")
    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        do_lower_case=False,
    )
    model = T5EncoderModel.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    ).eval().to(device)
    model = model.half() if device.type == "cuda" else model.float()
    if int(model.config.d_model) != EMBEDDING_DIM:
        raise RuntimeError(
            f"{MODEL_NAME} returned dimension {model.config.d_model}; "
            f"expected {EMBEDDING_DIM}"
        )

    vectors = [None] * len(sequences)
    short_records = [
        (index, sequence)
        for index, sequence in enumerate(sequences["fasta_seq"])
        if len(sequence) <= MAX_AA_FOR_DIRECT
    ]
    short_records.sort(key=lambda item: len(item[1]))
    batches = list(build_batches(short_records))
    for batch in tqdm(batches, desc="ProstT5 batches"):
        batch_sequences = [sequence for _, sequence in batch]
        try:
            embeddings = embed_batch(batch_sequences, tokenizer, model, device)
        except RuntimeError as error:
            if device.type != "cuda" or "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            embeddings = [
                embed_batch([sequence], tokenizer, model, device)[0]
                for sequence in batch_sequences
            ]
        for (row_index, _), embedding in zip(batch, embeddings):
            vectors[row_index] = embedding

    long_records = [
        (index, sequence)
        for index, sequence in enumerate(sequences["fasta_seq"])
        if len(sequence) > MAX_AA_FOR_DIRECT
    ]
    for row_index, sequence in tqdm(long_records, desc="ProstT5 long sequences"):
        vectors[row_index] = embed_long_sequence(sequence, tokenizer, model, device)

    if any(vector is None for vector in vectors):
        raise RuntimeError("Some ProstT5 embeddings were not generated")
    embeddings = pd.DataFrame(
        {
            "gene_name": sequences["gene_name"].tolist(),
            "ProstT5-Embeddings": vectors,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    embeddings.to_pickle(temporary_path)
    temporary_path.replace(output_path)
    validate_embeddings(output_path)
    print(f"Saved {len(embeddings):,} protein embeddings to {output_path}")


if __name__ == "__main__":
    main()
