"""Cache frozen layer-33 residues once, without changing the released BOS features.

Run through scripts/cscs/run_residue_ablation.py. Four workers write disjoint
protein ranges in one mmap; durable per-protein markers make restarts cheap.
"""
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

MODEL_NAME = "esm2_t36_3B_UR50D"
LAYER = 33
DIM = 2560
CHUNK = 1024


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(root, release):
    root, release = Path(root), Path(release)
    root.mkdir(parents=True, exist_ok=True)
    source = release / "data/raw/protein_sequences.csv.gz"
    features = release / "data/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk"
    graph = release / "data/networks_bulk/global_ppi_edgelist.txt"
    # Match exactly the original feature-covered reference-interactome cohort.
    available = set(pd.read_pickle(features).gene_name)
    genes = [gene for gene in nx.read_edgelist(graph).nodes if gene in available]
    table = pd.read_csv(source).set_index("gene_name")
    if table.index.has_duplicates:
        raise ValueError("Protein sequence identifiers must be unique")
    sequences = table.loc[genes, "fasta_seq"].str.strip().str.upper().tolist()
    if any(not isinstance(seq, str) or not seq or not seq.isalpha() for seq in sequences):
        raise ValueError("Missing/invalid protein sequences")
    metadata = dict(model=MODEL_NAME, layer=LAYER, embedding_dim=DIM, genes=genes,
                    sequence_sha256=sha256(source), bos_sha256=sha256(features),
                    graph_sha256=sha256(graph), chunk_size=CHUNK,
                    long_sequence_policy="consecutive nonoverlapping 1024-residue windows; keep every residue",
                    special_tokens="exclude BOS, EOS and padding", storage_dtype="float16",
                    inference_dtype="float32", sequences=sequences)
    manifest = root / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError("Existing residue cache belongs to different inputs/settings")
        return
    offsets = np.r_[0, np.cumsum([len(seq) for seq in sequences])].astype(np.int64)
    np.save(root / "offsets.npy", offsets)
    values = np.lib.format.open_memmap(root / "residues.npy", mode="w+", dtype=np.float16,
                                       shape=(int(offsets[-1]), DIM))
    values.flush()
    (root / "done").mkdir(exist_ok=True)
    save_json(manifest, metadata)


def extract_residues(sequence, model, converter, device):
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(sequence), CHUNK):
            piece = sequence[start:start + CHUNK]
            _, _, tokens = converter([("protein", piece)])
            if tokens.shape[1] != len(piece) + 2:
                raise ValueError("Expected one BOS, one EOS and one token per residue")
            result = model(tokens.to(device), repr_layers=[LAYER], return_contacts=False)
            values = result["representations"][LAYER][0, 1:len(piece) + 1]
            pieces.append(values.float().cpu().numpy().astype(np.float16))
    return np.concatenate(pieces)


def worker(root, rank, workers=4):
    import esm
    root = Path(root)
    metadata = json.loads((root / "manifest.json").read_text())
    offsets = np.load(root / "offsets.npy")
    values = np.load(root / "residues.npy", mmap_mode="r+")
    # Greedy length balancing also distributes the very longest proteins.
    assignments, loads = [[] for _ in range(workers)], [0] * workers
    for index in np.argsort(-np.diff(offsets), kind="stable"):
        target = int(np.argmin(loads))
        assignments[target].append(int(index))
        loads[target] += int(offsets[index + 1] - offsets[index])
    pending = [i for i in assignments[rank] if not (root / "done" / f"{i}.json").exists()]
    if not pending:
        return
    # Weights are downloaded before requesting GPUs. Only deserialize the pinned
    # official FAIR checkpoint, not arbitrary/user-supplied pickle files.
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / f"{MODEL_NAME}.pt"
    model_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    regression = torch.load(checkpoint.with_name(f"{MODEL_NAME}-contact-regression.pt"),
                            map_location="cpu", weights_only=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(MODEL_NAME, model_data, regression)
    del model_data, regression
    model = model.eval().requires_grad_(False).cuda()
    converter = alphabet.get_batch_converter()
    for i in pending:
        residues = extract_residues(metadata["sequences"][i], model, converter, "cuda")
        if residues.shape != (int(offsets[i + 1] - offsets[i]), DIM) or not np.isfinite(residues).all():
            raise ValueError(f"Invalid residue features for {metadata['genes'][i]}")
        values[offsets[i]:offsets[i + 1]] = residues
        values.flush()
        save_json(root / "done" / f"{i}.json", {"gene": metadata["genes"][i], "residues": len(residues)})
        print(f"Cached {metadata['genes'][i]} ({len(residues)} residues)", flush=True)


def finalize(root):
    root = Path(root)
    metadata = json.loads((root / "manifest.json").read_text())
    if not all((root / "done" / f"{i}.json").is_file() for i in range(len(metadata["genes"]))):
        raise ValueError("Some proteins are not cached yet")
    offsets = np.load(root / "offsets.npy")
    residues = np.load(root / "residues.npy", mmap_mode="r")
    means = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        values = np.asarray(residues[start:end], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite residues in cache")
        means.append(values.mean(axis=0))
    features = torch.from_numpy(np.stack(means))
    mean, std = features.mean(0).numpy(), features.std(0).numpy()
    if not np.all(std > 0):
        raise ValueError("Zero variance feature in residue-mean cache")
    np.savez(root / "normalization.npz", mean=mean, std=std)
    pd.DataFrame({"gene_name": metadata["genes"], "ESM2-Embeddings": means}).to_pickle(root / "mean.plk")
    save_json(root / "COMPLETE.json", dict(proteins=len(means), residues=int(offsets[-1]),
                                          manifest_sha256=sha256(root / "manifest.json")))
