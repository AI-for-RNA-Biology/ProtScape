"""Score separate ALS motor-neuron contexts and write the rewiring input matrix."""

from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from downstream_tasks.config import PATHS
from exploration.analysis.loss_consensus_string import open_text
from pretraining.s2gae_utils import (
    S2GAEDecoder, S2GAEDecoderSimple, undirected_decoder_logits,
)


PAIR_CHUNK = 200_000
SCORE_CHUNK = 8_192
CONTEXT_PATTERN = re.compile(r"CL_0000100_(CTRL|R155C|R191Q)_(cyto|nuc)_d(\d+)")


def pair_ids(a, b, n):
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return lo * (2 * n - lo - 1) // 2 + hi - lo - 1


def pair_chunks(n, chunk_size=PAIR_CHUNK):
    starts = np.arange(n, dtype=np.int64)
    starts = starts * (2 * n - starts - 1) // 2
    for start in range(0, n * (n - 1) // 2, chunk_size):
        rows = np.arange(start, min(start + chunk_size, n * (n - 1) // 2))
        a = np.searchsorted(starts, rows, side="right") - 1
        yield rows, a, a + 1 + rows - starts[a]


def read_contexts(ppi_dir):
    contexts = {}
    for path in sorted(Path(ppi_dir).glob("CL_0000100_*.txt")):
        match = CONTEXT_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        genotype, compartment, day = match.groups()
        with path.open() as handle:
            edges = {tuple(sorted(line.split()[:2])) for line in handle if line.strip()}
        nodes = set().union(*map(set, edges))
        edges = {edge for edge in edges if edge[0] != edge[1]}
        contexts[(genotype, compartment, int(day))] = (path.stem, nodes, edges)
    if not contexts:
        raise ValueError(f"No separate ALS motor-neuron graphs in {ppi_dir}")
    return contexts


def output_contexts(contexts):
    days = sorted({key[2] for key in contexts})
    groups = {}
    for genotype in ("CTRL", "VCP"):
        for compartment in ("cyto", "nuc"):
            for day in days:
                sources = [
                    (g, compartment, day)
                    for g in (("CTRL",) if genotype == "CTRL" else ("R155C", "R191Q"))
                ]
                for key in sources:
                    if key not in contexts:
                        raise ValueError(f"Missing separate ALS context: {key}")
                groups[f"{genotype}_{compartment}_d{day}"] = sources
    return groups


def load_decoder(checkpoint, device):
    protein = checkpoint["config"]["protein_config"]
    s2gae = checkpoint["config"].get("s2gae_config", {})
    if not s2gae.get("enabled"):
        raise ValueError("ALS pair scoring requires an S2GAE checkpoint.")
    if protein["jumping_knowledge"] not in ("concat", "concat_full"):
        raise ValueError("Export concatenated GNN-layer embeddings for S2GAE scoring.")
    layer_dim = protein["hidden_dim"]
    if protein["gnn_method"] == "GATv2":
        layer_dim *= protein["gat_heads"]
    raw_dim = protein["input_dim"] if protein["jumping_knowledge"] == "concat_full" else 0
    kwargs = dict(hidden_channels=layer_dim, num_encoder_layers=protein["n_layers"],
                  dropout=s2gae["decoder_dropout"])
    if s2gae["decoder_type"] == "cross_layer":
        decoder = S2GAEDecoder(**kwargs, decode_channels=s2gae["decode_channels"],
                              num_decoder_layers=s2gae["decoder_layers"], raw_input_dim=raw_dim)
    else:
        decoder = S2GAEDecoderSimple(**kwargs)
    prefix = "s2gae_decoder."
    decoder.load_state_dict({k[len(prefix):]: v for k, v in checkpoint["model_state_dict"].items()
                             if k.startswith(prefix)})
    return decoder.to(device).eval(), layer_dim, raw_dim


def string_scores(genes, info_path, links_path):
    """Map STRING preferred names, retaining the maximum score per gene pair."""
    gene_index = {gene: i for i, gene in enumerate(genes)}
    protein_index = {}
    with open_text(Path(info_path)) as handle:
        columns = handle.readline().rstrip().split("\t")
        protein_col, gene_col = columns.index("#string_protein_id"), columns.index("preferred_name")
        for line in handle:
            fields = line.rstrip().split("\t")
            gene = fields[gene_col].strip().upper()
            if gene in gene_index:
                protein_index[fields[protein_col]] = gene_index[gene]
    scores = np.zeros(len(genes) * (len(genes) - 1) // 2, dtype=np.float32)
    with open_text(Path(links_path)) as handle:
        columns = handle.readline().split()
        a_col, b_col, score_col = map(columns.index, ("protein1", "protein2", "combined_score"))
        for line in handle:
            fields = line.split()
            a, b = protein_index.get(fields[a_col]), protein_index.get(fields[b_col])
            if a is None or b is None or a == b:
                continue
            index = pair_ids(a, b, len(genes))
            scores[index] = max(scores[index], float(fields[score_col]) / 1000)
    return scores


@torch.no_grad()
def score_context(decoder, embedding, names, nodes, gene_index, output, layer_dim, raw_dim, device):
    names = [str(name).strip().upper() for name in names]
    if len(names) != len(embedding) or len(set(names)) != len(names):
        raise ValueError("Embedding rows must have unique, aligned protein names.")
    selected = [i for i, name in enumerate(names) if name in nodes]
    if len(selected) < 2:
        return
    local_names = [names[i] for i in selected]
    values = embedding[selected].to(device)
    layers = list(values[:, raw_dim:].split(layer_dim, dim=1))
    if len(layers) != decoder.num_encoder_layers or any(x.shape[1] != layer_dim for x in layers):
        raise ValueError("Embedding dimensions do not match the S2GAE checkpoint.")
    decoder_input = {"raw": values[:, :raw_dim], "layers": layers} if raw_dim else layers
    global_ids = np.array([gene_index[gene] for gene in local_names])
    for _, a, b in pair_chunks(len(local_names), SCORE_CHUNK):
        edges = torch.as_tensor(np.stack([a, b]), device=device)
        scores = torch.sigmoid(undirected_decoder_logits(decoder, decoder_input, edges))
        output[pair_ids(global_ids[a], global_ids[b], len(gene_index))] = scores.cpu().numpy()


def average_mutants(first, second, output):
    """Require both mutant scores; an absent score remains absent."""
    for start in range(0, len(output), PAIR_CHUNK):
        end = start + PAIR_CHUNK
        output[start:end] = (first[start:end] + second[start:end]) / 2


def build_matrix(paths):
    output = Path(paths["als_rewiring_raw_matrix_csv"])
    contexts = read_contexts(Path(paths["networks_bulk"]) / "ppi_edgelists")
    groups = output_contexts(contexts)
    genes = sorted(set().union(*(item[1] for item in contexts.values())))
    gene_index = {gene: i for i, gene in enumerate(genes)}
    n_pairs = len(genes) * (len(genes) - 1) // 2
    checkpoint = torch.load(paths["als_rewiring_checkpoint"], map_location="cpu", mmap=True, weights_only=True)
    exported = torch.load(Path(paths["als_rewiring_inference_dir"]) / "protein_embeddings.pt",
                          map_location="cpu", mmap=True, weights_only=False)
    names_to_ids = {name.replace(":", "_"): cell_id
                    for name, cell_id in zip(exported["cell_names"], exported["cell_ids"])}
    if exported.get("checkpoint_epoch", checkpoint["epoch"]) != checkpoint["epoch"]:
        raise ValueError("Use embeddings and decoder from the same checkpoint.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, layer_dim, raw_dim = load_decoder(checkpoint, device)
    print(f"{len(contexts)} separate contexts; {len(genes):,} proteins; {n_pairs:,} pairs; {device}", flush=True)
    print("Mapping STRING physical and combined scores", flush=True)
    physical = string_scores(genes, paths["string_protein_info"], paths["string_links_physical"])
    combined = string_scores(genes, paths["string_protein_info"], paths["string_links_detailed"])
    presence = {}
    for label, sources in groups.items():
        flags = np.zeros(n_pairs, dtype=np.uint8)
        edges = set().union(*(contexts[key][2] for key in sources))
        indices = np.array([(gene_index[a], gene_index[b]) for a, b in edges], dtype=np.int64)
        if len(indices):
            flags[pair_ids(indices[:, 0], indices[:, 1], len(genes))] = 1
        presence[label] = flags
    observed = np.maximum.reduce(list(presence.values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="als_scores_", dir=output.parent) as temporary:
        score_maps = {}
        for label, sources in groups.items():
            scores = np.memmap(Path(temporary) / f"{label}.bin", mode="w+", dtype=np.float32, shape=(n_pairs,))
            scores[:] = np.nan
            for index, key in enumerate(sources):
                target = scores
                if index:
                    target = np.memmap(Path(temporary) / "second_mutant.bin", mode="w+", dtype=np.float32, shape=(n_pairs,))
                    target[:] = np.nan
                name, nodes, _ = contexts[key]
                print(f"Scoring {name}", flush=True)
                cell_id = names_to_ids[name]
                score_context(decoder, exported["embeddings"][cell_id], exported["protein_names"][cell_id],
                              nodes, gene_index, target, layer_dim, raw_dim, device)
                if index:
                    average_mutants(scores, target, scores)
                    del target
            scores.flush()
            score_maps[label] = scores
        gene_array = np.array(genes)
        temporary_csv = Path(temporary) / output.name
        for rows, a, b in pair_chunks(len(genes)):
            data = dict(protA=gene_array[a], protB=gene_array[b], present_in_ppi=observed[rows],
                        stringdb_physical_score=physical[rows], stringdb_combined_score=combined[rows])
            data.update({f"ppi_present_{label}": flags[rows] for label, flags in presence.items()})
            data.update({f"s2gae_score_{label}": scores[rows] for label, scores in score_maps.items()})
            pd.DataFrame(data).to_csv(temporary_csv, index=False, mode="a", header=rows[0] == 0,
                                     na_rep="nan", float_format="%.6g")
        temporary_csv.replace(output)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    build_matrix(PATHS)
