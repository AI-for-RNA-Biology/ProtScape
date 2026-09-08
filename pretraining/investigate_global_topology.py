"""Validation-only topology interventions and pair-grouped structural probes.

One frozen checkpoint, train-only message graphs, topology-selected contexts.
Hard banks are HeaRT-inspired (RA/PPR rank fusion), not an exact reproduction.
No residue features, new GNN training, or test-set model selection occurs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .compare_inference_topologies import (
    load_context_data, encode_protscape_local, prepare_decoder_input, sha256_file,
)
from .checkpoints import load_protscape_model
from .diagnose_global_s2gae import validate_checkpoint_data, state_hash
from .evaluate_global_s2gae import build_model
from .global_s2gae import load_global_ppi_data
from .s2gae_utils import undirected_decoder_logits

K_VALUES = (1, 10, 50, 100, 500)
BANKS = ("random", "local_hard", "global_hard")
GNN_NAMES = ("CF global", "CF Cell-PPI", "CF boundary rewired", "CF outside features shuffled", "ProtScape Cell-PPI")
FEATURE_NAMES = ("degree_min", "degree_max", "CN", "RA", "PPR")


def adjacency(edges, size):
    edges = np.asarray(edges)
    matrix = sp.csr_matrix((np.ones(edges.shape[1], np.float32), edges), shape=(size, size))
    matrix = matrix.maximum(matrix.T)
    matrix.data[:] = 1
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def pair_keys(edges, size):
    return np.minimum(edges[0], edges[1]).astype(np.int64) * size + np.maximum(edges[0], edges[1])


def pair_fold(keys):
    """Stable symmetric pair grouping, including across different Cell-PPIs."""
    value = np.asarray(keys, dtype=np.uint64).copy()
    value ^= value >> np.uint64(16)
    value *= np.uint64(0x45D9F3B)
    value ^= value >> np.uint64(16)
    return (value % np.uint64(3)).astype(np.int8)


def rewire_boundary(edges, inside, size, seed, forbidden=frozenset()):
    """Bipartite double-edge swaps preserve every node's total/boundary degree."""
    unique = np.asarray(edges)[:, np.asarray(edges)[0] < np.asarray(edges)[1]]
    boundary = inside[unique[0]] != inside[unique[1]]
    crossing = unique[:, boundary].copy()
    reverse = ~inside[crossing[0]]
    crossing[:, reverse] = crossing[::-1, reverse]
    original = set((crossing[0] * size + crossing[1]).tolist())
    present = original.copy()
    rng = np.random.default_rng(seed)
    accepted = 0
    target = 5 * crossing.shape[1]
    for _ in range(50 * max(1, crossing.shape[1])):
        if accepted >= target or crossing.shape[1] < 2:
            break
        a, b = rng.integers(crossing.shape[1], size=2)
        u, x = crossing[:, a]
        v, y = crossing[:, b]
        if u == v or x == y or u * size + y in present or v * size + x in present:
            continue
        if min(u, y) * size + max(u, y) in forbidden or min(v, x) * size + max(v, x) in forbidden:
            continue
        present.remove(int(u * size + x))
        present.remove(int(v * size + y))
        present.add(int(u * size + y))
        present.add(int(v * size + x))
        crossing[1, a], crossing[1, b] = y, x
        accepted += 1
    new = np.concatenate([unique[:, ~boundary], crossing], axis=1)
    # The released symmetric edge list can contain two copies of a self-PPI.
    # Preserve those original occurrences; self-loops are never boundary edges.
    loops = np.asarray(edges)[:, np.asarray(edges)[0] == np.asarray(edges)[1]]
    result = np.concatenate([new, new[::-1], loops], axis=1)
    if not np.array_equal(np.bincount(result[0], minlength=size), np.bincount(np.asarray(edges)[0], minlength=size)):
        raise AssertionError("Boundary rewiring changed degrees")
    return result, {"boundary_edges": len(original), "accepted_swaps": accepted,
                    "boundary_fraction_changed": len(original - present) / max(1, len(original))}


@torch.no_grad()
def personalized_pagerank(matrix, sources, device, alpha=0.15, tolerance=2e-6):
    """PPR(source -> target), dangling walks restart at their own source.

    Batched sparse power iteration; reject unconverged results. Rows sum to one.
    Only training adjacency enters this computation.
    """
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inv = np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)
    transposed = (sp.diags(inv) @ matrix).T.tocoo()
    operator = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([transposed.row, transposed.col])),
        torch.from_numpy(transposed.data.astype(np.float32)), matrix.shape, device=device,
    ).coalesce()
    dangling = torch.from_numpy(degree == 0).to(device)
    output = np.empty((len(sources), matrix.shape[0]), np.float32)
    max_residual, max_steps = 0., 0
    for start in range(0, len(sources), 64):
        current_sources = sources[start:start + 64]
        restart = torch.zeros((matrix.shape[0], len(current_sources)), device=device)
        restart[torch.as_tensor(current_sources, device=device), torch.arange(len(current_sources), device=device)] = 1
        values = restart.clone()
        for step in range(160):
            updated = alpha * restart + (1 - alpha) * (
                torch.sparse.mm(operator, values) + restart * values[dangling].sum(0)
            )
            residual = (updated - values).abs().sum(0).max().item()
            values = updated
            if residual <= tolerance:
                break
        else:
            raise RuntimeError(f"PPR did not converge: {residual}")
        if not torch.allclose(values.sum(0), torch.ones(len(current_sources), device=device), atol=2e-5):
            raise AssertionError("PPR probability mass changed")
        output[start:start + len(current_sources)] = values.T.cpu().numpy()
        max_residual = max(max_residual, residual)
        max_steps = max(max_steps, step + 1)
    return output, {"alpha": alpha, "max_l1_iteration_residual": max_residual, "max_iterations": max_steps}


def topology_tables(matrix, sources, ppr):
    """Return degree, CN, RA, directed PPR for all anchor-target candidates."""
    degree = np.asarray(matrix.sum(1)).ravel()
    inv = np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)
    common = (matrix[sources] @ matrix).toarray().astype(np.float32)
    allocation = (matrix[sources] @ sp.diags(inv) @ matrix).toarray().astype(np.float32)
    return degree, common, allocation, ppr


def negative_targets(source, mapping, known, fold, local_ra, local_ppr, global_ra, global_ppr, rng, k=500):
    """Same-source corruptions, excluding all known reference PPIs and self.

    Each negative belongs to its positive's pair fold, so cross-fitting retains
    exact 1:k ratios with no pair leaking between probe fit/evaluation folds.
    Hard banks: rank-fuse RA/PPR, take 500 distinct candidates, shuffle once;
    smaller k values are nested prefixes, not increasingly hard top-k lists.
    """
    global_source = mapping[source]
    candidates = np.arange(len(mapping))
    keys = pair_keys(np.stack([np.full(len(mapping), global_source), mapping]), known.shape[0])
    allowed = (mapping != global_source) & (pair_fold(keys) == fold)
    allowed &= np.asarray(known[global_source, mapping].toarray()).ravel() == 0
    candidates = candidates[allowed]
    if len(candidates) < k:
        raise ValueError(f"Only {len(candidates)} distinct eligible candidates, need {k}")
    result = {"random": rng.choice(candidates, k, replace=True)}
    for name, ra, ppr in (("local_hard", local_ra, local_ppr), ("global_hard", global_ra, global_ppr)):
        ranks = np.minimum(rankdata(-ra[candidates], method="min"), rankdata(-ppr[candidates], method="min"))
        # Seeded random tie-breaking; no trained model score enters mining.
        chosen = candidates[np.lexsort((rng.random(len(candidates)), ranks))[:k]]
        result[name] = rng.permutation(chosen)
    return result


def prepare(args):
    output = args.output_dir
    if (output / "cache.pt").exists():
        raise FileExistsError("Prepared cache already exists; reuse it instead of reloading")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    contextual = torch.load(args.protscape_checkpoint, map_location="cpu", mmap=True, weights_only=True)
    data = load_global_ppi_data(args.networks_dir, args.esm2_embeddings, seed=0, verbose=False)
    validate_checkpoint_data(checkpoint, data)
    contexts, features, names, labels = load_context_data(contextual, args.networks_dir, args.esm2_embeddings)
    assert names == data.protein_names and torch.equal(features, data.features)
    size = len(names)
    global_train = adjacency(data.train_edge_index.numpy(), size)
    global_all = adjacency(data.all_edge_index.numpy(), size)
    degree = np.asarray(global_train.sum(1)).ravel()
    rows = []
    for cell, graph in contexts.items():
        mapping = graph.feature_index.numpy()
        local = adjacency(graph.edge_index[:, graph.train_mask].numpy(), len(mapping))
        local_all = adjacency(graph.edge_index.numpy(), len(mapping))
        local_degree = np.asarray(local.sum(1)).ravel()
        retained = np.divide(local_degree, degree[mapping], out=np.ones_like(local_degree), where=degree[mapping] > 0)
        rows.append({"cell_id": cell, "cell": labels[cell], "nodes": len(mapping),
                     "train_edges": local.nnz // 2, "retained_neighbor_mean": float(retained.mean()),
                     "retained_neighbor_median": float(np.median(retained)),
                     "induced_train": (local != global_train[mapping][:, mapping]).nnz == 0,
                     "induced_full": (local_all != global_all[mapping][:, mapping]).nnz == 0})
    frame = pd.DataFrame(rows)
    eligible = frame[frame.induced_train & frame.induced_full & (frame.nodes >= 2000)].sort_values(["retained_neighbor_mean", "cell_id"])
    selected = eligible.iloc[np.linspace(0, len(eligible) - 1, args.contexts).round().astype(int)].cell_id.tolist()
    assert len(set(selected)) == args.contexts
    frame["selected"] = frame.cell_id.isin(selected)
    frame.to_csv(output / "context_topology.csv", index=False)
    torch.save({"data": data, "contexts": contexts, "labels": labels, "selected": selected}, output / "cache.pt")
    metadata = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
                "protscape_checkpoint": str(args.protscape_checkpoint), "protscape_sha256": sha256_file(args.protscape_checkpoint),
                "selected": selected, "selection": "evenly spaced train-neighbor retention ranks among exact induced Cell-PPIs >=2000 nodes",
                "split": "validation only; all graph computations training-only", "max_positives_per_context": args.positives,
                "seed": 0, "probe_folds": 3, "hard_sampler": "HeaRT-inspired minimum RA/PPR rank, single randomly oriented positive anchor, fixed shuffled pool500",
                "negative_exclusion": "all known global reference PPIs and self; unknown pairs are not verified biological negatives",
                "positive_scope": "off-diagonal validation PPIs; homomers excluded from this diagnostic only",
                "heuristic_topology": "training-only simple graph without self-loops; GNN retains original training self-loop occurrences",
                "evaluation_unit": "sampled unique positive pair per context; pair-grouped cross-fit fold means then equal context means"}
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


@torch.no_grad()
def decoder_scores(decoder, layers, edges, device):
    output = []
    for start in range(0, edges.shape[1], 20000):
        logits = undirected_decoder_logits(decoder, layers, torch.from_numpy(edges[:, start:start + 20000]).long().to(device))
        output.append(logits.cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def gather_features(tables, source_rows, sources, targets):
    degree, cn, ra, ppr = tables
    source_degree, target_degree = degree[sources], degree[targets]
    return np.stack([np.minimum(source_degree, target_degree), np.maximum(source_degree, target_degree),
                     cn[source_rows, targets], ra[source_rows, targets], ppr[source_rows, targets]], axis=-1).astype(np.float32)


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(8)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Bulk frozen-model analysis must run on a compute GPU")
    cache = torch.load(args.output_dir / "cache.pt", map_location="cpu", weights_only=False)
    data, contexts = cache["data"], cache["contexts"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    metadata = json.loads((args.output_dir / "protocol.json").read_text())
    assert sha256_file(args.checkpoint) == metadata["checkpoint_sha256"]
    assert sha256_file(args.protscape_checkpoint) == metadata["protscape_sha256"]
    model = build_model(checkpoint, device).eval()
    initial_hash = state_hash(model)
    contextual_checkpoint = torch.load(args.protscape_checkpoint, map_location="cpu", mmap=True, weights_only=True)
    contextual = load_protscape_model(contextual_checkpoint, contexts, device=device).eval()
    contextual_hash = state_hash(contextual)
    features = data.features.to(device)
    train_edges = data.train_edge_index.numpy()
    size = features.shape[0]
    graph = adjacency(train_edges, size)
    known = adjacency(data.all_edge_index.numpy(), size)
    _, global_layers = model.encode(features, data.train_edge_index.to(device))
    global_val = set(pair_keys(data.val_edge_index.numpy(), size).tolist())
    selected = cache["selected"][args.shard::args.shards]
    for cell in selected:
        destination = args.output_dir / f"cell_{cell}.npz"
        if destination.exists():
            print(f"Reusing completed cell {cell}", flush=True)
            continue
        rng = np.random.default_rng(cell)
        context = contexts[cell]
        mapping = context.feature_index.numpy()
        local_train_edges = context.edge_index[:, context.train_mask]
        local = adjacency(local_train_edges.numpy(), len(mapping))
        positive = context.edge_index[:, context.val_mask].numpy()
        # Homomeric PPIs are valid biological targets but not informative about
        # between-protein neighbourhoods. Keep them out of this diagnostic;
        # the historical benchmark and all checkpoints remain unchanged.
        self_occurrences = int((positive[0] == positive[1]).sum())
        positive = positive[:, positive[0] != positive[1]]
        keys = pair_keys(mapping[positive], size)
        _, first = np.unique(keys, return_index=True)
        first = rng.choice(first, min(args.positives, len(first)), replace=False)
        positive = positive[:, first].copy()
        flip = rng.random(len(first)) < .5
        positive[:, flip] = positive[::-1, flip]
        pos_keys = pair_keys(mapping[positive], size)
        assert set(pos_keys.tolist()) <= global_val
        folds = pair_fold(pos_keys)
        sources, source_rows = np.unique(positive[0], return_inverse=True)
        local_ppr, local_ppr_info = personalized_pagerank(local, sources, device)
        global_ppr, global_ppr_info = personalized_pagerank(graph, mapping[sources], device)
        local_tables = topology_tables(local, sources, local_ppr)
        full_tables = topology_tables(graph, mapping[sources], global_ppr)
        # Restrict target columns, not the diffusion graph, to proteins in this cell.
        global_tables = (full_tables[0][mapping],) + tuple(values[:, mapping] for values in full_tables[1:])
        banks = {name: np.empty((len(first), 501), np.int64) for name in BANKS}
        for row, (source, target) in enumerate(positive.T):
            source_row = source_rows[row]
            negatives = negative_targets(source, mapping, known, folds[row],
                                         local_tables[2][source_row], local_tables[3][source_row],
                                         global_tables[2][source_row], global_tables[3][source_row], rng)
            for name in BANKS:
                banks[name][row, 0] = target
                banks[name][row, 1:] = negatives[name]
        target_array = np.stack([banks[name] for name in BANKS])
        source_array = np.broadcast_to(positive[0][None, :, None], target_array.shape)
        row_array = np.broadcast_to(source_rows[None, :, None], target_array.shape)
        edge_array = np.stack([source_array.ravel(), target_array.ravel()])
        global_keys = pair_keys(mapping[edge_array], size)
        expected_folds = np.broadcast_to(folds[None, :, None], target_array.shape).ravel()
        assert np.array_equal(pair_fold(global_keys), expected_folds)
        unique_keys, inverse = np.unique(pair_keys(edge_array, len(mapping)), return_inverse=True)
        unique_edges = np.stack([unique_keys // len(mapping), unique_keys % len(mapping)])
        unique_global_edges = mapping[unique_edges]
        scores = []
        scores.append(decoder_scores(model.decoder, global_layers, unique_global_edges, device)[inverse])
        _, local_layers = model.encode(features[mapping], local_train_edges.to(device))
        scores.append(decoder_scores(model.decoder, local_layers, unique_edges, device)[inverse])
        inside = np.zeros(size, bool)
        inside[mapping] = True
        boundary = inside[train_edges[0]] != inside[train_edges[1]]
        # Removing all boundary edges must yield the same local embeddings.
        _, cut_layers = model.encode(features, torch.from_numpy(train_edges[:, ~boundary]).to(device))
        cut_error = max(float((left - right[mapping]).abs().max()) for left, right in zip(local_layers, cut_layers))
        if not all(torch.allclose(left, right[mapping], atol=5e-5, rtol=5e-5) for left, right in zip(local_layers, cut_layers)):
            raise AssertionError(f"Global boundary-cut/local mismatch: {cut_error}")
        del cut_layers
        heldout_keys = set(pair_keys(torch.cat([data.val_edge_index, data.test_edge_index], 1).numpy(), size).tolist())
        rewired, rewire_info = rewire_boundary(train_edges, inside, size, cell, heldout_keys)
        assert not (set(pair_keys(rewired, size).tolist()) & heldout_keys)
        assert (adjacency(rewired, size)[mapping][:, mapping] != local).nnz == 0
        _, altered_layers = model.encode(features, torch.from_numpy(rewired).to(device))
        scores.append(decoder_scores(model.decoder, altered_layers, unique_global_edges, device)[inverse])
        outside = np.flatnonzero(~inside)
        permuted_features = features.clone()
        permuted_features[outside] = features[rng.permutation(outside)]
        _, altered_layers = model.encode(permuted_features, data.train_edge_index.to(device))
        scores.append(decoder_scores(model.decoder, altered_layers, unique_global_edges, device)[inverse])
        context_layers, raw = encode_protscape_local(contextual, data.features, context, cell, context.train_mask, device)
        decoder_input = prepare_decoder_input(contextual.s2gae_decoder, context_layers, raw_embeddings=raw)
        scores.append(decoder_scores(contextual.s2gae_decoder, decoder_input, unique_edges, device)[inverse])
        score_array = np.stack(scores).reshape(len(GNN_NAMES), *target_array.shape)
        local_features = gather_features(local_tables, row_array, source_array, target_array)
        global_features = gather_features(global_tables, row_array, source_array, target_array)
        assert np.all(global_features[..., 2] >= local_features[..., 2])
        info = {"cell_id": cell, "cell": cache["labels"][cell], "n_positive": len(first), "nodes": len(mapping),
                "excluded_validation_self_occurrences": self_occurrences,
                "boundary_cut_max_embedding_error": cut_error, **rewire_info,
                "local_ppr": local_ppr_info, "global_ppr": global_ppr_info,
                "global_local_random_logit_spearman": float(spearmanr(score_array[0, 0].ravel(), score_array[1, 0].ravel()).statistic)}
        assert state_hash(model) == initial_hash and state_hash(contextual) == contextual_hash
        # Publication marker last: incomplete jobs never masquerade as a reusable cell.
        temporary = destination.with_suffix(".partial.npz")
        np.savez_compressed(temporary, scores=score_array, local=local_features, global_features=global_features,
                            keys=global_keys.reshape(target_array.shape), folds=folds, positive_global=mapping[positive],
                            info=np.array(json.dumps(info)))
        temporary.replace(destination)
        print(json.dumps(info), flush=True)


PROBES = {
    "Local structure probe": [2, 3, 4, 5, 6],
    "Global structure probe": [7, 8, 9, 10, 11],
    "Local GNN + local structure": [1, 2, 3, 4, 5, 6],
    "Local GNN + both structures": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Global GNN + global structure": [0, 7, 8, 9, 10, 11],
    "Global GNN + both structures": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}


def probe_features(artifact):
    local, full = artifact["local"], artifact["global_features"]
    left, right = np.log1p(local.copy()), np.log1p(full.copy())
    left[..., 4], right[..., 4] = np.log1p(local[..., 4] * 10000), np.log1p(full[..., 4] * 10000)
    return np.concatenate([np.moveaxis(artifact["scores"][:2], 0, -1), left, right], -1)


def metric_rows(scores, folds, cell, bank, model):
    rows = []
    for fold in range(3):
        selected = scores[folds == fold]
        for k in K_VALUES:
            values = selected[:, :k + 1].ravel()
            labels = np.broadcast_to(np.arange(k + 1) == 0, (len(selected), k + 1)).ravel()
            rows.append({"cell_id": cell, "fold": fold, "bank": bank, "model": model, "k": k,
                         "n_positive": len(selected), "auprc": average_precision_score(labels, values),
                         "macro_f1_at_zero_logit": f1_score(labels, values >= 0, average="macro", zero_division=0)})
    return rows


def analyze(args):
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    cells = protocol["selected"]
    for cell in cells:
        if not (args.output_dir / f"cell_{cell}.npz").exists():
            raise FileNotFoundError(f"Missing completed context {cell}")
    fit_x, fit_y, fit_fold, fit_keys = [], [], [], []
    for cell in cells:
        with np.load(args.output_dir / f"cell_{cell}.npz") as item:
            features = probe_features(item)[0, :, :11]  # Random bank only, fixed 1:10 fit mixture.
            fit_x.append(features.reshape(-1, 12))
            fit_y.append(np.broadcast_to(np.arange(11) == 0, features.shape[:2]).ravel())
            fit_fold.append(np.repeat(item["folds"], 11))
            fit_keys.append(item["keys"][0, :, :11].ravel())
    x, y, fold_ids, keys = map(np.concatenate, (fit_x, fit_y, fit_fold, fit_keys))
    assert np.array_equal(pair_fold(keys), fold_ids)
    models, coefficients = {}, []
    for fold in range(3):
        mask = fold_ids != fold
        for name, columns in PROBES.items():
            estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, tol=1e-7, random_state=0))
            estimator.fit(x[mask][:, columns], y[mask])
            if estimator[-1].n_iter_.max() >= 2000:
                raise RuntimeError(f"Unconverged structural probe: {name}")
            models[fold, name] = estimator
            coefficients.append({"fold": fold, "probe": name, "columns": columns,
                                 "standardized_coefficients": estimator[-1].coef_[0].tolist(), "fit_count": int(mask.sum())})
    rows, topology, strata = [], [], []
    for cell in cells:
        with np.load(args.output_dir / f"cell_{cell}.npz") as item:
            features, folds = probe_features(item), item["folds"]
            topology.append(json.loads(str(item["info"])))
            for bank_id, bank in enumerate(BANKS):
                for model_id, name in enumerate(GNN_NAMES):
                    rows.extend(metric_rows(item["scores"][model_id, bank_id], folds, cell, bank, name))
                for offset, scope in ((2, "Local"), (7, "Global")):
                    for feature_id in (2, 3, 4):
                        rows.extend(metric_rows(features[bank_id, ..., offset + feature_id], folds, cell, bank, f"{scope} {FEATURE_NAMES[feature_id]}"))
                for name, columns in PROBES.items():
                    predictions = np.empty(features.shape[1:3], np.float32)
                    for fold in range(3):
                        selected = folds == fold
                        heldout = features[bank_id, selected]
                        assert np.all(pair_fold(item["keys"][bank_id, selected]) == fold)
                        predictions[selected] = models[fold, name].decision_function(heldout.reshape(-1, 12)[:, columns]).reshape(heldout.shape[:2])
                    rows.extend(metric_rows(predictions, folds, cell, bank, name))
                local_cn, global_cn = item["local"][bank_id, ..., 2], item["global_features"][bank_id, ..., 2]
                for label, slc in (("positive", slice(0, 1)), ("negative", slice(1, None))):
                    lcn, gcn = local_cn[:, slc], global_cn[:, slc]
                    strata.append({"cell_id": cell, "bank": bank, "label": label,
                                   "fraction_no_local_common_neighbor": float((lcn == 0).mean()),
                                   "fraction_no_global_common_neighbor": float((gcn == 0).mean()),
                                   "fraction_extra_global_common_neighbor": float((gcn > lcn).mean()),
                                   "fraction_only_outside_common_neighbor": float(((lcn == 0) & (gcn > 0)).mean())})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "metrics_by_fold.csv", index=False)
    per_context = frame.groupby(["cell_id", "bank", "model", "k"], as_index=False).auprc.mean()
    per_context.to_csv(args.output_dir / "metrics_by_context.csv", index=False)
    summary = per_context.groupby(["bank", "model", "k"], as_index=False).auprc.agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(strata).to_csv(args.output_dir / "topology_strata.csv", index=False)
    (args.output_dir / "interventions.json").write_text(json.dumps(topology, indent=2) + "\n")
    (args.output_dir / "probe_coefficients.json").write_text(json.dumps(coefficients, indent=2) + "\n")
    plot_results(summary, args.output_dir)
    print(summary[summary.k == 500].to_string(index=False), flush=True)


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    colors = ("#d7191c", "#7b3294", "#008837", "#666666", "#ff6600")
    for ax, bank in zip(axes, BANKS):
        for name, color in zip(GNN_NAMES, colors):
            subset = summary[(summary.bank == bank) & (summary.model == name)].sort_values("k")
            ax.plot(subset.k, subset["mean"] * 100, "o-", color=color, label=name, markersize=4)
        ax.set_xscale("log")
        ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
        ax.set_xlabel("Negatives per positive")
        ax.set_title(bank.replace("_", " ").title())
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Validation AUPRC (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(.5, -.10), frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"topology_interventions.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--protscape-checkpoint", type=Path)
    parser.add_argument("--networks-dir", type=Path)
    parser.add_argument("--esm2-embeddings", type=Path)
    parser.add_argument("--contexts", type=int, default=16)
    parser.add_argument("--positives", type=int, default=512)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "evaluate": evaluate, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
