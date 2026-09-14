"""Compare frozen-model predictions on complete Cell-PPI graphs, without labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpoints import load_protscape_model
from .compare_inference_topologies import (
    load_context_data, encode_context_free_local, encode_protscape_local,
    prepare_decoder_input, score_decoder, sha256_file,
)
from .evaluate_global_s2gae import build_model
from .global_s2gae import load_global_ppi_data, _canonical_keys
from .s2gae_utils import undirected_decoder_logits
from .summarize_context_rewiring import analyze

DATA = Path("/iopsstor/scratch/cscs/aloistho/protscape/release-data")
SELECTION = Path("/users/aloistho/projects/outputs/global_s2gae_convergence/selection.json")
CONTEXTUAL = Path("/capstor/store/cscs/swissai/a0204/aloistho/ProtScape_release/models/pretraining/protscape_main_state_dict.pt")


def shared_edge_panel(keys_by_context):
    """Unique edges in pairwise intersections, counting each edge once per context."""
    unique_sets = [np.unique(keys) for keys in keys_by_context]
    keys, counts = np.unique(np.concatenate(unique_sets), return_counts=True)
    return keys[counts >= 2], counts[counts >= 2]


def sample_unobserved_pairs(n_nodes, known_keys, presence, count, seed=0):
    """Uniform unique nonedges with endpoints co-present in at least two contexts."""
    rng, selected = np.random.default_rng(seed), set()
    for _ in range(1000):
        candidates = np.sort(rng.integers(n_nodes, size=(20000, 2)), axis=1)
        eligible = (presence[candidates[:, 0]] & presence[candidates[:, 1]]).sum(axis=1) >= 2
        for left, right in candidates[eligible]:
            key = int(left * n_nodes + right)
            if left != right and key not in known_keys:
                selected.add(key)
                if len(selected) == count:
                    return np.asarray(sorted(selected), dtype=np.int64)
    raise RuntimeError("Could not generate the requested unobserved-pair panel")


def load_inputs():
    """Load exactly the checkpoints, features and graphs used in the first comparison."""
    selected = json.loads(SELECTION.read_text())
    assert sha256_file(Path(selected["selected_checkpoint"])) == selected["checkpoint_sha256"]
    cf_checkpoint = torch.load(selected["selected_checkpoint"], map_location="cpu", weights_only=True)
    contextual = torch.load(CONTEXTUAL, map_location="cpu", weights_only=True)
    features_path = DATA / "sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk"
    global_data = load_global_ppi_data(DATA / "networks_bulk", features_path,
                                      seed=cf_checkpoint["training_config"]["split_seed"], verbose=False)
    for key in ["graph_fingerprint", "feature_fingerprint", "split_fingerprint"]:
        assert cf_checkpoint[key] == getattr(global_data, key), f"Changed {key}"
    contexts, features, names, cell_names = load_context_data(contextual, DATA / "networks_bulk", features_path)
    assert names == global_data.protein_names == cf_checkpoint["protein_names"]
    assert torch.equal(features, global_data.features)
    return selected, cf_checkpoint, contextual, global_data, contexts, features, names, cell_names


def all_unobserved_pairs(presence, known_keys, block_size=256):
    """Enumerate every unordered nonedge co-present in >=2 contexts; no sampling."""
    n = len(presence)
    known = np.zeros((n, n), dtype=bool)
    known_keys = np.asarray(known_keys, dtype=np.int64)
    known[known_keys // n, known_keys % n] = True
    present = presence.astype(np.float32)
    pair_blocks, count_blocks = [], []
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        # The sums are exact in float32 for these 207 binary context indicators.
        counts = present[start:stop] @ present.T
        keep = ((counts >= 2) & ~known[start:stop]
                & (np.arange(start, stop)[:, None] < np.arange(n)[None, :]))
        left, right = np.nonzero(keep)
        pair_blocks.append((left + start) * n + right)
        count_blocks.append(counts[keep].astype(np.uint16))
    return np.concatenate(pair_blocks), np.concatenate(count_blocks)


def prepare_exhaustive(root):
    selected, cf_checkpoint, _, global_data, contexts, _, names, cell_names = load_inputs()
    presence = np.zeros((len(names), len(contexts)), dtype=bool)
    for index, graph in enumerate(contexts.values()):
        presence[graph.feature_index.numpy(), index] = True
    keys, counts = all_unobserved_pairs(
        presence, _canonical_keys(global_data.all_edge_index, len(names)).numpy())
    np.save(root / "pair_keys.npy", keys)
    np.save(root / "context_counts.npy", counts)
    manifest = {"panel": "all_unobserved", "sampling": "none; exhaustive enumeration",
        "n_shared_pairs": len(keys), "n_contexts": len(contexts),
        "n_pair_context_observations": int(counts.sum()),
        "bank_sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
        "global_checkpoint": selected["selected_checkpoint"],
        "global_checkpoint_sha256": selected["checkpoint_sha256"],
        "contextual_checkpoint": str(CONTEXTUAL),
        "contextual_checkpoint_sha256": sha256_file(CONTEXTUAL),
        "graph_fingerprint": cf_checkpoint["graph_fingerprint"],
        "feature_fingerprint": cf_checkpoint["feature_fingerprint"],
        "cell_ids": list(contexts), "cell_names": [cell_names[cell] for cell in contexts],
        "input_graph": "full Cell-PPI, all edges; no evaluation split",
        "comparison": "prediction agreement, not accuracy",
        "eligibility": "absent from global PPI; distinct endpoints co-present in >=2 contexts"}
    (root / "BANK_COMPLETE.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


@torch.no_grad()
def evaluate_exhaustive(root, shard):
    """Encode each cell once and decode ALL eligible pairs in bounded-size chunks."""
    selected, cf_checkpoint, contextual, global_data, contexts, features, names, cell_names = load_inputs()
    manifest = json.loads((root / "BANK_COMPLETE.json").read_text())
    for key in ["graph_fingerprint", "feature_fingerprint"]:
        assert manifest[key] == cf_checkpoint[key]
    assert manifest["global_checkpoint_sha256"] == selected["checkpoint_sha256"]
    assert manifest["contextual_checkpoint"] == str(CONTEXTUAL)
    assert manifest["contextual_checkpoint_sha256"] == sha256_file(CONTEXTUAL)
    assert manifest["cell_ids"] == list(contexts)
    assert manifest["cell_names"] == [cell_names[cell] for cell in contexts]
    keys = np.load(root / "pair_keys.npy", mmap_mode="r")
    assert hashlib.sha256(keys).hexdigest() == manifest["bank_sha256"]
    n, device = len(names), torch.device("cuda")
    cf = build_model(cf_checkpoint, device).eval()
    prot = load_protscape_model(contextual, contexts, device=device).eval()
    previous_root = Path("/users/aloistho/projects/outputs/context_edge_pos_neg")
    previous_manifest = json.loads((previous_root / "COMPLETE.json").read_text())
    for key in ["global_checkpoint_sha256", "contextual_checkpoint", "input_graph"]:
        assert previous_manifest[key] == manifest[key]
    previous = pd.read_csv(previous_root / f"scores_{shard}.csv.gz",
                           usecols=["pair_key", "cell_id", "score_cell_context_free", "score_protscape"])
    for index, (cell, graph) in enumerate(contexts.items()):
        if index % 4 != shard:
            continue
        done = root / f"cell_{index}.json"
        if done.exists():
            saved = json.loads(done.read_text())
            for key in ["bank_sha256", "global_checkpoint_sha256", "contextual_checkpoint_sha256"]:
                assert saved[key] == manifest[key]
            print(f"Reusing completed context {index + 1}/{len(contexts)}", flush=True)
            continue
        lookup = np.full(n, -1, dtype=np.int64)
        lookup[graph.feature_index.numpy()] = np.arange(graph.num_nodes)
        indices = np.flatnonzero((lookup[keys // n] >= 0) & (lookup[keys % n] >= 0))
        np.save(root / f"cell_{index}_pair_indices.npy", indices.astype(np.uint32))
        scores = np.lib.format.open_memmap(root / f"cell_{index}_scores.npy", mode="w+",
                                          dtype=np.float32, shape=(len(indices), 2))
        message_mask = torch.ones(graph.edge_index.shape[1], dtype=torch.bool)
        cf_layers = encode_context_free_local(cf, features, graph, message_mask, device)
        prot_layers, raw = encode_protscape_local(prot, features, graph, cell, message_mask, device)
        prot_input = prepare_decoder_input(prot.s2gae_decoder, prot_layers, raw)
        for start in range(0, len(indices), 50000):
            batch_keys = keys[indices[start:start + 50000]]
            pairs = torch.from_numpy(np.stack([lookup[batch_keys // n], lookup[batch_keys % n]])).to(device)
            # Unique unordered candidates: use exactly the existing symmetric decoder.
            for column, decoder, inputs in [(0, cf.decoder, cf_layers), (1, prot.s2gae_decoder, prot_input)]:
                values = torch.sigmoid(undirected_decoder_logits(decoder, inputs, pairs)).cpu().numpy()
                assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
                scores[start:start + len(values), column] = values
        scores.flush()
        # The old sampled bank must be a subset, and its saved scores must agree.
        old = previous[previous.cell_id == cell]
        global_indices = np.searchsorted(keys, old.pair_key.to_numpy())
        np.testing.assert_array_equal(keys[global_indices], old.pair_key)
        positions = np.searchsorted(indices, global_indices)
        np.testing.assert_array_equal(indices[positions], global_indices)
        old_scores = old[["score_cell_context_free", "score_protscape"]].to_numpy()
        np.testing.assert_allclose(scores[positions], old_scores, atol=1e-5, rtol=1e-5)
        audit_error = float(np.abs(scores[positions] - old_scores).max()) if len(old) else 0.0
        done.write_text(json.dumps({"index": index, "cell_id": cell, "cell": cell_names[cell],
            "n_pairs": len(indices), "bank_sha256": manifest["bank_sha256"],
            "global_checkpoint_sha256": manifest["global_checkpoint_sha256"],
            "contextual_checkpoint_sha256": manifest["contextual_checkpoint_sha256"],
            "previous_sample_pairs_verified": len(old), "previous_sample_max_score_error": audit_error}, indent=2))
        del scores, cf_layers, prot_layers, raw, prot_input
        print(f"shard {shard}: ALL {len(indices):,} pairs scored in context {index + 1}/{len(contexts)}", flush=True)


def context_statistics(values):
    """Exact row-wise variance and Spearman; NaN marks absent contexts, not scores."""
    from scipy.stats import rankdata
    counts = np.isfinite(values[:, :, 0]).sum(axis=1)
    assert np.array_equal(np.isfinite(values[:, :, 0]), np.isfinite(values[:, :, 1]))
    assert (counts >= 2).all()
    means = np.nanmean(values, axis=1)
    centered = values - means[:, None, :]
    variances = np.nansum(centered ** 2, axis=1) / counts[:, None]
    ranks = rankdata(values, axis=1, method="average", nan_policy="omit")
    ranks -= np.nanmean(ranks, axis=1)[:, None, :]
    covariance = np.nansum(ranks[:, :, 0] * ranks[:, :, 1], axis=1)
    denominator = np.sqrt(np.nansum(ranks[:, :, 0] ** 2, axis=1)
                          * np.nansum(ranks[:, :, 1] ** 2, axis=1))
    rho = np.full(len(values), np.nan)
    valid = (counts >= 5) & (denominator > 0)
    np.divide(covariance, denominator, out=rho, where=valid)
    return variances, rho, np.nansum(centered ** 2, axis=0), counts


def summarize_exhaustive(root, shard):
    """Merge by pair in chunks, retaining exact ranks without a giant pandas table."""
    from scipy.stats import spearmanr
    manifest = json.loads((root / "BANK_COMPLETE.json").read_text())
    n_pairs, n_cells = manifest["n_shared_pairs"], manifest["n_contexts"]
    keys = np.load(root / "pair_keys.npy", mmap_mode="r")
    counts = np.load(root / "context_counts.npy", mmap_mode="r")
    assert len(keys) == len(counts) == n_pairs
    cell_indices, cell_scores, cell_metadata = [], [], []
    for index in range(n_cells):
        metadata = json.loads((root / f"cell_{index}.json").read_text())
        assert metadata["bank_sha256"] == manifest["bank_sha256"]
        indices = np.load(root / f"cell_{index}_pair_indices.npy", mmap_mode="r")
        scores = np.load(root / f"cell_{index}_scores.npy", mmap_mode="r")
        assert len(indices) == metadata["n_pairs"] and scores.shape == (len(indices), 2)
        assert np.all(indices[1:] > indices[:-1])
        cell_indices.append(indices)
        cell_scores.append(scores)
        cell_metadata.append(metadata)
    assert sum(m["n_pairs"] for m in cell_metadata) == manifest["n_pair_context_observations"]
    start_pair, stop_pair = n_pairs * shard // 4, n_pairs * (shard + 1) // 4
    stats = np.lib.format.open_memmap(root / f"pair_stats_{shard}.npy", mode="w+", dtype=np.float64,
                                     shape=(stop_pair - start_pair, 3))
    deviations = np.zeros((n_cells, 2), dtype=np.float64)
    observations = np.zeros(n_cells, dtype=np.int64)
    for start in range(start_pair, stop_pair, 32768):
        stop = min(stop_pair, start + 32768)
        values = np.full((stop - start, n_cells, 2), np.nan, dtype=np.float64)
        for cell, (indices, scores) in enumerate(zip(cell_indices, cell_scores)):
            left, right = np.searchsorted(indices, [start, stop])
            values[indices[left:right].astype(np.int64) - start, cell] = scores[left:right]
            observations[cell] += right - left
        variances, rho, squared_deviations, observed_counts = context_statistics(values)
        np.testing.assert_array_equal(observed_counts, counts[start:stop])
        stats[start - start_pair:stop - start_pair, :2] = variances
        stats[start - start_pair:stop - start_pair, 2] = rho
        deviations += squared_deviations
        if (start - start_pair) % (32768 * 50) == 0:
            print(f"summary shard {shard}: {stop - start_pair:,}/{stop_pair - start_pair:,} pairs", flush=True)
    stats.flush()
    np.savez(root / f"context_sums_{shard}.npz", deviations=deviations, observations=observations)
    correlations = []
    for index, scores in enumerate(cell_scores):
        if index % 4 == shard:
            rho = float(spearmanr(scores[:, 0], scores[:, 1]).statistic)
            correlations.append({**cell_metadata[index], "within_cell_spearman": rho})
    pd.DataFrame(correlations).to_csv(root / f"cell_correlations_{shard}.csv", index=False)
    (root / f"SUMMARY_{shard}_COMPLETE.json").write_text(json.dumps({
        "bank_sha256": manifest["bank_sha256"], "start_pair": start_pair, "stop_pair": stop_pair}))


def finish_exhaustive(root):
    manifest = json.loads((root / "BANK_COMPLETE.json").read_text())
    for shard in range(4):
        done = json.loads((root / f"SUMMARY_{shard}_COMPLETE.json").read_text())
        assert done["bank_sha256"] == manifest["bank_sha256"]
    stats = np.concatenate([np.load(root / f"pair_stats_{i}.npy", mmap_mode="r") for i in range(4)])
    assert len(stats) == manifest["n_shared_pairs"]
    cells = pd.concat([pd.read_csv(root / f"cell_correlations_{i}.csv") for i in range(4)]).sort_values("index")
    assert cells["index"].tolist() == list(range(manifest["n_contexts"]))
    sums = [np.load(root / f"context_sums_{i}.npz") for i in range(4)]
    observations = sum(s["observations"] for s in sums)
    np.testing.assert_array_equal(observations, cells.n_pairs)
    rms = 100 * np.sqrt(sum(s["deviations"] for s in sums) / observations[:, None])
    cells["rms_context_deviation_cell_context_free_pp"] = rms[:, 0]
    cells["rms_context_deviation_protscape_pp"] = rms[:, 1]
    cells.to_csv(root / "per_cell.csv", index=False)
    defined = stats[np.isfinite(stats[:, 2]), 2]
    q = np.quantile(defined, [.25, .5, .75])
    cq = cells.within_cell_spearman.quantile([.25, .5, .75]).to_numpy()
    row = {"panel": "Unobserved pairs", "n_pairs": len(stats), "n_contexts": len(cells),
        "pair_context_observations": int(observations.sum()),
        "cf_mean_variance": float(stats[:, 0].mean()), "protscape_mean_variance": float(stats[:, 1].mean()),
        "cf_median_pair_sd_pp": float(100 * np.sqrt(np.median(stats[:, 0]))),
        "protscape_median_pair_sd_pp": float(100 * np.sqrt(np.median(stats[:, 1]))),
        "median_within_cell_spearman": float(cq[1]), "within_cell_spearman_q25": float(cq[0]),
        "within_cell_spearman_q75": float(cq[2]), "median_within_pair_spearman": float(q[1]),
        "within_pair_spearman_q25": float(q[0]), "within_pair_spearman_q75": float(q[2]),
        "n_pairs_for_context_correlation": len(defined)}
    pd.DataFrame([row]).to_csv(root / "exhaustive_summary.csv", index=False)
    np.save(root / "context_correlations.npy", defined)
    (root / "COMPLETE.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(row, indent=2), flush=True)


@torch.no_grad()
def evaluate(root, shard, panel="shared_edges"):
    selected, cf_checkpoint, contextual, global_data, contexts, features, names, cell_names = load_inputs()
    n = len(names)
    keys_by_context = [np.unique(_canonical_keys(graph.feature_index[graph.edge_index], n).numpy())
                       for graph in contexts.values()]
    keys, counts = shared_edge_panel(keys_by_context)
    if panel == "unobserved":
        presence = np.zeros((n, len(contexts)), dtype=bool)
        for index, graph in enumerate(contexts.values()):
            presence[graph.feature_index.numpy(), index] = True
        known = set(_canonical_keys(global_data.all_edge_index, n).tolist())
        keys = sample_unobserved_pairs(n, known, presence, len(keys))
        counts = (presence[keys // n] & presence[keys % n]).sum(axis=1)
        assert not known.intersection(keys.tolist())
    print(f"{panel}: {len(keys)} pairs in >=2 contexts; strict intersection: {(counts == 207).sum()}", flush=True)
    bank_hash = hashlib.sha256(keys.tobytes()).hexdigest()
    device = torch.device("cuda")
    cf = build_model(cf_checkpoint, device).eval()
    prot = load_protscape_model(contextual, contexts, device=device).eval()
    _, global_layers = cf.encode(features.to(device), global_data.all_edge_index.to(device))
    global_pairs = torch.from_numpy(np.stack([keys // n, keys % n]))
    global_scores = score_decoder(cf.decoder, global_layers, global_pairs, device, 50000)
    rows = []
    for index, (cell, graph) in enumerate(contexts.items()):
        if index % 4 != shard:
            continue
        lookup = np.full(n, -1, dtype=np.int64)
        lookup[graph.feature_index.numpy()] = np.arange(graph.num_nodes)
        keep = (np.isin(keys, keys_by_context[index]) if panel == "shared_edges"
                else (lookup[keys // n] >= 0) & (lookup[keys % n] >= 0))
        local_pairs = torch.from_numpy(np.stack([lookup[keys[keep] // n], lookup[keys[keep] % n]]))
        message_mask = torch.ones(graph.edge_index.shape[1], dtype=torch.bool)
        cf_layers = encode_context_free_local(cf, features, graph, message_mask, device)
        prot_layers, raw = encode_protscape_local(prot, features, graph, cell, message_mask, device)
        cf_scores = score_decoder(cf.decoder, cf_layers, local_pairs, device, 50000)
        prot_scores = score_decoder(prot.s2gae_decoder, prepare_decoder_input(prot.s2gae_decoder, prot_layers, raw), local_pairs, device, 50000)
        rows.append(pd.DataFrame({"pair_key": keys[keep],
                                  "source_protein": np.asarray(names)[keys[keep] // n],
                                  "target_protein": np.asarray(names)[keys[keep] % n],
                                  "cell_id": cell, "cell": cell_names[cell],
                                  "score_global_context_free": global_scores[keep],
                                  "score_cell_context_free": cf_scores, "score_protscape": prot_scores}))
        print(f"shard {shard}: scored context {index + 1}/207 ({keep.sum()} pairs)", flush=True)
    pd.concat(rows, ignore_index=True).to_csv(root / f"scores_{shard}.csv.gz", index=False)
    (root / f"shard_{shard}.json").write_text(json.dumps({"bank_sha256": bank_hash,
        "n_shared_pairs": len(keys), "n_contexts": len(contexts), "shard": shard,
        "panel": panel,
        "strict_all_context_intersection": int((counts == len(contexts)).sum()),
        "global_checkpoint": selected["selected_checkpoint"], "global_checkpoint_sha256": selected["checkpoint_sha256"],
        "contextual_checkpoint": str(CONTEXTUAL), "seed": 0,
        "input_graph": "full Cell-PPI, all edges; no evaluation split",
        "comparison": "prediction agreement, not accuracy"}, indent=2))


def summarize(root):
    manifests = [json.loads((root / f"shard_{i}.json").read_text()) for i in range(4)]
    assert len({m["bank_sha256"] for m in manifests}) == 1, "Different query panels across shards"
    scores = pd.concat([pd.read_csv(root / f"scores_{i}.csv.gz") for i in range(4)], ignore_index=True)
    variation, pairs, cells, agreement, counts = analyze(scores, min_contexts=2)
    assert counts["n_contexts"] == 207
    assert counts["n_pairs"] == manifests[0]["n_shared_pairs"]
    assert pairs.sd_global_context_free_pp.max() < 1e-3, "Global CF must be context invariant"
    cells.to_csv(root / "per_cell.csv", index=False)
    pairs.to_csv(root / "per_pair.csv", index=False)
    headline = variation[variation.model_key != "global_context_free"].copy()
    headline["mean_within_pair_variance"] = [float(((pairs[f"sd_{key}_pp"] / 100) ** 2).mean())
                                            for key in headline.model_key]
    headline.to_csv(root / "results.csv", index=False)
    # Two-context Spearman can only be +/-1; summarize ranking agreement on >=5 contexts.
    stable = pairs.loc[pairs.n_contexts >= 5, "context_rank_spearman"].dropna()
    agreement = {"median_within_pair_spearman": float(stable.median()), "n_defined_pairs": len(stable),
                 "min_contexts_for_agreement": 5}
    (root / "agreement.json").write_text(json.dumps(agreement, indent=2))
    (root / "COMPLETE.json").write_text(json.dumps(manifests[0], indent=2))
    print(headline.to_string(index=False))
    print(json.dumps(agreement, indent=2))


def report_panels(root, reference_root):
    """Reuse reference predictions; compare the two panels without accuracy metrics."""
    summaries, cell_tables, pair_correlations = [], [], {}
    reference_manifest = json.loads((reference_root / "COMPLETE.json").read_text())
    unobserved_manifest = json.loads((root / "COMPLETE.json").read_text())
    for key in ["global_checkpoint_sha256", "contextual_checkpoint", "input_graph"]:
        assert reference_manifest[key] == unobserved_manifest[key], f"Unmatched {key}"
    for group, directory in [("Reference edges", reference_root), ("Unobserved pairs", root)]:
        if directory == root and unobserved_manifest["panel"] == "all_unobserved":
            summaries.append(pd.read_csv(root / "exhaustive_summary.csv").iloc[0].to_dict())
            cells = pd.read_csv(root / "per_cell.csv")
            cells.insert(0, "panel", group)
            cell_tables.append(cells)
            pair_correlations[group] = np.load(root / "context_correlations.npy", mmap_mode="r")
            continue
        pairs = pd.read_csv(directory / "per_pair.csv")
        cells = pd.read_csv(directory / "per_cell.csv")
        score_columns = ["cell_id", "score_cell_context_free", "score_protscape"]
        scores = pd.concat([pd.read_csv(directory / f"scores_{i}.csv.gz", usecols=score_columns)
                            for i in range(4)], ignore_index=True)
        correlations = []
        for cell, frame in scores.groupby("cell_id"):
            rho = frame.score_cell_context_free.corr(frame.score_protscape, method="spearman")
            correlations.append({"cell_id": cell, "within_cell_spearman": rho})
        cells = cells.merge(pd.DataFrame(correlations), on="cell_id", validate="one_to_one")
        cells.insert(0, "panel", group)
        cell_tables.append(cells)
        stable = pairs.loc[pairs.n_contexts >= 5, "context_rank_spearman"].dropna()
        pair_correlations[group] = stable.to_numpy()
        summaries.append({"panel": group, "n_pairs": len(pairs), "n_contexts": len(cells),
            "pair_context_observations": int(pairs.n_contexts.sum()),
            "cf_mean_variance": float(((pairs.sd_cell_context_free_pp / 100) ** 2).mean()),
            "protscape_mean_variance": float(((pairs.sd_protscape_pp / 100) ** 2).mean()),
            "cf_median_pair_sd_pp": float(pairs.sd_cell_context_free_pp.median()),
            "protscape_median_pair_sd_pp": float(pairs.sd_protscape_pp.median()),
            "median_within_cell_spearman": float(cells.within_cell_spearman.median()),
            "within_cell_spearman_q25": float(cells.within_cell_spearman.quantile(.25)),
            "within_cell_spearman_q75": float(cells.within_cell_spearman.quantile(.75)),
            "median_within_pair_spearman": float(stable.median()),
            "within_pair_spearman_q25": float(stable.quantile(.25)),
            "within_pair_spearman_q75": float(stable.quantile(.75)),
            "n_pairs_for_context_correlation": len(stable)})
        del scores
    table = pd.DataFrame(summaries)
    all_cells = pd.concat(cell_tables, ignore_index=True)
    table.to_csv(root / "comparison.csv", index=False)
    all_cells.to_csv(root / "per_context_comparison.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    colors = {"Reference edges": "#e6550d", "Unobserved pairs": "#1f77b4"}
    for ax, (group, frame) in zip(axes[:2], all_cells.groupby("panel", sort=False)):
        x, y = frame.rms_context_deviation_cell_context_free_pp, frame.rms_context_deviation_protscape_pp
        limit = 1.05 * max(x.max(), y.max())
        ax.plot([0, limit], [0, limit], "--", color="0.6", lw=1)
        ax.scatter(x, y, s=16, color=colors[group], alpha=.65, edgecolors="none")
        ax.set(xlabel="CF context deviation (pp)", ylabel="ProtScape context deviation (pp)",
               title=group + " — 207 contexts", xlim=(0, limit), ylim=(0, limit))
    for group, values in pair_correlations.items():
        axes[2].hist(values, bins=np.linspace(-1, 1, 41), histtype="step", lw=1.5,
                     weights=np.full(len(values), 100 / len(values)), color=colors[group], label=group)
    axes[2].set(xlabel="Same-pair context Spearman", ylabel="Pairs (%)", title="Agreement in context changes")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(root / "context_comparison.png", dpi=180)
    fig.savefig(root / "context_comparison.pdf")
    plt.close(fig)
    (root / "COMPARISON_COMPLETE.json").write_text(json.dumps({"reference_scores": str(reference_root),
                                                               "unobserved_scores": str(root)}, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["evaluate", "summarize", "report", "prepare-all", "summarize-all", "finish-all"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--shard", type=int, choices=range(4))
    parser.add_argument("--panel", choices=["shared_edges", "unobserved", "all_unobserved"], default="shared_edges")
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args()
    if args.stage == "evaluate":
        if args.shard is None:
            parser.error("--shard is required")
        if args.panel == "all_unobserved":
            evaluate_exhaustive(args.root, args.shard)
        else:
            evaluate(args.root, args.shard, args.panel)
    elif args.stage == "prepare-all":
        prepare_exhaustive(args.root)
    elif args.stage == "summarize-all":
        if args.shard is None:
            parser.error("--shard is required")
        summarize_exhaustive(args.root, args.shard)
    elif args.stage == "finish-all":
        finish_exhaustive(args.root)
    elif args.stage == "summarize":
        summarize(args.root)
    else:
        if args.reference_root is None:
            parser.error("--reference-root is required for the comparison report")
        report_panels(args.root, args.reference_root)
