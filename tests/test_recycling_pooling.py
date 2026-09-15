import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from pretraining.global_s2gae import GlobalS2GAE, masked_reconstruction_step
from pretraining.models.recycling_pooling import RecyclingResiduePooler, RECYCLING_MODES


@pytest.fixture
def cache(tmp_path):
    rng = np.random.default_rng(9)
    np.save(tmp_path / "residues.npy", rng.normal(size=(20, 6)).astype(np.float16))
    np.save(tmp_path / "offsets.npy", [0, 1, 5, 11, 20])
    np.savez(tmp_path / "normalization.npz", mean=np.zeros(6, np.float32), std=np.ones(6, np.float32))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(genes=["a", "b", "c", "d"], embedding_dim=6)))
    (tmp_path / "COMPLETE.json").write_text("{}")
    return tmp_path


def model_inputs(cache, mode, device="cpu"):
    torch.manual_seed(2)
    model = GlobalS2GAE(6, hidden_dim=8, num_layers=2, dropout=0, decode_channels=8,
        residue_ids=[0, 1, 2, 3], residue_pooling=dict(cache_root=str(cache), mode=mode,
                                                  hidden_dim=4, residue_budget=12)).to(device)
    values = torch.from_numpy(np.load(cache / "residues.npy")).float().to(device)
    means = torch.stack([r.mean(0) for r in values.split([1, 4, 6, 9])])
    edges = torch.tensor([[0, 1, 1, 2, 0, 1, 2, 3], [1, 0, 2, 1, 0, 1, 2, 3]], device=device)
    return model, means, edges


@pytest.mark.parametrize("mode", sorted(RECYCLING_MODES))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_graph_passes_gradients_and_reload(cache, mode, device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, means, edges = model_inputs(cache, mode, device)
    pool = model.residue_pooler
    torch.testing.assert_close(pool.protein_features(pool.read_residues(model.residue_ids), means), means)
    # Exercise the trained regime: zero output initialization deliberately
    # delays upstream gradients until the output projection takes a first step.
    torch.nn.init.normal_(pool.output_projection.weight, std=.1)
    if mode != "slots4":
        torch.nn.init.normal_(pool.query_update[-1].weight, std=.1)
    passes, reads = [], []
    original_graph, original_read = model._encode_graph, pool.read_residues
    def graph(features, adjacency):
        out = original_graph(features, adjacency)
        out[0].retain_grad()
        passes.append((adjacency, out))
        return out
    def read(*args):
        reads.append(True)
        return original_read(*args)
    monkeypatch.setattr(model, "_encode_graph", graph)
    monkeypatch.setattr(pool, "read_residues", read)
    embeddings, layers = model.encode(means, edges)
    assert len(passes) == (1 if mode == "slots4" else 2)
    assert all(adjacency is edges for adjacency, _ in passes)
    assert layers is passes[-1][1][1]  # preserve decoder cross-layer inputs
    assert len(reads) == (1 if mode in {"slots4", "recycle_memory"} else 2)
    logits = model.score_edges(layers, torch.tensor([[0, 0], [1, 3]], device=device))
    logits.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert pool.input_projection.weight.grad.abs().sum() > 0
    assert pool.queries.grad.abs().sum() > 0
    if mode in {"recycle_memory", "recycle_residues"}:
        assert pool.graph_projection.weight.grad.abs().sum() > 0
        assert passes[0][1][0].grad.abs().sum() > 0
    elif mode == "recycle_no_graph":
        assert pool.graph_projection.weight.grad is None
        assert passes[0][1][0].grad is None
        assert pool.query_update[-1].weight.grad.abs().sum() > 0
    monkeypatch.setattr(model, "_encode_graph", original_graph)
    monkeypatch.setattr(pool, "read_residues", original_read)
    model.eval()
    restored = GlobalS2GAE(**model.model_config).to(device).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.encode(means, edges)[0], model.encode(means, edges)[0])
    assert restored.residue_pooler._evaluation_values is None
    with pytest.raises(ValueError, match="pairwise-hidden"):
        masked_reconstruction_step(model, SimpleNamespace(), device=torch.device(device), mask_type="dm")


def test_attention_matches_manual_all_residues_and_chunking(cache):
    pool = RecyclingResiduePooler(cache, "slots4", graph_dim=16, hidden_dim=4, residue_budget=12)
    ids = torch.arange(4)
    actual = pool.read_residues(ids)
    values = torch.from_numpy(np.load(cache / "residues.npy")).float()
    expected = torch.cat([pool.attend(pool.queries[None], pool.input_projection(r)[None]) for r in values.split([1, 4, 6, 9])])
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    gradient = pool.input_projection.weight.grad.clone()
    pool.zero_grad()
    pool.residue_budget = 1000
    unchunked = pool.read_residues(ids)
    unchunked.square().sum().backward()
    torch.testing.assert_close(actual, unchunked)
    torch.testing.assert_close(pool.input_projection.weight.grad, gradient)


def test_controls_match_parameters_and_no_graph_really_ignores_feedback(cache):
    pools = [model_inputs(cache, m)[0].residue_pooler for m in ["recycle_memory", "recycle_residues", "recycle_no_graph"]]
    assert all([(k, tuple(v.shape)) for k, v in p.state_dict().items()] == [(k, tuple(v.shape)) for k, v in pools[0].state_dict().items()] for p in pools)
    for p in pools:
        torch.nn.init.normal_(p.query_update[-1].weight, std=.2)
    pools[2].load_state_dict(pools[1].state_dict())
    ids = torch.arange(4)
    slots = pools[1].read_residues(ids)
    a, b = torch.randn(4, 16), torch.randn(4, 16)
    assert not torch.allclose(pools[1].refine(ids, slots, a), pools[1].refine(ids, slots, b))
    torch.testing.assert_close(pools[2].refine(ids, slots, a), pools[2].refine(ids, slots, b))


@pytest.mark.parametrize("mode", sorted(RECYCLING_MODES))
def test_local_ids_node_permutation_and_no_graph_cache(cache, mode):
    model, means, edges = model_inputs(cache, mode)
    model.eval()
    torch.nn.init.normal_(model.residue_pooler.output_projection.weight, std=.2)
    if mode != "slots4":
        torch.nn.init.normal_(model.residue_pooler.query_update[-1].weight, std=.2)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = permutation.argsort()
    original = model.encode(means, edges)[0]
    reordered = model.encode(means[permutation], inverse[edges], permutation)[0]
    torch.testing.assert_close(reordered, original[permutation])
    subgraph = torch.tensor([[0, 1], [1, 0]])
    actual = model.encode(means[permutation[:2]], subgraph, permutation[:2])[0]
    assert actual.shape == (2, 16)
    assert model.residue_pooler._evaluation_values is None


def test_recycling_grid_and_paired_seed_commands(tmp_path):
    from scripts.cscs.run_cf_residue_ablation import REPO, trial_grid, train_command
    config = yaml.safe_load((REPO / "configs/cf_recycling_ablation.yaml").read_text())
    trials = trial_grid(config["pretraining"])
    assert len(trials) == len({t["name"] for t in trials}) == 30
    assert len({t["mode"] for t in trials}) == 8
    (tmp_path / "source/configs").mkdir(parents=True)
    (tmp_path / "source/configs/cf_recycling_ablation.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "inputs.json").write_text(json.dumps(dict(release="/release", config_file="cf_recycling_ablation.yaml")))
    for trial in trials:
        if trial["mode"] in RECYCLING_MODES:
            assert trial["hidden_dim"] == 128
        for seed in [0, 1, 2]:
            args = train_command(tmp_path, {**trial, "seed": seed})
            for flag, expected in {"--seed": str(seed), "--split-seed": "0", "--mask-type": "um", "--epochs": "5000", "--wandb-mode": "online"}.items():
                assert args[args.index(flag)+1] == expected


def test_masked_targets_are_hidden_in_both_passes(cache, monkeypatch):
    import pretraining.global_s2gae as module
    model, means, edges = model_inputs(cache, "recycle_residues")
    edges = edges[:, edges[0] != edges[1]]
    data = SimpleNamespace(features=means, train_edge_index=edges, all_edge_index=edges)
    passes, targets = [], []
    original_graph, original_loss = model._encode_graph, module.s2gae_loss
    def graph(features, adjacency):
        passes.append(adjacency)
        return original_graph(features, adjacency)
    def loss(decoder, layers, positive, negative):
        targets.append(positive)
        return original_loss(decoder, layers, positive, negative)
    monkeypatch.setattr(model, "_encode_graph", graph)
    monkeypatch.setattr(module, "s2gae_loss", loss)
    masked_reconstruction_step(model, data, device=torch.device("cpu"), mask_type="um", validate_targets=True)
    assert len(passes) == 2 and passes[0] is passes[1]
    visible = {tuple(sorted(e)) for e in passes[0].T.tolist()}
    assert all(tuple(sorted(e)) not in visible for e in targets[0].T.tolist())


def test_paired_reporting_uses_matching_seeds():
    import pandas as pd
    from scripts.cscs.run_cf_residue_ablation import paired_differences
    frame = pd.DataFrame([dict(pooling=m, seed=s, task="corum", auprc=v+s)
                          for s in [0, 1, 2] for m, v in [("recycle_memory", 60), ("recycle_residues", 62), ("recycle_no_graph", 61)]])
    result = paired_differences(frame, ["task"], "auprc")
    assert len(result) == 3
    assert (result.C_minus_B == 2).all() and (result.C_minus_D == 1).all()


def test_final_test_wandb_keys_are_unambiguous(monkeypatch):
    import wandb
    from scripts.cscs.run_cf_residue_ablation import log_test_summary
    saved = SimpleNamespace(summary={}, update=lambda: None)
    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(run=lambda path: saved))
    checkpoint = dict(wandb=dict(entity="e", project="p", run_id="r"), epoch=9)
    result = dict(global_unique_test={1: dict(ap=.9, f1=.8)}, rows=[
        dict(inference=i, k=1, auprc=.7, f1=.6) for i in ["global", "cell"]])
    log_test_summary(checkpoint, result)
    assert saved.summary["test/global_unique/ap_k1"] == .9
    assert saved.summary["test/cell_macro_cell_encoding/auprc_k1"] == .7
    assert saved.summary["test/cell_macro_global_encoding/f1_k1"] == .6
    assert saved.summary["test/checkpoint_update"] == 10
