import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pretraining.models.partner_pooling import PartnerResiduePooler, PARTNER_MODES, neighbour_rows
from pretraining.models.residue_pooling import ResiduePooler
from pretraining.global_s2gae import GlobalS2GAE, masked_reconstruction_step, protocol_metadata
from pretraining.s2gae_utils import edge_mask_per_graph


@pytest.fixture
def cache(tmp_path):
    rng = np.random.default_rng(12)
    values = rng.normal(size=(18, 6)).astype(np.float16)
    np.save(tmp_path / "residues.npy", values)
    np.save(tmp_path / "offsets.npy", [0, 2, 7, 10, 18])
    np.savez(tmp_path / "normalization.npz", mean=np.zeros(6, np.float32), std=np.ones(6, np.float32))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(genes=["A", "B", "C", "D"], embedding_dim=6)))
    (tmp_path / "COMPLETE.json").write_text("{}")
    return tmp_path


def inputs(cache, device="cpu"):
    values = torch.from_numpy(np.load(cache / "residues.npy")).float().to(device)
    means = torch.stack([x.mean(0) for x in torch.split(values, [2, 5, 3, 8])])
    edges = torch.tensor([[0, 1, 0, 2, 1, 2, 0, 1, 2, 3], [1, 0, 2, 0, 2, 1, 0, 1, 2, 3]], device=device)
    return torch.arange(4, device=device), edges, means


@pytest.mark.parametrize("mode", sorted(PARTNER_MODES))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initialization_gradient_reload_and_isolates(cache, mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = PartnerResiduePooler(cache, mode, hidden_dim=4, residue_budget=10).to(device)
    ids, edges, means = inputs(cache, device)
    actual = model(ids, edges, means)
    torch.testing.assert_close(actual, means)
    loss = (actual * torch.randn_like(actual)).sum()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    key = model.slot_output if mode == "pma4" else model.output
    assert key.weight.grad.abs().sum() > 0
    if mode in {"partner", "self_query", "partner_slots", "partner_dispersion", "partner_mean_mlp"}:
        assert model.query.weight.grad.abs().sum() > 0
    torch.optim.Adam(model.parameters(), lr=.02).step()
    trained = model.eval()(ids, edges, means)
    restored = PartnerResiduePooler(cache, mode, hidden_dim=4, residue_budget=10).to(device).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(ids, edges, means), trained)
    assert restored._evaluation_values is None  # no graph-derived inference cache
    if mode not in {"self_query", "pma4"}:
        baseline = ResiduePooler(cache, "attention", hidden_dim=4).to(device).eval()
        baseline.load_state_dict({k: v for k, v in model.state_dict().items() if k in baseline.state_dict()})
        torch.testing.assert_close(trained[3], baseline(ids)[3])


def test_neighbours_ignore_loops_duplicates_and_hidden_pairs():
    edges = torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 1, 0, 1, 2]])
    rows = neighbour_rows(edges, 3)
    assert [r.tolist() for r in rows] == [[1], [0], []]
    full = torch.tensor([[a, b] for a in range(20) for b in range(20) if a != b]).T
    sampled = neighbour_rows(full, 20, cap=5, seed=3)
    assert all(len(set(row)) == 5 for row in sampled)
    assert all(i not in row for i, row in enumerate(sampled))
    assert all(np.array_equal(a, b) for a, b in zip(sampled, neighbour_rows(full.flip(1), 20, 5, 3)))
    np.random.seed(0)
    visible, targets, _, _ = edge_mask_per_graph(full, 20, .5, "cpu", mask_type="um")
    rows = neighbour_rows(visible, 20)
    assert all(b not in rows[a] and a not in rows[b] for a, b in targets.T.tolist())


@pytest.mark.parametrize("mode", sorted(PARTNER_MODES))
def test_permutation_and_query_chunk_invariance(cache, mode):
    model = PartnerResiduePooler(cache, mode, hidden_dim=4, query_chunk_size=1).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, .1)
    ids, edges, means = inputs(cache)
    actual = model(ids, edges, means)
    model.query_chunk_size = 32
    torch.testing.assert_close(model(ids, edges, means), actual, atol=1e-6, rtol=1e-5)
    order = torch.tensor([3, 2, 0, 1])
    inverse = torch.argsort(order)
    permuted = model(ids[order], inverse[edges], means[order])
    torch.testing.assert_close(permuted[inverse], actual, atol=1e-6, rtol=1e-5)


def test_partner_identity_matters_and_self_control_ignores_graph(cache):
    ids, edges, means = inputs(cache)
    for mode in ["partner", "self_query", "pma4"]:
        model = PartnerResiduePooler(cache, mode, hidden_dim=4).eval()
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, .4)
        first = model(ids, edges, means)
        changed = edges.clone()
        changed[:, :2] = torch.tensor([[0, 3], [3, 0]])
        second = model(ids, changed, means)
        if mode == "partner":
            assert not torch.allclose(first, second)
        else:
            torch.testing.assert_close(first, second)


def test_dispersion_is_over_partner_views_with_matched_capacity(cache):
    dispersion = PartnerResiduePooler(cache, "partner_dispersion", hidden_dim=4)
    control = PartnerResiduePooler(cache, "partner_mean_mlp", hidden_dim=4)
    assert {k: tuple(v.shape) for k, v in dispersion.state_dict().items()} == {k: tuple(v.shape) for k, v in control.state_dict().items()}
    ids, _, _ = inputs(cache)
    with torch.no_grad():
        dispersion.query.weight.normal_(0, .3)
    seen = []
    handle = dispersion.residual.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    queries = torch.randn(1, 2, 4)
    dispersion._load_residues()
    dispersion._pool_batch(ids[:1], queries, torch.ones(1, 2, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    values, _ = dispersion._values(ids[:1])
    gate = dispersion.output(torch.tanh(dispersion.V(values)) * torch.sigmoid(dispersion.U(values))).squeeze(-1)
    logits = gate[:, None] + queries @ dispersion.key(dispersion.norm(values)).transpose(1, 2) / 2
    views = logits.softmax(-1) @ values
    expected = (views.var(1, correction=0) + 1e-8).sqrt() - 1e-4
    torch.testing.assert_close(seen[-1], expected, atol=1e-6, rtol=1e-4)
    handle.remove()


def test_private_sampling_stream_resumes_and_does_not_change_torch_rng(cache):
    model = PartnerResiduePooler(cache, "partner", hidden_dim=4, max_partners=1)
    ids, edges, means = inputs(cache)
    rng = torch.get_rng_state().clone()
    model(ids, edges, means)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    restored = PartnerResiduePooler(cache, "partner", hidden_dim=4, max_partners=1)
    restored.load_state_dict(state)
    torch.testing.assert_close(model(ids, edges, means), restored(ids, edges, means))


def test_global_partner_model_rejects_dm_and_records_pair_protocol(cache):
    from types import SimpleNamespace
    model = GlobalS2GAE(6, hidden_dim=4, num_layers=2, decode_channels=4,
                       residue_pooling=dict(cache_root=str(cache), mode="partner", hidden_dim=4), residue_ids=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="pairwise-hidden"):
        masked_reconstruction_step(model, SimpleNamespace(), device=torch.device("cpu"), mask_type="dm")
    assert protocol_metadata("um")["primary_mask_type"] == "um"
    assert protocol_metadata()["primary_mask_type"] == "dm"


def test_separate_panel_grid_and_pairwise_commands(tmp_path):
    from scripts.cscs.run_cf_residue_ablation import REPO, trial_grid, train_command
    config = yaml.safe_load((REPO / "configs/cf_partner_ablation.yaml").read_text())
    trials = trial_grid(config["pretraining"])
    assert len(trials) == 66 and len({t["name"] for t in trials}) == 66
    assert len(config["pretraining"]["modes"]) == 9
    (tmp_path / "source/configs").mkdir(parents=True)
    (tmp_path / "source/configs/cf_partner_ablation.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "inputs.json").write_text(json.dumps(dict(release="/release", config_file="cf_partner_ablation.yaml")))
    for trial in trials:
        args = train_command(tmp_path, trial)
        assert args[args.index("--mask-type") + 1] == "um"
        assert args[args.index("--wandb-group") + 1] == "cf_partner_pooling_pairmasked"
        assert args[args.index("--epochs") + 1] == "5000"
