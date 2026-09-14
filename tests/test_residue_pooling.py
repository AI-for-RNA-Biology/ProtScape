import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from pretraining.cache_esm2_residues import extract_residues
from pretraining.models.residue_pooling import ResiduePooler, attach_residue_ids


@pytest.fixture
def cache(tmp_path):
    values = np.arange(28, dtype=np.float16).reshape(7, 4) / 10
    np.save(tmp_path / "residues.npy", values)
    np.save(tmp_path / "offsets.npy", [0, 2, 5, 7])
    np.savez(tmp_path / "normalization.npz", mean=np.ones(4, np.float32), std=np.full(4, 2, np.float32))
    (tmp_path / "manifest.json").write_text(json.dumps({"genes": ["A", "B", "C"], "embedding_dim": 4}))
    (tmp_path / "COMPLETE.json").write_text("{}")
    return tmp_path


@pytest.mark.parametrize("mode", ["mlp", "attention"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mean_initialization_dedup_gradient_and_reload(cache, mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pooler = ResiduePooler(cache, mode, hidden_dim=3, residue_budget=3).to(device)
    values = torch.from_numpy(np.load(cache / "residues.npy")).float().to(device)
    expected = torch.stack([values[:2].mean(0), values[2:5].mean(0), values[5:].mean(0)])
    expected = (expected - 1) / 2
    ids = torch.tensor([2, 0, 1, 0], device=device)
    output = pooler(ids)
    torch.testing.assert_close(output, expected[ids])
    output.square().sum().backward()
    assert pooler.output.weight.grad.abs().sum() > 0
    optimizer = torch.optim.Adam(pooler.parameters(), lr=.01)
    optimizer.step()
    with torch.no_grad():
        trained = pooler(ids)
    assert not torch.allclose(trained, expected[ids])
    pooler.eval()
    torch.testing.assert_close(pooler(ids), trained)
    torch.testing.assert_close(pooler(ids), trained)  # evaluation cache
    restored = ResiduePooler(cache, mode, hidden_dim=3).to(device).eval()
    restored(ids)  # ensure loading invalidates a populated cache
    restored.load_state_dict(pooler.state_dict())
    torch.testing.assert_close(restored(ids), trained)
    torch.save(pooler, cache / "model.pt")
    loaded = torch.load(cache / "model.pt", weights_only=False)
    assert loaded._residues is None and loaded._evaluation_values is None
    torch.testing.assert_close(loaded(ids), trained)


def test_node_mapping_and_batch_offsets(cache):
    graphs = {7: Data(x=torch.zeros(2, 4)), 8: Data(x=torch.zeros(2, 4))}
    layers = {"one": SimpleNamespace(nodes=lambda: ["C", "A"]),
              "two": SimpleNamespace(nodes=lambda: ["A", "B"])}
    attach_residue_ids(graphs, layers, {"one": 7, "two": 8}, cache)
    batch = Batch.from_data_list(list(graphs.values()))
    assert batch.residue_id.tolist() == [2, 0, 0, 1]
    pooler = ResiduePooler(cache, "attention")
    torch.testing.assert_close(pooler(batch.residue_id)[1], pooler(batch.residue_id)[2])


def test_extraction_excludes_special_tokens_and_keeps_long_sequence():
    def converter(items):
        length = len(items[0][1])
        return None, None, torch.arange(length + 2)[None]

    def model(tokens, **kwargs):
        return {"representations": {33: tokens[:, :, None].float()}}

    actual = extract_residues("A" * 1030, model, converter, "cpu")
    np.testing.assert_array_equal(actual[:, 0], np.r_[np.arange(1, 1025), np.arange(1, 7)])


def test_backbone_initialization_is_unchanged(cache):
    from pretraining.models.protein_modules import prot_module
    config = dict(input_dim=4, gnn_method="ACM_RandomWalk", hidden_dim=8, n_layers=2,
                  gnn_dropout=0, gnn_batchnorm=True, gnn_activation="leaky_relu", jumping_knowledge="concat")
    torch.manual_seed(42)
    baseline = prot_module(config, "cpu")
    after_baseline = torch.rand(5)
    torch.manual_seed(42)
    pooled = prot_module(dict(config, residue_pooling=dict(mode="attention", hidden_dim=3, cache_root=str(cache))), "cpu")
    torch.testing.assert_close(torch.rand(5), after_baseline)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(pooled.state_dict()[key], value)
    graph = Data(x=ResiduePooler(cache, "attention")(torch.arange(3)).detach(),
                 residue_id=torch.arange(3), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
    torch.testing.assert_close(baseline.eval()({0: graph.clone()})[0], pooled.eval()({0: graph.clone()})[0])


def test_graphsaint_preserves_cache_ids(cache):
    from torch_geometric.loader import GraphSAINTEdgeSampler
    graph = Data(x=torch.ones(3, 4), residue_id=torch.tensor([2, 0, 1]), n_id=torch.arange(3),
                 edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]))
    sample = next(iter(GraphSAINTEdgeSampler(graph, batch_size=4, num_steps=1)))
    torch.testing.assert_close(sample.residue_id, graph.residue_id[sample.n_id])


def test_sweep_only_varies_pooling_and_learning_rates():
    import yaml
    from scripts.cscs.run_residue_ablation import grid, readouts, REPO
    config = yaml.safe_load((REPO / "configs/residue_ablation.yaml").read_text())
    trials = grid(config["pretraining"])
    assert len(trials) == 18 and len({t["name"] for t in trials}) == 18
    assert {t["mode"] for t in trials} == {"mean", "mlp", "attention"}
    assert len(list(readouts(config["downstream"]))) == 22
    assert config["pretraining"]["seed"] == 0


def test_cache_preserves_isoforms_gaps_and_gnn_normalization(tmp_path):
    import pandas as pd
    from pretraining.cache_esm2_residues import prepare, finalize
    release, root = tmp_path / "release", tmp_path / "cache"
    for folder in ["raw", "networks_bulk", "sequence_embeddings"]:
        (release / "data" / folder).mkdir(parents=True)
    pd.DataFrame({"gene_name": ["A", "B", "A"], "fasta_seq": ["AA", ".A", "AAA"]}).to_csv(
        release / "data/raw/protein_sequences.csv.gz", index=False)
    pd.DataFrame({"gene_name": ["A", "B", "A"], "ESM2-Embeddings": [np.zeros(2560)] * 3}).to_pickle(
        release / "data/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk")
    (release / "data/networks_bulk/global_ppi_edgelist.txt").write_text("A B\n")
    prepare(root, release)
    data = np.load(root / "residues.npy", mmap_mode="r+")
    data[:] = np.arange(7, dtype=np.float16)[:, None]
    data.flush()
    for index in range(3):
        (root / "done" / f"{index}.json").write_text("{}")
    finalize(root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["genes"] == ["A", "B", "A"]
    assert manifest["gnn_indices"] == [0, 1]
    assert manifest["sequences"] == ["AA", ".A", "AAA"]
    features = pd.read_pickle(root / "mean.plk")
    assert [v[0] for v in features["ESM2-Embeddings"]] == [.5, 2.5, 5.]
    stats = np.load(root / "normalization.npz")
    np.testing.assert_allclose(stats["mean"], 1.5)
    np.testing.assert_allclose(stats["std"], np.sqrt(2))
    graphs = {0: Data(x=torch.zeros(2, 2560))}
    attach_residue_ids(graphs, {"cell": SimpleNamespace(nodes=lambda: ["B", "A"])}, {"cell": 0}, root)
    assert graphs[0].residue_id.tolist() == [1, 0]
