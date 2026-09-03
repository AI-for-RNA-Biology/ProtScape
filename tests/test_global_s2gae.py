import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch_geometric.data import Data

import pretraining.global_s2gae as global_s2gae
from pretraining.evaluate_global_s2gae import (
    _score_context_edges,
    _validate_complete_summary,
    contextwise_protocol_metadata,
    evaluate_contextwise_edges,
    select_checkpoint,
)
from pretraining.global_s2gae import (
    GlobalPPIData,
    GlobalS2GAE,
    _assign_context_splits,
    _canonical_keys,
    evaluate_global_edges,
    masked_reconstruction_step,
    protocol_metadata,
    sample_negative_bank,
    validate_masked_targets,
)
from pretraining.train_global_s2gae import (
    capture_rng_state,
    checkpoint_payload,
    restore_rng_state,
    validate_archived_best,
    validate_completed_extension,
)


def symmetric(edge_index):
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def tiny_data() -> GlobalPPIData:
    unique_edges = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 3, 4],
            [1, 2, 3, 2, 3, 4, 3, 4, 4, 5],
        ],
        dtype=torch.long,
    )
    train = unique_edges[:, :6]
    val = unique_edges[:, 6:8]
    test = unique_edges[:, 8:]
    features = torch.randn(6, 4)
    return GlobalPPIData(
        features=features,
        feature_mean=torch.zeros(4),
        feature_std=torch.ones(4),
        protein_names=[f"p{index}" for index in range(6)],
        all_edge_index=symmetric(unique_edges),
        train_edge_index=symmetric(train),
        train_val_edge_index=symmetric(torch.cat([train, val], dim=1)),
        val_edge_index=val,
        test_edge_index=test,
        source_context_count=2,
        graph_fingerprint="graph",
        feature_fingerprint="features",
        split_fingerprint="split",
        split_counts={
            "reference_only_train": 1,
            "shared_train": 5,
            "shared_val": 2,
            "shared_test": 2,
        },
    )


def tiny_model() -> GlobalS2GAE:
    return GlobalS2GAE(
        4,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        decode_channels=8,
        decoder_layers=2,
        decoder_dropout=0.0,
    )


def test_protocol_is_global_only_v3():
    protocol = protocol_metadata()
    assert protocol["version"] == "global_s2gae_unique_ppi_v3"
    assert protocol["selection_metric"] == "global_val_ap"
    assert protocol["evaluation_unit"] == "unique_undirected_global_pair"
    assert protocol["negative_scope"] == "global_reference_nonedge"
    assert protocol["validation_topology"] == "train"
    assert protocol["test_topology"] == "train_plus_validation"
    assert not any("context_macro" in key for key in protocol)


def test_final_evaluation_requires_every_reported_k():
    multiplicity = {
        "directed_positive_occurrences_scored": 2,
        "unique_pair_context_occurrences": 2,
        "unique_global_test_pairs_observed": 1,
        "unique_global_test_pairs_total": 1,
        "pairs_observed_in_multiple_contexts": 1,
        "fraction_observed_pairs_in_multiple_contexts": 1.0,
        "mean_contexts_per_observed_pair": 2.0,
        "median_contexts_per_observed_pair": 2.0,
        "max_contexts_per_observed_pair": 2,
    }
    summary = {
        "protocol": protocol_metadata(),
        "global_test": {str(k): {} for k in (1, 10, 50, 100, 500)},
        "context_ppi_protocol": contextwise_protocol_metadata(),
        "context_ppi_macro_test": {
            str(k): {} for k in (1, 10, 50, 100, 500)
        },
        "context_ppi_multiplicity": multiplicity,
    }
    _validate_complete_summary(summary)
    summary["global_test"].pop("500")
    with pytest.raises(ValueError, match="every required K"):
        _validate_complete_summary(summary)


def test_final_evaluation_requires_context_specific_results():
    summary = {
        "protocol": protocol_metadata(),
        "global_test": {str(k): {} for k in (1, 10, 50, 100, 500)},
        "context_ppi_protocol": contextwise_protocol_metadata(),
        "context_ppi_macro_test": {
            str(k): {} for k in (1, 10, 50, 100, 500)
        },
        "context_ppi_multiplicity": {
            "directed_positive_occurrences_scored": 2,
            "unique_pair_context_occurrences": 2,
            "unique_global_test_pairs_observed": 1,
            "unique_global_test_pairs_total": 1,
            "pairs_observed_in_multiple_contexts": 1,
            "fraction_observed_pairs_in_multiple_contexts": 1.0,
            "mean_contexts_per_observed_pair": 2.0,
            "median_contexts_per_observed_pair": 2.0,
            "max_contexts_per_observed_pair": 2,
        },
    }
    summary["context_ppi_macro_test"].pop("500")
    with pytest.raises(ValueError, match="context-specific K"):
        _validate_complete_summary(summary)


def test_context_splits_are_assigned_by_undirected_pair():
    unique_edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    context = Data(
        edge_index=symmetric(torch.tensor([[0, 1], [1, 2]], dtype=torch.long)),
        feature_index=torch.tensor([0, 1, 2]),
        train_mask=torch.tensor([True, False, True, False]),
        val_mask=torch.tensor([False, True, False, True]),
        test_mask=torch.zeros(4, dtype=torch.bool),
    )
    assert _assign_context_splits(unique_edges, {0: context}, 4).tolist() == [0, 1, -1]


def test_loader_uses_contexts_only_for_the_global_split(monkeypatch, tmp_path):
    networks_dir = tmp_path / "networks"
    (networks_dir / "ppi_edgelists").mkdir(parents=True)
    (networks_dir / "global_ppi_edgelist.txt").write_text(
        "p0 p1\np1 p2\np2 p3\np3 p4\n", encoding="utf-8"
    )
    (networks_dir / "mg_edgelist.txt").touch()
    embeddings_path = tmp_path / "esm2.pt"
    embeddings_path.touch()
    with (networks_dir / "count_edge_dict.pkl").open("wb") as handle:
        pickle.dump(
            {("p0", "p1"): 2, ("p1", "p2"): 1, ("p2", "p3"): 1},
            handle,
        )

    context_edges = symmetric(
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    )
    first_context = Data(
        edge_index=context_edges,
        feature_index=torch.arange(5),
        train_mask=torch.tensor([True, False, False, True, False, False]),
        val_mask=torch.tensor([False, True, False, False, True, False]),
        test_mask=torch.tensor([False, False, True, False, False, True]),
    )
    repeated_train = Data(
        edge_index=symmetric(torch.tensor([[0], [1]], dtype=torch.long)),
        feature_index=torch.arange(5),
        train_mask=torch.ones(2, dtype=torch.bool),
        val_mask=torch.zeros(2, dtype=torch.bool),
        test_mask=torch.zeros(2, dtype=torch.bool),
    )
    mg_data = SimpleNamespace(
        global_protein_names=[f"p{index}" for index in range(5)],
        global_protein_features=torch.arange(20, dtype=torch.float32).reshape(5, 4),
        global_feature_mean=torch.zeros(4),
        global_feature_std=torch.ones(4),
    )

    def fake_read_data(*args, **kwargs):
        return (
            {0: first_context, 1: repeated_train},
            mg_data,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    monkeypatch.setattr(global_s2gae, "read_data", fake_read_data)
    data = global_s2gae.load_global_ppi_data(
        networks_dir, embeddings_path, seed=0, verbose=False
    )

    assert data.source_context_count == 2
    assert data.split_counts == {
        "reference_only_train": 1,
        "shared_train": 1,
        "shared_val": 1,
        "shared_test": 1,
    }
    assert data.all_edge_index.size(1) == 8
    assert data.train_edge_index.size(1) == 4
    assert data.train_val_edge_index.size(1) == 6
    assert data.val_edge_index.size(1) == 1
    assert data.test_edge_index.size(1) == 1
    assert not hasattr(data, "contexts")


def test_pairwise_validation_distinguishes_dm_and_um():
    message_edges = torch.tensor([[1], [0]])
    positive_edges = torch.tensor([[0], [1]])
    negative_edges = torch.tensor([[0], [2]])
    all_positive = symmetric(positive_edges)

    validate_masked_targets(
        message_edges,
        positive_edges,
        negative_edges,
        all_positive,
        3,
        require_pairwise_hidden=False,
    )
    with pytest.raises(ValueError, match="masked interaction"):
        validate_masked_targets(
            message_edges,
            positive_edges,
            negative_edges,
            all_positive,
            3,
            require_pairwise_hidden=True,
        )


@pytest.mark.parametrize("mask_type", ["dm", "um"])
def test_masked_step_has_only_encoder_decoder_and_valid_targets(mask_type):
    np.random.seed(0)
    torch.manual_seed(0)
    data = tiny_data()
    model = tiny_model()
    loss, logits, labels = masked_reconstruction_step(
        model,
        data,
        device=torch.device("cpu"),
        mask_ratio=0.5,
        mask_type=mask_type,
        validate_targets=True,
    )
    assert torch.isfinite(loss)
    assert logits.shape == labels.shape
    assert set(name.split(".", 1)[0] for name, _ in model.named_parameters()) == {
        "encoder",
        "decoder",
    }
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_negative_bank_is_deterministic_and_excludes_global_positives():
    data = tiny_data()
    first = sample_negative_bank(
        data.val_edge_index, data.all_edge_index, 6, 500, seed=7, return_k=1
    )
    second = sample_negative_bank(
        data.val_edge_index, data.all_edge_index, 6, 500, seed=7, return_k=1
    )
    assert torch.equal(first, second)
    assert not (first[0] == first[1]).any()
    assert not torch.isin(
        _canonical_keys(first, 6),
        torch.unique(_canonical_keys(data.all_edge_index, 6)),
    ).any()


def test_negative_bank_first_column_is_independent_of_requested_k():
    data = tiny_data()
    first = sample_negative_bank(
        data.val_edge_index, data.all_edge_index, 6, 500, seed=7, return_k=1
    )
    full = sample_negative_bank(
        data.val_edge_index, data.all_edge_index, 6, 500, seed=7
    ).reshape(2, data.val_edge_index.size(1), 500)
    assert torch.equal(first, full[:, :, :1].reshape(2, -1))


class RecordingModel:
    def __init__(self):
        self.encoded_edges = []
        self.scored_edges = []

    def eval(self):
        return self

    def encode(self, features, edge_index):
        self.encoded_edges.append(edge_index.detach().cpu().clone())
        return features, [features]

    def score_edges(self, layer_embeddings, edge_index):
        self.scored_edges.append(edge_index.detach().cpu().clone())
        return torch.zeros(edge_index.size(1), device=edge_index.device)


def test_context_score_cache_preserves_order_and_uses_undirected_uniques():
    class SymmetricModel:
        def __init__(self):
            self.scored_edges = []

        def score_edges(self, _layers, edge_index):
            self.scored_edges.append(edge_index.detach().cpu())
            return edge_index.sum(dim=0).float()

    model = SymmetricModel()
    edges = torch.tensor([[1, 2, 1], [2, 1, 4]], dtype=torch.long)

    scores = _score_context_edges(model, [], edges, torch.device("cpu"))

    assert len(model.scored_edges) == 1
    assert model.scored_edges[0].size(1) == 2
    np.testing.assert_allclose(
        scores,
        torch.sigmoid(torch.tensor([3.0, 3.0, 5.0])).numpy(),
    )


def test_contextwise_evaluation_counts_repeated_pair_occurrences():
    data = tiny_data()
    repeated_test = torch.tensor([[3], [4]], dtype=torch.long)
    contexts = {
        11: Data(
            edge_index=torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long),
            test_mask=torch.tensor([False, False, True]),
            feature_index=torch.arange(6),
            num_nodes=6,
        ),
        12: Data(
            edge_index=torch.tensor(
                [[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long
            ),
            test_mask=torch.tensor([False, False, True, True]),
            feature_index=torch.arange(6),
            num_nodes=6,
        ),
    }
    assert torch.equal(
        contexts[11].edge_index[:, contexts[11].test_mask], repeated_test
    )
    model = RecordingModel()

    metrics, rows, diagnostics, multiplicity_rows = evaluate_contextwise_edges(
        model,
        data,
        contexts,
        {11: "cell_a", 12: "cell_b"},
        k_values=[1, 10, 50, 100, 500],
        device=torch.device("cpu"),
    )

    assert len(model.encoded_edges) == 1
    assert torch.equal(model.encoded_edges[0], data.train_val_edge_index)
    assert metrics[1]["n_cells"] == 2
    assert metrics[1]["n_pos"] == 3
    assert metrics[500]["n_neg"] == 1_500
    assert len(rows) == 10
    assert {row["edgelist"] for row in rows} == {"cell_a", "cell_b"}
    assert diagnostics == {
        "directed_positive_occurrences_scored": 3,
        "unique_pair_context_occurrences": 3,
        "unique_global_test_pairs_observed": 2,
        "unique_global_test_pairs_total": 2,
        "pairs_observed_in_multiple_contexts": 1,
        "fraction_observed_pairs_in_multiple_contexts": 0.5,
        "mean_contexts_per_observed_pair": 1.5,
        "median_contexts_per_observed_pair": 1.5,
        "max_contexts_per_observed_pair": 2,
    }
    assert multiplicity_rows == [
        {
            "global_source_index": 3,
            "global_target_index": 4,
            "source_protein": "p3",
            "target_protein": "p4",
            "n_contexts": 2,
        },
        {
            "global_source_index": 4,
            "global_target_index": 5,
            "source_protein": "p4",
            "target_protein": "p5",
            "n_contexts": 1,
        },
    ]


@pytest.mark.parametrize(
    ("split", "positive_attribute", "topology_attribute"),
    [
        ("val", "val_edge_index", "train_edge_index"),
        ("test", "test_edge_index", "train_val_edge_index"),
    ],
)
def test_global_evaluation_scores_each_positive_once_on_the_correct_topology(
    split, positive_attribute, topology_attribute
):
    data = tiny_data()
    model = RecordingModel()
    positive_edges = getattr(data, positive_attribute)
    negative_bank = torch.tensor([[[0], [1]], [[5], [5]]], dtype=torch.long)

    metrics = evaluate_global_edges(
        model,
        data,
        split=split,
        k_values=[1],
        device=torch.device("cpu"),
        negative_bank=negative_bank,
    )

    assert len(model.encoded_edges) == 1
    assert torch.equal(model.encoded_edges[0], getattr(data, topology_attribute))
    assert len(model.scored_edges) == 2
    assert torch.equal(model.scored_edges[0], positive_edges)
    assert model.scored_edges[0].size(1) == positive_edges.size(1)
    assert torch.unique(_canonical_keys(model.scored_edges[0], 6)).numel() == (
        positive_edges.size(1)
    )
    assert metrics[1]["n_pos"] == positive_edges.size(1)
    assert metrics[1]["n_neg"] == positive_edges.size(1)


def test_decoder_scores_are_symmetric():
    torch.manual_seed(0)
    data = tiny_data()
    model = tiny_model().eval()
    with torch.no_grad():
        _, layers = model.encode(data.features, data.train_edge_index)
        forward = model.score_edges(layers, data.val_edge_index)
        reverse = model.score_edges(layers, data.val_edge_index.flip(0))
    torch.testing.assert_close(forward, reverse)


def test_portable_checkpoint_roundtrip_is_weights_only_safe(tmp_path):
    torch.manual_seed(0)
    data = tiny_data()
    model = tiny_model().eval()
    args = SimpleNamespace(
        epochs=3,
        lr=0.01,
        mask_ratio=0.5,
        mask_type="dm",
        k_negatives=1,
        seed=0,
        split_seed=0,
        run_name="test",
        wandb_entity="entity",
        wandb_project="project",
        wandb_group="group",
        experiment_role="sweep",
    )
    payload = checkpoint_payload(
        args,
        model,
        data,
        epoch=1,
        best_epoch=1,
        best_global_val_ap=0.5,
        val_metrics={"ap": 0.5},
    )
    assert payload["format_version"] == 3
    assert payload["best_global_val_ap"] == 0.5
    assert payload["protocol"]["selection_metric"] == "global_val_ap"
    assert "best_context_val_ap_macro" not in payload

    path = tmp_path / "best.pt"
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    restored = GlobalS2GAE(**loaded["model_config"]).eval()
    restored.load_state_dict(loaded["model_state_dict"], strict=True)

    with torch.no_grad():
        _, original_layers = model.encode(data.features, data.train_edge_index)
        _, restored_layers = restored.encode(data.features, data.train_edge_index)
        original = model.score_edges(original_layers, data.val_edge_index)
        recovered = restored.score_edges(restored_layers, data.val_edge_index)
    torch.testing.assert_close(original, recovered)


def test_rng_restore_accepts_checkpoint_tensors_on_any_device():
    state = capture_rng_state()
    state["torch"] = state["torch"].to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if state["torch_cuda"] is not None:
        state["torch_cuda"] = [value.to("cuda") for value in state["torch_cuda"]]

    restore_rng_state(state)


def test_completed_run_extension_changes_only_epoch_budget_and_code_version():
    training = {
        "epochs": 300,
        "lr": 0.01,
        "mask_ratio": 0.5,
        "mask_type": "dm",
        "k_negatives": 1,
        "seed": 0,
        "split_seed": 0,
    }
    checkpoint = {
        "epoch": 299,
        "best_epoch": 297,
        "best_global_val_ap": 0.92,
        "training_config": training,
        "git_commit": "old",
        "protocol": protocol_metadata(),
        "graph_fingerprint": "graph",
        "feature_fingerprint": "features",
        "split_fingerprint": "split",
    }
    completion = {
        "completed_epochs": 300,
        "last_epoch": 299,
        "best_epoch": 297,
        "best_global_val_ap": 0.92,
        "git_commit": "old",
        "protocol": protocol_metadata(),
        "graph_fingerprint": "graph",
        "feature_fingerprint": "features",
        "split_fingerprint": "split",
    }
    previous_config = {
        "epochs": 300,
        "git_commit": "old",
        "protocol": protocol_metadata(),
        "graph_fingerprint": "graph",
        "feature_fingerprint": "features",
        "split_fingerprint": "split",
    }

    continuation = validate_completed_extension(
        checkpoint,
        completion,
        previous_config,
        {**training, "epochs": 500},
        "new",
    )

    assert continuation == {
        "from_completed_epochs": 300,
        "to_epochs": 500,
        "from_git_commit": "old",
        "to_git_commit": "new",
    }
    with pytest.raises(ValueError, match="training parameter lr changed"):
        validate_completed_extension(
            checkpoint,
            completion,
            previous_config,
            {**training, "epochs": 500, "lr": 0.02},
            "new",
        )


def test_archived_best_must_match_latest_completion_state():
    shared = {
        "format_version": 3,
        "experiment_role": "sweep",
        "model_config": {"hidden_dim": 128},
        "training_config": {"epochs": 300},
        "wandb": {"run_id": "id"},
        "graph_fingerprint": "graph",
        "feature_fingerprint": "features",
        "split_fingerprint": "split",
        "protocol": protocol_metadata(),
        "git_commit": "old",
        "best_epoch": 297,
        "best_global_val_ap": 0.92,
    }
    validate_archived_best({**shared, "epoch": 297}, {**shared, "epoch": 299})
    with pytest.raises(ValueError, match="best/latest best_global_val_ap"):
        validate_archived_best(
            {**shared, "epoch": 297, "best_global_val_ap": 0.91},
            {**shared, "epoch": 299},
        )


def test_manifest_selection_requires_primaries_but_not_sensitivities(tmp_path):
    data = tiny_data()
    defaults = {
        "experiment_role": "sweep",
        "hidden_dim": 8,
        "num_layers": 2,
        "dropout": 0.0,
        "decode_channels": 8,
        "decoder_layers": 2,
        "decoder_dropout": 0.0,
        "mask_ratio": 0.5,
        "mask_type": "dm",
        "k_negatives": 1,
        "epochs": 3,
        "lr": 0.01,
        "seed": 0,
        "split_seed": 0,
        "wandb_entity": "entity",
        "wandb_project": "project",
        "wandb_group": "group",
    }
    runs = [
        {"name": "primary_a"},
        {"name": "primary_b", "dropout": 0.1},
        {
            "name": "optional_um",
            "mask_type": "um",
            "experiment_role": "sensitivity",
        },
    ]
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "selection_metric": "global_val_ap",
                "defaults": defaults,
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )

    for name, dropout, score in (
        ("primary_a", 0.0, 0.5),
        ("primary_b", 0.1, 0.6),
    ):
        run_dir = tmp_path / "runs" / name
        run_dir.mkdir(parents=True)
        model = GlobalS2GAE(
            4,
            hidden_dim=8,
            num_layers=2,
            dropout=dropout,
            decode_channels=8,
            decoder_layers=2,
            decoder_dropout=0.0,
        )
        args = SimpleNamespace(**{**defaults, "dropout": dropout, "run_name": name})
        payload = checkpoint_payload(
            args,
            model,
            data,
            epoch=2,
            best_epoch=2,
            best_global_val_ap=score,
            val_metrics={"ap": score},
        )
        torch.save(payload, run_dir / "best_model_state_dict.pt")
        (run_dir / "completed.json").write_text(
            json.dumps(
                {
                    "run_name": name,
                    "experiment_role": "sweep",
                    "completed_epochs": 3,
                    "last_epoch": 2,
                    "best_epoch": 2,
                    "best_global_val_ap": score,
                    "parameter_count": payload["parameter_count"],
                    "selection_metric": "global_val_ap",
                    "graph_fingerprint": data.graph_fingerprint,
                    "feature_fingerprint": data.feature_fingerprint,
                    "split_fingerprint": data.split_fingerprint,
                    "protocol": protocol_metadata(),
                    "git_commit": payload["git_commit"],
                }
            ),
            encoding="utf-8",
        )

    args = SimpleNamespace(
        checkpoint=None,
        runs_root=tmp_path / "runs",
        sweep_config=sweep_path,
    )
    selected_path, _, rows = select_checkpoint(args)
    assert selected_path.parent.name == "primary_b"
    assert [row["validation_rank"] for row in rows[:2]] == [1, 2]
    assert rows[0]["best_global_val_ap"] == 0.6
    assert rows[-1]["run_name"] == "optional_um"
    assert rows[-1]["status"] == "incomplete"

    (tmp_path / "runs" / "primary_a" / "completed.json").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete primary"):
        select_checkpoint(args)
