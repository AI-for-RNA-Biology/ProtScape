import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pretraining.run_global_s2gae_sweep import command, load_sweep
from pretraining.prepare_global_s2gae_sweep import prepare


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_production_sweep_is_complete_seed_zero_encoder_grid():
    _, configurations = load_sweep(REPO_ROOT / "configs/global_s2gae_sweep.yaml")

    assert len(configurations) == 54
    assert {config["hidden_dim"] for config in configurations} == {128, 256, 512}
    assert {config["num_layers"] for config in configurations} == {2, 3, 4}
    assert {config["dropout"] for config in configurations} == {
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    }
    assert {config["epochs"] for config in configurations} == {500}
    assert {config["seed"] for config in configurations} == {0}
    assert {config["split_seed"] for config in configurations} == {0}
    assert {config["experiment_role"] for config in configurations} == {"sweep"}

    points = {
        (config["hidden_dim"], config["num_layers"], config["dropout"])
        for config in configurations
    }
    assert len(points) == 3 * 3 * 6


def test_completed_shorter_run_is_continued_not_restarted(tmp_path, monkeypatch):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "selection_metric": "global_val_ap",
                "defaults": {
                    "experiment_role": "sweep",
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "decode_channels": 512,
                    "decoder_layers": 2,
                    "decoder_dropout": 0.0,
                    "mask_ratio": 0.5,
                    "mask_type": "dm",
                    "k_negatives": 1,
                    "epochs": 500,
                    "lr": 0.01,
                    "seed": 0,
                    "split_seed": 0,
                    "wandb_entity": "entity",
                    "wandb_project": "project",
                    "wandb_group": "group",
                },
                "runs": [{"name": "existing"}],
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "runs"
    run_dir = output_root / "existing"
    run_dir.mkdir(parents=True)
    (run_dir / "latest_checkpoint.pt").touch()
    (run_dir / "completed.json").write_text(
        json.dumps({"completed_epochs": 300}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        config=config_path,
        index=0,
        networks_dir=tmp_path / "networks",
        esm2_embeddings=tmp_path / "esm2.plk",
        output_root=output_root,
        wandb_mode="disabled",
        device="cpu",
        dry_run=False,
    )
    captured = []
    monkeypatch.setattr(
        "pretraining.run_global_s2gae_sweep.parse_args",
        lambda: args,
    )
    monkeypatch.setattr(
        "pretraining.run_global_s2gae_sweep.subprocess.run",
        lambda cmd, check: captured.append((cmd, check)),
    )

    from pretraining.run_global_s2gae_sweep import main

    main()

    submitted, check = captured.pop()
    assert check
    assert "--resume" in submitted
    assert "--extend-completed" in submitted
    assert submitted[submitted.index("--epochs") + 1] == "500"


def test_completed_target_run_is_validated_before_skip(tmp_path, monkeypatch):
    _, configs = load_sweep(REPO_ROOT / "configs/global_s2gae_sweep.yaml")
    target = configs[0]
    output_root = tmp_path / "runs"
    run_dir = output_root / target["name"]
    run_dir.mkdir(parents=True)
    (run_dir / "completed.json").write_text(
        json.dumps({"completed_epochs": 500, "git_commit": "commit"}),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps({**target, "lr": 0.02}), encoding="utf-8"
    )
    args = SimpleNamespace(
        config=REPO_ROOT / "configs/global_s2gae_sweep.yaml",
        index=0,
        networks_dir=tmp_path / "networks",
        esm2_embeddings=tmp_path / "esm2.plk",
        output_root=output_root,
        wandb_mode="disabled",
        device="cpu",
        dry_run=False,
    )
    monkeypatch.setattr(
        "pretraining.run_global_s2gae_sweep.parse_args",
        lambda: args,
    )

    from pretraining.run_global_s2gae_sweep import main

    with pytest.raises(ValueError, match="changed lr"):
        main()


def test_command_does_not_mark_new_run_as_resume(tmp_path):
    args = SimpleNamespace(
        networks_dir=tmp_path / "networks",
        esm2_embeddings=tmp_path / "esm2.plk",
        output_root=tmp_path / "runs",
        wandb_mode="disabled",
        device="cpu",
    )
    _, configs = load_sweep(REPO_ROOT / "configs/global_s2gae_sweep.yaml")

    submitted = command(args, configs[0])

    assert "--resume" not in submitted
    assert "--extend-completed" not in submitted


def test_prepare_reuses_only_compatible_shorter_grid_runs(tmp_path):
    _, configs = load_sweep(REPO_ROOT / "configs/global_s2gae_sweep.yaml")
    reusable = configs[0]
    source = tmp_path / "old" / reusable["name"]
    source.mkdir(parents=True)
    for filename in (
        "best_model_state_dict.pt",
        "history.csv",
        "latest_checkpoint.pt",
    ):
        (source / filename).touch()
    (source / "config.json").write_text(
        json.dumps({**reusable, "epochs": 300}), encoding="utf-8"
    )
    (source / "completed.json").write_text(
        json.dumps(
            {
                "run_name": reusable["name"],
                "completed_epochs": 300,
                "last_epoch": 299,
            }
        ),
        encoding="utf-8",
    )

    reused = prepare(
        REPO_ROOT / "configs/global_s2gae_sweep.yaml",
        tmp_path / "old",
        tmp_path / "new",
    )

    assert reused == [
        {
            "run_name": reusable["name"],
            "from_epochs": 300,
            "to_epochs": 500,
        }
    ]
    copied = tmp_path / "new" / reusable["name"]
    assert all((copied / filename).is_file() for filename in (
        "best_model_state_dict.pt",
        "completed.json",
        "config.json",
        "history.csv",
        "latest_checkpoint.pt",
    ))


def test_prepare_rejects_a_changed_fixed_parameter(tmp_path):
    _, configs = load_sweep(REPO_ROOT / "configs/global_s2gae_sweep.yaml")
    reusable = configs[0]
    source = tmp_path / "old" / reusable["name"]
    source.mkdir(parents=True)
    for filename in (
        "best_model_state_dict.pt",
        "history.csv",
        "latest_checkpoint.pt",
    ):
        (source / filename).touch()
    (source / "config.json").write_text(
        json.dumps({**reusable, "epochs": 300, "lr": 0.02}), encoding="utf-8"
    )
    (source / "completed.json").write_text(
        json.dumps(
            {
                "run_name": reusable["name"],
                "completed_epochs": 300,
                "last_epoch": 299,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed lr"):
        prepare(
            REPO_ROOT / "configs/global_s2gae_sweep.yaml",
            tmp_path / "old",
            tmp_path / "new",
        )
