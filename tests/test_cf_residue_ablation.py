import json
from pathlib import Path
import shutil

import yaml

from scripts.cscs.run_cf_residue_ablation import REPO, grid, train_command, downstream_command


def setup_root(tmp_path):
    (tmp_path / "source/configs").mkdir(parents=True)
    shutil.copy2(REPO / "configs/cf_residue_ablation.yaml", tmp_path / "source/configs/cf_residue_ablation.yaml")
    (tmp_path / "inputs.json").write_text(json.dumps(dict(release="/release")))
    return yaml.safe_load((REPO / "configs/cf_residue_ablation.yaml").read_text())


def test_cf_grid_online_early_stop_and_correct_backbone(tmp_path):
    config = setup_root(tmp_path)
    trials = grid(config["pretraining"])
    assert len(trials) == 26
    for trial in trials:
        args = train_command(tmp_path, trial)
        assert args[2] == "pretraining.train_global_s2gae"
        for flag, expected in {"--wandb-mode": "online", "--wandb-group": "cf_residue_pooling",
                               "--epochs": "5000", "--early-stopping-patience": "200",
                               "--early-stopping-min-delta": "0.0005", "--hidden-dim": "512",
                               "--num-layers": "2", "--dropout": "0.4", "--seed": "0"}.items():
            assert args[args.index(flag) + 1] == expected
        assert ("--residue-config" in args) == (trial["mode"] != "mean")
        assert "train_protscape" not in " ".join(args)
    folder = tmp_path / "pretraining" / trials[0]["name"]
    folder.mkdir(parents=True)
    (folder / "latest_checkpoint.pt").touch()
    assert "--resume" in train_command(tmp_path, trials[0])


def test_cf_downstream_no_cell_vectors_and_fixed_settings(tmp_path):
    setup_root(tmp_path)
    args = downstream_command(tmp_path, "mean", dict(task="corum", csv="/corum.csv"))
    assert args[args.index("--model") + 1] == "lr_global,lr_global_ext_embed"
    assert "--global-inference" in args and "--cell-embedding-file" not in args
    assert args[args.index("--epochs") + 1] == "300"
    assert args[args.index("--patience") + 1] == "50"
    assert "gene_protein_embeddings_esm2_3B_layer33.plk" in " ".join(args)
