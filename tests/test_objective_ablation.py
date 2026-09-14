"""Lightweight checks of the fixed ablation dispatch (no GPU/data required)."""
import ast
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/cscs/run_objective_ablation.py"


def test_three_loss_controls_keep_architecture_and_seed(tmp_path):
    module = runpy.run_path(str(RUNNER))
    commands = []
    module["train"].__globals__["run"] = lambda command, log: commands.append(command)
    for arm, cci, tissue in [("neither", 0, 0), ("cci_only", 1, 0), ("tissue_only", 0, 1)]:
        module["train"](tmp_path, arm)
        command = commands[-1]
        assert command[8] == str(tissue)
        for flag, value in [("--metagraph-loss-weight", str(cci)), ("--seed", "0"),
                            ("--use-metagraph", "true"), ("--epochs", "300"),
                            ("--checkpoint-selection", "ppi")]:
            assert command[command.index(flag) + 1] == value
    assert len(commands) == 3


def test_full_model_cannot_be_retrained(tmp_path):
    import pytest
    module = runpy.run_path(str(RUNNER))
    with pytest.raises(ValueError, match="Reuse"):
        module["train"](tmp_path, "both")
    assert module["BASELINE"].name == "protscape_main_state_dict.pt"


def test_ppi_selection_resume_does_not_add_cci(tmp_path):
    # The training CLI executes at import time: load only its pure helper.
    import csv
    tree = ast.parse((ROOT / "pretraining/train_protscape.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_best_metric_value")
    namespace = {"csv": csv, "os": __import__("os")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "helper", "exec"), namespace)
    metrics = tmp_path / "val.csv"
    metrics.write_text("best_val_ap_ppi,best_val_ap_meta\n0.8,0.9\n")
    paths = {"val_metrics_path": str(metrics), "model_path": str(tmp_path / "missing.pt")}
    assert namespace["_load_best_metric_value"]("ap", paths, "ppi") == 0.8
    assert abs(namespace["_load_best_metric_value"]("ap", paths) - 1.7) < 1e-10


def test_recovery_skips_completed_training_and_full_model(tmp_path):
    module = runpy.run_path(str(RUNNER))
    calls = []
    namespace = module["recover"].__globals__
    namespace["train"] = lambda root, arm: calls.append(("train", arm))
    namespace["evaluate"] = lambda root, arm: calls.append(("evaluate", arm))
    completed = module["checkpoint_path"](tmp_path, "neither")
    completed.parent.mkdir(parents=True)
    completed.touch()
    for arm in ["neither", "cci_only", "both"]:
        module["recover"](tmp_path, arm)
    assert calls == [("evaluate", "neither"), ("train", "cci_only"),
                     ("evaluate", "cci_only"), ("evaluate", "both")]
