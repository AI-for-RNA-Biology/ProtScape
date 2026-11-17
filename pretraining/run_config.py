"""Train a named configuration from configs/pretraining.yaml."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "configs" / "pretraining.yaml"


def main():
    """Combine a named YAML configuration with explicit overrides, then generate features and train."""
    with CONFIG_FILE.open(encoding="utf-8") as handle:
        configs = yaml.safe_load(handle)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", choices=list(configs), nargs="?", default="protscape")
    parser.add_argument("--esm2-pooling", choices=["bos", "mean"])
    parser.add_argument("--device")
    args, overrides = parser.parse_known_args()

    config = configs[args.configuration]
    arguments = shlex.split(config["args"]) + overrides
    feature_mode, pooling, device = "ESM2", "bos", "auto"
    for index, argument in enumerate(arguments):
        option, separator, value = argument.partition("=")
        if option not in {"--features-mode", "--esm2-pooling", "--device"}:
            continue
        if not separator:
            value = arguments[index + 1]
        if option == "--features-mode":
            feature_mode = value
        elif option == "--esm2-pooling":
            pooling = value
        elif option == "--device":
            device = value
    pooling = args.esm2_pooling or pooling
    device = args.device or device
    if feature_mode == "ESM2":
        subprocess.run(
            [sys.executable, "-m", "pretraining.generate_esm2_embeddings", "--pooling", pooling, "--device", device],
            cwd=REPO_ROOT,
            check=True,
        )
        arguments += ["--esm2-pooling", pooling]
    elif feature_mode == "ProstT5":
        subprocess.run([sys.executable, "-m", "pretraining.generate_prostt5_embeddings", "--device", device], cwd=REPO_ROOT, check=True)

    command = [sys.executable, "-m", config["module"], *arguments, "--device", device]
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
