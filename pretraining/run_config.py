"""Train one pretraining configuration from configs/pretraining.yaml."""

import shlex
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "configs" / "pretraining.yaml"


def main():
    with CONFIG_FILE.open(encoding="utf-8") as handle:
        configs = yaml.safe_load(handle)

    if len(sys.argv) != 2 or sys.argv[1] not in configs:
        choices = "\n  ".join(configs)
        raise SystemExit(
            "Usage: python -m pretraining.run_config <configuration>\n"
            f"Available configurations:\n  {choices}"
        )

    config = configs[sys.argv[1]]
    arguments = shlex.split(config["args"])
    if "ESM2" in arguments:
        subprocess.run(
            [sys.executable, "-m", "pretraining.generate_esm2_embeddings"],
            cwd=REPO_ROOT,
            check=True,
        )

    command = [sys.executable, "-m", config["module"], *arguments]
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
