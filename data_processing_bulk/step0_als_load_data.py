#!/usr/bin/env python
"""Step 0 (ALS): stage the released count matrices and sample metadata."""

import argparse
import logging
from pathlib import Path
from shutil import copy2

from .config import (
    ALS_ASTRO_COUNTS,
    ALS_ASTRO_INTERMEDIATE,
    ALS_ASTRO_METADATA,
    ALS_MN_COUNTS,
    ALS_MN_INTERMEDIATE,
    ALS_MN_METADATA,
)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DATASETS = {
    "motor_neuron": {
        "counts_path": ALS_MN_COUNTS,
        "metadata_path": ALS_MN_METADATA,
        "output_dir": ALS_MN_INTERMEDIATE,
        "counts_file": "als_motor_neurons_raw_counts.csv",
        "metadata_file": "als_motor_neurons_raw_metadata.csv",
    },
    "astrocyte": {
        "counts_path": ALS_ASTRO_COUNTS,
        "metadata_path": ALS_ASTRO_METADATA,
        "output_dir": ALS_ASTRO_INTERMEDIATE,
        "counts_file": "als_astrocytes_raw_counts.csv",
        "metadata_file": "als_astrocytes_raw_metadata.csv",
    },
}


def process_dataset(dataset: str):
    config = DATASETS[dataset]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = output_dir / config["counts_file"]
    metadata_path = output_dir / config["metadata_file"]
    copy2(config["counts_path"], counts_path)
    copy2(config["metadata_path"], metadata_path)
    logger.info("Staged %s counts at %s", dataset, counts_path)
    logger.info("Staged %s metadata at %s", dataset, metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-type",
        choices=["motor_neuron", "astrocyte", "both"],
        default="both",
    )
    args = parser.parse_args()

    selected = DATASETS if args.cell_type == "both" else [args.cell_type]
    for dataset in selected:
        process_dataset(dataset)


if __name__ == "__main__":
    main()
