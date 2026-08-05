#!/usr/bin/env python
"""Step 0 (ALS): load Kallisto counts and average replicates per context."""

import argparse
import logging
from pathlib import Path

from .als_bulk_loader import load_als_bulk
from .config import (
    ALS_ASTRO_INTERMEDIATE,
    ALS_ASTRO_KALLISTO,
    ALS_ASTRO_METADATA,
    ALS_CELL_TYPE_MAPPINGS,
    ALS_MN_INTERMEDIATE,
    ALS_MN_KALLISTO,
    ALS_MN_METADATA,
)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DATASETS = {
    "motor_neuron": {
        "kallisto_dir": ALS_MN_KALLISTO,
        "metadata_path": ALS_MN_METADATA,
        "output_dir": ALS_MN_INTERMEDIATE,
        "counts_file": "als_motor_neurons_raw_counts.csv",
        "metadata_file": "als_motor_neurons_raw_metadata.csv",
    },
    "astrocyte": {
        "kallisto_dir": ALS_ASTRO_KALLISTO,
        "metadata_path": ALS_ASTRO_METADATA,
        "output_dir": ALS_ASTRO_INTERMEDIATE,
        "counts_file": "als_astrocytes_raw_counts.csv",
        "metadata_file": "als_astrocytes_raw_metadata.csv",
    },
}


def process_dataset(dataset: str):
    config = DATASETS[dataset]
    counts, metadata = load_als_bulk(
        kallisto_dir=config["kallisto_dir"],
        metadata_path=config["metadata_path"],
        dataset=dataset,
        cl_id=ALS_CELL_TYPE_MAPPINGS[dataset],
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = output_dir / config["counts_file"]
    metadata_path = output_dir / config["metadata_file"]
    counts.to_csv(counts_path)
    metadata.to_csv(metadata_path, index=False)
    logger.info("Saved %s counts to %s", dataset, counts_path)
    logger.info("Saved %s metadata to %s", dataset, metadata_path)
    return counts, metadata


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
