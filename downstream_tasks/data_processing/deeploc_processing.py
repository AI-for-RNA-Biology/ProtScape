#!/usr/bin/env python3
"""Build the broad human DeepLoc localization task from HGNC-mapped tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


DEEPLOC_LABELS = (
    "Cell membrane",
    "Cytoplasm",
    "Endoplasmic reticulum",
    "Extracellular",
    "Golgi apparatus",
    "Lysosome/Vacuole",
    "Mitochondrion",
    "Nucleus",
    "Peroxisome",
    "Plastid",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapped_table(path: Path, labels: Sequence[str]) -> pd.DataFrame:
    table = pd.read_csv(path)
    if "hgnc_symbol" not in table:
        raise ValueError(f"Missing hgnc_symbol column: {path}")

    result = pd.DataFrame({"protein": table["hgnc_symbol"]})
    for label in labels:
        values = table[label] if label in table else pd.Series(0, index=table.index)
        result[label] = pd.to_numeric(values, errors="raise").fillna(0).astype(int)
        if not result[label].isin((0, 1)).all():
            raise ValueError(f"Label {label!r} is not binary in {path}")
    return result.dropna(subset=["protein"])


def process_deeploc(
    test_csv: Path,
    train_csv: Path,
    output_csv: Path,
    min_positive_count: int = 50,
) -> dict:
    """Merge mapped DeepLoc tables and write a canonical membership CSV.

    The HPA test rows take precedence when a mapped HGNC symbol occurs in both
    inputs, matching the original ProtScape preprocessing script.
    """
    if min_positive_count < 1:
        raise ValueError("min_positive_count must be at least one")

    test = _load_mapped_table(test_csv, DEEPLOC_LABELS)
    train = _load_mapped_table(train_csv, DEEPLOC_LABELS)
    combined = pd.concat((test, train), ignore_index=True)
    combined["protein"] = combined["protein"].astype(str).str.strip().str.upper()
    combined = combined[combined["protein"] != ""]
    combined = combined.drop_duplicates("protein", keep="first")

    positive_counts = {
        label: int(combined[label].sum()) for label in DEEPLOC_LABELS
    }
    retained_labels = [
        label
        for label in DEEPLOC_LABELS
        if positive_counts[label] >= min_positive_count
    ]
    if not retained_labels:
        raise ValueError("No DeepLoc labels passed the positive-count threshold")

    annotated = combined[combined[retained_labels].any(axis=1)]
    rows = []
    for record in annotated.itertuples(index=False):
        protein = record[0]
        for column, label in enumerate(DEEPLOC_LABELS, start=1):
            if label in retained_labels and record[column] == 1:
                rows.append(
                    {
                        "protein": protein,
                        "label": label,
                        "label_id": f"DeepLoc2:{label}",
                        "source": "DeepLoc2",
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("protein", "label", "label_id", "source")
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["protein"], row["label_id"])))

    manifest = {
        "format_version": 1,
        "dataset": "deeploc2_broad_human_localization",
        "inputs": {
            "hpa_test_sha256": _sha256(test_csv),
            "swissprot_train_validation_sha256": _sha256(train_csv),
        },
        "parameters": {
            "candidate_labels": list(DEEPLOC_LABELS),
            "duplicate_gene_policy": "HPA test row first",
            "min_positive_count": min_positive_count,
            "retained_labels": retained_labels,
        },
        "statistics": {
            "mapped_unique_proteins": int(len(combined)),
            "retained_proteins": int(len(annotated)),
            "memberships": len(rows),
            "positive_counts_before_filter": positive_counts,
        },
        "output": {
            "columns": ["protein", "label", "label_id", "source"],
            "csv_sha256": _sha256(output_csv),
        },
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--min-positive-count", type=int, default=50)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = process_deeploc(
        test_csv=args.test_csv,
        train_csv=args.train_csv,
        output_csv=args.output_csv,
        min_positive_count=args.min_positive_count,
    )
    print(json.dumps(manifest["statistics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
