"""Label loaders for the two downstream tasks reported in the paper."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class BaseTaskLoader:
    def __init__(self, label_csv: Path):
        self.label_csv = label_csv

    def load(self) -> Tuple[List[str], np.ndarray, List[str]]:
        raise NotImplementedError


class CORUMLoader(BaseTaskLoader):
    def __init__(
        self,
        label_csv: Path,
        min_pos_per_complex: int = 10,
        max_pos_per_complex: int = 300,
    ):
        super().__init__(label_csv)
        self.min_pos = min_pos_per_complex
        self.max_pos = max_pos_per_complex

    def load(self) -> Tuple[List[str], np.ndarray, List[str]]:
        df = pd.read_csv(self.label_csv)

        gene_col = next(
            (column for column in ("protein", "gene_name_raw", "uniprot_id") if column in df),
            None,
        )
        complex_col = next(
            (column for column in ("complex_id", "complex_name") if column in df),
            None,
        )
        if gene_col is None or complex_col is None:
            raise ValueError(
                f"Expected protein and complex columns in {self.label_csv}"
            )

        gene_complexes: Dict[str, set] = {}
        for _, row in df.iterrows():
            gene = str(row[gene_col]).upper()
            gene_complexes.setdefault(gene, set()).add(str(row[complex_col]))

        complex_counts: Dict[str, int] = {}
        for complexes in gene_complexes.values():
            for complex_id in complexes:
                complex_counts[complex_id] = complex_counts.get(complex_id, 0) + 1

        class_names = sorted(
            complex_id
            for complex_id, count in complex_counts.items()
            if self.min_pos <= count <= self.max_pos
        )
        if not class_names:
            raise ValueError(
                f"No complexes with {self.min_pos}-{self.max_pos} proteins found"
            )

        class_index = {complex_id: index for index, complex_id in enumerate(class_names)}
        genes = sorted(gene_complexes)
        labels = np.zeros((len(genes), len(class_names)), dtype=np.float32)
        for gene_index, gene in enumerate(genes):
            for complex_id in gene_complexes[gene]:
                if complex_id in class_index:
                    labels[gene_index, class_index[complex_id]] = 1.0

        keep = labels.sum(axis=1) > 0
        return [gene for index, gene in enumerate(genes) if keep[index]], labels[keep], class_names


class TherapeuticTargetLoader(BaseTaskLoader):
    def load(self) -> Tuple[List[str], np.ndarray, List[str]]:
        df = pd.read_csv(self.label_csv)
        gene_col = next(
            (column for column in ("protein", "Protein", "gene", "Gene") if column in df),
            None,
        )
        label_col = next(
            (column for column in ("label", "Label", "target", "is_target") if column in df),
            None,
        )
        if gene_col is None or label_col is None:
            raise ValueError(f"Expected gene and binary-label columns in {self.label_csv}")

        df = df[[gene_col, label_col]].copy()
        df[gene_col] = df[gene_col].astype(str).str.strip().str.upper()
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df[(df[gene_col] != "") & df[label_col].notna()]
        df[label_col] = (df[label_col] > 0).astype(np.float32)
        df = df.groupby(gene_col, as_index=False)[label_col].max().sort_values(gene_col)

        return (
            df[gene_col].tolist(),
            df[[label_col]].to_numpy(dtype=np.float32),
            ["therapeutic_target"],
        )


def get_task_loader(task: str, label_csv: Path) -> BaseTaskLoader:
    task = task.lower()
    if task == "corum":
        return CORUMLoader(label_csv)
    if task.startswith("therapeutic_target_"):
        return TherapeuticTargetLoader(label_csv)
    raise ValueError(f"Unknown paper task: {task}")
