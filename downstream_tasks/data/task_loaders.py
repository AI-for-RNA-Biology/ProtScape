"""Strict label loaders for downstream protein tasks."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class BaseTaskLoader:
    def __init__(self, label_csv: Path):
        self.label_csv = label_csv

    def load(self) -> Tuple[List[str], np.ndarray, List[str]]:
        raise NotImplementedError


class MultiLabelMembershipLoader(BaseTaskLoader):
    """Load a canonical long-form protein-to-label membership table.

    Accepted columns are ``protein,label`` plus optional ``label_id,source``.
    ``label_id`` is the stable class identity when present; otherwise the label
    text itself is used.  Duplicate memberships are harmless, but conflicting
    ID/name mappings are rejected.
    """

    REQUIRED_COLUMNS = frozenset({"protein", "label"})
    OPTIONAL_COLUMNS = frozenset({"label_id", "source"})

    def load(self) -> Tuple[List[str], np.ndarray, List[str]]:
        table = pd.read_csv(self.label_csv, dtype=str)
        missing = sorted(self.REQUIRED_COLUMNS.difference(table.columns))
        if missing:
            raise ValueError(
                f"Membership table is missing required column(s): {', '.join(missing)}"
            )
        unexpected = sorted(
            set(table.columns).difference(self.REQUIRED_COLUMNS | self.OPTIONAL_COLUMNS)
        )
        if unexpected:
            raise ValueError(
                f"Membership table has unexpected column(s): {', '.join(unexpected)}"
            )
        if table.empty or table.isna().any(axis=None):
            raise ValueError(
                "Membership table is empty or contains missing values: "
                f"{self.label_csv}"
            )

        for column in table.columns:
            table[column] = table[column].str.strip()
        if (table == "").any(axis=None):
            raise ValueError(
                f"Membership table contains missing values: {self.label_csv}"
            )
        table["protein"] = table["protein"].str.upper()

        class_column = "label_id" if "label_id" in table.columns else "label"
        if class_column == "label_id":
            ids_per_label = table.groupby("label", sort=False)["label_id"].nunique()
            labels_per_id = table.groupby("label_id", sort=False)["label"].nunique()
            if (ids_per_label > 1).any() or (labels_per_id > 1).any():
                raise ValueError("label and label_id must have a one-to-one mapping")

        memberships = table[["protein", class_column]].drop_duplicates()
        genes = sorted(memberships["protein"].unique())
        class_names = sorted(memberships[class_column].unique())
        if not genes or not class_names:
            raise ValueError(f"Membership table has no usable rows: {self.label_csv}")
        gene_index = {gene: index for index, gene in enumerate(genes)}
        class_index = {
            class_name: index for index, class_name in enumerate(class_names)
        }
        labels = np.zeros((len(genes), len(class_names)), dtype=np.float32)
        for protein, class_name in memberships.itertuples(index=False, name=None):
            labels[gene_index[protein], class_index[class_name]] = 1.0
        return genes, labels, class_names


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
        if df.empty:
            raise ValueError(f"Therapeutic-target label table is empty: {self.label_csv}")
        if df[gene_col].isna().any() or df[label_col].isna().any():
            raise ValueError(
                f"Therapeutic-target labels contain missing values: {self.label_csv}"
            )
        df[gene_col] = df[gene_col].astype(str).str.strip().str.upper()
        if (df[gene_col] == "").any():
            raise ValueError(
                f"Therapeutic-target labels contain empty gene names: {self.label_csv}"
            )
        numeric_labels = pd.to_numeric(df[label_col], errors="coerce")
        if numeric_labels.isna().any() or not set(numeric_labels.unique()).issubset(
            {0, 1}
        ):
            raise ValueError(
                f"Therapeutic-target labels must be binary 0/1: {self.label_csv}"
            )
        df[label_col] = numeric_labels.astype(np.float32)
        conflicts = df.groupby(gene_col)[label_col].nunique()
        if (conflicts > 1).any():
            raise ValueError(
                f"Therapeutic-target genes have conflicting labels: {self.label_csv}"
            )
        df = (
            df.drop_duplicates(subset=[gene_col])
            .sort_values(gene_col)
            .reset_index(drop=True)
        )
        if set(df[label_col].unique()) != {0.0, 1.0}:
            raise ValueError(
                f"Therapeutic-target labels must contain both classes: {self.label_csv}"
            )

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
    if task in {"protein_localization", "pathway"}:
        return MultiLabelMembershipLoader(label_csv)
    raise ValueError(f"Unknown downstream task: {task}")
