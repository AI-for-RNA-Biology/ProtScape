"""Shared downstream universe and context metadata for global protein probes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ContextPresence:
    """Cell-PPI membership patterns used only to stratify protein folds."""

    gene_to_contexts: Dict[str, Tuple[str, ...]]
    context_names: Tuple[str, ...]
    fingerprint: str


def load_context_presence(
    ppi_edgelists: Path,
    *,
    expected_context_count: int | None = None,
) -> ContextPresence:
    """Index protein presence in released Cell-PPI edgelists without loading edges."""
    ppi_edgelists = Path(ppi_edgelists).expanduser()
    if not ppi_edgelists.is_dir():
        raise FileNotFoundError(f"Cell-PPI edgelist directory not found: {ppi_edgelists}")

    paths = sorted(path for path in ppi_edgelists.iterdir() if path.suffix == ".txt")
    if not paths:
        raise ValueError(f"No .txt Cell-PPI edgelists found in {ppi_edgelists}")
    if expected_context_count is not None and len(paths) != int(expected_context_count):
        raise ValueError(
            f"Expected {int(expected_context_count)} Cell-PPI contexts, found {len(paths)} "
            f"in {ppi_edgelists}."
        )

    context_names = tuple(path.stem for path in paths)
    if len(context_names) != len(set(context_names)):
        raise ValueError("Cell-PPI filenames do not define unique context names.")

    gene_to_contexts: Dict[str, List[str]] = {}
    digest = hashlib.sha256()
    for context_name, path in zip(context_names, paths):
        genes = set()
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 2:
                    raise ValueError(
                        f"Malformed Cell-PPI edge at {path}:{line_number}: {line!r}"
                    )
                genes.update((fields[0].upper(), fields[1].upper()))
        if not genes:
            raise ValueError(f"Cell-PPI context is empty: {path}")

        digest.update(context_name.encode("utf-8"))
        digest.update(b"\0")
        for gene in sorted(genes):
            digest.update(gene.encode("utf-8"))
            digest.update(b"\0")
            gene_to_contexts.setdefault(gene, []).append(context_name)

    return ContextPresence(
        gene_to_contexts={
            gene: tuple(contexts) for gene, contexts in gene_to_contexts.items()
        },
        context_names=context_names,
        fingerprint=digest.hexdigest(),
    )


def restrict_global_probe_universe(
    genes: Sequence[str],
    labels: np.ndarray,
    *,
    global_genes: Iterable[str],
    sequence_genes: Iterable[str],
    context_presence: ContextPresence,
) -> tuple[List[str], np.ndarray, List[List[str]]]:
    """Return task ∩ global ∩ sequence ∩ Cell-PPI-union in task order."""
    labels = np.asarray(labels)
    if labels.shape[0] != len(genes):
        raise ValueError(
            f"Label rows ({labels.shape[0]}) do not match genes ({len(genes)})."
        )

    genes_upper = [str(gene).upper() for gene in genes]
    if len(genes_upper) != len(set(genes_upper)):
        raise ValueError("Task gene names are not unique after uppercasing.")
    global_set = {str(gene).upper() for gene in global_genes}
    sequence_set = {str(gene).upper() for gene in sequence_genes}
    cellppi_union = set(context_presence.gene_to_contexts)
    shared = set(genes_upper) & global_set & sequence_set & cellppi_union
    keep_indices = [index for index, gene in enumerate(genes_upper) if gene in shared]

    shared_genes = [genes_upper[index] for index in keep_indices]
    shared_labels = labels[keep_indices]
    contexts = [
        list(context_presence.gene_to_contexts[gene]) for gene in shared_genes
    ]
    return shared_genes, shared_labels, contexts


def split_fingerprint(genes: Sequence[str], folds: Sequence[np.ndarray]) -> str:
    """Fingerprint the ordered protein universe and all fold memberships."""
    digest = hashlib.sha256()
    for gene in genes:
        digest.update(str(gene).upper().encode("utf-8"))
        digest.update(b"\0")
    for fold_index, fold in enumerate(folds):
        indices = np.asarray(fold, dtype=np.int64)
        digest.update(int(fold_index).to_bytes(4, "little", signed=False))
        digest.update(indices.tobytes())
    return digest.hexdigest()
