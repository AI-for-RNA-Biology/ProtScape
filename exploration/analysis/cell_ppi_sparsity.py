#!/usr/bin/env python3
"""Compute the edge-density range of the released cell-PPI networks."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
with (REPO_ROOT / "configs" / "paths.yaml").open(encoding="utf-8") as handle:
    PATHS = yaml.safe_load(handle)
PPI_DIR = Path(PATHS["networks_bulk"]).expanduser() / "ppi_edgelists"


def graph_counts(path: Path) -> tuple[int, int]:
    nodes = set()
    edges = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            source, target = parts[:2]
            nodes.update((source, target))
            if source != target:
                edges.add(tuple(sorted((source, target))))
    return len(nodes), len(edges)


def main() -> None:
    densities = []
    for path in sorted(PPI_DIR.glob("*.txt")):
        n_nodes, n_edges = graph_counts(path)
        density = 100.0 * 2.0 * n_edges / (n_nodes * (n_nodes - 1))
        densities.append((density, path.stem, n_nodes, n_edges))

    minimum = min(densities)
    maximum = max(densities)
    print(f"minimum\t{minimum[1]}\t{minimum[2]} nodes\t{minimum[3]} edges\t{minimum[0]:.6f}%")
    print(f"maximum\t{maximum[1]}\t{maximum[2]} nodes\t{maximum[3]} edges\t{maximum[0]:.6f}%")
    print(f"rounded range\t{minimum[0]:.2f}%--{maximum[0]:.2f}%")


if __name__ == "__main__":
    main()
