"""Export observed ALS PPI neighbourhoods for the configured gene.

Union nuclear and cytoplasmic edges within each condition/day; combine the
two mutant lines as VCP. This descriptive graph export does not run inference.
"""

import csv
from collections import defaultdict, deque
from pathlib import Path


EDGE_COLUMNS = [
    "context", "condition", "day", "source_contexts", "source", "target",
    "source_hop", "target_hop", "first_included_at_k",
]


def source_contexts(condition, day):
    genotypes = ("CTRL",) if condition == "CTRL" else ("R155C", "R191Q")
    return [
        f"{genotype}_{compartment}_d{day}"
        for genotype in genotypes
        for compartment in ("cyto", "nuc")
    ]


def read_union_edges(ppi_dir, contexts):
    edges = set()
    for context in contexts:
        with (Path(ppi_dir) / f"CL_0000100_{context}.txt").open() as handle:
            for line in handle:
                protein_a, protein_b = line.split()[:2]
                if protein_a != protein_b:
                    edges.add(tuple(sorted((protein_a, protein_b))))
    return edges


def shortest_hops(edges, gene, max_hop=3):
    adjacency = defaultdict(set)
    for protein_a, protein_b in edges:
        adjacency[protein_a].add(protein_b)
        adjacency[protein_b].add(protein_a)
    if gene not in adjacency:
        return {}

    distances = {gene: 0}
    queue = deque([gene])
    while queue:
        protein = queue.popleft()
        if distances[protein] == max_hop:
            continue
        for neighbor in adjacency[protein]:
            if neighbor not in distances:
                distances[neighbor] = distances[protein] + 1
                queue.append(neighbor)
    return distances


def prepare_neighborhood(ppi_dir, output_csv, gene, days=(22, 35)):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDGE_COLUMNS)
        writer.writeheader()
        for condition in ("CTRL", "VCP"):
            for day in days:
                context = f"{condition}_d{day}"
                sources = source_contexts(condition, day)
                edges = read_union_edges(ppi_dir, sources)
                distances = shortest_hops(edges, gene)
                if not distances:
                    print(f"{context}: {gene} absent; no neighbourhood edges")
                    continue
                for protein_a, protein_b in sorted(edges):
                    if protein_a not in distances or protein_b not in distances:
                        continue
                    writer.writerow({
                        "context": context,
                        "condition": condition,
                        "day": day,
                        "source_contexts": ";".join(sources),
                        "source": protein_a,
                        "target": protein_b,
                        "source_hop": distances[protein_a],
                        "target_hop": distances[protein_b],
                        "first_included_at_k": max(distances[protein_a], distances[protein_b]),
                    })
    print(f"Saved {output_csv}")


def main():
    from downstream_tasks.config import PATHS
    from exploration.analysis.ALS_rewiring import config

    prepare_neighborhood(
        Path(PATHS["networks_bulk"]) / "ppi_edgelists",
        config.NEIGHBORHOOD_EDGES_CSV,
        config.NEIGHBORHOOD_GENE,
    )


if __name__ == "__main__":
    main()
