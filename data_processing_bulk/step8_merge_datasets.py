#!/usr/bin/env python
"""Step 8: merge the HBCA+Tabula and ALS networks used for model training."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List

from .config import (
    ALS_ASTRO_INTERMEDIATE,
    ALS_BULK_DIR,
    ALS_MN_INTERMEDIATE,
    CELLTYPE_CLASS_MAPPING,
    GLOBAL_PPI,
    HBCA_INTERMEDIATE,
    MERGED_INTERMEDIATE,
    PINNACLE_BASE,
    TABULA_INTERMEDIATE,
)
from .network_merging import (
    check_celltype_tissue_coverage,
    create_summary,
    log_final_summary,
    merge_cci_networks,
    merge_metagraphs,
    merge_ppi_networks,
    summarize_cci_vs_metagraph,
    summarize_metagraph,
    summarize_ppi_dir,
    write_count_edge_dict,
)
from .reliable_gene_merging import (
    merge_reliable_genes_multi,
    prefer_specific_file,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MERGED_OUTPUT = os.path.join(PINNACLE_BASE, "networks_bulk")

ALS_MN_REL = os.path.relpath(ALS_MN_INTERMEDIATE, ALS_BULK_DIR)
ALS_ASTRO_REL = os.path.relpath(ALS_ASTRO_INTERMEDIATE, ALS_BULK_DIR)


def _validate_output_path(output: Path, input_roots: List[Path]) -> None:
    def overlaps(left: Path, right: Path) -> bool:
        return left == right or left in right.parents or right in left.parents

    repository = Path(__file__).resolve().parents[1]
    home = Path.home().resolve()
    pinnacle = Path(PINNACLE_BASE).resolve()
    unsafe = (
        output == Path("/")
        or output == home
        or output in home.parents
        or output == pinnacle
        or output in pinnacle.parents
        or overlaps(output, repository)
        or any(overlaps(output, root) for root in input_roots)
    )
    if unsafe:
        raise ValueError(f"Refusing to replace unsafe output directory: {output}")


def _require_inputs(paths: List[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Step 8 inputs:\n" + "\n".join(missing))


def _als_gene_files(als_root: Path) -> List[str]:
    combined = als_root / "reliable_genes.json"
    if combined.exists():
        return [str(combined)]
    return [
        prefer_specific_file(str(als_root / ALS_MN_REL / "reliable_genes.json")),
        prefer_specific_file(str(als_root / ALS_ASTRO_REL / "reliable_genes.json")),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(MERGED_OUTPUT))
    parser.add_argument("--hbca-root", type=Path, default=Path(HBCA_INTERMEDIATE))
    parser.add_argument("--tabula-root", type=Path, default=Path(TABULA_INTERMEDIATE))
    parser.add_argument("--als-root", type=Path, default=Path(ALS_BULK_DIR))
    parser.add_argument("--merged-root", type=Path, default=Path(MERGED_INTERMEDIATE))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output_dir.resolve()
    hbca = args.hbca_root.resolve()
    tabula = args.tabula_root.resolve()
    als = args.als_root.resolve()
    merged = args.merged_root.resolve()
    input_roots = [hbca, tabula, als, merged]
    _validate_output_path(output, input_roots)

    gene_files = [
        prefer_specific_file(str(hbca / "reliable_genes.json")),
        prefer_specific_file(str(tabula / "reliable_genes.json")),
        *_als_gene_files(als),
    ]
    _require_inputs(
        [
            Path(GLOBAL_PPI),
            Path(CELLTYPE_CLASS_MAPPING),
            *(Path(path) for path in gene_files),
            hbca / "ppi" / "ppi_edgelists",
            tabula / "ppi" / "ppi_edgelists",
            merged / "cci_edgelist.txt",
            merged / "metagraph.txt",
            als / "ppi" / "ppi_edgelists",
            als / "cci_edgelist.txt",
            als / "metagraph.txt",
        ]
    )

    if output.exists():
        logger.info("Removing existing output directory: %s", output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    ppi_stats = merge_ppi_networks(
        output_root=str(output),
        hbca_ppi_root=str(hbca / "ppi"),
        tabula_ppi_root=str(tabula / "ppi"),
        als_ppi_roots=[str(als / "ppi")],
    )
    cci_graph = merge_cci_networks(
        [str(merged / "cci_edgelist.txt"), str(als / "cci_edgelist.txt")],
        str(output / "cci_edgelist.txt"),
    )
    mg_graph = merge_metagraphs(
        [str(merged / "metagraph.txt"), str(als / "metagraph.txt")],
        str(output / "mg_edgelist.txt"),
        str(output),
    )
    genes = merge_reliable_genes_multi(
        gene_files,
        str(output / "reliable_genes.json"),
        str(output / "reliable_genes_per_celltype.json"),
    )

    output_global_ppi = output / "global_ppi_edgelist.txt"
    shutil.copy(GLOBAL_PPI, output_global_ppi)
    shutil.copy(CELLTYPE_CLASS_MAPPING, output / "celltype_class_mapping.csv")
    write_count_edge_dict(output_global_ppi, output, output / "count_edge_dict.pkl")
    ppi_summary = summarize_ppi_dir(str(output))
    mg_summary = summarize_metagraph(mg_graph)
    coverage = check_celltype_tissue_coverage(mg_graph, str(output))
    cci_summary = summarize_cci_vs_metagraph(cci_graph, mg_graph)
    create_summary(
        str(output),
        ppi_stats,
        cci_graph,
        mg_graph,
        genes,
        ppi_summary,
        mg_summary,
        str(output_global_ppi),
        coverage,
        cci_summary,
    )
    log_final_summary(ppi_summary, mg_summary, len(genes), str(output_global_ppi))
    logger.info("Merged data saved to %s", output)


if __name__ == "__main__":
    main()
