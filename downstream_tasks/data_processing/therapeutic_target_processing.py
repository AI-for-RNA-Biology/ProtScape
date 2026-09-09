#!/usr/bin/env python
"""Build therapeutic-target labels from fixed Open Targets clinical and lookup tables."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import pandas as pd

from ..config import (
    DEFAULT_GLOBAL_PPI,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_THERAPEUTIC_TARGET_DRUGBANK_TARGETS,
    DEFAULT_THERAPEUTIC_TARGET_EVIDENCE_DIR,
    DEFAULT_THERAPEUTIC_TARGET_OT_ASSOCIATIONS_DIR,
    DEFAULT_THERAPEUTIC_TARGET_OT_DISEASES_DIR,
    DEFAULT_THERAPEUTIC_TARGET_OT_RELEASE,
    DEFAULT_THERAPEUTIC_TARGET_OT_TARGETS_DIR,
    THERAPEUTIC_TARGET_IDS,
)


def detect_output_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / "data" / "therapeutic_target_dataset"


def read_global_ppi_genes(path: Path) -> Set[str]:
    genes: Set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                genes.update(gene.upper() for gene in parts[:2])
    return genes


def load_druggable_targets(path: Path) -> Set[str]:
    """Load the October 2022 approved-human DrugBank target symbols."""
    table = pd.read_csv(path)

    # Accept a gene-symbol table or the annotated DrugBank target export.
    if "gene_symbol" in table.columns:
        values = table["gene_symbol"]
    else:
        if "Species" not in table.columns:
            table = pd.read_csv(path, index_col=0)
        required = {"Species", "Gene Name", "GenAtlas ID"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Missing DrugBank columns: {sorted(missing)}")
        human = table[table["Species"].astype(str).str.lower() == "humans"]
        values = pd.concat([human["Gene Name"], human["GenAtlas ID"]])

    return {
        str(value).strip().upper()
        for value in values.dropna()
        if str(value).strip() and str(value).strip().upper() != "NAN"
    }


def detect_table_files(path: Path, requested_format: str) -> Tuple[List[Path], str]:
    """Find the parts of one Open Targets table."""
    if not path.exists():
        raise FileNotFoundError(f"Open Targets input not found: {path}")

    roots = [path] if path.is_file() else []
    parquet = roots if path.suffix == ".parquet" else []
    json_files = roots if path.suffix == ".json" else []
    if path.is_dir():
        parquet = sorted(path.rglob("*.parquet"))
        json_files = sorted(path.rglob("*.json"))

    if requested_format == "auto":
        if parquet:
            return parquet, "parquet"
        if json_files:
            return json_files, "json"
    elif requested_format == "parquet" and parquet:
        return parquet, "parquet"
    elif requested_format == "json" and json_files:
        return json_files, "json"

    raise ValueError(f"No {requested_format} table parts found in {path}")


def iter_records(files: Sequence[Path], table_format: str) -> Iterator[Dict[str, object]]:
    if table_format == "json":
        for path in files:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        return

    for path in files:
        yield from pd.read_parquet(path).to_dict(orient="records")


def load_disease_descendants(
    path: Path,
    diseases: Iterable[str],
    table_format: str,
) -> Dict[str, Set[str]]:
    requested = set(diseases)
    files, detected = detect_table_files(path, table_format)
    descendants: Dict[str, Set[str]] = {}
    for record in iter_records(files, detected):
        disease_id = str(record.get("id", ""))
        if disease_id not in requested:
            continue
        values = record.get("descendants")
        if values is None:
            values = []
        descendants[disease_id] = {str(value) for value in values}
        descendants[disease_id].add(disease_id)

    missing = requested - set(descendants)
    if missing:
        raise ValueError(f"Disease IDs absent from the Open Targets disease table: {sorted(missing)}")
    return descendants


def load_target_symbols(
    path: Path, table_format: str, ppi_genes: Set[str] | None = None,
) -> Dict[str, str]:
    """Align approved names to unique HGNC obsolete names in the PPI, without merging targets."""
    files, detected = detect_table_files(path, table_format)
    symbols: Dict[str, str] = {}
    obsolete = {}
    for record in iter_records(files, detected):
        target_id = str(record.get("id", "")).strip()
        symbol = str(record.get("approvedSymbol", "")).strip().upper()
        if target_id and symbol:
            symbols[target_id] = symbol
            if ppi_genes is not None and symbol not in ppi_genes:
                entries = record.get("obsoleteSymbols")
                obsolete[target_id] = {
                    str(entry["label"]).strip().upper()
                    for entry in ([] if entries is None else entries)
                    if entry.get("source") == "HGNC"
                } & ppi_genes
    if not symbols:
        raise ValueError(f"No target symbols found in {path}")
    if ppi_genes is None:
        return symbols

    approved_owners = defaultdict(set)
    for target_id, symbol in symbols.items():
        approved_owners[symbol].add(target_id)
    candidates, ambiguous = {}, set()
    for target_id, aliases in obsolete.items():
        if len(aliases) > 1:
            ambiguous.add(target_id)
        elif len(aliases) == 1:
            alias = next(iter(aliases))
            if approved_owners[alias] - {target_id}:
                ambiguous.add(target_id)
            else:
                candidates[target_id] = alias
    alias_owners = defaultdict(set)
    for target_id, alias in candidates.items():
        alias_owners[alias].add(target_id)
    for target_id, alias in candidates.items():
        if len(alias_owners[alias]) == 1:
            symbols[target_id] = alias
        else:
            ambiguous.add(target_id)
    if ambiguous:
        warnings.warn(
            f"Retained approved names outside the PPI for {len(ambiguous)} targets with "
            f"ambiguous HGNC aliases: {', '.join(sorted(ambiguous))}", stacklevel=2,
        )
    return symbols


def load_association_scores(
    path: Path,
    diseases: Iterable[str],
    target_symbols: Dict[str, str],
    table_format: str,
    ot_release: str | None = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load indirect datatype scores, taking the maximum per gene and datatype."""
    requested = set(diseases)
    associations = {disease: {} for disease in sorted(requested)}
    unresolved = set()
    files, detected = detect_table_files(path, table_format)

    for record in iter_records(files, detected):
        disease_id = str(record.get("diseaseId", ""))
        if disease_id not in requested:
            continue
        if "datatypeId" in record and "score" in record:
            schema_release, datatype_field, score_field = "24.03", "datatypeId", "score"
        elif (
            record.get("aggregationType") == "datatypeId"
            and "aggregationValue" in record and "associationScore" in record
        ):
            schema_release, datatype_field, score_field = "26.03", "aggregationValue", "associationScore"
        else:
            raise ValueError(f"Unsupported Open Targets association schema in {path}")
        if ot_release is not None and schema_release != ot_release:
            raise ValueError(
                f"Association schema matches {schema_release}, not the configured release {ot_release}"
            )
        datatype = str(record[datatype_field] or "").strip().lower()
        if not datatype:
            raise ValueError(f"Missing association datatype in {path}")
        target_id = str(record.get("targetId", "")).strip()
        if target_id not in target_symbols:
            unresolved.add(target_id)
            continue
        scores = associations[disease_id].setdefault(target_symbols[target_id], {})
        score = float(record[score_field])
        scores[datatype] = max(score, scores.get(datatype, score))

    if unresolved:
        raise ValueError(
            f"{len(unresolved)} associated target IDs are absent from the Open Targets target table"
        )
    return associations


def non_literature_targets(
    associations: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Set[str]]:
    """Exclude genes with support beyond the literature (Europe PMC text-mining) datatype."""
    return {
        disease: {
            protein for protein, scores in proteins.items()
            if any(datatype != "literature" for datatype in scores)
        }
        for disease, proteins in associations.items()
    }


def load_associated_targets(
    path: Path,
    diseases: Iterable[str],
    target_symbols: Dict[str, str],
    table_format: str,
) -> Dict[str, Set[str]]:
    """Load non-literature indirect associations for each root disease."""
    return non_literature_targets(
        load_association_scores(path, diseases, target_symbols, table_format)
    )


def clinically_relevant(phase: object, status: object) -> bool:
    try:
        phase_value = float(phase)
    except (TypeError, ValueError):
        return False
    return phase_value >= 3 or (
        phase_value == 2 and str(status).strip().lower() == "completed"
    )


def collect_evidence(
    files: Sequence[Path],
    table_format: str,
    disease_ids: Set[str],
) -> pd.DataFrame:
    rows = []
    for record in iter_records(files, table_format):
        mapped_id = str(record.get("diseaseFromSourceMappedId", ""))
        disease_id = str(record.get("diseaseId", ""))
        if mapped_id not in disease_ids and disease_id not in disease_ids:
            continue
        if not clinically_relevant(
            record.get("clinicalPhase"), record.get("clinicalStatus")
        ):
            continue
        target_id = record.get("targetId")
        if target_id is not None and not pd.isna(target_id):
            rows.append(
                {
                    "diseaseFromSourceMappedId": record.get(
                        "diseaseFromSourceMappedId"
                    ),
                    "diseaseId": record.get("diseaseId"),
                    "targetId": target_id,
                    "targetFromSourceId": record.get("targetFromSourceId"),
                    "clinicalPhase": record.get("clinicalPhase"),
                    "clinicalStatus": record.get("clinicalStatus"),
                    "drugId": record.get("drugId"),
                }
            )
    return pd.DataFrame(rows).drop_duplicates()


def resolve_positive_targets(
    evidence: pd.DataFrame,
    target_symbols: Dict[str, str],
) -> Set[str]:
    target_ids = set(evidence["targetId"].dropna().astype(str))
    missing = sorted(target_ids - set(target_symbols))
    if missing:
        raise ValueError(
            f"{len(missing)} ChEMBL target IDs are absent from the Open Targets target table"
        )
    return {target_symbols[target_id] for target_id in target_ids}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_dataset(
    disease: str,
    descendants: Set[str],
    associated_targets: Set[str],
    target_symbols: Dict[str, str],
    evidence_files: Sequence[Path],
    evidence_format: str,
    ppi_genes: Set[str],
    druggable_targets: Set[str],
    output_dir: Path,
    processed_dir: Path,
    min_proteins_per_label: int,
    lookup_release: str = DEFAULT_THERAPEUTIC_TARGET_OT_RELEASE,
) -> Dict[str, object]:
    evidence = collect_evidence(evidence_files, evidence_format, descendants)
    if evidence.empty:
        raise RuntimeError(f"No clinically relevant 24.03 evidence for {disease}")

    positive_raw = resolve_positive_targets(evidence, target_symbols)
    positive = positive_raw & ppi_genes
    negative = (druggable_targets - associated_targets) & ppi_genes
    negative -= positive

    if len(positive) < min_proteins_per_label or len(negative) < min_proteins_per_label:
        raise RuntimeError(
            f"{disease}: insufficient labels ({len(positive)} positive, "
            f"{len(negative)} negative)"
        )

    write_json(
        processed_dir / f"positive_proteins_global_{disease}.json",
        {"GLOBAL": sorted(positive)},
    )
    write_json(
        processed_dir / f"negative_proteins_global_{disease}.json",
        {"GLOBAL": sorted(negative)},
    )
    write_json(
        processed_dir / f"raw_targets_global_{disease}.json",
        sorted(positive_raw),
    )

    records = [
        {"protein": protein, "label": 1, "disease_id": disease}
        for protein in sorted(positive)
    ]
    records += [
        {"protein": protein, "label": 0, "disease_id": disease}
        for protein in sorted(negative)
    ]
    output_csv = output_dir / f"therapeutic_target_{disease}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values("protein").to_csv(output_csv, index=False)

    print(
        f"{disease}: {len(positive)} positive, {len(negative)} negative "
        f"-> {output_csv}"
    )
    return {
        "disease_id": disease,
        "n_descendants_including_root": len(descendants),
        "n_evidence_rows": len(evidence),
        "n_positive_raw": len(positive_raw),
        "n_positive": len(positive),
        "n_negative": len(negative),
        "n_total": len(positive) + len(negative),
        "dataset_csv": output_csv.name,
        "clinical_evidence_release": "24.03",
        "lookup_release": lookup_release,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build therapeutic-target labels from fixed Open Targets tables."
    )
    parser.add_argument(
        "--ot-release", choices=["24.03", "26.03"],
        default=DEFAULT_THERAPEUTIC_TARGET_OT_RELEASE,
        help="Release of the disease, target and association lookup tables.",
    )
    parser.add_argument(
        "--drugbank-targets",
        type=Path,
        default=DEFAULT_THERAPEUTIC_TARGET_DRUGBANK_TARGETS,
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_THERAPEUTIC_TARGET_EVIDENCE_DIR,
    )
    parser.add_argument(
        "--ot-diseases-dir",
        type=Path,
        default=DEFAULT_THERAPEUTIC_TARGET_OT_DISEASES_DIR,
    )
    parser.add_argument(
        "--ot-targets-dir",
        type=Path,
        default=DEFAULT_THERAPEUTIC_TARGET_OT_TARGETS_DIR,
    )
    parser.add_argument(
        "--ot-associations-dir",
        type=Path,
        default=DEFAULT_THERAPEUTIC_TARGET_OT_ASSOCIATIONS_DIR,
    )
    parser.add_argument("--global-ppi-path", type=Path, default=DEFAULT_GLOBAL_PPI)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--diseases", nargs="+", default=list(THERAPEUTIC_TARGET_IDS)
    )
    parser.add_argument(
        "--evidence-format", choices=["auto", "json", "parquet"], default="auto"
    )
    parser.add_argument(
        "--ot-format", choices=["auto", "json", "parquet"], default="auto"
    )
    parser.add_argument("--processed-subdir", default="processed")
    parser.add_argument("--min-proteins-per-label", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diseases = [disease.strip() for disease in args.diseases if disease.strip()]
    output_dir = args.output_dir or detect_output_dir()
    processed_dir = output_dir / args.processed_subdir

    evidence_files, evidence_format = detect_table_files(
        args.evidence_dir, args.evidence_format
    )
    descendants = load_disease_descendants(
        args.ot_diseases_dir, diseases, args.ot_format
    )
    ppi_genes = read_global_ppi_genes(args.global_ppi_path)
    target_symbols = load_target_symbols(args.ot_targets_dir, args.ot_format, ppi_genes)
    associations = load_association_scores(
        args.ot_associations_dir, diseases, target_symbols, args.ot_format, args.ot_release
    )
    associated_targets = non_literature_targets(associations)
    druggable_targets = load_druggable_targets(args.drugbank_targets)

    rows = [
        build_dataset(
            disease=disease,
            descendants=descendants[disease],
            associated_targets=associated_targets[disease],
            target_symbols=target_symbols,
            evidence_files=evidence_files,
            evidence_format=evidence_format,
            ppi_genes=ppi_genes,
            druggable_targets=druggable_targets,
            output_dir=output_dir,
            processed_dir=processed_dir,
            min_proteins_per_label=args.min_proteins_per_label,
            lookup_release=args.ot_release,
        )
        for disease in diseases
    ]
    pd.DataFrame(rows).to_csv(
        output_dir / "therapeutic_target_summary.csv", index=False
    )
    if "MONDO_0005180" in associations:
        annotation_rows = [
            {
                "protein": protein,
                "datatype_scores_json": json.dumps(
                    [{"id": datatype, "score": score}
                     for datatype, score in sorted(scores.items())],
                    separators=(",", ":"),
                ),
                "open_targets_release": args.ot_release,
            }
            for protein, scores in sorted(associations["MONDO_0005180"].items())
        ]
        pd.DataFrame(annotation_rows, columns=[
            "protein", "datatype_scores_json", "open_targets_release"
        ]).to_csv(output_dir / "opentargets_parkinson_associations.csv", index=False)


if __name__ == "__main__":
    main()
