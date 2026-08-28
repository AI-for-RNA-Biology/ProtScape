#!/usr/bin/env python
"""
Build therapeutic target downstream datasets (split by disease EFO code).

This script is embedding-independent:
- Inputs come from therapeutic target evidence files + global PPI universe.
- Outputs are per-disease binary CSVs consumed by the downstream training pipeline.
- Positives use phase >= 3 or completed phase 2 evidence over the disease subtree.
- Negatives are approved-human DrugBank targets without a non-literature
  Open Targets association for the root disease query.
- ``--static-release-dir`` resolves disease descendants, target symbols, and
  direct or indirect associations entirely from a frozen Open Targets export.

Generated outputs:
- therapeutic_target_EFO_0000685.csv
- therapeutic_target_EFO_0003767.csv
- therapeutic_target_summary.csv

The script also writes intermediate processed label files in:
- processed/positive_proteins_global_<DISEASE>.json
- processed/negative_proteins_global_<DISEASE>.json
- processed/raw_targets_global_<DISEASE>.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
try:
    import requests
except ImportError as exc:
    raise SystemExit("Please install `requests` (pip install requests).") from exc


from ..config import (
    DEFAULT_GLOBAL_PPI,
    DEFAULT_THERAPEUTIC_TARGET_DATASET_DIR,
    DEFAULT_THERAPEUTIC_TARGET_DRUGBANK_TARGETS,
    DEFAULT_THERAPEUTIC_TARGET_EVIDENCE_DIR,
)

DEFAULT_DISEASES = (
    "EFO_0003767",
    "EFO_0000685",
    "EFO_0000305",
    "EFO_1001207",
    "EFO_0000571",
    "EFO_0000676",
    "EFO_0001361",
    "MONDO_0005148",
    "MONDO_0007915",
    "EFO_0000274",
    "MONDO_0004979",
    "EFO_0000341",
    "EFO_1001249",
    "EFO_0003884",
    "MONDO_0005180",
)

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
UNIPROT_IDMAPPING_BASE = "https://rest.uniprot.org/idmapping"
STATIC_ASSOCIATION_DATASETS = {
    "direct": "associationByDatatypeDirect",
    "indirect": "associationByDatatypeIndirect",
}
STATIC_PARQUET_ASSOCIATION_DATASETS = {
    "direct": "association_by_datatype_direct",
    "indirect": "association_by_datatype_indirect",
}


@dataclass(frozen=True)
class StaticOpenTargetsRelease:
    """Open Targets records needed to build labels without network access."""

    release_dir: Path
    release_name: str
    storage_format: str
    association_scope: str
    disease_dataset: str
    target_dataset: str
    association_dataset: str
    descendants: Dict[str, Set[str]]
    ensg_to_symbol: Dict[str, str]
    associated_targets: Dict[str, Set[str]]
    unmapped_association_targets: Dict[str, int]


def detect_output_dir() -> Path:
    """Return the configured directory for generated therapeutic-target tables."""
    return DEFAULT_THERAPEUTIC_TARGET_DATASET_DIR


def read_global_ppi_genes(ppi_path: Path) -> Set[str]:
    """Load all genes from a global PPI edge list."""
    genes: Set[str] = set()
    with open(ppi_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            genes.add(parts[0].upper())
            genes.add(parts[1].upper())
    return genes


def load_druggable_targets(all_drug_targets_path: Path) -> Set[str]:
    """
    Load druggable targets from approved-drug table.

    Returns uppercase union of 'Gene Name' and 'GenAtlas ID' for human rows.
    """
    df = pd.read_csv(all_drug_targets_path)
    if "Species" not in df.columns:
        # Some files are saved with an index column.
        df = pd.read_csv(all_drug_targets_path, index_col=0)

    required = {"Species", "Gene Name", "GenAtlas ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {all_drug_targets_path}: {sorted(missing)}"
        )

    human = df[df["Species"].astype(str).str.lower() == "humans"]
    targets: Set[str] = set()
    for col in ("Gene Name", "GenAtlas ID"):
        for val in human[col].dropna().astype(str):
            gene = val.strip().upper()
            if gene and gene != "NAN":
                targets.add(gene)
    return targets


def detect_evidence_files(evidence_dir: Path, evidence_format: str) -> Tuple[List[Path], str]:
    """Find evidence files recursively."""
    if not evidence_dir.exists():
        raise FileNotFoundError(f"Evidence directory not found: {evidence_dir}")

    parquet_files = sorted(
        set(evidence_dir.rglob("*.snappy.parquet")) | set(evidence_dir.rglob("*.parquet"))
    )
    json_files = sorted(evidence_dir.rglob("*.json"))

    if evidence_format == "auto":
        if parquet_files:
            return parquet_files, "parquet"
        if json_files:
            return json_files, "json"
        raise ValueError(f"No evidence files found in {evidence_dir}")

    if evidence_format == "parquet":
        if not parquet_files:
            raise ValueError(f"No parquet evidence files found in {evidence_dir}")
        return parquet_files, "parquet"

    if evidence_format == "json":
        if not json_files:
            raise ValueError(f"No json evidence files found in {evidence_dir}")
        return json_files, "json"

    raise ValueError(f"Unknown evidence_format={evidence_format}")


def _static_jsonl_files(release_dir: Path, dataset: str) -> List[Path]:
    """Return a complete Open Targets Spark JSON export."""
    dataset_dir = release_dir / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Missing Open Targets static dataset directory: {dataset_dir}"
        )
    if not (dataset_dir / "_SUCCESS").exists():
        raise FileNotFoundError(
            f"Incomplete Open Targets static dataset (missing _SUCCESS): {dataset_dir}"
        )
    files = sorted(dataset_dir.glob("part-*.json"))
    if not files:
        raise FileNotFoundError(f"No part-*.json files found in {dataset_dir}")
    return files


def _iter_static_jsonl(files: Sequence[Path]):
    """Yield records from static JSONL parts, failing on corrupt snapshots."""
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected a JSON object in {path}:{line_number}"
                    )
                yield record


def _detect_static_release_format(release_dir: Path) -> str:
    """Detect one release layout without mixing JSON and Parquet snapshots."""
    parquet_paths = (
        release_dir / "disease",
        release_dir / "target",
        release_dir / "association_by_datatype_direct",
        release_dir / "association_by_datatype_indirect",
    )
    json_paths = (
        release_dir / "diseases",
        release_dir / "searchTarget",
        release_dir / "associationByDatatypeDirect",
        release_dir / "associationByDatatypeIndirect",
    )
    has_parquet = any(path.exists() for path in parquet_paths)
    has_json = any(path.exists() for path in json_paths)
    if has_parquet and has_json:
        raise ValueError(
            "Static Open Targets directory mixes JSON and Parquet release layouts: "
            f"{release_dir}"
        )
    if has_parquet:
        return "parquet"
    if has_json:
        return "json"
    raise FileNotFoundError(
        "No supported Open Targets static release layout found under "
        f"{release_dir}; expected diseases/ (legacy JSON) or disease/ "
        "(official Parquet)."
    )


def _static_parquet_files(release_dir: Path, dataset: str) -> List[Path]:
    """Return deterministic, non-empty files for an official Parquet dataset."""
    dataset_dir = release_dir / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Missing Open Targets Parquet dataset directory: {dataset_dir}"
        )
    if dataset == "disease":
        disease_file = dataset_dir / "disease.parquet"
        if not disease_file.is_file():
            raise FileNotFoundError(
                f"Missing official Open Targets disease file: {disease_file}"
            )
        files = [disease_file]
    else:
        if not (dataset_dir / "_SUCCESS").is_file():
            raise FileNotFoundError(
                "Incomplete Open Targets Parquet dataset (missing _SUCCESS): "
                f"{dataset_dir}"
            )
        files = sorted(dataset_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No *.parquet files found in {dataset_dir}")
    empty = [str(path) for path in files if path.stat().st_size == 0]
    if empty:
        raise ValueError(
            f"Empty Parquet files in Open Targets dataset {dataset}: "
            + ", ".join(empty)
        )
    return files


def _parquet_schema_columns(path: Path, dataset: str) -> Set[str]:
    """Read Parquet metadata so schema aliases can be validated before scanning."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading an Open Targets Parquet release requires pyarrow"
        ) from exc
    try:
        return set(pq.ParquetFile(path).schema_arrow.names)
    except Exception as exc:
        raise ValueError(f"Invalid Parquet file in {dataset}: {path}") from exc


def _iter_static_parquet(
    files: Sequence[Path],
    columns: Sequence[str],
    dataset: str,
    filters: Optional[Sequence[Tuple[str, str, object]]] = None,
):
    """Yield projected Parquet parts, failing explicitly on schema/read errors."""
    required = set(columns)
    for path in files:
        missing = required.difference(_parquet_schema_columns(path, dataset))
        if missing:
            raise ValueError(
                f"Open Targets Parquet dataset {dataset} is missing columns "
                f"{sorted(missing)} in {path}"
            )
        try:
            yield pd.read_parquet(
                path,
                columns=list(columns),
                filters=list(filters) if filters else None,
                engine="pyarrow",
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to read Open Targets Parquet dataset {dataset}: {path}"
            ) from exc


def _string_values(value: object) -> Set[str]:
    """Normalize a scalar or Arrow/Pandas list value to non-empty strings."""
    if value is None:
        return set()
    if isinstance(value, str):
        values = [value]
    elif hasattr(value, "tolist"):
        values = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    out: Set[str] = set()
    for item in values:
        if item is None:
            continue
        try:
            if bool(pd.isna(item)):
                continue
        except (TypeError, ValueError):
            pass
        text = str(item).strip()
        if text:
            out.add(text)
    return out


def _load_static_descendants(
    release_dir: Path,
    diseases: Sequence[str],
    storage_format: str,
) -> Dict[str, Set[str]]:
    requested = {str(disease).strip() for disease in diseases if str(disease).strip()}
    if not requested:
        raise ValueError("At least one disease ID is required")
    descendants: Dict[str, Set[str]] = {}
    if storage_format == "json":
        files = _static_jsonl_files(release_dir, "diseases")
        for record in _iter_static_jsonl(files):
            disease_id = str(record.get("id", "")).strip()
            if disease_id not in requested:
                continue
            descendants[disease_id] = {disease_id}
            descendants[disease_id].update(_string_values(record.get("descendants")))
    elif storage_format == "parquet":
        files = _static_parquet_files(release_dir, "disease")
        filters = [("id", "in", sorted(requested))]
        for frame in _iter_static_parquet(
            files, ("id", "descendants"), "disease", filters
        ):
            for disease_id, values in frame.itertuples(index=False, name=None):
                disease_id = str(disease_id).strip()
                if disease_id not in requested:
                    continue
                if disease_id in descendants:
                    raise ValueError(
                        f"Duplicate disease record in static release: {disease_id}"
                    )
                descendants[disease_id] = {disease_id}
                descendants[disease_id].update(_string_values(values))
    else:
        raise ValueError(f"Unsupported static release format: {storage_format}")

    missing = sorted(requested.difference(descendants))
    if missing:
        raise KeyError(
            "Disease IDs absent from the Open Targets static diseases export: "
            + ", ".join(missing)
        )
    return descendants


def _load_static_target_symbols(
    release_dir: Path,
    storage_format: str,
) -> Dict[str, str]:
    symbols: Dict[str, str] = {}
    if storage_format == "json":
        files = _static_jsonl_files(release_dir, "searchTarget")
        record_groups = [
            (
                (record.get("id"), record.get("name"))
                for record in _iter_static_jsonl(files)
                if str(record.get("entity", "")).strip().lower() == "target"
            )
        ]
        dataset = "searchTarget"
    elif storage_format == "parquet":
        files = _static_parquet_files(release_dir, "target")
        record_groups = (
            frame.itertuples(index=False, name=None)
            for frame in _iter_static_parquet(
                files, ("id", "approvedSymbol"), "target"
            )
        )
        dataset = "target"
    else:
        raise ValueError(f"Unsupported static release format: {storage_format}")

    for records in record_groups:
        for raw_target_id, raw_symbol in records:
            if pd.isna(raw_target_id) or pd.isna(raw_symbol):
                continue
            target_id = str(raw_target_id).strip()
            symbol = str(raw_symbol).strip().upper()
            if not target_id or not symbol or symbol == "NAN":
                continue
            previous = symbols.get(target_id)
            if previous is not None and previous != symbol:
                raise ValueError(
                    f"Conflicting symbols for {target_id} in {dataset}: "
                    f"{previous!r} and {symbol!r}"
                )
            symbols[target_id] = symbol
    if not symbols:
        raise ValueError(f"The Open Targets static {dataset} export contained no targets")
    return symbols


def _load_static_associated_targets(
    release_dir: Path,
    diseases: Sequence[str],
    association_scope: str,
    ensg_to_symbol: Dict[str, str],
    storage_format: str,
) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    if association_scope not in STATIC_ASSOCIATION_DATASETS:
        raise ValueError(
            f"Invalid association_scope={association_scope!r}; expected direct or indirect"
        )

    requested = {str(disease).strip() for disease in diseases if str(disease).strip()}
    associated_ids: Dict[str, Set[str]] = {disease: set() for disease in requested}
    if storage_format == "json":
        dataset = STATIC_ASSOCIATION_DATASETS[association_scope]
        files = _static_jsonl_files(release_dir, dataset)
        record_groups = [
            (
                (
                    record.get("diseaseId"),
                    record.get("targetId"),
                    record.get("datatypeId"),
                    record.get("score"),
                )
                for record in _iter_static_jsonl(files)
            )
        ]
    elif storage_format == "parquet":
        dataset = STATIC_PARQUET_ASSOCIATION_DATASETS[association_scope]
        files = _static_parquet_files(release_dir, dataset)
        schema = _parquet_schema_columns(files[0], dataset)
        if {"aggregationValue", "associationScore"}.issubset(schema):
            datatype_column, score_column = "aggregationValue", "associationScore"
        elif {"datatypeId", "score"}.issubset(schema):
            datatype_column, score_column = "datatypeId", "score"
        else:
            raise ValueError(
                f"Open Targets Parquet dataset {dataset} must contain either "
                "aggregationValue/associationScore or datatypeId/score"
            )
        columns = ("diseaseId", "targetId", datatype_column, score_column)
        filters = [("diseaseId", "in", sorted(requested))]
        record_groups = (
            frame.itertuples(index=False, name=None)
            for frame in _iter_static_parquet(files, columns, dataset, filters)
        )
    else:
        raise ValueError(f"Unsupported static release format: {storage_format}")

    for records in record_groups:
        for raw_disease_id, raw_target_id, raw_datatype, raw_score in records:
            disease_id = str(raw_disease_id).strip()
            if disease_id not in requested:
                continue
            if str(raw_datatype).strip().lower() == "literature":
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if pd.isna(score) or score <= 0:
                continue
            if pd.isna(raw_target_id):
                continue
            target_id = str(raw_target_id).strip()
            if target_id and target_id != "nan":
                associated_ids[disease_id].add(target_id)

    associated_symbols: Dict[str, Set[str]] = {}
    unmapped: Dict[str, int] = {}
    for disease, target_ids in associated_ids.items():
        associated_symbols[disease] = {
            ensg_to_symbol[target_id]
            for target_id in target_ids
            if target_id in ensg_to_symbol
        }
        unmapped[disease] = len(target_ids.difference(ensg_to_symbol))
    return associated_symbols, unmapped


def load_static_open_targets_release(
    release_dir: Path,
    diseases: Sequence[str],
    association_scope: str,
    release_name: Optional[str] = None,
) -> StaticOpenTargetsRelease:
    """Load the frozen inputs used in place of every live Open Targets query."""
    release_dir = Path(release_dir).expanduser()
    release_name = str(release_name or release_dir.name).strip()
    if not release_name:
        raise ValueError("The static Open Targets release name cannot be empty")
    storage_format = _detect_static_release_format(release_dir)
    descendants = _load_static_descendants(release_dir, diseases, storage_format)
    ensg_to_symbol = _load_static_target_symbols(release_dir, storage_format)
    associated_targets, unmapped = _load_static_associated_targets(
        release_dir,
        diseases,
        association_scope,
        ensg_to_symbol,
        storage_format,
    )
    if storage_format == "json":
        disease_dataset = "diseases"
        target_dataset = "searchTarget"
        association_dataset = STATIC_ASSOCIATION_DATASETS[association_scope]
    else:
        disease_dataset = "disease"
        target_dataset = "target"
        association_dataset = STATIC_PARQUET_ASSOCIATION_DATASETS[association_scope]
    return StaticOpenTargetsRelease(
        release_dir=release_dir,
        release_name=release_name,
        storage_format=storage_format,
        association_scope=association_scope,
        disease_dataset=disease_dataset,
        target_dataset=target_dataset,
        association_dataset=association_dataset,
        descendants=descendants,
        ensg_to_symbol=ensg_to_symbol,
        associated_targets=associated_targets,
        unmapped_association_targets=unmapped,
    )


def get_disease_descendants_efo(disease: str) -> List[str]:
    """Get descendants from EFO OLS API."""
    if disease.startswith("EFO_"):
        encoded = (
            "https://www.ebi.ac.uk/ols/api/ontologies/efo/terms/"
            f"http%253A%252F%252Fwww.ebi.ac.uk%252Fefo%252F{disease}"
            "/hierarchicalDescendants?size=5000"
        )
    elif disease.startswith("MONDO_"):
        encoded = (
            "https://www.ebi.ac.uk/ols/api/ontologies/efo/terms/"
            f"http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252F{disease}"
            "/hierarchicalDescendants?size=5000"
        )
    else:
        raise ValueError(f"Unsupported disease code for EFO descendants: {disease}")

    resp = requests.get(encoded, timeout=120)
    resp.raise_for_status()
    payload = resp.json()

    out: Set[str] = {disease}
    for term in payload.get("_embedded", {}).get("terms", []):
        short_form = term.get("short_form")
        if isinstance(short_form, str) and short_form.startswith("EFO_"):
            out.add(short_form)
        refs = term.get("annotation", {}).get("database_cross_reference", [])
        if isinstance(refs, str):
            refs = [refs]
        if isinstance(refs, list):
            for ref in refs:
                ref_norm = str(ref).replace(":", "_")
                if ref_norm.startswith("EFO_"):
                    out.add(ref_norm)

    return sorted(out)


def get_disease_descendants_ot_api(disease: str) -> List[str]:
    """Get descendants from the OpenTargets GraphQL disease API."""
    query = """
    query DiseaseDesc($id: String!) {
      disease(efoId: $id) {
        id
        name
        descendants
      }
    }
    """
    resp = requests.post(
        OT_URL,
        json={"query": query, "variables": {"id": disease}},
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()
    disease_obj = (payload.get("data") or {}).get("disease")
    if disease_obj is None:
        raise RuntimeError(f"OpenTargets disease query returned no record for {disease}: {payload}")

    out: Set[str] = {disease}
    descendants = disease_obj.get("descendants") or []
    if isinstance(descendants, str):
        descendants = [descendants]
    for descendant in descendants:
        desc_id = str(descendant).strip()
        if desc_id:
            out.add(desc_id)
    return sorted(out)


def _is_clinically_relevant(phase: object, status: object) -> bool:
    try:
        phase_val = float(phase)
    except Exception:
        return False

    if phase_val >= 3:
        return True
    if phase_val == 2 and str(status).strip().lower() == "completed":
        return True
    return False


def _extract_evidence_row(rec: Dict[str, object]) -> Optional[Dict[str, object]]:
    target_id = rec.get("targetId")
    target_src_id = rec.get("targetFromSourceId")
    if pd.isna(target_id) and pd.isna(target_src_id):
        return None
    return {
        "diseaseFromSourceMappedId": rec.get("diseaseFromSourceMappedId"),
        "diseaseId": rec.get("diseaseId"),
        "targetId": target_id,
        "targetFromSourceId": target_src_id,
        "clinicalPhase": rec.get("clinicalPhase"),
        "clinicalStatus": rec.get("clinicalStatus"),
        "drugId": rec.get("drugId"),
    }


def collect_clinically_relevant_evidence(
    evidence_files: Sequence[Path],
    evidence_format: str,
    disease_ids: Set[str],
) -> pd.DataFrame:
    """Collect clinically relevant evidence rows for a disease subtree."""
    rows: List[Dict[str, object]] = []

    if evidence_format == "json":
        for fp in evidence_files:
            with open(fp, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in evidence file {fp}:{line_number}: {exc}"
                        ) from exc
                    if not isinstance(rec, dict):
                        raise ValueError(
                            f"Expected a JSON object in evidence file "
                            f"{fp}:{line_number}"
                        )
                    did_mapped = str(rec.get("diseaseFromSourceMappedId", ""))
                    did = str(rec.get("diseaseId", ""))
                    if did_mapped not in disease_ids and did not in disease_ids:
                        continue
                    if not _is_clinically_relevant(
                        rec.get("clinicalPhase"),
                        rec.get("clinicalStatus"),
                    ):
                        continue
                    row = _extract_evidence_row(rec)
                    if row is not None:
                        rows.append(row)
    elif evidence_format == "parquet":
        for fp in evidence_files:
            df = pd.read_parquet(fp)
            if df.empty:
                continue
            disease_mask = df["diseaseFromSourceMappedId"].astype(str).isin(disease_ids)
            if "diseaseId" in df.columns:
                disease_mask = disease_mask | df["diseaseId"].astype(str).isin(disease_ids)
            df = df.loc[disease_mask]
            if df.empty:
                continue
            phase = pd.to_numeric(df["clinicalPhase"], errors="coerce")
            if "clinicalStatus" in df.columns:
                status = df["clinicalStatus"].astype(str).str.lower()
            else:
                status = pd.Series([""] * len(df), index=df.index)
            clin_mask = (phase >= 3) | ((phase == 2) & (status == "completed"))
            df = df.loc[clin_mask]
            if df.empty:
                continue
            keep_cols = [
                "diseaseFromSourceMappedId",
                "diseaseId",
                "targetId",
                "targetFromSourceId",
                "clinicalPhase",
                "clinicalStatus",
                "drugId",
            ]
            keep_cols = [c for c in keep_cols if c in df.columns]
            rows.extend(df[keep_cols].to_dict(orient="records"))
    else:
        raise ValueError(f"Unsupported evidence format: {evidence_format}")

    if not rows:
        return pd.DataFrame(
            columns=[
                "diseaseFromSourceMappedId",
                "diseaseId",
                "targetId",
                "targetFromSourceId",
                "clinicalPhase",
                "clinicalStatus",
                "drugId",
            ]
        )

    out = pd.DataFrame(rows).drop_duplicates()
    return out


def get_ot_associated_targets(
    disease: str,
    page_size: int = 2048,
    max_pages: int = 12,
) -> Tuple[Set[str], Dict[str, str]]:
    """
    Fetch associated targets from OpenTargets for one disease.

    Returns:
    - associated_target_symbols: all symbols except pure-literature-only associations
    - ensg_to_symbol: mapping from Ensembl gene id to approved symbol
    """
    query = """
    query disease($efoId: String!, $index: Int!, $size: Int!) {
      disease(efoId: $efoId) {
        associatedTargets(page: { index: $index, size: $size }) {
          rows {
            datatypeScores { id score }
            target { id approvedSymbol }
          }
        }
      }
    }
    """

    associated: Set[str] = set()
    ensg_to_symbol: Dict[str, str] = {}

    for page_idx in range(max_pages):
        payload = {
            "query": query,
            "variables": {"efoId": disease, "index": page_idx, "size": page_size},
        }
        resp = requests.post(OT_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        disease_obj = (data.get("data") or {}).get("disease") or {}
        rows = (disease_obj.get("associatedTargets") or {}).get("rows")

        if rows is None:
            break
        if isinstance(rows, dict):
            rows = [rows]
        if not rows:
            break

        for row in rows:
            target = row.get("target") or {}
            ensg = str(target.get("id", "")).strip()
            sym = str(target.get("approvedSymbol", "")).strip().upper()
            if ensg and sym:
                ensg_to_symbol[ensg] = sym

            datatype_scores = row.get("datatypeScores") or []
            if not datatype_scores:
                continue
            if (
                len(datatype_scores) == 1
                and str(datatype_scores[0].get("id", "")).strip().lower() == "literature"
            ):
                continue
            if sym:
                associated.add(sym)

        if len(rows) < page_size:
            break

    return associated, ensg_to_symbol


def _wait_for_uniprot_job(job_id: str, poll_seconds: float = 3.0, timeout_seconds: float = 600.0) -> None:
    """Poll UniProt id-mapping job until ready."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        r = requests.get(f"{UNIPROT_IDMAPPING_BASE}/status/{job_id}", timeout=30)
        r.raise_for_status()
        status = r.json()
        if "jobStatus" not in status:
            return
        js = str(status["jobStatus"]).upper()
        if js == "RUNNING" or js == "NEW":
            time.sleep(poll_seconds)
            continue
        raise RuntimeError(f"UniProt mapping job {job_id} failed with status={js}")
    raise TimeoutError(f"Timed out waiting for UniProt mapping job {job_id}")


def map_uniprot_to_gene_names(uniprot_ids: Iterable[str]) -> Dict[str, str]:
    """Map UniProt accessions to gene symbols via UniProt idmapping API."""
    ids = sorted({str(x).strip().upper().split("-")[0] for x in uniprot_ids if str(x).strip()})
    if not ids:
        return {}

    submit = requests.post(
        f"{UNIPROT_IDMAPPING_BASE}/run",
        data={"from": "UniProtKB_AC-ID", "to": "Gene_Name", "ids": ",".join(ids)},
        timeout=60,
    )
    submit.raise_for_status()
    job_id = submit.json().get("jobId")
    if not job_id:
        raise RuntimeError("UniProt mapping submission did not return a jobId")

    _wait_for_uniprot_job(job_id)
    stream_url = f"{UNIPROT_IDMAPPING_BASE}/stream/{job_id}?format=json"
    resp = requests.get(stream_url, timeout=180)
    resp.raise_for_status()
    payload = resp.json()

    mapping: Dict[str, str] = {}
    for rec in payload.get("results", []):
        src = str(rec.get("from", "")).strip().upper().split("-")[0]
        to = rec.get("to")
        gene = None
        if isinstance(to, str):
            gene = to
        elif isinstance(to, dict):
            gene = to.get("value") or to.get("geneName")
            if gene is None:
                genes = to.get("genes")
                if isinstance(genes, list) and genes:
                    g0 = genes[0]
                    if isinstance(g0, dict):
                        g_name = g0.get("geneName")
                        if isinstance(g_name, dict):
                            gene = g_name.get("value")
                        elif isinstance(g_name, str):
                            gene = g_name
        if src and gene:
            mapping[src] = str(gene).strip().upper()
    return mapping


def map_ensembl_to_gene_symbols(ensembl_ids: Iterable[str], batch_size: int = 200) -> Dict[str, str]:
    """Map Ensembl gene ids to symbols via Ensembl REST lookup."""
    ids = sorted({str(x).strip() for x in ensembl_ids if str(x).strip().startswith("ENSG")})
    if not ids:
        return {}

    out: Dict[str, str] = {}
    url = "https://rest.ensembl.org/lookup/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        resp = requests.post(
            url,
            data=json.dumps({"ids": batch}),
            headers=headers,
            timeout=120,
        )
        if resp.status_code != 200:
            continue
        payload = resp.json()
        if not isinstance(payload, dict):
            continue
        for ensg, rec in payload.items():
            if not isinstance(rec, dict):
                continue
            sym = rec.get("display_name")
            if sym:
                out[str(ensg)] = str(sym).upper()
        time.sleep(0.05)
    return out


def resolve_positive_targets(
    evidence_df: pd.DataFrame,
    ensg_to_symbol_from_ot: Dict[str, str],
) -> Tuple[Set[str], Dict[str, int]]:
    """Resolve evidence target ids to gene symbols."""
    if evidence_df.empty:
        return set(), {
            "n_uniprot_ids": 0,
            "n_uniprot_mapped": 0,
            "n_ensg_ids": 0,
            "n_ensg_mapped_ot": 0,
            "n_ensg_mapped_ensembl": 0,
        }

    uniprot_ids = set(
        evidence_df["targetFromSourceId"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    ensg_ids = set(
        evidence_df["targetId"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    uniprot_map = map_uniprot_to_gene_names(uniprot_ids)
    positives: Set[str] = set(uniprot_map.values())

    n_mapped_ot = 0
    missing_ensg: List[str] = []
    for ensg in ensg_ids:
        symbol = ensg_to_symbol_from_ot.get(ensg)
        if symbol:
            positives.add(symbol.upper())
            n_mapped_ot += 1
        else:
            missing_ensg.append(ensg)

    ensembl_map = map_ensembl_to_gene_symbols(missing_ensg) if missing_ensg else {}
    positives.update(ensembl_map.values())

    stats = {
        "n_uniprot_ids": len(uniprot_ids),
        "n_uniprot_mapped": len(uniprot_map),
        "n_ensg_ids": len(ensg_ids),
        "n_ensg_mapped_ot": n_mapped_ot,
        "n_ensg_mapped_ensembl": len(ensembl_map),
    }
    return positives, stats


def resolve_positive_targets_static(
    evidence_df: pd.DataFrame,
    ensg_to_symbol: Dict[str, str],
) -> Tuple[Set[str], Dict[str, int]]:
    """Resolve ChEMBL evidence through the frozen searchTarget export only."""
    if evidence_df.empty:
        return set(), {
            "n_uniprot_ids": 0,
            "n_uniprot_mapped": 0,
            "n_ensg_ids": 0,
            "n_ensg_mapped_ot": 0,
            "n_ensg_mapped_ensembl": 0,
        }

    uniprot_ids = {
        str(value).strip()
        for value in evidence_df["targetFromSourceId"].dropna()
        if str(value).strip()
    }
    ensg_ids = {
        str(value).strip()
        for value in evidence_df["targetId"].dropna()
        if str(value).strip()
    }
    mapped = {
        target_id: ensg_to_symbol[target_id]
        for target_id in ensg_ids
        if target_id in ensg_to_symbol
    }
    positives = {symbol.upper() for symbol in mapped.values()}
    return positives, {
        "n_uniprot_ids": len(uniprot_ids),
        "n_uniprot_mapped": 0,
        "n_ensg_ids": len(ensg_ids),
        "n_ensg_mapped_ot": len(mapped),
        "n_ensg_mapped_ensembl": 0,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _require_fresh_static_outputs(
    output_dir: Path,
    diseases: Sequence[str],
    force: bool,
) -> None:
    """Prevent an existing reconstruction from receiving new provenance."""
    if force:
        return
    artifacts = [
        output_dir / f"therapeutic_target_{str(disease).strip()}.csv"
        for disease in diseases
        if str(disease).strip()
    ]
    artifacts.extend(
        [
            output_dir / "therapeutic_target_summary.csv",
            output_dir / "therapeutic_target_manifest.json",
        ]
    )
    existing = sorted(path for path in artifacts if path.exists())
    if existing:
        raise FileExistsError(
            "Static therapeutic-target outputs already exist; rerun with --force "
            "to rebuild them with the requested provenance: "
            + ", ".join(str(path) for path in existing)
        )


def _static_manifest_provenance(
    static_release: StaticOpenTargetsRelease,
    evidence_release_name: str,
) -> Dict[str, object]:
    """Return explicit evidence and association release provenance."""
    evidence_release_name = str(evidence_release_name).strip()
    if not evidence_release_name:
        raise ValueError("The ChEMBL evidence release name cannot be empty")
    return {
        # Retained for downstream compatibility; this is the association release.
        "open_targets_release": static_release.release_name,
        "open_targets_association_release": static_release.release_name,
        "open_targets_evidence_release": evidence_release_name,
        "open_targets_release_dir": str(static_release.release_dir.resolve()),
        "open_targets_storage_format": static_release.storage_format,
    }


def build_dataset_for_disease(
    disease: str,
    evidence_files: Sequence[Path],
    evidence_format: str,
    descendants_source: str,
    ppi_genes: Set[str],
    druggable_targets: Set[str],
    out_dir: Path,
    processed_dir: Path,
    min_proteins_per_label: int = 10,
    force: bool = False,
    static_release: Optional[StaticOpenTargetsRelease] = None,
) -> Dict[str, object]:
    """Build one disease CSV and intermediate processed files."""
    out_csv = out_dir / f"therapeutic_target_{disease}.csv"
    pos_json = processed_dir / f"positive_proteins_global_{disease}.json"
    neg_json = processed_dir / f"negative_proteins_global_{disease}.json"
    raw_json = processed_dir / f"raw_targets_global_{disease}.json"

    if out_csv.exists() and not force:
        if static_release is not None:
            raise FileExistsError(
                f"Static therapeutic-target output already exists: {out_csv}. "
                "Use force=True (CLI: --force) to rebuild it."
            )
        df_existing = pd.read_csv(out_csv)
        n_pos = int((df_existing["label"] == 1).sum())
        n_neg = int((df_existing["label"] == 0).sum())
        print(f"[SKIP] {disease}: using existing dataset {out_csv}")
        return {
            "disease_id": disease,
            "n_descendants": None,
            "n_evidence_rows": None,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_total": int(df_existing.shape[0]),
            "dataset_csv": out_csv.name,
            "status": "existing",
        }

    if static_release is not None:
        if disease not in static_release.descendants:
            raise KeyError(f"Disease {disease} was not loaded from the static release")
        descendants = static_release.descendants[disease]
    elif descendants_source == "ot":
        descendants = set(get_disease_descendants_ot_api(disease))
    elif descendants_source == "efo":
        descendants = set(get_disease_descendants_efo(disease))
    elif descendants_source == "none":
        descendants = {disease}
    else:
        raise ValueError(
            f"Unsupported descendants_source={descendants_source}; valid: ot, efo, none"
        )
    print(f"[INFO] {disease}: {len(descendants)} disease IDs (with descendants)")

    evidence_df = collect_clinically_relevant_evidence(
        evidence_files=evidence_files,
        evidence_format=evidence_format,
        disease_ids=descendants,
    )
    print(f"[INFO] {disease}: {len(evidence_df)} clinically relevant evidence rows")
    if evidence_df.empty:
        raise RuntimeError(
            f"No clinically relevant evidence found for {disease}. "
            f"Check evidence files and disease code."
        )

    if static_release is not None:
        all_associated_targets = static_release.associated_targets[disease]
        positive_raw, map_stats = resolve_positive_targets_static(
            evidence_df,
            static_release.ensg_to_symbol,
        )
    else:
        all_associated_targets, ensg_to_symbol = get_ot_associated_targets(disease)
        positive_raw, map_stats = resolve_positive_targets(evidence_df, ensg_to_symbol)
    all_associated_targets = {g.upper() for g in all_associated_targets}
    positive_raw = {g.upper() for g in positive_raw}

    positive = positive_raw.intersection(ppi_genes)
    negative = (
        druggable_targets
        .difference(all_associated_targets)
        .intersection(ppi_genes)
    )
    negative = negative.difference(positive)

    if not positive:
        raise RuntimeError(
            f"{disease}: empty positive set after mapping/filtering to global PPI."
        )
    if not negative:
        raise RuntimeError(
            f"{disease}: empty negative set after filtering to druggable/PPI."
        )
    if len(positive) < min_proteins_per_label:
        raise RuntimeError(
            f"{disease}: only {len(positive)} positive proteins after filtering; "
            f"minimum required is {min_proteins_per_label}."
        )
    if len(negative) < min_proteins_per_label:
        raise RuntimeError(
            f"{disease}: only {len(negative)} negative proteins after filtering; "
            f"minimum required is {min_proteins_per_label}."
        )

    # Write the processed label tables.
    write_json(pos_json, {"GLOBAL": sorted(positive)})
    write_json(neg_json, {"GLOBAL": sorted(negative)})
    write_json(raw_json, sorted(positive_raw))

    records = (
        [{"protein": g, "label": 1, "disease_id": disease} for g in sorted(positive)]
        + [{"protein": g, "label": 0, "disease_id": disease} for g in sorted(negative)]
    )
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["protein"], keep="first").sort_values("protein")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(
        f"[OK] {disease}: positives={len(positive)} negatives={len(negative)} "
        f"total={len(df)} -> {out_csv}"
    )

    return {
        "disease_id": disease,
        "n_descendants": len(descendants),
        "n_evidence_rows": int(len(evidence_df)),
        "n_positive": int(len(positive)),
        "n_negative": int(len(negative)),
        "min_proteins_per_label": int(min_proteins_per_label),
        "n_total": int(len(df)),
        "n_uniprot_ids": map_stats["n_uniprot_ids"],
        "n_uniprot_mapped": map_stats["n_uniprot_mapped"],
        "n_ensg_ids": map_stats["n_ensg_ids"],
        "n_ensg_mapped_ot": map_stats["n_ensg_mapped_ot"],
        "n_ensg_mapped_ensembl": map_stats["n_ensg_mapped_ensembl"],
        "open_targets_mode": (
            f"static_{static_release.association_scope}"
            if static_release is not None
            else "live"
        ),
        "open_targets_release_dir": (
            str(static_release.release_dir) if static_release is not None else None
        ),
        "open_targets_release": (
            static_release.release_name if static_release is not None else None
        ),
        "n_unmapped_association_targets": (
            static_release.unmapped_association_targets[disease]
            if static_release is not None
            else None
        ),
        "dataset_csv": out_csv.name,
        "status": "rebuilt",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build therapeutic target downstream datasets (split by EFO disease)."
    )
    parser.add_argument(
        "--drugbank-targets",
        type=Path,
        default=DEFAULT_THERAPEUTIC_TARGET_DRUGBANK_TARGETS,
        help=(
            "Approved-drug target table. Defaults to "
            "therapeutic_target_drugbank_targets in configs/paths.yaml."
        ),
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_THERAPEUTIC_TARGET_EVIDENCE_DIR,
        help=(
            "Open Targets ChEMBL evidence directory. Defaults to "
            "therapeutic_target_evidence_dir in configs/paths.yaml."
        ),
    )
    parser.add_argument(
        "--evidence-release-name",
        default="24.03",
        help=(
            "Open Targets release identifier for the frozen ChEMBL evidence. "
            "Defaults to the 24.03 evidence used by the released pipeline."
        ),
    )
    parser.add_argument(
        "--static-release-dir",
        type=Path,
        default=None,
        help=(
            "Frozen Open Targets release root in legacy JSON or official Parquet "
            "layout. When supplied, no live Open Targets, UniProt, Ensembl, or "
            "ontology API is queried."
        ),
    )
    parser.add_argument(
        "--association-scope",
        choices=sorted(STATIC_ASSOCIATION_DATASETS),
        default="indirect",
        help=(
            "Static association table used to exclude disease-associated negatives. "
            "Indirect includes evidence propagated from disease descendants."
        ),
    )
    parser.add_argument(
        "--static-release-name",
        default=None,
        help=(
            "Stable release identifier recorded in reconstruction metadata. "
            "Defaults to the static release directory name."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated per-disease labels and summaries.",
    )
    parser.add_argument(
        "--global-ppi-path",
        type=Path,
        default=None,
        help="Path to global_ppi_edgelist.txt",
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        default=list(DEFAULT_DISEASES),
        help="Disease IDs to build (default: all 15 configured disease areas).",
    )
    parser.add_argument(
        "--descendants-source",
        choices=["ot", "efo", "none"],
        default="ot",
        help="How to expand disease IDs in evidence filtering.",
    )
    parser.add_argument(
        "--evidence-format",
        choices=["auto", "json", "parquet"],
        default="auto",
        help="Evidence file format.",
    )
    parser.add_argument(
        "--processed-subdir",
        type=str,
        default="processed",
        help="Subdirectory under the output directory for processed label JSON files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild datasets even if target CSV files already exist.",
    )
    parser.add_argument(
        "--min-proteins-per-label",
        type=int,
        default=10,
        help="Minimum number of proteins required per class (positive and negative).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir or detect_output_dir()
    global_ppi_path = args.global_ppi_path or DEFAULT_GLOBAL_PPI

    all_drug_targets_path = args.drugbank_targets
    evidence_dir = args.evidence_dir
    processed_dir = output_dir / args.processed_subdir

    if args.static_release_dir is not None:
        _require_fresh_static_outputs(output_dir, args.diseases, args.force)

    print("=" * 80)
    print("Therapeutic Target Dataset Processing")
    print("=" * 80)
    print(f"DrugBank targets: {all_drug_targets_path}")
    print(f"OT evidence:      {evidence_dir}")
    print(f"OT static release:{args.static_release_dir or 'disabled'}")
    if args.static_release_dir is not None:
        print(f"Association scope:{args.association_scope}")
    print(f"Output dir:       {output_dir}")
    print(f"Global PPI:       {global_ppi_path}")
    print(f"Diseases:         {args.diseases}")
    print(f"Evidence format request: {args.evidence_format}")
    print()

    required_paths = [all_drug_targets_path, global_ppi_path, evidence_dir]
    if args.static_release_dir is not None:
        required_paths.append(args.static_release_dir)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    evidence_files, evidence_format = detect_evidence_files(evidence_dir, args.evidence_format)
    print(f"Detected evidence format: {evidence_format}")
    print(f"Evidence files used: {len(evidence_files)}")

    ppi_genes = read_global_ppi_genes(global_ppi_path)
    print(f"Global PPI genes: {len(ppi_genes)}")

    druggable_targets = load_druggable_targets(all_drug_targets_path)
    print(f"Druggable targets (human): {len(druggable_targets)}")

    static_release = None
    if args.static_release_dir is not None:
        static_release = load_static_open_targets_release(
            args.static_release_dir,
            args.diseases,
            args.association_scope,
            args.static_release_name,
        )
        print(
            "Static target symbols: "
            f"{len(static_release.ensg_to_symbol)}; "
            f"association table: {static_release.association_scope}"
        )

    summary_rows: List[Dict[str, object]] = []
    for disease in args.diseases:
        disease = disease.strip()
        if not disease:
            continue
        row = build_dataset_for_disease(
            disease=disease,
            evidence_files=evidence_files,
            evidence_format=evidence_format,
            descendants_source=args.descendants_source,
            ppi_genes=ppi_genes,
            druggable_targets=druggable_targets,
            out_dir=output_dir,
            processed_dir=processed_dir,
            min_proteins_per_label=args.min_proteins_per_label,
            force=args.force,
            static_release=static_release,
        )
        summary_rows.append(row)

    summary_path = output_dir / "therapeutic_target_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    if summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        if "disease_id" in existing_summary.columns:
            summary = pd.concat([existing_summary, summary], ignore_index=True)
            summary = summary.drop_duplicates(subset=["disease_id"], keep="last")
        else:
            summary = pd.concat([existing_summary, summary], ignore_index=True)
    summary = summary.reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    manifest_path = None
    if static_release is not None:
        manifest = {
            "format_version": 1,
            "dataset": "therapeutic_target",
            "reconstruction": "frozen_open_targets",
            **_static_manifest_provenance(
                static_release,
                args.evidence_release_name,
            ),
            "association_scope": static_release.association_scope,
            "association_dataset": static_release.association_dataset,
            "paper_class_balances_enforced": False,
            "disease_hierarchy_dataset": static_release.disease_dataset,
            "target_mapping_dataset": static_release.target_dataset,
            "evidence_dir": str(evidence_dir.resolve()),
            "evidence_format": evidence_format,
            "evidence_file_count": len(evidence_files),
            "drugbank_targets": str(all_drug_targets_path.resolve()),
            "global_ppi": str(global_ppi_path.resolve()),
            "diseases": [str(disease).strip() for disease in args.diseases],
            "label_rule": {
                "positive": "phase>=3 or completed phase=2 over root and descendants",
                "negative": (
                    "approved human DrugBank target without a non-literature "
                    f"{static_release.association_scope} Open Targets association"
                ),
            },
            "label_files": [
                f"therapeutic_target_{str(disease).strip()}.csv"
                for disease in args.diseases
                if str(disease).strip()
            ],
            "summary_file": summary_path.name,
        }
        manifest_path = output_dir / "therapeutic_target_manifest.json"
        write_json(manifest_path, manifest)

    print()
    print("=" * 80)
    print(f"[DONE] Summary saved to: {summary_path}")
    if manifest_path is not None:
        print(f"[DONE] Reconstruction manifest saved to: {manifest_path}")
    print(summary)
    print("=" * 80)


if __name__ == "__main__":
    main()
