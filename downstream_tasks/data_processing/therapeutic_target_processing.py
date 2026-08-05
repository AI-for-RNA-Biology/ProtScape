#!/usr/bin/env python
"""
Build therapeutic target downstream datasets (split by disease EFO code).

This script is embedding-independent:
- Inputs come from therapeutic target evidence files + global PPI universe.
- Outputs are per-disease binary CSVs consumed by the downstream training pipeline.

Default outputs (under downstream_tasks/therapeutic_target_dataset):
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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
try:
    import requests
except ImportError as exc:
    raise SystemExit("Please install `requests` (pip install requests).") from exc


from ..config import DEFAULT_DATA_ROOT, DEFAULT_GLOBAL_PPI, DEFAULT_OUTPUT_ROOT

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


def detect_data_root() -> Path:
    """Return the configured input-data root."""
    return DEFAULT_DATA_ROOT


def detect_output_dir() -> Path:
    """Return the configured directory for generated therapeutic-target tables."""
    return DEFAULT_OUTPUT_ROOT / "data" / "therapeutic_target_dataset"


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


def load_chembl_to_drugbank_map(path: Path) -> Dict[str, str]:
    """Load ChEMBL -> DrugBank map (for compatibility; not required for labels)."""
    if not path.exists():
        return {}
    df = pd.read_table(path)
    if df.shape[1] < 2:
        return {}
    df = df.iloc[:, :2].copy()
    df.columns = ["chembl", "db"]
    return dict(zip(df["chembl"].astype(str), df["db"].astype(str)))


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
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


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
) -> Dict[str, object]:
    """Build one disease CSV and intermediate processed files."""
    out_csv = out_dir / f"therapeutic_target_{disease}.csv"
    pos_json = processed_dir / f"positive_proteins_global_{disease}.json"
    neg_json = processed_dir / f"negative_proteins_global_{disease}.json"
    raw_json = processed_dir / f"raw_targets_global_{disease}.json"

    if out_csv.exists() and not force:
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
            "dataset_csv": str(out_csv),
            "status": "existing",
        }

    if descendants_source == "ot":
        descendants = set(get_disease_descendants_ot_api(disease))
    elif descendants_source == "efo":
        descendants = set(get_disease_descendants_efo(disease))
    elif descendants_source == "none":
        descendants = {disease}
    else:
        raise ValueError(
            f"Unsupported descendants_source={descendants_source}; valid: efo, none"
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

    all_associated_targets, ensg_to_symbol = get_ot_associated_targets(disease)
    all_associated_targets = {g.upper() for g in all_associated_targets}

    positive_raw, map_stats = resolve_positive_targets(evidence_df, ensg_to_symbol)
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

    # Save processed labels for reproducibility.
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
        "dataset_csv": str(out_csv),
        "status": "rebuilt",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build therapeutic target downstream datasets (split by EFO disease)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Base data directory (auto-detected if omitted).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Therapeutic dataset directory containing all_approved_oct2022.csv, src1src2.txt and evidence_data/.",
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
        help="Disease IDs to build (default: the 15 paper disease areas).",
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
        "--limit-evidence-files",
        type=int,
        default=None,
        help="Optional debug limit on number of evidence files scanned.",
    )
    parser.add_argument(
        "--processed-subdir",
        type=str,
        default="processed",
        help="Subdirectory name under dataset-dir for processed label json files.",
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

    data_root = args.data_root or detect_data_root()
    dataset_dir = args.dataset_dir or (data_root / "downstream_tasks" / "therapeutic_target_dataset")
    output_dir = args.output_dir or detect_output_dir()
    global_ppi_path = args.global_ppi_path or DEFAULT_GLOBAL_PPI

    all_drug_targets_path = dataset_dir / "all_approved_oct2022.csv"
    chembl2db_path = dataset_dir / "src1src2.txt"
    evidence_dir = dataset_dir / "evidence_data"
    processed_dir = output_dir / args.processed_subdir

    print("=" * 80)
    print("Therapeutic Target Dataset Processing")
    print("=" * 80)
    print(f"Data root:      {data_root}")
    print(f"Dataset dir:    {dataset_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Global PPI:     {global_ppi_path}")
    print(f"Diseases:       {args.diseases}")
    print(f"Evidence format request: {args.evidence_format}")
    print()

    for path in (all_drug_targets_path, chembl2db_path, global_ppi_path, evidence_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    evidence_files, evidence_format = detect_evidence_files(evidence_dir, args.evidence_format)
    if args.limit_evidence_files is not None:
        evidence_files = evidence_files[: args.limit_evidence_files]
    print(f"Detected evidence format: {evidence_format}")
    print(f"Evidence files used: {len(evidence_files)}")

    ppi_genes = read_global_ppi_genes(global_ppi_path)
    print(f"Global PPI genes: {len(ppi_genes)}")

    druggable_targets = load_druggable_targets(all_drug_targets_path)
    print(f"Druggable targets (human): {len(druggable_targets)}")

    # Loaded for compatibility/debug parity with old task.
    chembl_map = load_chembl_to_drugbank_map(chembl2db_path)
    print(f"ChEMBL->DrugBank map entries: {len(chembl_map)}")
    print()

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
    summary.to_csv(summary_path, index=False)

    print()
    print("=" * 80)
    print(f"[DONE] Summary saved to: {summary_path}")
    print(summary)
    print("=" * 80)


if __name__ == "__main__":
    main()
