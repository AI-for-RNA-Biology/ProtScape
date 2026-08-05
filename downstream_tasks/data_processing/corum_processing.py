import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_DATA_ROOT, DEFAULT_GLOBAL_PPI, DEFAULT_OUTPUT_ROOT

ORGANISM_KEEP = {"Human"}
MIN_MEMBERS = 10
MAX_MEMBERS = 300

REDUCE_REDUNDANCY = True
MAX_OVERLAP_COEF = 0.90   # |A∩B| / min(|A|,|B|)
MAX_JACCARD = 0.80        # |A∩B| / |A∪B|


def default_base_dir():
    return DEFAULT_DATA_ROOT


def default_output_dir():
    return DEFAULT_OUTPUT_ROOT / "data" / "corum_dataset"


def resolve_corum_json_path(base_dir):
    candidates = [
        base_dir / "corum_dataset" / "corum_humanComplexes.json",
        base_dir / "downstream_tasks" / "corum_dataset" / "corum_humanComplexes.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Build CORUM downstream datasets.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base_dir(),
        help="Base PINNACLE data directory containing corum_dataset/, networks/, downstream_tasks/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory for generated CORUM tables.",
    )
    parser.add_argument(
        "--global-ppi-path",
        type=Path,
        default=None,
        help="Override the global PPI edgelist used for universe filtering and overlap removal.",
    )
    return parser.parse_args()


def sanitize_symbol(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = s.upper()
    s = re.split(r"[,\s;/|()]+", s)[0].strip()
    return s if s else None


def split_symbols(s):
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    parts = re.split(r"[;,|]\s*", s)
    out = []
    for p in parts:
        sym = sanitize_symbol(p)
        if sym:
            out.append(sym)
    return out


def read_ppi_universe_hgnc(ppi_path):
    proteins = set()
    with open(ppi_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            a, b = parts[0].upper(), parts[1].upper()
            proteins.add(a)
            proteins.add(b)
    return proteins


def select_nonredundant_ids(id_to_members, max_overlap_coef, max_jaccard, has_pmid=None):
    if not id_to_members:
        return [], []

    sizes = {i: len(s) for i, s in id_to_members.items()}

    def pmid_key(i):
        if not has_pmid:
            return 0
        return int(has_pmid.get(i, 0))

    ordered = sorted(id_to_members.keys(), key=lambda i: (-pmid_key(i), sizes[i], str(i)))

    kept = []
    dropped = []

    for cid in ordered:
        members = id_to_members[cid]
        if not members:
            continue

        drop_info = None
        for kept_id in kept:
            kept_members = id_to_members[kept_id]
            inter = len(members & kept_members)
            if inter == 0:
                continue

            overlap_coef = inter / min(len(members), len(kept_members))
            jaccard = inter / len(members | kept_members)

            if overlap_coef >= max_overlap_coef or jaccard >= max_jaccard:
                drop_info = (cid, kept_id, overlap_coef, jaccard)
                break

        if drop_info is not None:
            dropped.append(drop_info)
            continue

        kept.append(cid)

    return kept, dropped


def build_corum_tables(complexes, universe):
    complex_rows = []
    membership_rows = []

    for complex_data in complexes:
        if str(complex_data.get("organism", "")).strip() not in ORGANISM_KEEP:
            continue

        complex_id = complex_data.get("complex_id")
        complex_name = complex_data.get("complex_name")
        pmid = complex_data.get("pmid")
        members = []
        seen = set()

        for subunit in complex_data.get("subunits", []) or []:
            swissprot = subunit.get("swissprot", {}) or {}
            gene = swissprot.get("gene", {}) or {}

            gene_name = sanitize_symbol(swissprot.get("gene_name"))
            candidates = [gene_name]
            candidates.extend(split_symbols(swissprot.get("gene_name_synonyms")))
            candidates.append(sanitize_symbol(gene.get("genname")))

            symbol = next((s for s in candidates if s in universe), None)
            if symbol is None or symbol in seen:
                continue
            seen.add(symbol)

            members.append(
                {
                    "protein": symbol,
                    "gene_name": gene_name,
                    "uniprot_id": swissprot.get("uniprot_id"),
                }
            )

        n = len(members)
        if n < MIN_MEMBERS or n > MAX_MEMBERS:
            continue

        complex_rows.append(
            {
                "complex_id": complex_id,
                "complex_name": complex_name,
                "pmid": pmid,
                "n_members_in_ppi_universe": n,
                "member_hgnc": ";".join([m["protein"] for m in members]),
                "member_uniprots": ";".join([str(m["uniprot_id"]) for m in members if m.get("uniprot_id")]),
            }
        )

        for m in members:
            membership_rows.append(
                {
                    "protein": m["protein"],
                    "gene_name_raw": m.get("gene_name"),
                    "uniprot_id": m.get("uniprot_id"),
                    "complex_id": complex_id,
                    "complex_name": complex_name,
                    "pmid": pmid,
                }
            )

    complexes_df = pd.DataFrame(complex_rows)
    memberships_df = pd.DataFrame(membership_rows)

    if not complexes_df.empty:
        complexes_df = complexes_df.sort_values(
            ["n_members_in_ppi_universe", "complex_id"], ascending=[False, True]
        ).reset_index(drop=True)

    if not memberships_df.empty:
        memberships_df = memberships_df.sort_values(
            ["protein", "complex_id"], ascending=[True, True]
        ).reset_index(drop=True)

    return complexes_df, memberships_df


def build_protein_to_complexes(memberships_df):
    if memberships_df.empty:
        return pd.DataFrame(columns=["protein", "complex_ids", "complex_names", "n_complexes"])

    rows = []
    for prot, dfp in memberships_df.groupby("protein"):
        cids = dfp["complex_id"].astype(str).tolist()
        cnames = dfp["complex_name"].astype(str).tolist()
        rows.append(
            {
                "protein": prot,
                "complex_ids": ";".join(cids),
                "complex_names": ";".join(cnames),
                "n_complexes": len(cids),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["n_complexes", "protein"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_positive_pairs(complexes_df):
    if complexes_df.empty:
        return pd.DataFrame(columns=["protein_a", "protein_b", "complex_id", "complex_name"])

    rows = []
    for _, r in complexes_df.iterrows():
        cid = r["complex_id"]
        cname = r["complex_name"]
        members = [x for x in str(r["member_hgnc"]).split(";") if x]
        members = sorted(set(members))
        for a, b in combinations(members, 2):
            rows.append({"protein_a": a, "protein_b": b, "complex_id": cid, "complex_name": cname})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    base_dir = args.base_dir
    corum_json_path = resolve_corum_json_path(base_dir)
    global_ppi_txt_path = args.global_ppi_path or DEFAULT_GLOBAL_PPI
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not corum_json_path.exists():
        raise FileNotFoundError(f"Missing CORUM JSON: {corum_json_path}")
    if not global_ppi_txt_path.exists():
        raise FileNotFoundError(f"Missing PPI TXT: {global_ppi_txt_path}")

    print(f"[1/4] Reading PPI universe (HGNC): {global_ppi_txt_path}")
    universe = read_ppi_universe_hgnc(global_ppi_txt_path)
    print(f"      Unique HGNC symbols in PPI: {len(universe):,}")

    print(f"[2/4] Loading CORUM JSON: {corum_json_path}")
    with open(corum_json_path, "r", encoding="utf-8") as f:
        complexes = json.load(f)
    n_human = sum(str(c.get("organism", "")).strip() in ORGANISM_KEEP for c in complexes)
    print(f"      Human complexes loaded: {n_human:,}")

    print("[3/4] Filtering complexes to universe")
    complexes_df, memberships_df = build_corum_tables(complexes, universe)

    if REDUCE_REDUNDANCY and not memberships_df.empty:
        cid_to_members = memberships_df.groupby("complex_id")["protein"].agg(lambda s: set(s)).to_dict()
        cid_has_pmid = memberships_df.groupby("complex_id")["pmid"].apply(
            lambda values: int(any(pd.notna(v) and str(v) != "None" for v in values))
        ).to_dict()

        kept_ids, dropped = select_nonredundant_ids(
            cid_to_members,
            MAX_OVERLAP_COEF,
            MAX_JACCARD,
            has_pmid=cid_has_pmid,
        )
        kept_set = set(kept_ids)

        before = complexes_df["complex_id"].nunique() if not complexes_df.empty else 0
        complexes_df = complexes_df[complexes_df["complex_id"].isin(kept_set)].copy().reset_index(drop=True)
        memberships_df = memberships_df[memberships_df["complex_id"].isin(kept_set)].copy().reset_index(drop=True)
        after = complexes_df["complex_id"].nunique() if not complexes_df.empty else 0

        print(f"[CORUM] redundancy filter: kept {after:,} / {before:,} complexes; dropped {before-after:,}")
        if dropped:
            prev = dropped[:10]
            print("[CORUM] dropped examples (dropped_id -> kept_id, overlap, jaccard):")
            for d_id, k_id, oc, j in prev:
                print(f"  - {d_id} -> {k_id} | overlap={oc:.3f}, jaccard={j:.3f}")

    print("[4/4] Writing CSVs")
    complexes_csv = out_dir / "corum_complexes_filtered.csv"
    memberships_csv = out_dir / "corum_memberships_filtered.csv"
    prot2c_csv = out_dir / "protein_to_complexes.csv"
    pairs_csv = out_dir / "corum_positive_pairs.csv"

    complexes_df.to_csv(complexes_csv, index=False)
    memberships_df.to_csv(memberships_csv, index=False)

    prot2c_df = build_protein_to_complexes(memberships_df)
    prot2c_df.to_csv(prot2c_csv, index=False)

    pairs_df = build_positive_pairs(complexes_df)
    pairs_df.to_csv(pairs_csv, index=False)

    print(f"      Complexes kept: {len(complexes_df):,}")
    print(f"      Membership rows: {len(memberships_df):,}")
    print(f"      Proteins with >=1 complex label: {len(prot2c_df):,}")
    print(f"      Positive co-complex pairs (per-complex, not deduped across complexes): {len(pairs_df):,}")

    print("\n[OK] Wrote files:")
    print(f"  - {complexes_csv}")
    print(f"  - {memberships_csv}")
    print(f"  - {prot2c_csv}")
    print(f"  - {pairs_csv}")


if __name__ == "__main__":
    main()
