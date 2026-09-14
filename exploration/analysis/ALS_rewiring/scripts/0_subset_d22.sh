#!/usr/bin/env bash
## Generates config.py's D22_CSV (stage 01's input) from the full, pre-subsetting
## S2GAE/STRING matrix (`als_rewiring_raw_matrix_csv` in configs/paths.yaml).
##
## Subsets a large CSV to:
##   - day 22 only (cyto & nuc), for all two genotypes (CTRL, VCP)
##     - drops d0/d3/d7/d14/d35 entirely, keeping just the 4 d22 score cols
##   - keeps a row only if, for EACH of the two genotypes independently,
##     AT LEAST ONE of its two d22 values (cyto OR nuc) is non-NaN:
##       CTRL:  (cyto_d22 present OR nuc_d22 present)   AND
##       VCP: (cyto_d22 present OR nuc_d22 present)
##     i.e. cyto/nuc are combined with OR *within* a genotype, but both
##     genotypes (CTRL + VCP) are combined with AND *between* them -
##     every genotype must have at least one of its two compartments present.
## Streams line-by-line with awk -> constant memory, no matter the file size.
##
## Reuses an existing subset only when it is newer than the source matrix.
##
## Usage (from anywhere):
##   bash 0_subset_d22.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PATHS_YAML="${REPO_ROOT}/configs/paths.yaml"

get_path() {
    python3 -c "import yaml; print(yaml.safe_load(open('${PATHS_YAML}'))['$1'])"
}

# Paths come from configs/paths.yaml, not hardcoded here, so this always
# matches what config.py itself reads (als_rewiring_raw_matrix_csv,
# als_rewiring_data_root -> D22_CSV).
input_csv="$(get_path als_rewiring_raw_matrix_csv)"
output_csv="$(get_path als_rewiring_data_root)/subset_d22_pergenotype_presentp.csv"

if [ -f "$input_csv" ] && [ "$output_csv" -nt "$input_csv" ]; then
    echo "Up to date: $output_csv"
    exit 0
fi

mkdir -p "$(dirname "$output_csv")"
subset_tmp="$(mktemp "${output_csv}.XXXXXX")"
trap 'rm -f "$subset_tmp"' EXIT

awk -F',' '
function is_missing(v) {
    return (v == "" || v == "NA" || v == "NaN" || v == "nan")
}

BEGIN { OFS="," }
NR==1 {
    for (i=1;i<=NF;i++) col[$i]=i

    # metadata columns kept as-is, in this order
    n_meta = split("protA protB present_in_ppi stringdb_physical_score stringdb_combined_score", meta, " ")

    # day-22 columns kept in the OUTPUT, all two genotypes, in this order
    n_out = split("s2gae_score_CTRL_cyto_d22 s2gae_score_CTRL_nuc_d22 s2gae_score_VCP_cyto_d22 s2gae_score_VCP_nuc_d22", out_cols, " ")

    # genotype groups for the presence check (cyto/nuc = OR within a group,
    # the two groups themselves = AND)
    n_ctrl  = split("s2gae_score_CTRL_cyto_d22 s2gae_score_CTRL_nuc_d22",   ctrl_cols,  " ")
    n_vcp = split("s2gae_score_VCP_cyto_d22 s2gae_score_VCP_nuc_d22", vcp_cols, " ")


    # Numeric loops preserve the declared column order.
    for (k=1;k<=n_meta;k++) {
        name = meta[k]
        if (!(name in col)) { print "Missing expected column: " name > "/dev/stderr"; exit 1 }
        meta_idx[k] = col[name]
    }
    for (k=1;k<=n_out;k++) {
        name = out_cols[k]
        if (!(name in col)) { print "Missing expected column: " name > "/dev/stderr"; exit 1 }
        out_idx[k] = col[name]
    }
    for (k=1;k<=n_ctrl;k++)  { if (!(ctrl_cols[k]  in col)) { print "Missing expected column: " ctrl_cols[k]  > "/dev/stderr"; exit 1 }; ctrl_idx[k]  = col[ctrl_cols[k]] }
    for (k=1;k<=n_vcp;k++) { if (!(vcp_cols[k] in col)) { print "Missing expected column: " vcp_cols[k] > "/dev/stderr"; exit 1 }; vcp_idx[k] = col[vcp_cols[k]] }

    # print header for the reduced output
    hdr = meta[1]
    for (k=2;k<=n_meta;k++) hdr = hdr OFS meta[k]
    for (k=1;k<=n_out;k++) hdr = hdr OFS out_cols[k]
    print hdr
    next
}
{
    ctrl_present = 0
    for (k=1;k<=n_ctrl;k++)  { if (!is_missing($(ctrl_idx[k])))  { ctrl_present = 1;  break } }

    vcp_present = 0
    for (k=1;k<=n_vcp;k++) { if (!is_missing($(vcp_idx[k]))) { vcp_present = 1; break } }

    keep = (ctrl_present && vcp_present)

    if (keep) {
        line = $(meta_idx[1])
        for (k=2;k<=n_meta;k++) line = line OFS $(meta_idx[k])
        for (k=1;k<=n_out;k++) line = line OFS $(out_idx[k])
        print line
    }
}
' "$input_csv" > "$subset_tmp"
mv "$subset_tmp" "$output_csv"

echo "Done. Rows kept: $(($(wc -l < "$output_csv") - 1))"
