#!/usr/bin/env bash
# Download the tables used for labels from one fixed Open Targets release.
set -euo pipefail
release=${1:-24.03}
root=${2:-data/reference_data/opentargets}
base="https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/$release"

download_table() {
    local remote=$1 local_name=$2
    mkdir -p "$root/$release/$local_name"
    wget --recursive --level=1 --no-parent --no-directories --continue \
        --accept='*.parquet' --directory-prefix="$root/$release/$local_name" \
        "$base/$remote/"
}

case "$release" in
    24.03)
        download_table output/etl/parquet/diseases disease
        download_table output/etl/parquet/targets target
        download_table output/etl/parquet/evidence/sourceId=chembl evidence
        download_table output/etl/parquet/associationByDatatypeIndirect association_by_datatype_indirect
        ;;
    26.03)
        download_table output/disease disease
        download_table output/target target
        download_table output/evidence_clinical_precedence evidence
        download_table output/clinical_report clinical_report
        download_table output/association_by_datatype_indirect association_by_datatype_indirect
        ;;
    *) echo "Usage: bash $0 {24.03|26.03} [download-root]" >&2; exit 1 ;;
esac
