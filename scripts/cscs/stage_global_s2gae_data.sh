#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="${PROTSCAPE_RELEASE:-/users/aloistho/projects/ProtScape_release}"
STAGE_ROOT="${PROTSCAPE_DATA:-/iopsstor/scratch/cscs/aloistho/protscape/release-data}"
NETWORK_ARCHIVE="${RELEASE_ROOT}/data/networks_bulk.zip"
FEATURE_SOURCE="${RELEASE_ROOT}/data/sequence_embeddings/gene_protein_embeddings_esm2_3B_layer33.plk"
FEATURE_DIR="${STAGE_ROOT}/sequence_embeddings"
FEATURE_TARGET="${FEATURE_DIR}/gene_protein_embeddings_esm2_3B_layer33.plk"

unzip -tq "${NETWORK_ARCHIVE}" >/dev/null
mkdir -p "${STAGE_ROOT}" "${FEATURE_DIR}"
if [[ ! -f "${STAGE_ROOT}/networks_bulk/global_ppi_edgelist.txt" ]]; then
    unzip -q "${NETWORK_ARCHIVE}" -d "${STAGE_ROOT}"
fi
if [[ ! -f "${FEATURE_TARGET}" ]] || ! cmp -s "${FEATURE_SOURCE}" "${FEATURE_TARGET}"; then
    cp "${FEATURE_SOURCE}" "${FEATURE_TARGET}"
fi

test -f "${STAGE_ROOT}/networks_bulk/count_edge_dict.pkl"
test -d "${STAGE_ROOT}/networks_bulk/ppi_edgelists"
cmp -s "${FEATURE_SOURCE}" "${FEATURE_TARGET}"
printf 'networks=%s\nfeatures=%s\n' \
    "${STAGE_ROOT}/networks_bulk" "${FEATURE_TARGET}"
