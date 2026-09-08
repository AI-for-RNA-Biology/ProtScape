#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: bash run_pipeline.sh {hbca|tabula|als|merged|all} [start_step] [end_step]"
    echo "Steps: 0, 1, 2, 2b, 3, 4, 5, 6, 7, 8"
}

if (( $# > 3 )); then
    usage
    exit 2
fi

DATASET="${1:-all}"
case "${DATASET}" in
    hbca|als)
        default_start=0
        default_end=7
        ;;
    tabula)
        default_start=1
        default_end=7
        ;;
    merged)
        default_start=4
        default_end=7
        ;;
    all)
        default_start=0
        default_end=8
        ;;
    *)
        usage
        exit 2
        ;;
esac

START_STEP="${2:-${default_start}}"
END_STEP="${3:-${default_end}}"

stage_rank() {
    case "${1}" in
        0) echo 0 ;;
        1) echo 1 ;;
        2) echo 2 ;;
        2b) echo 3 ;;
        3) echo 4 ;;
        4) echo 5 ;;
        5) echo 6 ;;
        6) echo 7 ;;
        7) echo 8 ;;
        8) echo 9 ;;
        *) return 1 ;;
    esac
}

if ! start_rank="$(stage_rank "${START_STEP}")"; then
    echo "Unknown start step: ${START_STEP}" >&2
    usage
    exit 2
fi
if ! end_rank="$(stage_rank "${END_STEP}")"; then
    echo "Unknown end step: ${END_STEP}" >&2
    usage
    exit 2
fi
if (( start_rank > end_rank )); then
    echo "start_step must not come after end_step" >&2
    exit 2
fi

min_rank="$(stage_rank "${default_start}")"
max_rank="$(stage_rank "${default_end}")"
if (( start_rank < min_rank || end_rank > max_rank )); then
    echo "${DATASET} supports Steps ${default_start} through ${default_end}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

PATHS_CONFIG="${REPO_ROOT}/configs/paths.yaml"
OUTPUT_ROOT="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["output_root"])' "${PATHS_CONFIG}")"
CPDB_DATABASE="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["cellphonedb_database"])' "${PATHS_CONFIG}")"

export PIPELINE_THREADS="${PIPELINE_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${PIPELINE_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${PIPELINE_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${PIPELINE_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${PIPELINE_THREADS}}"

SCRNA_ENV="${SCRNA_ENV:-protscape-scrna}"
CPDB_ENV="${CPDB_ENV:-protscape-cellphonedb}"
PINNACLE_BASE="${OUTPUT_ROOT%/}/data_processing_bulk"

HBCA_DIR="${PINNACLE_BASE}/hbca_pseudobulk"
TABULA_DIR="${PINNACLE_BASE}/tabula_pseudobulk"
ALS_DIR="${PINNACLE_BASE}/als_bulk"
ALS_MN_DIR="${ALS_DIR}/motor_neurons"
ALS_ASTRO_DIR="${ALS_DIR}/astrocytes"
MERGED_DIR="${PINNACLE_BASE}/merged_single_cell"
FINAL_DIR="${PINNACLE_BASE}/networks_bulk"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available on PATH" >&2
    exit 1
fi

cd "${REPO_ROOT}"

run_scrna() {
    conda run --no-capture-output -n "${SCRNA_ENV}" python -m "data_processing_bulk.${1}" "${@:2}"
}

run_cpdb_python() {
    conda run --no-capture-output -n "${CPDB_ENV}" python -m "data_processing_bulk.${1}" "${@:2}"
}

selected() {
    [[ "${DATASET}" == all || "${DATASET}" == "${1}" ]]
}

prepare_cellphonedb() {
    local dataset="${1}"
    local input_dir

    case "${dataset}" in
        hbca) input_dir="${HBCA_DIR}/cellphonedb_input" ;;
        tabula) input_dir="${TABULA_DIR}/cellphonedb_input" ;;
        als) input_dir="${ALS_DIR}/cellphonedb_input" ;;
        merged) input_dir="${MERGED_DIR}/cellphonedb_input" ;;
    esac

    run_scrna step4_prep_cellphonedb --dataset "${dataset}" --max-cells-per-type 100

    local pickle_path="${input_dir}/counts_${dataset}_CellPhoneDB.pkl"
    local h5ad_path="${input_dir}/counts_${dataset}_CellPhoneDB.h5ad"
    if [[ ! -f "${pickle_path}" ]]; then
        echo "Missing CellPhoneDB counts pickle: ${pickle_path}" >&2
        exit 1
    fi

    run_cpdb_python convert_cellphonedb_counts "${pickle_path}" "${h5ad_path}"
}

clean_cpdb_results() {
    local output_dir="${1}"
    case "${output_dir}" in
        "${PINNACLE_BASE}"/*/cellphonedb_results) ;;
        *)
            echo "Refusing to clean unexpected path: ${output_dir}" >&2
            exit 1
            ;;
    esac
    rm -rf -- "${output_dir}"
    mkdir -p "${output_dir}"
}

check_cellphonedb() {
    if [[ ! -f "${CPDB_DATABASE}" ]]; then
        echo "CellPhoneDB database not found: ${CPDB_DATABASE}" >&2
        exit 1
    fi
    if ! conda run -n "${CPDB_ENV}" cellphonedb --help >/dev/null 2>&1; then
        echo "CellPhoneDB is unavailable in conda environment: ${CPDB_ENV}" >&2
        exit 1
    fi
}

run_cellphonedb() {
    local dataset="${1}"
    local base_dir

    case "${dataset}" in
        hbca) base_dir="${HBCA_DIR}" ;;
        tabula) base_dir="${TABULA_DIR}" ;;
        als) base_dir="${ALS_DIR}" ;;
        merged) base_dir="${MERGED_DIR}" ;;
    esac

    local input_dir="${base_dir}/cellphonedb_input"
    local output_dir="${base_dir}/cellphonedb_results"
    local meta_path="${input_dir}/meta_${dataset}_CellPhoneDB.txt"
    local counts_path="${input_dir}/counts_${dataset}_CellPhoneDB.h5ad"

    if [[ ! -f "${meta_path}" || ! -f "${counts_path}" ]]; then
        echo "Missing CellPhoneDB inputs for ${dataset}; run Step 4 first" >&2
        exit 1
    fi
    clean_cpdb_results "${output_dir}"

    local iterations="${CELLPHONEDB_ITERATIONS:-100}"
    local threads="${CELLPHONEDB_THREADS:-${PIPELINE_THREADS}}"
    local threshold="${CELLPHONEDB_EXPRESSION_THRESHOLD:-0.1}"
    local counts_data="${CELLPHONEDB_COUNTS_DATA:-hgnc_symbol}"
    local run_ids="${CELLPHONEDB_RUN_IDS:-0}"
    local run_id

    for run_id in ${run_ids}; do
        local run_dir="${output_dir}/run_${run_id}"
        local seed=$((run_id + 1))
        mkdir -p "${run_dir}"
        local command=(
            conda run --no-capture-output -n "${CPDB_ENV}"
            cellphonedb method statistical_analysis
            "${meta_path}" "${counts_path}"
            --counts-data "${counts_data}"
            --output-path "${run_dir}"
            --iterations "${iterations}"
            --threads "${threads}"
            --threshold "${threshold}"
            --debug-seed "${seed}"
            --database "${CPDB_DATABASE}"
        )

        # Subsample individual cells, not the single bulk profile per ALS context.
        if [[ "${dataset}" != als ]]; then
            command+=(--subsampling --subsampling-log false)
            if [[ -n "${CELLPHONEDB_SUBSAMPLING_NUM_CELLS:-}" ]]; then
                command+=(--subsampling-num-cells "${CELLPHONEDB_SUBSAMPLING_NUM_CELLS}")
            fi
            if [[ -n "${CELLPHONEDB_SUBSAMPLING_NUM_PC:-}" ]]; then
                command+=(--subsampling-num-pc "${CELLPHONEDB_SUBSAMPLING_NUM_PC}")
            fi
        fi

        echo "[Step 5] ${dataset}, run ${run_id}"
        "${command[@]}"
        if [[ -z "$(find "${run_dir}" -type f -name '*pvalues*.txt' -print -quit)" ]]; then
            echo "CellPhoneDB did not produce a p-values file for ${dataset}, run ${run_id}" >&2
            exit 1
        fi
    done
}

run_stage() {
    local stage="${1}"
    echo
    echo "===== Step ${stage} ====="

    case "${stage}" in
        0)
            if selected hbca; then
                run_scrna step0_merge_single_cell
            fi
            if selected als; then
                run_scrna step0_als_load_data --cell-type both
            fi
            ;;
        1)
            if selected hbca; then
                run_scrna step1_build_pseudobulk --dataset hbca
            fi
            if selected tabula; then
                run_scrna step1_build_pseudobulk --dataset tabula
            fi
            if selected als; then
                run_scrna step1_als_pseudobulk --cell-type both
            fi
            ;;
        2)
            if selected hbca; then
                run_scrna step2_filter_reliable_genes --dataset hbca --qc-dir "${HBCA_DIR}/qc" --output-dir "${HBCA_DIR}"
            fi
            if selected tabula; then
                run_scrna step2_filter_reliable_genes --dataset tabula --qc-dir "${TABULA_DIR}/qc" --output-dir "${TABULA_DIR}"
            fi
            if selected als; then
                run_scrna step2_als_reliable_genes --cell-type both
            fi
            ;;
        2b)
            if selected hbca; then
                run_scrna step2b_select_specific_genes --dataset hbca --output-dir "${HBCA_DIR}"
            fi
            if selected tabula; then
                run_scrna step2b_select_specific_genes --dataset tabula --output-dir "${TABULA_DIR}"
            fi
            if selected als; then
                run_scrna step2b_select_specific_genes --dataset als_mn --output-dir "${ALS_MN_DIR}"
                run_scrna step2b_select_specific_genes --dataset als_astro --output-dir "${ALS_ASTRO_DIR}"
            fi
            ;;
        3)
            if selected hbca; then
                run_scrna step3_construct_ppi --dataset hbca --output-dir "${HBCA_DIR}/ppi"
            fi
            if selected tabula; then
                run_scrna step3_construct_ppi --dataset tabula --output-dir "${TABULA_DIR}/ppi"
            fi
            if selected als; then
                run_scrna step3_als_ppi --cell-type both
            fi
            ;;
        4)
            if selected hbca; then
                prepare_cellphonedb hbca
            fi
            if selected tabula; then
                prepare_cellphonedb tabula
            fi
            if selected als; then
                prepare_cellphonedb als
            fi
            if selected merged; then
                prepare_cellphonedb merged
            fi
            ;;
        5)
            check_cellphonedb
            if selected hbca; then
                run_cellphonedb hbca
            fi
            if selected tabula; then
                run_cellphonedb tabula
            fi
            if selected als; then
                run_cellphonedb als
            fi
            if selected merged; then
                run_cellphonedb merged
            fi
            ;;
        6)
            if selected hbca; then
                run_scrna step6_construct_cci --dataset hbca
            fi
            if selected tabula; then
                run_scrna step6_construct_cci --dataset tabula
            fi
            if selected als; then
                run_scrna step6_construct_cci --dataset als
            fi
            if selected merged; then
                run_scrna step6_construct_cci --dataset merged
            fi
            ;;
        7)
            if selected hbca; then
                run_scrna step7_construct_metagraph --dataset hbca
            fi
            if selected tabula; then
                run_scrna step7_construct_metagraph --dataset tabula
            fi
            if selected als; then
                run_scrna step7_construct_metagraph --dataset als
            fi
            if selected merged; then
                run_scrna step7_construct_metagraph --dataset merged
            fi
            ;;
        8)
            if [[ "${DATASET}" == all ]]; then
                run_scrna step8_merge_datasets --merged-root "${MERGED_DIR}" --output-dir "${FINAL_DIR}"
            fi
            ;;
    esac
}

STAGES=(0 1 2 2b 3 4 5 6 7 8)
echo "Dataset: ${DATASET}"
echo "Steps: ${START_STEP} through ${END_STEP}"
echo "scRNA environment: ${SCRNA_ENV}"
echo "CellPhoneDB environment: ${CPDB_ENV}"

for stage in "${STAGES[@]}"; do
    rank="$(stage_rank "${stage}")"
    if (( rank >= start_rank && rank <= end_rank )); then
        run_stage "${stage}"
    fi
done

echo
echo "Pipeline run complete."
if [[ "${DATASET}" == all ]] && (( end_rank >= 9 )); then
    echo "Merged dataset: ${FINAL_DIR}"
fi
