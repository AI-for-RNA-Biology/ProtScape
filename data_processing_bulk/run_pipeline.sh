#!/usr/bin/env bash
# Build pseudobulks, context PPIs, CellPhoneDB interactions, and the metagraph.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET="${1:-all}"
case "$DATASET" in
    hbca|als) first=0; last=7 ;;
    tabula) first=1; last=7 ;;
    merged) first=4; last=7 ;;
    all) first=0; last=8 ;;
    *) echo "Usage: bash data_processing_bulk/run_pipeline.sh {hbca|tabula|als|merged|all} [first_step] [last_step]" >&2; exit 2 ;;
esac

# Step 2b sits between reliable-gene selection and PPI construction.
STEPS=(0 1 2 2b 3 4 5 6 7 8)
declare -A STEP_INDEX=([0]=0 [1]=1 [2]=2 [2b]=3 [3]=4 [4]=5 [5]=6 [6]=7 [7]=8 [8]=9)
START="${2:-$first}"
END="${3:-$last}"
if (( $# > 3 )) || [[ ! -v STEP_INDEX[$START] || ! -v STEP_INDEX[$END] ]]; then
    echo "Choose steps from: ${STEPS[*]}" >&2
    exit 2
fi
if (( STEP_INDEX[$START] < STEP_INDEX[$first] || STEP_INDEX[$END] > STEP_INDEX[$last] || STEP_INDEX[$START] > STEP_INDEX[$END] )); then
    echo "$DATASET supports steps $first through $last, in order." >&2
    exit 2
fi

OUTPUT_ROOT="$(python -c 'from pathlib import Path; import yaml; print(Path(yaml.safe_load(open("configs/paths.yaml"))["output_root"]).expanduser())')"
CPDB_DATABASE="$(python -c 'from pathlib import Path; import yaml; print(Path(yaml.safe_load(open("configs/paths.yaml"))["cellphonedb_database"]).expanduser())')"
PROCESSING_ROOT="${OUTPUT_ROOT%/}/data_processing_bulk"
declare -A DATASET_DIR=(
    [hbca]="$PROCESSING_ROOT/hbca_pseudobulk"
    [tabula]="$PROCESSING_ROOT/tabula_pseudobulk"
    [als]="$PROCESSING_ROOT/als_bulk"
    [merged]="$PROCESSING_ROOT/merged_single_cell"
)
SCRNA_ENV="${SCRNA_ENV:-protscape-scrna}"
CPDB_ENV="${CPDB_ENV:-protscape-cellphonedb}"
export PIPELINE_THREADS="${PIPELINE_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$PIPELINE_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$PIPELINE_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$PIPELINE_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$PIPELINE_THREADS}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

run_scrna() {
    conda run --no-capture-output -n "$SCRNA_ENV" python -m "data_processing_bulk.$1" "${@:2}"
}

prepare_cellphonedb() {
    local dataset="$1"
    local input="${DATASET_DIR[$dataset]}/cellphonedb_input"
    run_scrna step4_prep_cellphonedb --dataset "$dataset" --max-cells-per-type 100
    conda run --no-capture-output -n "$CPDB_ENV" python -m data_processing_bulk.convert_cellphonedb_counts \
        "$input/counts_${dataset}_CellPhoneDB.pkl" "$input/counts_${dataset}_CellPhoneDB.h5ad"
}

run_cellphonedb() {
    local dataset="$1"
    local input="${DATASET_DIR[$dataset]}/cellphonedb_input"
    local output="${DATASET_DIR[$dataset]}/cellphonedb_results"
    for file in "$CPDB_DATABASE" "$input/meta_${dataset}_CellPhoneDB.txt" "$input/counts_${dataset}_CellPhoneDB.h5ad"; do
        if [[ ! -f "$file" ]]; then
            echo "Missing CellPhoneDB input: $file" >&2
            exit 1
        fi
    done
    # Clear only this step's results so Step 6 cannot pick up an earlier run.
    rm -rf -- "$output"
    mkdir -p "$output/run_0"
    local command=(
        conda run --no-capture-output -n "$CPDB_ENV" cellphonedb method statistical_analysis
        "$input/meta_${dataset}_CellPhoneDB.txt" "$input/counts_${dataset}_CellPhoneDB.h5ad"
        --counts-data hgnc_symbol --output-path "$output/run_0"
        --iterations 100 --threads "$PIPELINE_THREADS" --threshold 0.1
        --debug-seed 1 --database "$CPDB_DATABASE"
    )
    # ALS has one bulk profile per context; subsampling applies to single cells.
    if [[ "$dataset" != als ]]; then
        command+=(--subsampling --subsampling-log false)
    fi
    "${command[@]}"
    if [[ -z "$(find "$output/run_0" -name '*pvalues*.txt' -type f -print -quit)" ]]; then
        echo "CellPhoneDB produced no p-values for $dataset." >&2
        exit 1
    fi
}

if [[ "$DATASET" == all ]]; then
    datasets=(hbca tabula als merged)
else
    datasets=("$DATASET")
fi

for step in "${STEPS[@]}"; do
    if (( STEP_INDEX[$step] < STEP_INDEX[$START] || STEP_INDEX[$step] > STEP_INDEX[$END] )); then
        continue
    fi
    echo "Step $step"
    if [[ "$step" == 8 ]]; then
        run_scrna step8_merge_datasets --merged-root "${DATASET_DIR[merged]}" --output-dir "$PROCESSING_ROOT/networks_bulk"
        continue
    fi
    for dataset in "${datasets[@]}"; do
        root="${DATASET_DIR[$dataset]}"
        case "$step:$dataset" in
            0:hbca) run_scrna step0_merge_single_cell ;;
            0:als) run_scrna step0_als_load_data --cell-type both ;;
            1:hbca|1:tabula) run_scrna step1_build_pseudobulk --dataset "$dataset" ;;
            1:als) run_scrna step1_als_pseudobulk --cell-type both ;;
            2:hbca|2:tabula)
                run_scrna step2_filter_reliable_genes --dataset "$dataset" --qc-dir "$root/qc" --output-dir "$root" ;;
            2:als) run_scrna step2_als_reliable_genes --cell-type both ;;
            2b:hbca|2b:tabula)
                run_scrna step2b_select_specific_genes --dataset "$dataset" --output-dir "$root" ;;
            2b:als)
                run_scrna step2b_select_specific_genes --dataset als_mn --output-dir "$root/motor_neurons"
                run_scrna step2b_select_specific_genes --dataset als_astro --output-dir "$root/astrocytes" ;;
            3:hbca|3:tabula)
                run_scrna step3_construct_ppi --dataset "$dataset" --output-dir "$root/ppi" ;;
            3:als) run_scrna step3_als_ppi --cell-type both ;;
            4:*) prepare_cellphonedb "$dataset" ;;
            5:*) run_cellphonedb "$dataset" ;;
            6:*) run_scrna step6_construct_cci --dataset "$dataset" ;;
            7:*) run_scrna step7_construct_metagraph --dataset "$dataset" ;;
        esac
    done
done
