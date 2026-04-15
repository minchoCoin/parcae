#!/bin/bash
set -e

export PYTHONUNBUFFERED=1

# ============================================================================
# Batch Evaluation Script
# ============================================================================
# Usage:
#   ./runs/run_batch_eval.sh
#
# Configure the MODELS array below with your model output directories.
# For parcae (looped) models, set RECURRENCE_DEPTHS to evaluate at different
# recurrence counts for core_extended eval.
# ============================================================================

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# List of model output directories to evaluate
MODELS=(
    "outputs/parcae-small-140m"
)

# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------
# Each benchmark is defined as: "eval_name:eval_tasks:eval_config"
#   - eval_name: Name for the results file
#   - eval_tasks: Comma-separated tasks (lm_eval, bpb, sample, core, core_extended)
#   - eval_config: Path to eval config yaml (optional, use "" for defaults)
#
# For parcae models with RECURRENCE_DEPTHS set, recurrence sweep only applies
# to benchmarks listed in RECURRENCE_SWEEP_BENCHMARKS
# ---------------------------------------------------------------------------

BENCHMARKS=(
    "core_extended:core_extended:eval_configs/eval-core-extended.yaml"
    # "val_loss:bpb:eval_configs/eval-val-loss.yaml"
)

# Which benchmarks should be run at each recurrence depth for parcae models
# Others will only run once at the default recurrence
RECURRENCE_SWEEP_BENCHMARKS=("core_extended")

# Number of GPUs to use
NUM_GPUS="${NUM_GPUS:-8}"

# For parcae models: recurrence depths to evaluate at
# Leave empty to use model's default mean_recurrence
RECURRENCE_DEPTHS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

is_parcae_model() {
    local out_dir="$1"
    local config_file=""
    
    # Find config file
    for f in "run_config.json" "model_config.json" "config.yaml" "config.json"; do
        if [[ -f "${out_dir}/${f}" ]]; then
            config_file="${out_dir}/${f}"
            break
        fi
    done
    
    if [[ -z "$config_file" ]]; then
        return 1
    fi
    
    # Check if model_class_name is Parcae or if model_name contains "parcae"
    if grep -q '"model_class_name".*"Parcae"' "$config_file" 2>/dev/null || \
       grep -q 'model_class_name.*Parcae' "$config_file" 2>/dev/null || \
       grep -q 'parcae' "$config_file" 2>/dev/null; then
        return 0
    fi
    return 1
}

get_default_recurrence() {
    local out_dir="$1"
    local config_file=""
    
    for f in "run_config.json" "model_config.json" "config.yaml" "config.json"; do
        if [[ -f "${out_dir}/${f}" ]]; then
            config_file="${out_dir}/${f}"
            break
        fi
    done
    
    if [[ -n "$config_file" ]]; then
        # Try to extract mean_recurrence from config
        local rec=$(grep -o '"mean_recurrence"[[:space:]]*:[[:space:]]*[0-9]*' "$config_file" 2>/dev/null | grep -o '[0-9]*$')
        if [[ -z "$rec" ]]; then
            rec=$(grep -o 'mean_recurrence[[:space:]]*:[[:space:]]*[0-9]*' "$config_file" 2>/dev/null | grep -o '[0-9]*$')
        fi
        if [[ -n "$rec" ]]; then
            echo "$rec"
            return
        fi
    fi
    echo "32"  # default
}

eval_already_done() {
    local out_dir="$1"
    local eval_name="$2"
    [[ -f "${out_dir}/eval/${eval_name}.json" ]]
}

run_eval() {
    local out_dir="$1"
    local eval_name="$2"
    local eval_tasks="$3"
    local eval_config="$4"
    local override_recurrence="$5"
    
    if eval_already_done "$out_dir" "$eval_name"; then
        echo "SKIP: ${out_dir} / ${eval_name} (already complete)"
        return 0
    fi

    echo ""
    echo "=============================================="
    echo "Evaluating: ${out_dir}"
    echo "Eval Name: ${eval_name}"
    echo "Tasks: ${eval_tasks}"
    if [[ -n "$eval_config" ]]; then
        echo "Config: ${eval_config}"
    fi
    if [[ -n "$override_recurrence" ]]; then
        echo "Override Recurrence: ${override_recurrence}"
    fi
    echo "GPUs: ${NUM_GPUS}"
    echo "=============================================="
    
    local master_port=$((29500 + RANDOM % 1000))
    local cmd="torchrun --nproc_per_node=${NUM_GPUS} --master_port=${master_port} scripts/eval.py"
    
    # Add config file FIRST (so command-line args override it)
    if [[ -n "$eval_config" ]] && [[ -f "$eval_config" ]]; then
        cmd="$cmd --config ${eval_config}"
    fi
    
    # Add command-line overrides AFTER config
    cmd="$cmd --out_dir ${out_dir} --eval_name ${eval_name} --eval_tasks ${eval_tasks}"
    
    # For parcae models with recurrence override
    if [[ -n "$override_recurrence" ]]; then
        cmd="$cmd --override_mean_recurrence ${override_recurrence}"
    fi
    
    eval $cmd
    
    echo "=============================================="
    echo "Completed: ${out_dir} (${eval_name})"
    echo "=============================================="
}

# Check if a benchmark should have recurrence sweep
should_sweep_recurrence() {
    local benchmark_name="$1"
    for sweep_bench in "${RECURRENCE_SWEEP_BENCHMARKS[@]}"; do
        if [[ "$benchmark_name" == "$sweep_bench" ]]; then
            return 0
        fi
    done
    return 1
}

echo "=============================================="
echo "Batch Evaluation"
echo "=============================================="
echo "Models: ${#MODELS[@]}"
echo "Benchmarks: ${#BENCHMARKS[@]}"
for bench in "${BENCHMARKS[@]}"; do
    IFS=':' read -r name tasks config <<< "$bench"
    echo "  - ${name} (${tasks})"
done
if [[ ${#RECURRENCE_DEPTHS[@]} -gt 0 ]]; then
    echo "Recurrence depths: ${RECURRENCE_DEPTHS[*]}"
    echo "Recurrence sweep benchmarks: ${RECURRENCE_SWEEP_BENCHMARKS[*]}"
fi
echo "=============================================="

TOTAL_EVALS=0
COMPLETED_EVALS=0
SKIPPED_EVALS=0

for model_dir in "${MODELS[@]}"; do
    if [[ ! -d "$model_dir" ]]; then
        echo "WARNING: Model directory not found: ${model_dir}"
        continue
    fi
    
    echo ""
    echo "======================================================================"
    echo "MODEL: ${model_dir}"
    echo "======================================================================"
    
    is_parcae=false
    if is_parcae_model "$model_dir"; then
        echo "Type: Parcae (looped) model"
        is_parcae=true
    else
        echo "Type: Standard model"
    fi
    
    # Run each benchmark
    for benchmark in "${BENCHMARKS[@]}"; do
        IFS=':' read -r bench_name eval_tasks eval_config <<< "$benchmark"
        
        if [[ "$is_parcae" == "true" ]] && should_sweep_recurrence "$bench_name" && [[ ${#RECURRENCE_DEPTHS[@]} -gt 0 ]]; then
            # Parcae model with recurrence sweep for this benchmark
            for rec in "${RECURRENCE_DEPTHS[@]}"; do
                eval_name="${bench_name}_rec${rec}"
                TOTAL_EVALS=$((TOTAL_EVALS + 1))
                if eval_already_done "$model_dir" "$eval_name"; then
                    SKIPPED_EVALS=$((SKIPPED_EVALS + 1))
                fi
                run_eval "$model_dir" "$eval_name" "$eval_tasks" "$eval_config" "$rec"
                COMPLETED_EVALS=$((COMPLETED_EVALS + 1))
            done
        elif [[ "$is_parcae" == "true" ]]; then
            # Parcae model but no recurrence sweep for this benchmark
            default_rec=$(get_default_recurrence "$model_dir")
            eval_name="${bench_name}_rec${default_rec}"
            TOTAL_EVALS=$((TOTAL_EVALS + 1))
            if eval_already_done "$model_dir" "$eval_name"; then
                SKIPPED_EVALS=$((SKIPPED_EVALS + 1))
            fi
            run_eval "$model_dir" "$eval_name" "$eval_tasks" "$eval_config" ""
            COMPLETED_EVALS=$((COMPLETED_EVALS + 1))
        else
            # Standard model
            eval_name="${bench_name}"
            TOTAL_EVALS=$((TOTAL_EVALS + 1))
            if eval_already_done "$model_dir" "$eval_name"; then
                SKIPPED_EVALS=$((SKIPPED_EVALS + 1))
            fi
            run_eval "$model_dir" "$eval_name" "$eval_tasks" "$eval_config" ""
            COMPLETED_EVALS=$((COMPLETED_EVALS + 1))
        fi
    done
done

echo ""
echo "=============================================="
echo "Batch Evaluation Complete"
echo "Completed: ${COMPLETED_EVALS}/${TOTAL_EVALS} evaluations (${SKIPPED_EVALS} skipped)"
echo "=============================================="

