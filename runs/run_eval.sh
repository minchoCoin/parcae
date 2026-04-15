#!/bin/bash
set -e

export PYTHONUNBUFFERED=1

OUT_DIR="${1:-outputs/gpt-small-140m-final}"
EVAL_CONFIG="${2:-eval_configs/eval-core.yaml}"
NUM_GPUS="${3:-8}"

echo "=============================================="
echo "Evaluating: ${OUT_DIR}"
echo "Config: ${EVAL_CONFIG}"
echo "GPUs: ${NUM_GPUS}"
echo "=============================================="

torchrun --nproc_per_node=${NUM_GPUS} \
    scripts/eval.py \
    --out_dir ${OUT_DIR} \
    --config ${EVAL_CONFIG}

echo "=============================================="
echo "Evaluation complete"
echo "=============================================="
