#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_ROOT="${OUT_DIR:-}"
if [ -z "$EXPERIMENT_ROOT" ]; then
  d="$SCRIPT_DIR"
  while [ "$d" != "/" ] && [ ! -d "$d/experiments/factowl_eval" ]; do d="$(dirname "$d")"; done
  EXPERIMENT_ROOT="$d"
fi
OUT_DIR="$EXPERIMENT_ROOT/experiments/factowl_eval"
mkdir -p "$SCRIPT_DIR/logs" "$OUT_DIR"
export VLLM_USE_RAY=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GPU="${GPU:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
K_LIST="${K_LIST:-1,2,4,8,16,32,64,99}"
MODEL="${MODEL:-unsloth/Llama-3.1-8B-Instruct}"
PYTHON="${PYTHON:-/opt/conda/envs/factowl/bin/python}"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
MASTER_LOG="$SCRIPT_DIR/logs/factowl_zh_llmagg_curve_details_${RUN_TS}.log"
export CUDA_VISIBLE_DEVICES="$GPU"
for domain in rivers cars disasters; do
  echo "=== $(date -u) domain=${domain} ===" | tee -a "$MASTER_LOG"
  "$PYTHON" run_llmagg_curve.py \
    --domain "$domain" \
    --lang zh \
    --final-only \
    --gpu "$CUDA_VISIBLE_DEVICES" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --k-list "$K_LIST" \
    --model "$MODEL" \
    --save-topic-details \
    --log "$OUT_DIR/factowl_llmagg_curve_zh.log" \
    --data-dir "$SCRIPT_DIR/data" \
    --output-dir "$OUT_DIR" \
    2>&1 | tee -a "$MASTER_LOG"
done
echo "=== $(date -u) done ===" | tee -a "$MASTER_LOG"
