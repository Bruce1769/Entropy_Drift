#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

CONFIG_PATH="${CONFIG_PATH:-config/Qwen3-0.6B+Qwen3-8B_entropy_js_topk_sparse.yaml}"
DATASET="${DATASET:-aime}"
OUTPUT_DIR="${OUTPUT_DIR:-output/eval/qwen3_0_6b_qwen3_8b_${DATASET}_sparse_$(date +%Y%m%d_%H%M%S)}"

SLM_TP_SIZE="${SLM_TP_SIZE:-1}"
LLM_TP_SIZE="${LLM_TP_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"

cmd=(
  python script/evaluate/hf_dataset_sglang.py
  --dataset "$DATASET"
  --config-path "$CONFIG_PATH"
  --use_hybrid
  --slm_tp_size "$SLM_TP_SIZE"
  --llm_tp_size "$LLM_TP_SIZE"
  --batch_size "$BATCH_SIZE"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --top_k "$TOP_K"
  --output_dir "$OUTPUT_DIR"
)

if [[ "${OVERLAP_TP_SCHEDULE:-0}" == "1" ]]; then
  cmd+=(--overlap_tp_schedule)
fi

if [[ "${DEBUG:-0}" == "1" ]]; then
  cmd+=(--debug)
fi

if [[ -n "${NUM_PROBLEMS:-}" ]]; then
  cmd+=(--num_problems "$NUM_PROBLEMS")
fi

echo "Running AIME24+25 sparse top-k evaluation with:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  CONFIG_PATH=$CONFIG_PATH"
echo "  DATASET=$DATASET"
echo "  SLM_TP_SIZE=$SLM_TP_SIZE"
echo "  LLM_TP_SIZE=$LLM_TP_SIZE"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n\n'

"${cmd[@]}"
