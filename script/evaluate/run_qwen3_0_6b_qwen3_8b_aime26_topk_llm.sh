#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export CUDA_VISIBLE_DEVICES
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_PATH="${CONFIG_PATH:-config/Qwen3-0.6B+Qwen3-8B_entropy_js_topk_llm.yaml}"
DATASET="${DATASET:-aime26}"
OUTPUT_DIR="${OUTPUT_DIR:-output/eval/qwen3_0_6b_qwen3_8b_${DATASET}_topk_llm_$(date +%Y%m%d_%H%M%S)}"

SLM_TP_SIZE="${SLM_TP_SIZE:-1}"
LLM_TP_SIZE="${LLM_TP_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
RESUME="${RESUME:-0}"
SKIP_PROBLEM_IDS="${SKIP_PROBLEM_IDS:-}"
RERUN_MISSING_ANSWERS="${RERUN_MISSING_ANSWERS:-1}"
RERUN_MAX_NEW_TOKENS="${RERUN_MAX_NEW_TOKENS:-24576}"

build_cmd() {
  local max_new_tokens="$1"
  local problem_ids_value="${2:-}"
  local include_resume="${3:-0}"

  cmd=(
    "$PYTHON_BIN" script/evaluate/hf_dataset_sglang.py
    --dataset "$DATASET"
    --config-path "$CONFIG_PATH"
    --use_hybrid
    --slm_tp_size "$SLM_TP_SIZE"
    --llm_tp_size "$LLM_TP_SIZE"
    --batch_size "$BATCH_SIZE"
    --max_new_tokens "$max_new_tokens"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
    --output_dir "$OUTPUT_DIR"
  )

  if [[ -n "${DATASET_PATH:-}" ]]; then
    cmd+=(--dataset_path "$DATASET_PATH")
  fi

  if [[ "$include_resume" == "1" ]]; then
    if [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "TRUE" ]]; then
      cmd+=(--resume)
    fi
  fi

  if [[ -n "${NUM_PROBLEMS:-}" ]]; then
    cmd+=(--num_problems "$NUM_PROBLEMS")
  fi

  if [[ -n "${REPEAT_INPUT_NUM:-}" ]]; then
    cmd+=(--repeat_input_num "$REPEAT_INPUT_NUM")
  fi

  if [[ -n "$problem_ids_value" ]]; then
    cmd+=(--problem_ids "$problem_ids_value")
  fi

  if [[ -n "${SKIP_PROBLEM_IDS:-}" ]]; then
    cmd+=(--skip_problem_ids "$SKIP_PROBLEM_IDS")
  fi

  if [[ "${OVERLAP_TP_SCHEDULE:-0}" == "1" ]]; then
    cmd+=(--overlap_tp_schedule)
  fi

  if [[ "${DEBUG:-0}" == "1" ]]; then
    cmd+=(--debug)
  fi
}

maybe_rerun_missing_answers() {
  if [[ "$RERUN_MISSING_ANSWERS" != "1" && "$RERUN_MISSING_ANSWERS" != "true" && "$RERUN_MISSING_ANSWERS" != "TRUE" ]]; then
    return
  fi

  if (( RERUN_MAX_NEW_TOKENS <= MAX_NEW_TOKENS )); then
    echo "Skipping missing-answer rerun because RERUN_MAX_NEW_TOKENS=$RERUN_MAX_NEW_TOKENS <= MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
    return
  fi

  local rerun_problem_ids
  rerun_problem_ids="$("$PYTHON_BIN" script/evaluate/find_missing_answer_problem_ids.py --output_dir "$OUTPUT_DIR")"

  if [[ -z "$rerun_problem_ids" ]]; then
    echo "No problems without extracted answers found. Skipping rerun."
    return
  fi

  echo
  echo "Rerunning problems without extracted answers with a larger token budget:"
  echo "  PROBLEM_IDS=$rerun_problem_ids"
  echo "  MAX_NEW_TOKENS=$RERUN_MAX_NEW_TOKENS"
  echo

  build_cmd "$RERUN_MAX_NEW_TOKENS" "$rerun_problem_ids" 0
  printf 'Rerun command:'
  printf ' %q' "${cmd[@]}"
  printf '\n\n'

  "${cmd[@]}"
}

build_cmd "$MAX_NEW_TOKENS" "${PROBLEM_IDS:-}" 1

echo "Running ${DATASET} entropy_js_topk_llm evaluation with:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  CONFIG_PATH=$CONFIG_PATH"
echo "  DATASET=$DATASET"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  RESUME=$RESUME"
echo "  SKIP_PROBLEM_IDS=${SKIP_PROBLEM_IDS:-<none>}"
echo "  RERUN_MISSING_ANSWERS=$RERUN_MISSING_ANSWERS"
echo "  RERUN_MAX_NEW_TOKENS=$RERUN_MAX_NEW_TOKENS"
echo
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n\n'

"${cmd[@]}"
maybe_rerun_missing_answers
