#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG_PATH="${1:-configs/rl_qwen25_3b.yaml}"
CHECKPOINT_PATH="${2:-outputs/stage2-qwen25-3b/best}"
SPLIT="${3:-test}"
NUM_PROCESSES="${SIEVE_NUM_GPUS:-2}"

"$PYTHON_BIN" -m accelerate.commands.launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision bf16 \
  -m sieve.cli.evaluate_stage2 \
  --root "$ROOT_DIR" \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --split "$SPLIT" \
  --num-processes "$NUM_PROCESSES"
