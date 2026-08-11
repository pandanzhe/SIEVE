#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG_PATH="${1:-configs/sft_qwen25_3b.yaml}"
NUM_PROCESSES="${SIEVE_NUM_GPUS:-1}"

if [[ "${SIEVE_VALIDATE_ONLY:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m sieve.cli.train_sft \
    --root "$ROOT_DIR" \
    --config "$CONFIG_PATH" \
    --num-processes "$NUM_PROCESSES" \
    --validate-only
  exit 0
fi

"$PYTHON_BIN" -m accelerate.commands.launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision bf16 \
  -m sieve.cli.train_sft \
  --root "$ROOT_DIR" \
  --config "$CONFIG_PATH" \
  --num-processes "$NUM_PROCESSES"
