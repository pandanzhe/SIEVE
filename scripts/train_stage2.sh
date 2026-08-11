#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG_PATH="${1:-configs/rl_qwen25_3b.yaml}"
NUM_PROCESSES="${SIEVE_NUM_GPUS:-2}"

"$PYTHON_BIN" -m accelerate.commands.launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision bf16 \
  -m sieve.cli.train_rl \
  --root "$ROOT_DIR" \
  --config "$CONFIG_PATH" \
  --num-processes "$NUM_PROCESSES"
