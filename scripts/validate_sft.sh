#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG_PATH="${1:-configs/sft_qwen3_4b.yaml}"
"$PYTHON_BIN" -m sieve.cli.train_sft \
  --root "$ROOT_DIR" \
  --config "$CONFIG_PATH" \
  --num-processes "${SIEVE_NUM_GPUS:-1}" \
  --validate-only
