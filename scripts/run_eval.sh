#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG_PATH="${1:-configs/toy.yaml}"
CHECKPOINT_PATH="${2:-outputs/toy/checkpoints/grpo_policy.npz}"
"$PYTHON_BIN" -m sieve.cli.evaluate \
  --root "$ROOT_DIR" \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH"
