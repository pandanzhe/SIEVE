#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG_PATH="${1:-configs/rl.yaml}"
"$PYTHON_BIN" -m sieve.cli.train_rl --root "$ROOT_DIR" --config "$CONFIG_PATH"
