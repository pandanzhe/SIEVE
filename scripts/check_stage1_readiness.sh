#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG_PATH="${1:-configs/rl_qwen25_3b.yaml}"
"$PYTHON_BIN" -m sieve.cli.check_stage1_readiness \
  --root "$ROOT_DIR" \
  --config "$CONFIG_PATH" \
  --max-samples "${SIEVE_READINESS_SAMPLES:-300}" \
  --probe-scenarios "${SIEVE_READINESS_PROBES:-8}" \
  --closed-loop-scenarios "${SIEVE_READINESS_CLOSED_LOOP:-100}"
