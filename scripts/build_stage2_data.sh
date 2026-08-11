#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PYTHON_BIN" -m sieve.cli.build_rl_data \
  --root "$ROOT_DIR" \
  --source "${SIEVE_STAGE2_SOURCE:-data/sft/source/records.jsonl}" \
  --output-dir "${SIEVE_STAGE2_DATA_DIR:-data/rl}" \
  --seed "${SIEVE_SEED:-42}" \
  --train-count "${SIEVE_RL_TRAIN_COUNT:-1600}" \
  --dev-count "${SIEVE_RL_DEV_COUNT:-200}" \
  --test-count "${SIEVE_RL_TEST_COUNT:-300}"
