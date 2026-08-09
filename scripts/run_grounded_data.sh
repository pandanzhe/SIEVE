#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG_PATH="${1:-configs/grounded_full_dry_run.json}"
WORK_DIR="${SIEVE_DATA_WORK_DIR:-tmp/data_factory}"
NORMALIZED_DIR="${NORMALIZED_DIR:-$WORK_DIR/normalized-full}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/stage1-trajectory-v2}"
PER_SOURCE_LIMIT="${PER_SOURCE_LIMIT:-500}"
SEED="${SEED:-42}"
CANONICAL_DIR="${CANONICAL_DIR:-data/sft}"
PROMPT_VERSION="${PROMPT_VERSION:-stage1-system-v2}"

"$PYTHON_BIN" -m sieve.cli.prepare_grounded_sources \
  --root "$ROOT_DIR" \
  --output-dir "$NORMALIZED_DIR" \
  --per-source-limit "$PER_SOURCE_LIMIT"

"$PYTHON_BIN" -m sieve.cli.generate_grounded \
  --root "$ROOT_DIR" \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --source-manifest "$NORMALIZED_DIR/manifest.json" \
  --dry-run \
  --confirm-full 6000

"$PYTHON_BIN" -m sieve.cli.export_grounded \
  --root "$ROOT_DIR" \
  --input "$OUTPUT_DIR/accepted.jsonl" \
  --output-dir "$OUTPUT_DIR/training" \
  --seed "$SEED"

"$PYTHON_BIN" -m sieve.cli.publish_stage1_data \
  --root "$ROOT_DIR" \
  --input "$OUTPUT_DIR/accepted.jsonl" \
  --split-dir "$OUTPUT_DIR/training" \
  --output-dir "$CANONICAL_DIR" \
  --prompt-version "$PROMPT_VERSION" \
  --quality-report "$OUTPUT_DIR/quality_report.json"