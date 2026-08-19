#!/usr/bin/env bash
# Stage-1 SFT v3 Data Build Pipeline
# Usage: ./scripts/build_stage1_v3.sh [--preview] [--full]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PREVIEW=0
FULL=0
for arg in "$@"; do
  case "$arg" in
    --preview) PREVIEW=1 ;;
    --full)    FULL=1 ;;
  esac
done

# Step 1: Ensure τ³-bench source data is in data/raw/
echo "=== Step 1: Verify source data ==="
if [[ ! -f "data/raw/tau2-bench/data/tau2/domains/airline/tasks.json" ]]; then
  echo "ERROR: τ³-bench source data not found at data/raw/tau2-bench/"
  echo "Run: git clone https://github.com/sierra-research/tau2-bench.git data/raw/tau2-bench"
  exit 1
fi
echo "τ³-bench source data: OK"

# Step 2: Serialize CAR-bench tasks if not already done
echo "=== Step 2: Verify CAR-bench serialized data ==="
if [[ ! -f "data/raw/car-bench/serialized/base_train.jsonl" ]]; then
  echo "Serializing CAR-bench tasks..."
  "$PYTHON_BIN" scripts/serialize_car_bench_tasks.py \
    --car-bench-repo tmp/car-bench-repo \
    --output-dir data/raw/car-bench/serialized
fi
echo "CAR-bench serialized data: OK"

# Step 3: Normalize sources into GroundedSourceRecord format
echo "=== Step 3: Prepare grounded sources ==="
"$PYTHON_BIN" -m sieve.cli.prepare_grounded_sources \
  --root "$ROOT_DIR" \
  --output-dir tmp/sft-v3-build/normalized

# Step 4: Generate data
BUILD_DIR="tmp/sft-v3-build"
mkdir -p "$BUILD_DIR"

if [[ "$PREVIEW" -eq 1 ]]; then
  echo "=== Step 4: Preview generation (50 records, dry-run) ==="
  "$PYTHON_BIN" -m sieve.cli.generate_grounded \
    --root "$ROOT_DIR" \
    --config configs/stage1_sft_v3_generation.json \
    --total 50 \
    --dry-run \
    --output-dir "$BUILD_DIR/-preview"
  echo "Preview complete. Inspect $BUILD_DIR/preview/"
  exit 0
fi

# Full generation requires ZAI_API_KEY
if [[ -z "${ZAI_API_KEY:-}" ]]; then
  echo "ERROR: ZAI_API_KEY not set. Set it before running full generation:"
  echo "  export ZAI_API_KEY='your-key'"
  exit 1
fi

if [[ "$FULL" -eq 1 ]]; then
  echo "=== Step 4: Full live generation (6000 records) ==="
  "$PYTHON_BIN" -m sieve.cli.generate_grounded \
    --root "$ROOT_DIR" \
    --config configs/stage1_sft_v3_generationA.json \
    --total 6000 \
    --confirm-full 6000 \
    --output-dir "$BUILD_DIR"
else
  echo "=== Step 4: Preview generation with live LLM (100 records) ==="
  "$PYTHON_BIN" -m sieve.cli.generate_grounded \
    --root "$ROOT_DIR" \
    --config configs/stage1_sft_v3_generation.json \
    --total 100 \
    --output-dir "$BUILD_DIR/preview-live"
  echo "Live preview complete. Review before running --full."
  exit 0
fi

# Step 5: Export training splits
echo "=== Step 5: Export training splits ==="
"$PYTHON_BIN" -m sieve.cli.export_grounded \
  --root "$ROOT_DIR" \
  --input "$BUILD_DIR/accepted.jsonl" \
  --output-dir "$BUILD_DIR/clean"

# Step 6: Validate
echo "=== Step 6: Validate ==="
"$PYTHON_BIN" -m sieve.cli.validate_stage1_data \
  --root "$ROOT_DIR" \
  --accepted "$BUILD_DIR/accepted.jsonl" \
  --train "$BUILD_DIR/clean/train.jsonl" \
  --dev "$BUILD_DIR/clean/dev.jsonl"

# Step 7: Atomic publish
echo "=== Step 7: Atomic publish ==="
if [[ -d "data/sft" ]]; then
  echo "Backing up existing data/sft → data/sft.old.tmp"
  mv data/sft data/sft.old.tmp
fi

"$PYTHON_BIN" -m sieve.cli.publish_stage1_data \
  --root "$ROOT_DIR" \
  --input "$BUILD_DIR/accepted.jsonl" \
  --split-dir "$BUILD_DIR/clean" \
  --output-dir data/sft \
  --prompt-version stage1-system-v2 \
  --quality-report "$BUILD_DIR/quality_report.json"

# Step 8: Post-publish validation
echo "=== Step 8: Post-publish validation ==="
"$PYTHON_BIN" -m sieve.cli.train_sft \
  --root "$ROOT_DIR" \
  --config configs/sft_qwen3_4b.yaml \
  --num-processes 1 \
  --validate-only

if [[ $? -eq 0 ]]; then
  echo "===: Publish successful! ==="
  if [[ -d "data/sft.old.tmp" ]]; then
    echo "Removing old backup..."
    rm -rf data/sft.old.tmp
  fi
else
  echo "!!! Post-publish validation FAILED. Rolling back..."
  rm -rf data/sft
  mv data/sft.old.tmp data/sft
  exit 1
fi

echo "=== Stage-1 SFT v3 build complete ==="
