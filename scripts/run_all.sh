#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

bash "$SCRIPT_DIR/run_generate.sh" "${1:-configs/toy.yaml}"
bash "$SCRIPT_DIR/run_sft.sh" "${2:-configs/sft.yaml}"
bash "$SCRIPT_DIR/run_rl.sh" "${3:-configs/rl.yaml}"
bash "$SCRIPT_DIR/run_eval.sh" "${1:-configs/toy.yaml}" \
  "${4:-outputs/toy/checkpoints/grpo_policy.npz}"
