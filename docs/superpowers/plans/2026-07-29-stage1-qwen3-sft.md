# Stage 1 Qwen3-4B Structured SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a portable, testable Stage-1 structured SFT pipeline for Qwen3-4B that defaults to two A100 40G GPUs and can also run on one or four GPUs without code changes.

**Architecture:** Keep the existing NumPy reference trainer intact. Add an HF backend built from Transformers, PEFT LoRA, and Accelerate. The collator constructs causal-LM targets plus gated structured labels; the policy uses a causally safe prompt-boundary representation for decision, affected-field, verification, and per-slot patch-operation heads; the training service owns distributed preparation, evaluation, best-checkpoint selection, and resume state.

**Tech Stack:** Python 3.10+, PyTorch, Transformers >= 4.51, PEFT, Accelerate, PyYAML, unittest.

---

### Task 1: Stage-1 configuration contract

**Files:**
- Create: `src/sieve/training/hf_sft_config.py`
- Create: `tests/test_hf_sft_config.py`
- Create: `configs/sft_qwen3_4b.yaml`

- [ ] **Step 1: Write failing tests**

Test that the Qwen configuration resolves repository-relative data/output paths, defaults to BF16 LoRA and two processes, rejects unsupported precision and invalid batch accumulation, and never requires four GPUs.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_hf_sft_config -v`

Expected: FAIL because `sieve.training.hf_sft_config` does not exist.

- [ ] **Step 3: Implement the contract**

Add immutable configuration dataclasses and `parse_hf_sft_config`. Keep paths relative in YAML and resolve them only against the CLI root. Define Qwen3-4B, `enable_thinking: false`, BF16, LoRA rank 16, effective batch size 32, and `num_processes: 2`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_hf_sft_config -v`

Expected: PASS.

### Task 2: Causally safe structured collation

**Files:**
- Modify: `src/sieve/policies/hf_data.py`
- Create: `tests/test_hf_data_contract.py`

- [ ] **Step 1: Write failing tests**

Use a lightweight tokenizer and records from the toy generator to verify prompt-boundary pooling, UPDATE-only per-slot patch labels, HOLD-only verification labels, field-slot masks, and Qwen non-thinking template arguments.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_hf_data_contract -v`

Expected: FAIL because the current collator has no prompt pool index and patch labels are one-dimensional.

- [ ] **Step 3: Implement the collator**

Return `pool_indices`, `slot_mask`, two-dimensional `patch_operation_labels`, and `patch_operation_mask`. Pool at the last prompt token so structured heads cannot see the gold completion. Use the chat template with `enable_thinking=False` when supported and retain a deterministic fallback.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_hf_data_contract -v`

Expected: PASS.

### Task 3: Multi-head loss and checkpoint bundle

**Files:**
- Modify: `src/sieve/policies/hf_lora_policy.py`
- Create: `src/sieve/training/hf_sft_metrics.py`
- Create: `tests/test_hf_sft_metrics.py`

- [ ] **Step 1: Write failing tests**

Test pure metric accumulation for decision macro-F1, affected-field F1, verification F1, patch accuracy, false-update rate, and executable proxy. Test that invalid labels are excluded.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_hf_sft_metrics -v`

Expected: FAIL because the metric module does not exist.

- [ ] **Step 3: Implement metrics and update model semantics**

Make patch-operation logits per slot, use prompt-boundary pooling, apply all branch masks before loss reduction, expose detached component losses, load Qwen in explicit BF16/FP32 mode, disable cache during training, enable gradient checkpointing by configuration, and save adapter/head/schema metadata.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_hf_sft_metrics -v`

Expected: PASS.

### Task 4: Accelerate training service

**Files:**
- Replace: `src/sieve/training/hf_sft.py`
- Create: `tests/test_hf_sft_runtime.py`

- [ ] **Step 1: Write failing tests**

Test dependency reporting, optimizer parameter grouping, constrained checkpoint scoring, dataset audit, and a validate-only execution path without importing torch.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_hf_sft_runtime -v`

Expected: FAIL because the runtime APIs do not exist.

- [ ] **Step 3: Implement the service**

Build lazy HF imports, deterministic data loaders, Accelerate gradient accumulation, AdamW parameter groups, cosine scheduling, evaluation aggregation, best/final adapter bundles, trainer-state checkpoints, JSONL metrics, and exact resume support. Do not download or instantiate a model in validate-only mode.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_hf_sft_runtime -v`

Expected: PASS.

### Task 5: CLI and portable launch scripts

**Files:**
- Modify: `src/sieve/cli/train_sft.py`
- Modify: `scripts/run_sft.sh`
- Create: `scripts/validate_sft.sh`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/training_design.md`
- Modify: `tests/test_portability.py`

- [ ] **Step 1: Write failing tests**

Verify the launch script uses `SIEVE_NUM_GPUS` with a default of two, delegates root setup to `common.sh`, invokes Accelerate only for HF training, and supports offline validation.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_portability -v`

Expected: FAIL until the Stage-1 launch contract is present.

- [ ] **Step 3: Implement CLI and documentation**

Dispatch the legacy NumPy and new HF backends from one CLI. Add `--validate-only`; keep all paths relative outside `common.sh`; pin Qwen-compatible minimum dependency versions; document one-, two-, and four-GPU commands and explain why two GPUs are the default.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_portability -v`

Expected: PASS.

### Task 6: Repository verification

**Files:**
- No production changes unless a failing test exposes a defect.

- [ ] **Step 1: Compile**

Run: `python -m compileall -q src tests`

Expected: exit code 0.

- [ ] **Step 2: Run all tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass; optional HF/GPU integration tests may be explicitly skipped when dependencies are absent.

- [ ] **Step 3: Validate the real data/config chain**

Run: `python -m sieve.cli.train_sft --root . --config configs/sft_qwen3_4b.yaml --validate-only`

Expected: reports the existing 4,874/605/521 split, configuration, and missing local HF dependencies without starting training or modifying the dataset.

- [ ] **Step 4: Inspect portability**

Run: `rg -n "[A-Za-z]:[\\\\/]" src configs scripts`

Expected: no hard-coded Windows absolute path in portable source/configuration/launch files.
