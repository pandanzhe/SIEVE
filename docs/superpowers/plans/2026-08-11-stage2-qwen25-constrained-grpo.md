# Stage-2 Qwen2.5-3B Constrained GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable Stage-2 scenario corpus and a server-ready Qwen2.5-3B constrained multi-step GRPO learner initialized from the Stage-1 PEFT adapter.

**Architecture:** Compile provenance-grounded Stage-1 source records into task-disjoint, action-conditioned scenario graphs. Run those graphs through a deterministic environment locally and through an autoregressive Qwen2.5 PEFT rollout/update loop on the server. Keep oracle fields inside the environment and update only LoRA parameters.

**Tech Stack:** Python 3.10+, dataclasses, NumPy, PyYAML, PyTorch 2.7, Transformers 4.53, PEFT 0.16, Accelerate 1.8, Bash.

---

### Task 1: Scenario schema and deterministic builder

**Files:**
- Create: `src/sieve/rl_data/schema.py`
- Create: `src/sieve/rl_data/build.py`
- Create: `src/sieve/rl_data/io.py`
- Create: `src/sieve/rl_data/__init__.py`
- Test: `tests/test_rl_data.py`

- [ ] Write tests that construct accepted authoritative audit rows and assert task-level split isolation, required event kinds, three belief slots, and no top-level SFT `target`.
- [ ] Run `python -m unittest tests.test_rl_data -v` and confirm import failure for the missing package.
- [ ] Implement immutable `RLScenario` and `ScenarioEvent` types with explicit public context and private oracle/event fields.
- [ ] Implement stable SHA-256 base-task splitting and exact domain-stratified variant allocation.
- [ ] Implement atomic JSONL writers, SHA-256 manifests, and quality counters.
- [ ] Re-run `python -m unittest tests.test_rl_data -v` and confirm all tests pass.

### Task 2: Action-conditioned scenario environment

**Files:**
- Create: `src/sieve/environments/scenario_env.py`
- Modify: `src/sieve/environments/__init__.py`
- Test: `tests/test_scenario_env.py`

- [ ] Write tests for reset policy-view isolation, HOLD+VERIFY branching, unavailable verification, wrong UPDATE persistence, budget decrement, and terminal success.
- [ ] Run `python -m unittest tests.test_scenario_env -v` and confirm the missing environment failure.
- [ ] Implement `ScenarioRevisionEnvironment` using `StateExecutor` as the unique state writer.
- [ ] Add `invalid_format` to the scenario cost vector without changing the legacy replay cost contract.
- [ ] Re-run `python -m unittest tests.test_scenario_env -v` and confirm all tests pass.

### Task 3: Strict JSON action protocol and readiness metrics

**Files:**
- Create: `src/sieve/policies/hf_revision_io.py`
- Create: `src/sieve/evaluation/readiness.py`
- Test: `tests/test_hf_revision_io.py`
- Test: `tests/test_readiness.py`

- [ ] Write parser tests for valid UPDATE/HOLD/IGNORE, code fences, trailing prose, unknown fields, invalid patch operations, and safe fallback behavior.
- [ ] Run both focused tests and confirm missing-module failures.
- [ ] Implement prompt rendering from `RevisionContext`, strict JSON decoding, schema conversion, canonical serialization, and completion-level metrics.
- [ ] Implement threshold evaluation that returns individual gate results and an overall `ready` flag.
- [ ] Re-run focused tests and confirm all pass.

### Task 4: HF constrained GRPO configuration and pure math

**Files:**
- Create: `src/sieve/training/hf_grpo_config.py`
- Create: `src/sieve/training/hf_grpo_math.py`
- Create: `configs/rl_qwen25_3b.yaml`
- Test: `tests/test_hf_grpo_config.py`
- Test: `tests/test_hf_grpo_math.py`

- [ ] Write tests for repository-relative paths, two-GPU batch divisibility, non-negative constraints, group size, advantage normalization, clipping, and sampled KL.
- [ ] Run focused tests and confirm missing-module failures.
- [ ] Implement frozen configuration dataclasses and validation.
- [ ] Implement NumPy reference functions for group advantages, clipped surrogate, and non-negative sampled KL.
- [ ] Re-run focused tests and confirm all pass.

### Task 5: Qwen2.5 PEFT rollout and update loop

**Files:**
- Create: `src/sieve/training/hf_constrained_grpo.py`
- Test: `tests/test_hf_grpo_runtime.py`

- [ ] Write dependency-free tests for dataset audit, adapter layout audit, trainable-parameter filtering, and validate-only behavior.
- [ ] Run the focused test and confirm missing-module failures.
- [ ] Implement lazy Torch/Transformers/PEFT imports, policy/reference loading, seeded JSON generation, completion log-probability calculation, trajectory collection, clipped updates, KL regularization, Lagrange updates, checkpointing, and metrics JSONL.
- [ ] Ensure invalid generations use a safe transition fallback while retaining `invalid_format` cost.
- [ ] Re-run the focused test and confirm it passes without loading a model.

### Task 6: CLI, shell entrypoints, and dependencies

**Files:**
- Create: `src/sieve/cli/build_rl_data.py`
- Modify: `src/sieve/cli/train_rl.py`
- Create: `scripts/build_rl_data.sh`
- Modify: `scripts/run_rl.sh`
- Create: `scripts/validate_rl.sh`
- Create: `requirements/rl.txt`
- Test: `tests/test_stage2_cli_contract.py`

- [ ] Write tests that inspect defaults, relative paths, validate-only execution, shell root handling, and dependency pins.
- [ ] Run the focused test and confirm expected failures.
- [ ] Implement CLI argument/config plumbing only; keep generation and training logic in library modules.
- [ ] Re-run the focused test and confirm all pass.

### Task 7: Generate and audit the canonical Stage-2 corpus

**Files:**
- Generate: `data/rl/scenarios/train.jsonl`
- Generate: `data/rl/scenarios/dev.jsonl`
- Generate: `data/rl/scenarios/test.jsonl`
- Generate: `data/rl/manifest.json`
- Generate: `data/rl/quality_report.json`

- [ ] Run `bash scripts/build_rl_data.sh configs/rl_qwen25_3b.yaml` or the equivalent Python module command.
- [ ] Assert exact 1,600/200/300 counts, zero base-task overlap, deterministic checksums, required domain quotas, and parseable scenarios.
- [ ] Run the generator a second time into `tmp/rl-repro` and compare checksums.

### Task 8: Documentation and regression verification

**Files:**
- Modify: `README.md`
- Modify: `docs/training_design.md`
- Create: `docs/stage2_rl_experiment_plan.md`

- [ ] Document Stage-1 graduation gates, Qwen2.5-3B checkpoint layout, dataset semantics, reward/cost definitions, one/two-GPU commands, and the internal-test boundary.
- [ ] Run focused Stage-2 tests.
- [ ] Run the entire legacy and Stage-2 unit-test suite with `PYTHONPATH=src`.
- [ ] Run `python -m compileall -q src tests`.
- [ ] Run both Stage-2 CLIs in validate-only mode and confirm no model or GPU is required.
- [ ] Inspect `git diff --check`, `git status --short`, generated manifest counts, and absence of absolute local paths.
