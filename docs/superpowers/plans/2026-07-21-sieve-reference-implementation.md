# SIEVE Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable, end-to-end SIEVE reference implementation that generates paired observation data, trains a structured revision policy with SFT, improves it with constrained trajectory optimization, and evaluates step/state/trajectory metrics.

**Architecture:** The default backend is a deterministic NumPy research harness so the entire pipeline runs without GPU packages or model downloads. It exercises the same typed state, factorized revision action, deterministic executor, reward costs, Lagrange constraints, and checkpoint interfaces as the optional Hugging Face LoRA backend. Python code contains no machine-specific paths; shell entrypoints compute the repository root from their own location.

**Tech Stack:** Python 3.10+, standard library, NumPy, PyYAML; optional PyTorch, Transformers, and PEFT for the 7B/8B LoRA backend; `unittest` for dependency-free tests.

**Local execution boundary:** Do not start SFT, GRPO, model downloads, or end-to-end training on the current machine. Local verification is limited to syntax, data contracts, state logic, and non-training tests.

---

### Task 1: Project skeleton and portable configuration

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `configs/toy.yaml`
- Create: `configs/sft.yaml`
- Create: `configs/rl.yaml`
- Create: `src/sieve/__init__.py`
- Create: `src/sieve/config.py`
- Test: `tests/test_config.py`

- [ ] Write a failing `unittest` proving relative paths resolve against an explicit repository root and reject paths escaping that root.
- [ ] Run `python -m unittest tests.test_config -v` and confirm failure because `sieve.config` does not exist.
- [ ] Implement YAML loading, recursive override merging, root-relative path resolution, deterministic seeding, and output-directory creation.
- [ ] Add package metadata and optional `hf` dependencies without making GPU libraries mandatory.
- [ ] Run the config test and confirm it passes.

### Task 2: Typed state and deterministic executor

**Files:**
- Create: `src/sieve/core/__init__.py`
- Create: `src/sieve/core/types.py`
- Create: `src/sieve/core/executor.py`
- Test: `tests/test_executor.py`

- [ ] Write failing tests for IGNORE immutability, HOLD candidate isolation, affected-field locality, invalid patch rejection, fixed ledger capacity, and pending-dependency action blocking.
- [ ] Run `python -m unittest tests.test_executor -v` and confirm the import failure.
- [ ] Implement enums and dataclasses for belief slots, observations, risk envelopes, patches, verification requests, revision outputs, budgets, ledgers, and executor results.
- [ ] Implement a pure deterministic executor that deep-copies state, validates actions, records costs, and never mutates caller-owned objects.
- [ ] Run executor tests and confirm all invariants pass.

### Task 3: Event transition, toy environment, and paired perturbations

**Files:**
- Create: `src/sieve/environments/__init__.py`
- Create: `src/sieve/environments/protocol.py`
- Create: `src/sieve/environments/toy.py`
- Create: `src/sieve/environments/perturbations.py`
- Create: `src/sieve/core/transition.py`
- Create: `src/sieve/data/__init__.py`
- Create: `src/sieve/data/schema.py`
- Create: `src/sieve/data/generate.py`
- Test: `tests/test_transition.py`
- Test: `tests/test_data.py`

- [ ] Write failing tests showing revision time increments per atomic observation, environment time increments only on external action, verification produces a later observation, and scenario groups do not cross dataset splits.
- [ ] Run transition/data tests and confirm failure for missing modules.
- [ ] Implement a small order-fulfilment environment with oracle state, reversible/irreversible actions, verification tools, and delayed consequences.
- [ ] Implement clean, stale, wrong-entity, weak-source, conflict, irrelevant, delayed-correction, duplicate, and out-of-order perturbations.
- [ ] Generate JSONL SFT records and grouped train/dev/test splits with deterministic seeds.
- [ ] Run transition/data tests and confirm they pass.

### Task 4: Structured policy and optional HF LoRA adapter

**Files:**
- Create: `src/sieve/policies/__init__.py`
- Create: `src/sieve/policies/protocol.py`
- Create: `src/sieve/policies/features.py`
- Create: `src/sieve/policies/structured_policy.py`
- Create: `src/sieve/policies/hf_lora_policy.py`
- Test: `tests/test_policy.py`

- [ ] Write failing tests for fixed feature dimension, action masks, conditional log-probability, deterministic greedy prediction, and checkpoint round-trip.
- [ ] Run `python -m unittest tests.test_policy -v` and confirm failure for missing modules.
- [ ] Implement a hashed, fixed-dimensional featurizer and a NumPy factorized policy with decision, affected-field, and verification heads.
- [ ] Implement masked sampling and greedy prediction; derive typed patches from the selected valid branch.
- [ ] Implement lazy optional imports and a Hugging Face PEFT policy factory that attaches LoRA and structured heads without importing GPU dependencies during the toy run.
- [ ] Run policy tests and confirm they pass.

### Task 5: SFT training

**Files:**
- Create: `src/sieve/training/__init__.py`
- Create: `src/sieve/training/sft.py`
- Create: `src/sieve/cli/__init__.py`
- Create: `src/sieve/cli/generate.py`
- Create: `src/sieve/cli/train_sft.py`
- Test: `tests/test_sft.py`

- [ ] Write a failing test that trains on a small deterministic dataset and requires lower final loss plus improved decision accuracy.
- [ ] Run `python -m unittest tests.test_sft -v` and confirm failure.
- [ ] Implement masked multi-task cross entropy, mini-batch gradient updates, gradient clipping, class weighting, seeded shuffling, JSON metrics, and checkpoints.
- [ ] Implement generate and SFT CLI entrypoints that contain only argument/config plumbing.
- [ ] Run the SFT test and confirm it passes.

### Task 6: Rewards, Lagrangian constraints, and trajectory optimization

**Files:**
- Create: `src/sieve/training/rewards.py`
- Create: `src/sieve/training/lagrangian.py`
- Create: `src/sieve/training/constrained_grpo.py`
- Create: `src/sieve/cli/train_rl.py`
- Test: `tests/test_rewards.py`
- Test: `tests/test_rl.py`

- [ ] Write failing tests for potential-difference shaping, false-update/unsafe/verify/stall costs, non-negative multiplier updates, clipped ratios, and a seeded RL smoke run.
- [ ] Run reward/RL tests and confirm failure.
- [ ] Implement task reward, state potential, costs, generalized returns, group-relative clipped policy loss, KL-to-SFT penalty, and projected Lagrange updates.
- [ ] Implement trajectory collection in the toy environment and update only revision policy parameters.
- [ ] Implement the RL CLI with resumable checkpoints and constraint metrics.
- [ ] Run reward/RL tests and confirm they pass.

### Task 7: Evaluation and reports

**Files:**
- Create: `src/sieve/evaluation/__init__.py`
- Create: `src/sieve/evaluation/metrics.py`
- Create: `src/sieve/evaluation/evaluator.py`
- Create: `src/sieve/cli/evaluate.py`
- Test: `tests/test_metrics.py`

- [ ] Write failing tests for macro-F1, false/missed update rates, affected-field F1, patch validity, state consistency, contamination area, recovery latency, task success, verification cost, and pass-at-k.
- [ ] Run `python -m unittest tests.test_metrics -v` and confirm failure.
- [ ] Implement pure metric functions and an evaluator that writes JSON and Markdown summaries.
- [ ] Add bootstrap confidence intervals with seeded resampling.
- [ ] Run metric tests and confirm they pass.

### Task 8: Shell orchestration and end-to-end verification

**Files:**
- Create: `scripts/common.sh`
- Create: `scripts/run_generate.sh`
- Create: `scripts/run_sft.sh`
- Create: `scripts/run_rl.sh`
- Create: `scripts/run_eval.sh`
- Create: `scripts/run_all.sh`
- Create: `tests/test_portability.py`

- [ ] Write a failing portability test that scans Python/config files for local absolute paths and checks every shell script derives `ROOT_DIR` through `common.sh`.
- [ ] Run `python -m unittest tests.test_portability -v` and confirm failure.
- [ ] Implement shell entrypoints using only a root derived from `BASH_SOURCE`, editable config arguments, and module execution.
- [ ] Run `bash -n scripts/*.sh` and the portability test.
- [ ] Run `python -m unittest discover -s tests -v` and require zero failures.
- [ ] Defer `bash scripts/run_all.sh` to a configured training machine; on the local machine run only syntax, data-contract, and non-training tests.
- [ ] Inspect generated checkpoints and `outputs/toy/evaluation/metrics.json` for finite values and required metrics.

### Task 9: Documentation and requirement audit

**Files:**
- Modify: `README.md`
- Modify: `docs/training_design.md`

- [ ] Document installation, toy execution, 7B/8B LoRA configuration, external environment adapters, outputs, reproducibility, and hardware profiles.
- [ ] Link the implementation modules to the mathematical variables \(s_t,y_t,B_t,U,P_{env},P_{obs},R,C\).
- [ ] Scan docs and code for unfinished placeholder markers and machine-specific paths.
- [ ] Re-run the full tests and end-to-end pipeline after documentation changes.
