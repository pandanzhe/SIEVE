# Stage-1 Trajectory-Grounded SFT Data Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Rebuild the 6,000-record Stage-1 SFT corpus as leakage-free, domain-adapted state-action supervision with a 70/15/15 UPDATE/HOLD/IGNORE distribution and real HOLD-to-verification successor transitions.

**Architecture:** Keep the upstream open-source snapshots and normalized grounded records. Add observable source metadata and prompt-safe serialization, generate 900 two-step HOLD→UPDATE episodes plus 4,200 singleton episodes across commerce, airline/telecom, and tool/workflow domains, validate every executor transition, and export trajectory-grouped train/dev splits. The fixed system prompt is owned by the HF prompt builder; datasets store context and target separately.

**Tech Stack:** Python 3.10+, standard library, existing SIEVE dataclasses/executor, unittest, Hugging Face tokenizer integration.

---

### Task 1: Prompt-safe observation schema

**Files:**
- Modify: `src/sieve/core/types.py`
- Modify: `src/sieve/data/io.py`
- Modify: `src/sieve/data/schema.py`
- Test: `tests/test_data.py`

- [x] Add failing tests proving observable `source_authority` and `authenticated` round-trip while `condition`, `relevant`, and `perturbation` are excluded from policy prompt serialization.
- [x] Run `python -m unittest tests.test_data -v` and verify the new tests fail for missing prompt-safe behavior.
- [x] Add optional source metadata to `Observation`, defaults for internal-only generation labels, and a bounded JSON prompt serializer.
- [x] Re-run `python -m unittest tests.test_data -v` and verify it passes.

### Task 2: Fixed system prompt and chat placement

**Files:**
- Modify: `src/sieve/policies/hf_data.py`
- Test: `tests/test_hf_data_contract.py`

- [x] Add failing tests that require one fixed system message, one JSON user context, `add_generation_prompt=True`, no leakage keys, and target-only LM labels ending in native EOS.
- [x] Run `python -m unittest tests.test_hf_data_contract -v` and verify the new tests fail.
- [x] Introduce `SIEVE_SYSTEM_PROMPT`, render prompt-safe context JSON as the user message, and keep the target out of the prompt.
- [x] Re-run `python -m unittest tests.test_hf_data_contract -v` and verify it passes.

### Task 3: Domain-aware normalization

**Files:**
- Modify: `src/sieve/data_factory/normalize.py`
- Test: `tests/test_data_factory_normalize.py`

- [x] Add failing WebArena tests requiring order-number entities and semantic fields such as `billing_address`, while preserving provenance.
- [x] Run `python -m unittest tests.test_data_factory_normalize -v` and verify failure.
- [x] Derive safe snake-case field IDs and order entities from observable source fields, retaining fallback behavior for non-order tasks.
- [x] Re-run the normalization tests and verify success.

### Task 4: Scenario-aware rule construction and adjacent verification

**Files:**
- Modify: `src/sieve/data_factory/models.py`
- Modify: `src/sieve/data_factory/rules.py`
- Create: `src/sieve/data_factory/trajectories.py`
- Test: `tests/test_data_factory_rules.py`
- Create: `tests/test_data_factory_trajectories.py`

- [x] Add failing tests for authoritative UPDATE, same-entity secondary-source HOLD, cross-entity IGNORE, and a HOLD successor whose state equals executor output from the preceding step.
- [x] Run the focused rule and trajectory tests and verify expected failures.
- [x] Implement domain source profiles, empty-slot consistency, wrong-entity pairing, HOLD ledger transitions, budget decrement, and authoritative verification successors.
- [x] Re-run focused tests and verify they pass.

### Task 5: Exact corpus quotas and trajectory-safe export

**Files:**
- Modify: `src/sieve/data_factory/quotas.py`
- Modify: `src/sieve/data_factory/export.py`
- Modify: `src/sieve/data_factory/pipeline.py`
- Test: `tests/test_data_factory_quotas.py`
- Test: `tests/test_data_factory_export.py`
- Test: `tests/test_data_factory_pipeline.py`

- [x] Add failing tests for exactly 4,200 UPDATE, 900 HOLD, and 900 IGNORE decisions, 900 adjacent HOLD-successor pairs, and indivisible scenario splits.
- [x] Run the focused tests and verify failures.
- [x] Implement the 6,000-record schedule, keep paired records under one scenario ID with meaningful step indices, and split by scenario ID.
- [x] Re-run focused tests and verify success.

### Task 6: Leakage-free realization and validation

**Files:**
- Modify: `src/sieve/data_factory/glm.py`
- Modify: `src/sieve/data_factory/quality.py`
- Test: `tests/test_data_factory_glm.py`
- Test: `tests/test_data_factory_quality.py`

- [x] Add failing tests rejecting `relevant`, `condition`, `perturbation`, decision names, and reason codes in model-visible observations.
- [x] Run focused tests and verify failures.
- [x] Update deterministic and live realization prompts so models verbalize only observable facts; add leakage checks to validation.
- [x] Re-run focused tests and verify success.

### Task 7: Regeneration command and canonical outputs

**Files:**
- Modify: `src/sieve/cli/generate_grounded.py`
- Modify: `scripts/run_grounded_data.sh`
- Modify: `configs/grounded_full_dry_run.json`
- Modify: `README.md`
- Test: `tests/test_data_factory_cli.py`

- [x] Add failing CLI tests for trajectory mode, deterministic local generation, canonical output paths, and the fixed 6,000 confirmation guard.
- [x] Run CLI tests and verify failures.
- [x] Wire the new pipeline into the existing relative-path CLI and shell entry point.
- [x] Re-run CLI tests and verify success.
- [x] Run the full local generation command without GPU or online model calls.

### Task 8: Corpus audit and regression verification

**Files:**
- Regenerate: `data/source_cache/normalized-full/*.jsonl`
- Regenerate: `data/sft/source/records.jsonl`
- Regenerate: `data/sft/clean/train.jsonl`
- Regenerate: `data/sft/clean/dev.jsonl`
- Regenerate: `data/sft/manifest.json`
- Create: `data/sft/quality_report.json`

- [x] Validate JSONL parseability, exact class ratios, source/domain counts, trajectory adjacency, executor validity, no split overlap, and absence of leakage keys/text.
- [x] Run the complete CPU test suite.
- [x] Inspect representative UPDATE, HOLD, successor UPDATE, and IGNORE records.
- [x] Record hashes, counts, prompt version, rule version, and generation method in the manifest and quality report.
