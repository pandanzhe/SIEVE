# Grounded Data Production Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Finish the local source-grounded SFT data production chain, export leakage-free train/dev/test JSONL, and produce a verified 6,000-record deterministic full dataset without starting model training.

**Architecture:** Keep rule labels and oracle transitions deterministic. Permit one grounded source group to yield several distinct transformation cells while preventing duplicate `(group_id, transformation)` derivations. Export accepted audit records through a separate adapter that replaces structured observation values with the realized text, removes production-only fields, and assigns splits by stable group hash.

**Tech Stack:** Python 3.10+, standard library, existing SIEVE dataclasses and executor, unittest, JSONL.

---

### Task 1: Natural-language value validation

**Files:**
- Modify: `src/sieve/data_factory/quality.py`
- Test: `tests/test_data_factory_quality.py`

- [x] **Step 1: Run the existing list-value test and observe `missing_grounded_value` failure.**
- [x] **Step 2: Parse JSON/Python container strings and validate their scalar atoms.**
- [x] **Step 3: Run `python -m unittest tests.test_data_factory_quality -v`; expect 5 tests PASS.**

### Task 2: Full-run grounded derivation scheduling

**Files:**
- Modify: `src/sieve/data_factory/pipeline.py`
- Test: `tests/test_data_factory_pipeline.py`

- [x] **Step 1: Add a failing test with a small source pool showing that different quota transformations may reuse one group while duplicate derivations are forbidden.**
- [x] **Step 2: Run the focused test and confirm the current `not enough distinct normalized records` failure.**
- [x] **Step 3: Replace group-only exclusion with `(group_id, observation_type, decision, verification_action)` exclusion and deterministic source rotation.**
- [x] **Step 4: Run pipeline tests and expect PASS.**

### Task 3: Training-view export and group-isolated splits

**Files:**
- Create: `src/sieve/data_factory/export.py`
- Create: `src/sieve/cli/export_grounded.py`
- Create: `tests/test_data_factory_export.py`

- [x] **Step 1: Write failing tests asserting oracle/provenance/generation/validation fields are absent, realized observation text is used, exported JSONL loads through `record_from_dict`, and a group occurs in only one split.**
- [x] **Step 2: Run `python -m unittest tests.test_data_factory_export -v`; expect import failure.**
- [x] **Step 3: Implement `audit_to_sft_dict`, stable SHA-256 group split assignment, and atomic JSONL writers for train/dev/test.**
- [x] **Step 4: Implement `python -m sieve.cli.export_grounded --input ... --output-dir ... --seed 42`.**
- [x] **Step 5: Run export tests and expect PASS.**

### Task 4: Portable full-production entry points

**Files:**
- Create: `configs/grounded_full_dry_run.json`
- Create: `scripts/run_grounded_data.sh`
- Modify: `README.md`
- Test: `tests/test_portability.py`

- [x] **Step 1: Add a failing portability test for the new script and root-relative config.**
- [x] **Step 2: Add a Bash wrapper whose only absolute root is derived in `scripts/common.sh`; all Python paths remain root-relative.**
- [x] **Step 3: Document normalize, preview, guarded full generation, export, and live resume commands.**
- [x] **Step 4: Run portability tests and expect PASS.**

### Task 5: Produce and verify local data

**Files produced:**
- `data/source_cache/normalized-full/`
- `data/generated/full-dry-run/`
- `data/generated/full-dry-run/training/`

- [x] **Step 1: Normalize all locally cached sources with `--per-source-limit 500`; observed counts must be recorded in the manifest.**
- [x] **Step 2: Run guarded deterministic production with `--total 6000 --confirm-full 6000 --dry-run`.**
- [x] **Step 3: Export group-isolated train/dev/test JSONL.**
- [x] **Step 4: Verify exact total/quota/source counts, patch execution, split isolation, secret absence, JSONL loadability, and resume idempotence.**
- [x] **Step 5: Run the full unittest suite and `python -m compileall -q src tests`.**

The real GLM preview may resume from cache only when `ZAI_API_KEY` is present. Absence of that environment variable must never block deterministic production or cause the key to be logged.

