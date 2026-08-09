# SIEVE Data Layout Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep only auditable raw source files and directly trainable Stage-1 SFT data under `data/`.

**Architecture:** Copy the exact upstream files consumed by normalization into `data/raw/`, describe them with a hash manifest, and make production intermediates disposable under `tmp/data_factory/`. Preserve `data/sft/` byte-for-byte and remove duplicate or reproducible data only after hash validation.

**Tech Stack:** Python 3, PowerShell, Bash, JSON, unittest

---

### Task 1: Lock the clean path contract

**Files:**
- Modify: `tests/test_portability.py`
- Modify: `src/sieve/cli/prepare_grounded_sources.py`
- Modify: `scripts/run_grounded_data.sh`
- Modify: `configs/grounded_full_dry_run.json`
- Modify: `configs/grounded_preview.json`

- [ ] Add a failing test asserting that source preparation reads `data/raw`, persistent outputs use `data/sft`, and intermediates default to `tmp/data_factory`.
- [ ] Run `python -m unittest tests.test_portability -v` with `PYTHONPATH=src` and confirm failure on the old paths.
- [ ] Change the source root and intermediate defaults to the approved paths.
- [ ] Re-run the portability test and confirm it passes.

### Task 2: Preserve the minimum raw source set

**Files:**
- Create: `data/raw/manifest.json`
- Copy: the three tau2 `tasks.json` files, 15 ToolBench answer JSON files, AgentBench `standard.jsonl`, WebArena `test.raw.json`, and four LICENSE files

- [ ] Copy each exact source file while preserving the relative path under its dataset directory.
- [ ] Generate `data/raw/manifest.json` with pinned versions, licenses, relative paths, file sizes and SHA-256 values.
- [ ] Verify that every manifest entry exists and matches its recorded hash.
- [ ] Run source normalization into `tmp/data_factory/normalized-smoke` with a small per-source limit and confirm all four sources produce records.

### Task 3: Delete duplicated and reproducible data

**Files:**
- Delete: `data/generated/`
- Delete: `data/source_cache/`
- Delete: `data/source_manifests/`
- Preserve unchanged: `data/sft/`

- [ ] Verify the three `data/sft/manifest.json` hashes immediately before deletion.
- [ ] Resolve and validate every deletion target remains below the workspace `data/` directory.
- [ ] Delete only the three approved targets.
- [ ] Verify `data/` contains only `raw/` and `sft/`.
- [ ] Re-check the SFT hashes and record that they are unchanged.

### Task 4: Update documentation and verify the repository

**Files:**
- Modify: `README.md`

- [ ] Replace obsolete persistent generated/source-cache paths with `data/raw`, `data/sft`, and disposable `tmp/data_factory` paths.
- [ ] Run `python -m compileall -q src`.
- [ ] Run `python -m unittest discover -s tests` with `PYTHONPATH=src`.
- [ ] Run `python -m sieve.cli.train_sft --root . --config configs/sft_qwen3_4b.yaml --validate-only` and confirm the canonical record counts and hashes remain valid.
