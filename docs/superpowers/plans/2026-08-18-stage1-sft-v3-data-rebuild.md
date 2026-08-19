# Stage-1 SFT v3 Data Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, publish, and synchronize a 6,000-record live-LLM Stage-1 corpus grounded in τ³-bench and CAR-bench training tasks.

**Architecture:** Pin source snapshots locally, adapt upstream tasks into deterministic grounded anchors, construct gold transitions with SIEVE rules and Executor replay, and use a live teacher only for observation realization. Publish only after exact quota, leakage, provenance, execution, and hash audits pass.

**Tech Stack:** Python 3.10+, stdlib JSON/HTTP, zai-sdk, PyYAML, unittest/pytest, Hugging Face/Qwen3 training stack, Bash entry scripts.

**Spec:** `docs/superpowers/specs/2026-08-18-stage1-sft-v3-data-rebuild-design.md`

## Global Constraints

- Canonical total is exactly 6,000: 5,400 train and 600 dev.
- Decision counts are exactly UPDATE 2,700, HOLD+VERIFY 1,800, IGNORE 1,500.
- Training sources are τ³ airline/retail/telecom train and CAR Base/Disambiguation train only.
- Gold actions are deterministic; the live LLM realizes observation text only.
- Canonical publication rejects fake generators and zero token usage.
- `data/sft` is not replaced until the staged corpus passes every audit.
- `data/rl`, `model`, and `outputs` are out of scope.

---

### Task 1: Source acquisition and manifests

**Files:**
- Create: `src/sieve/data_factory/v3_sources.py`
- Create: `src/sieve/cli/download_stage1_sources.py`
- Create: `configs/stage1_sources_v3.json`
- Create: `tests/test_data_factory_v3_sources.py`

**Interfaces:**
- Produces: `SourceTask`, `load_tau3_tasks(...)`, `load_car_tasks(...)`, `audit_source_snapshot(...)`.

- [ ] Write tests with literal τ³ and CAR fixtures that reject held-out IDs and preserve task/tool/state provenance.
- [ ] Run the tests and verify they fail because the v3 adapter does not exist.
- [ ] Implement source adapters and checksum-aware acquisition with repository-relative paths.
- [ ] Run focused tests, then download pinned sources to `data/raw` and audit their hashes.

### Task 2: Deterministic transition construction

**Files:**
- Create: `src/sieve/data_factory/v3_candidates.py`
- Modify: `src/sieve/data_factory/trajectories.py`
- Create: `tests/test_data_factory_v3_candidates.py`

**Interfaces:**
- Consumes: `SourceTask`.
- Produces: `build_v3_candidates(tasks, total, seed)` returning ordered `GenerationCandidate` records with parent-task provenance.

- [ ] Write failing tests for exact 45/30/25 decision counts, HOLD→UPDATE replay, wrong-entity/stale IGNORE, visible tool contracts, and parent-task provenance.
- [ ] Implement the minimal candidate schedule and Executor-backed successor construction.
- [ ] Run focused tests and refactor shared transition helpers without changing legacy behavior.

### Task 3: Live teacher generation boundary

**Files:**
- Modify: `src/sieve/data_factory/glm.py`
- Create: `src/sieve/data_factory/v3_generation.py`
- Modify: `src/sieve/cli/generate_grounded.py`
- Create: `tests/test_data_factory_v3_generation.py`

**Interfaces:**
- Produces: grounded observation realizations plus generator/prompt/request/token metadata.

- [ ] Write failing tests that reject target leakage, missing grounding terms, fake generator publication, duplicate response IDs, and zero live token usage.
- [ ] Implement a live-only canonical mode, resumable response cache, retry policy, and provenance capture.
- [ ] Run focused tests and a small fake preview; never publish the preview.

### Task 4: Exact grouped split and canonical audit

**Files:**
- Create: `src/sieve/data_factory/v3_audit.py`
- Modify: `src/sieve/data_factory/export.py`
- Modify: `src/sieve/data_factory/publish.py`
- Create: `src/sieve/cli/validate_stage1_data.py`
- Create: `tests/test_data_factory_v3_audit.py`
- Modify: `tests/test_data_factory_publish.py`

**Interfaces:**
- Produces: exact 5,400/600 group-safe split, provenance JSONL, quality report, and schema-v3 manifest.

- [ ] Write failing tests for exact split cells, task overlap, held-out leakage, private prompt leakage, non-executable targets, duplicate records, incomplete provenance, and manifest hash mismatch.
- [ ] Implement exact stratified group assignment and full audit reporting.
- [ ] Implement staged publication that rejects incomplete or fake corpora.
- [ ] Run focused tests and validate a small controlled corpus end to end.

### Task 5: Qwen3 training alignment

**Files:**
- Modify: `src/sieve/policies/hf_data.py`
- Modify: `src/sieve/policies/hf_lora_policy.py`
- Modify: `src/sieve/training/hf_sft.py`
- Modify: `src/sieve/training/hf_sft_config.py`
- Modify: `configs/sft_qwen3_4b.yaml`
- Modify: `scripts/run_sft.sh`
- Modify: `scripts/validate_sft.sh`
- Modify: relevant `tests/test_hf_*.py` and `tests/test_stage1_cli_contract.py`

**Interfaces:**
- Produces: decision, structure-token, and patch-value-token loss components.

- [ ] Write failing tests for structure/value token masks, three loss weights, Qwen3 default scripts, and model-independent data preflight.
- [ ] Implement token-span labeling and remove obsolete affected/verification/patch-operation MLP supervision.
- [ ] Update checkpoint metrics and configuration parsing to the three-objective contract.
- [ ] Run focused HF tests without loading model weights.

### Task 6: Live corpus production and replacement

**Files:**
- Create: `configs/stage1_sft_v3_generation.json`
- Create: `scripts/build_stage1_v3.sh`
- Replace after audit: `data/sft/**`

**Interfaces:**
- Consumes: local pinned sources and `ZAI_API_KEY`.
- Produces: canonical schema-v3 corpus.

- [ ] Generate 50 live preview records and inspect grounding, action balance, and token metadata.
- [ ] Generate/resume all 6,000 accepted records in `tmp/sft-v3-build`.
- [ ] Run the full canonical audit and compare exact hashes/counts.
- [ ] Transactionally replace `data/sft`, rerun the audit through canonical paths, then remove the temporary old copy.

### Task 7: Repository and server synchronization

**Files:**
- Commit intended code, config, test, documentation, and canonical data files only.

**Interfaces:**
- Produces: updated GitHub `main` and `/root/SIEVE` checkout with local server assets preserved.

- [ ] Run the complete local test suite and validate-only data audit.
- [ ] Commit and push the verified revision to GitHub.
- [ ] SSH to the server, inspect dirty/untracked files, and preserve `model`, `outputs`, and local credentials.
- [ ] Pull with a non-destructive strategy, validate server hashes and imports, and report that training remains blocked only by missing model assets if applicable.
