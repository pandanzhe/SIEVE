# Stage-1 SFT v3 Data Rebuild Design (v2)

> 基于 2026-08-18 原始 design v1，经代码库全面验证后更新。主要变化：标注 6 处代码缺口、明确 Approach A 实现策略、补充 token-span loss 细节、增加 12 项审计清单。

## Goal

Replace the current template-generated Stage-1 corpus with 6,000 source-grounded records derived only from the training partitions of τ³-bench and CAR-bench. Natural-language observations must be produced by a live LLM (glm-4.7), while gold actions remain deterministic and executable.

## Non-goals

- Do not train Qwen3-4B on the local machine.
- Do not use official benchmark test tasks for generation, tuning, or internal validation.
- Do not let the teacher LLM choose UPDATE, HOLD, IGNORE, patches, or verification tools.
- Do not overwrite `data/sft` until a complete staged corpus passes every audit.
- Do not modify `data/rl`, `model`, or `outputs`.
- Do not rebuild RL data in this cycle (separate spec later).

## Implementation approach: Approach A — New v3 modules alongside existing code

Create `v3_sources.py`, `v3_candidates.py`, `v3_generation.py`, `v3_audit.py` as new files in `data_factory/`. Existing v2 code stays untouched until the v3 corpus passes audit, then we swap training pipeline pointers.

**Rationale:**
- Zero risk to existing v2 pipeline during development.
- Can run v2 and v3 in parallel for comparison.
- Natural rollback: just don't swap if v3 fails audit.
- Shared logic (Executor, GLM client, core types, `build_candidate`, `build_verification_successor`) imported from existing modules — no duplication of 90%+ reusable code.

## Validated code gaps

| # | Gap | Current Code | V3 Requirement |
|---|-----|-------------|----------------|
| 1 | Source adapters | tau2-bench, ToolBench, AgentBench, WebArena | τ³-bench (airline/retail/telecom train), CAR-bench (Base/Disambiguation train) |
| 2 | Quota system | Observation-type × decision matrix, 7 types | Exact decision counts: 2700 UPDATE / 1800 HOLD+VERIFY / 1500 IGNORE |
| 3 | Source composition | Macro-domain 50/30/20 (commerce/service/workflow) | 3900 τ³ / 1500 CAR / 600 counterfactual |
| 4 | Split assignment | Stochastic sha256 80/10/10 | Exact 5400/600 by parent_task_id, zero overlap |
| 5 | Loss weights | 5 auxiliary heads (decision/affected/verification/patch_operation/patch_value) | 3 objectives (decision/structure/patch_value) via token-span masks |
| 6 | Corpus audit | Per-candidate validation only | Full 12-point audit: test leakage, provenance, manifest hash, LLM usage, duplicates |

## Source boundary

The source snapshot contains:

- τ³-bench text domains: airline, retail, and telecom — **train split only**;
- CAR-bench: Base Train and Disambiguation Train — **train split only**;
- source licenses, versions, task split definitions, file sizes, and SHA-256 hashes.

All source artifacts live under `data/raw`. An immutable source manifest (`configs/stage1_sources_v3.json`) identifies each selected file. Test task IDs may be downloaded for later official evaluation, but the candidate builder and generator reject them.

### New file: `src/sieve/data_factory/v3_sources.py`

Provides:
- `SourceTask` dataclass: `task_id, split, domain, goal, initial_state, tool_contracts, reference_actions, is_test`. This is the v3 adapter's output. Each `SourceTask` is then converted to a `GroundedSourceRecord` (from `models.py`) for compatibility with the existing `rules.build_candidate()` and `trajectories.build_verification_successor()` pipeline. The conversion preserves task ID, domain, goal, entity, field, value, and tool semantics.
- `load_tau3_tasks(raw_dir) -> list[SourceTask]`: reads airline/retail/telecom train only, rejects test task IDs
- `load_car_tasks(raw_dir) -> list[SourceTask]`: reads Base Train + Disambiguation Train only
- `audit_source_snapshot(tasks, manifest)`: verifies commit, license, file hashes

### New file: `src/sieve/cli/download_stage1_sources.py`

Downloads pinned sources to `data/raw/tau3-bench/` and `data/raw/car-bench/`. Reuses `acquire.py` `fetch_artifact()` mechanism.

### New file: `configs/stage1_sources_v3.json`

Records both datasets' URLs, versions, SHA-256 hashes.

## Conversion boundary

The conversion has four distinct stages:

1. A source adapter (`v3_sources.py`) converts an upstream task into an auditable `SourceTask`.
2. A deterministic candidate builder (`v3_candidates.py`) creates atomic belief transitions and counterfactual observations. It fixes entity, field, value, authority, tool, timestamp, and target action before the teacher call.
3. A live teacher LLM (`v3_generation.py` wrapping `glm.py`) realizes only the observation text. Every required entity and value term must survive verbatim; private labels and oracle fields are excluded from its prompt.
4. The SIEVE Executor (`executor.py`, reused unchanged) replays gold actions, produces successor states, and rejects targets that cannot execute.

The LLM may improve language diversity but cannot alter grounded fields or targets. Fake clients are permitted only in unit tests and preview runs (total ≤ 100); canonical publication requires live generator metadata and positive token usage.

### New file: `src/sieve/data_factory/v3_candidates.py`

Provides:
- `build_v3_candidates(tau3_tasks, car_tasks, total=6000, seed=42) -> list[GenerationCandidate]`

**Exact action allocation (45/30/25 per source block):**

| Source | Total | UPDATE | HOLD+VERIFY | IGNORE |
|--------|-------|--------|-------------|--------|
| τ³-derived | 3900 | 1755 | 1170 | 975 |
| CAR-derived | 1500 | 675 | 450 | 375 |
| Safety counterfactual | 600 | 270 | 180 | 150 |
| **Total** | **6000** | **2700** | **1800** | **1500** |

**Split assignment by parent_task_id:**
- Group all candidates by `parent_task_id` before generating variants.
- Every step and counterfactual from one parent remains in one split.
- Dev quotas are exact 10% stratified slices of each (source, decision) cell.
- No test file is produced.

**HOLD→UPDATE pairs:** Reuses `trajectories.build_verification_successor()` — no duplication.

**Counterfactuals:** Reuses `rules.build_candidate()` with distractor for wrong-entity IGNORE and stale-conflict IGNORE — no duplication.

## Corpus contract

The canonical corpus contains exactly 6,000 records:

- 5,400 train;
- 600 dev;
- no internally generated test file.

The exact action distribution is:

- 2,700 UPDATE;
- 1,800 HOLD+VERIFY;
- 1,500 IGNORE.

The source contribution is:

- 3,900 τ³-derived records;
- 1,500 CAR-derived records;
- 600 safety counterfactual records inheriting the parent source provenance.

## LLM generation boundary

### New file: `src/sieve/data_factory/v3_generation.py`

Provides:
- `CanonicalGlmClient`: wraps `CachedGlmClient`, enforces:
  - Inner client must be `ZhipuGlmClient` (not `FakeGlmClient`)
  - `ZAI_API_KEY` must exist in environment
  - After each `realize()`, checks `total_tokens > 0` — rejects otherwise
- `capture_generation_metadata(client, candidates, realizations) -> dict`:
  - `generator`: client.model (`glm-4.7`)
  - `prompt_version`: from `glm.PROMPT_VERSION`
  - `source_task_id`: from candidate.provenance
  - `source_file_sha256`: candidate.provenance.source_sha256
  - `request_hash`: realization.request_hash
  - `token_usage`: realization.usage
  - `result_hash`: SHA-256 of observation_text

### Modified: `src/sieve/cli/generate_grounded.py`

- V3 mode: `--dry-run` only allowed for preview (total ≤ 100)
- Full generation (total = 6000) automatically uses `CanonicalGlmClient`

## Canonical sample

Each training row stores `scenario_id`, `step_index`, `context`, and `target`. The fixed system prompt is injected by the collator. Provenance is stored separately and contains source dataset, source version, source task ID, source hash, transition type, transformation, generator, prompt version, request hash, and token usage.

The policy-visible context contains only executable belief, current observation, goal, risk, ledger, budget, and visible verification contract. It excludes oracle values, gold decision, perturbation labels, rule reason codes, and held-out split metadata.

## Exact grouped split and canonical audit

### New file: `src/sieve/data_factory/v3_audit.py`

Provides:
- `exact_grouped_split(candidates, train=5400, dev=600, seed=42)`: Stratified exact split by parent_task_id:
  - Group by (source_dataset, decision) for stratification
  - Within each stratum, sort by parent_task_id — all steps/counterfactuals from one parent stay together
  - Dev takes exactly 10% of each stratum
  - Train takes the remainder
  - No test file produced

- `audit_corpus(accepted_records, candidates, source_manifest, test_task_ids) -> AuditReport`: Full 12-point audit:

| # | Check | Required |
|---|-------|----------|
| 1 | Exact total, split, source, and decision counts | 6000/5400/600, 2700/1800/1500 |
| 2 | JSON Schema pass rate | 100% |
| 3 | Executor gold-action pass rate | 100% |
| 4 | Parent-task overlap between train and dev | 0 |
| 5 | Official-test task IDs in provenance | 0 |
| 6 | Private-label leakage in rendered prompts | 0 |
| 7 | Provenance completeness for every row | 100% |
| 8 | Unique record IDs, no duplicate samples | 0 duplicates |
| 9 | Live generator identity and positive aggregate token usage | > 0 |
| 10 | All file hashes matching manifest | exact match |
| 11 | Near-duplicate sample rate (Jaccard > 0.9) | below 1% of corpus |
| 12 | Source composition (τ³/CAR/counterfactual) | 3900/1500/600 |

### Modified: `src/sieve/data_factory/export.py`

- V3 mode uses `exact_grouped_split()` instead of `assign_split()`

### Modified: `src/sieve/data_factory/publish.py`

- Publication requires `audit_corpus()` passing all 12 checks before any file copy

## Publication layout

The final directory is:

```text
data/sft/
├── clean/train.jsonl
├── clean/dev.jsonl
├── provenance.jsonl
├── manifest.json
└── quality_report.json
```

Intermediate candidates, raw LLM responses, retries, and rejected rows remain under `tmp/sft-v3-build` and are not published.

### Atomic replacement procedure

```
Phase 1: Build → tmp/sft-v3-build/
  ├── accepted.jsonl, rejected.jsonl, response_cache.jsonl
  ├── clean/{train.jsonl, dev.jsonl}
  ├── provenance.jsonl, manifest.json, quality_report.json

Phase 2: Audit → all 12 checks pass

Phase 3: Atomic replace
  data/sft → data/sft.old.tmp          (rename)
  tmp/sft-v3-build/published → data/sft (rename)
  Re-run formal path validation
  Success → delete data/sft.old.tmp
  Failure → rollback: data/sft.old.tmp → data/sft
```

Old data is never deleted before replacement succeeds. The build script must **never** use `--dry-run` or `FakeGlmClient` for canonical publication.

## Training alignment

Stage-1 uses local `model/` assets for Qwen3-4B-Instruct-2507. `scripts/run_sft.sh` and `scripts/validate_sft.sh` default to `configs/sft_qwen3_4b.yaml`.

### Three-objective loss

The training objective has three logical components:

$$\mathcal{L} = \lambda_{dec}\mathcal{L}_{decision} + \lambda_{struct}\mathcal{L}_{structure} + \lambda_{value}\mathcal{L}_{patch\ value}$$

- **decision**: three-way pooled classification loss (unchanged).
- **structure**: causal-LM loss on JSON structure tokens — affected_fields, patch operations, and verification request tokens. Implemented as a **token-span mask** over the completion. Specifically, the completion JSON is tokenized; structure tokens are those outside quoted patch-value strings (i.e., `"decision":`, `"affected_fields":`, `"op":`, `"field_id":`, `"verification":` keys and their non-value tokens); value tokens are the quoted string/number tokens that appear as patch `"value"` fields.
- **patch_value**: causal-LM loss on writable patch-value tokens. Implemented as a **token-span mask** over the completion. These are exactly the tokens that appear as the `"value"` field in each patch object within the `"patches"` array.

The structure and value objectives share the frozen base LM head and train LoRA parameters; they are token masks, not separate vocabulary heads. The old `affected_head`, `verification_head`, and `patch_operation_head` MLP layers are removed.

**Token-span mask construction algorithm:**
1. Serialize the `RevisionOutput` to JSON (same as current `_target_json()`).
2. Tokenize the full completion string without special tokens.
3. Use a JSON path walker to identify the byte offsets of each patch `[i].value` field's content.
4. Map byte offsets to token indices via the tokenizer's byte-token alignment.
5. Structure mask = 1 for all completion tokens whose byte offset is NOT inside a patch value; value mask = 1 for tokens inside a patch value. Both masks are 0 for prompt tokens.

### Modified files for training alignment

- **`hf_data.py`**: New `token_span_labels()` function identifying structure vs. value token positions in completion. `HFRevisionCollator` outputs `structure_labels`, `structure_mask`, `value_labels`, `value_mask`.
- **`hf_lora_policy.py`**: Remove `affected_head`, `verification_head`, `patch_operation_head`. Keep `decision_head` + shared LM head. `loss_weights` becomes 3 entries.
- **`hf_sft.py`**: Forward computes three-objective loss. Checkpoint metrics: `decision_macro_f1`, `structure_accuracy`, `value_exact_match`.
- **`hf_sft_config.py`**: Parse `loss_weights` with 3 keys: `decision`, `structure`, `patch_value`.
- **`sft_qwen3_4b.yaml`**: `loss_weights: {decision: 1.0, structure: 1.0, patch_value: 1.0}`.
- **`run_sft.sh` / `validate_sft.sh`**: Default config → `configs/sft_qwen3_4b.yaml`.
- **`validate_sft.sh`**: Add `--check-manifest`, `--check-llm-usage`, `--check-test-leakage`.
- **`hf_sft.py` `audit_sft_inputs()`**: Pre-training manifest + file hash check.

## New scripts and configs

- **`scripts/build_stage1_v3.sh`**: Full build pipeline (download → adapt → preview 50–100 → inspect → generate 6000 → audit → replace)
- **`configs/stage1_sft_v3_generation.json`**: V3 generation configuration
- **`src/sieve/cli/validate_stage1_data.py`**: Standalone v3 data validator

## Failure behavior

- Missing source file, checksum mismatch, or split ambiguity stops before generation.
- Missing live API credentials stops before any canonical replacement.
- Malformed or ungrounded LLM output is retried up to the configured limit and then rejected.
- Insufficient accepted rows, quota mismatch, leakage, or non-executable targets blocks publication.
- Failed post-replacement validation triggers rollback to the renamed old directory.

## Test coverage

- **`tests/test_data_factory_v3_sources.py`**: τ³ and CAR fixtures, test ID rejection, provenance
- **`tests/test_data_factory_v3_candidates.py`**: Exact 45/30/25 counts, HOLD→UPDATE replay, wrong-entity/stale IGNORE, parent-task provenance
- **`tests/test_data_factory_v3_generation.py`**: Leakage rejection, grounding checks, fake generator rejection, duplicate response IDs, zero token rejection
- **`tests/test_data_factory_v3_audit.py`**: Exact split cells, task overlap, test leakage, private prompt leakage, non-executable targets, duplicate records, provenance completeness, manifest hash mismatch
- Update `tests/test_hf_lora_policy.py`, `tests/test_hf_data_contract.py`, `tests/test_stage1_cli_contract.py` for three-objective loss
