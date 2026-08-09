# Source-Grounded SFT Data Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a resumable local pipeline that converts traceable open-source records into rule-labelled SIEVE JSONL data, uses GLM-4.7 only for grounded observation realization, and stops after a 140-record preview.

**Architecture:** A source-neutral `GroundedSourceRecord` enters a deterministic quota and rule layer, producing a fully labelled `GenerationCandidate`. A pluggable real or fake GLM client realizes only the observation text. Validation executes patches against the oracle target, checks provenance and leakage, then appends accepted and rejected audit records plus run reports.

**Tech Stack:** Python 3.10+, standard library dataclasses/JSON/hashlib/argparse/unittest, existing SIEVE core executor, optional `zai-sdk` for live GLM-4.7 calls.

---

## File structure

- Create `src/sieve/data_factory/models.py`: audit schema, source record, quota cell, and serialization types.
- Create `src/sieve/data_factory/quotas.py`: preview/full matrices and deterministic quota expansion.
- Create `src/sieve/data_factory/sources.py`: local JSON/JSONL adapters and source manifests.
- Create `src/sieve/data_factory/rules.py`: deterministic candidate construction and counterfactual rules.
- Create `src/sieve/data_factory/glm.py`: fake/real client protocol, prompt building, JSON parsing, request hashes.
- Create `src/sieve/data_factory/quality.py`: schema, executor, provenance, leakage, and duplicate validation.
- Create `src/sieve/data_factory/pipeline.py`: checkpointed generation, JSONL output, reports, and full-run guard.
- Create `src/sieve/data_factory/__init__.py`: public imports.
- Create `src/sieve/cli/generate_grounded.py`: CLI entry point.
- Create `configs/grounded_preview.json`: preview runtime defaults and source-manifest path.
- Create `data/source_manifests/example.json`: documented local-source manifest shape.
- Modify `pyproject.toml`: optional `generation` dependency containing `zai-sdk`.
- Modify `README.md`: local dry-run/live-preview commands and secret handling.
- Create focused tests under `tests/test_data_factory_*.py`.

### Task 1: Quota contracts and audit models

**Files:**
- Create: `tests/test_data_factory_quotas.py`
- Create: `src/sieve/data_factory/models.py`
- Create: `src/sieve/data_factory/quotas.py`
- Create: `src/sieve/data_factory/__init__.py`

- [ ] **Step 1: Write failing quota tests**

```python
class QuotaTests(unittest.TestCase):
    def test_preview_matrix_matches_approved_margins(self):
        cells = preview_quota_cells()
        self.assertEqual(sum(cell.count for cell in cells), 140)
        self.assertEqual(count_by(cells, "decision"), {
            "UPDATE": 56, "IGNORE": 49, "HOLD": 35,
        })
        self.assertEqual(count_by(cells, "observation_type"), {
            "new_consistent": 21,
            "explicit_conflict": 21,
            "implicit_conflict": 21,
            "stale": 21,
            "irrelevant": 21,
            "tool_error_or_low_trust": 21,
            "insufficient_or_ambiguous": 14,
        })

    def test_full_quota_has_six_thousand_records(self):
        self.assertEqual(sum(c.count for c in full_quota_cells()), 6000)
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `python -m unittest tests.test_data_factory_quotas -v`

Expected: FAIL because `sieve.data_factory` does not exist.

- [ ] **Step 3: Implement immutable model types and approved matrices**

Define enums for observation type, scenario type, verification action, difficulty, and source name. Define `QuotaCell`, `Provenance`, `GroundedSourceRecord`, `GenerationCandidate`, `AuditRecord`, and `ValidationResult`, each with explicit `to_dict()` serialization. Encode the approved 140 cross-matrix and the 6,000 marginal matrix.

- [ ] **Step 4: Run quota tests**

Run: `python -m unittest tests.test_data_factory_quotas -v`

Expected: PASS.

### Task 2: Local open-source adapters and provenance

**Files:**
- Create: `tests/test_data_factory_sources.py`
- Create: `src/sieve/data_factory/sources.py`
- Create: `data/source_manifests/example.json`

- [ ] **Step 1: Write failing adapter tests**

```python
class SourceAdapterTests(unittest.TestCase):
    def test_jsonl_adapter_preserves_traceable_identity(self):
        records = load_source_manifest(self.manifest_path)
        record = records[0]
        self.assertEqual(record.provenance.source_dataset, "tau2-bench")
        self.assertEqual(record.provenance.source_record_id, "retail-001")
        self.assertEqual(len(record.provenance.source_sha256), 64)

    def test_missing_license_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "license"):
            load_source_manifest(self.missing_license_manifest)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_data_factory_sources -v`

Expected: FAIL because the adapter is absent.

- [ ] **Step 3: Implement manifest and JSON/JSONL adapters**

The manifest explicitly maps local files and JSON selectors to normalized fields. Hash canonicalized source content with SHA-256. Reject missing source version, record ID, locator, hashable content, or license note. Do not download data implicitly in this task.

- [ ] **Step 4: Run adapter tests**

Run: `python -m unittest tests.test_data_factory_sources -v`

Expected: PASS.

### Task 3: Deterministic rules and oracle state

**Files:**
- Create: `tests/test_data_factory_rules.py`
- Create: `src/sieve/data_factory/rules.py`

- [ ] **Step 1: Write failing rule tests**

```python
class RuleTests(unittest.TestCase):
    def test_authoritative_newer_value_updates(self):
        candidate = build_candidate(self.source, quota("new_consistent", "UPDATE"))
        self.assertEqual(candidate.target.decision, Decision.UPDATE)
        self.assertEqual(candidate.target.affected_fields, ("status",))

    def test_wrong_entity_conflict_ignores_without_patch(self):
        candidate = build_candidate(self.source, quota("explicit_conflict", "IGNORE"))
        self.assertEqual(candidate.target, RevisionOutput(Decision.IGNORE))

    def test_low_trust_high_risk_holds_for_verification(self):
        candidate = build_candidate(
            self.source, quota("tool_error_or_low_trust", "HOLD", "VERIFY")
        )
        self.assertEqual(candidate.target.decision, Decision.HOLD)
        self.assertIsNotNone(candidate.target.verification)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_data_factory_rules -v`

Expected: FAIL because `build_candidate` is absent.

- [ ] **Step 3: Implement minimal deterministic rule construction**

Map each quota cell to an explicit transformation over a grounded source record. Construct `RevisionContext`, `RevisionOutput`, oracle target state, reason code, conflict type, and structured event. All UPDATE patches use existing `PatchOp`; IGNORE has no fields or patches; HOLD sets pending status and optionally creates a verification request.

- [ ] **Step 4: Add counterintuitive-cell tests**

Test stale-but-newer-than-belief UPDATE, authoritative explicit-conflict UPDATE, duplicate new-consistent IGNORE, and active-subgoal-irrelevant but maintained-long-term-field UPDATE.

- [ ] **Step 5: Run rule tests**

Run: `python -m unittest tests.test_data_factory_rules -v`

Expected: PASS.

### Task 4: GLM protocol, prompt, parsing, and secret safety

**Files:**
- Create: `tests/test_data_factory_glm.py`
- Create: `src/sieve/data_factory/glm.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing GLM tests**

```python
class GlmTests(unittest.TestCase):
    def test_fake_client_realizes_without_network(self):
        result = FakeGlmClient().realize([self.candidate])
        self.assertEqual(result[0].record_id, self.candidate.record_id)
        self.assertTrue(result[0].observation_text)

    def test_real_client_requires_environment_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ZAI_API_KEY"):
                ZhipuGlmClient()

    def test_request_hash_is_stable_and_excludes_key(self):
        digest = request_hash(self.candidate, prompt_version="v1")
        self.assertEqual(digest, request_hash(self.candidate, prompt_version="v1"))
        self.assertNotIn("api", digest.lower())
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_data_factory_glm -v`

Expected: FAIL because the GLM module is absent.

- [ ] **Step 3: Implement fake client and bounded prompt**

The prompt receives provenance-safe source material and the rule-generated structured event. It instructs the model to return a JSON array containing only `record_id` and `observation_text`, preserving entity, value, source, time, polarity, and condition. It explicitly forbids decision labels and oracle fields in the text.

- [ ] **Step 4: Implement lazy real client**

Read `ZAI_API_KEY` only at construction, lazily import `ZhipuAiClient`, call `glm-4.7` with thinking enabled, `temperature=0.6`, and a bounded default `max_tokens=4096`. Extract `response.choices[0].message.content`; usage extraction is optional and defensive. Never serialize the key or exception request headers.

- [ ] **Step 5: Add optional dependency**

Add:

```toml
generation = ["zai-sdk>=0.2.3"]
```

- [ ] **Step 6: Run GLM tests**

Run: `python -m unittest tests.test_data_factory_glm -v`

Expected: PASS without installing `zai-sdk`.

### Task 5: Quality validation

**Files:**
- Create: `tests/test_data_factory_quality.py`
- Create: `src/sieve/data_factory/quality.py`

- [ ] **Step 1: Write failing quality tests**

```python
class QualityTests(unittest.TestCase):
    def test_valid_record_executes_to_oracle(self):
        result = validate_candidate(self.candidate, self.realization)
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason_codes, ())

    def test_label_leakage_is_rejected(self):
        realization = realization_with_text("decision=UPDATE, 地址已更新")
        result = validate_candidate(self.candidate, realization)
        self.assertIn("label_leakage", result.reason_codes)

    def test_missing_parent_for_counterfactual_is_rejected(self):
        result = validate_candidate(self.counterfactual_without_parent, self.realization)
        self.assertIn("missing_parent_record_id", result.reason_codes)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_data_factory_quality -v`

Expected: FAIL because quality validation is absent.

- [ ] **Step 3: Implement deterministic validators**

Validate branch contracts, apply the existing `StateExecutor`, canonicalize the resulting belief state, compare it with the oracle state, require provenance, scan for label/reason leakage, and reject exact normalized-text duplicates. Semantic checks require expected entity/value tokens or explicitly recorded aliases from the source record.

- [ ] **Step 4: Run quality tests**

Run: `python -m unittest tests.test_data_factory_quality -v`

Expected: PASS.

### Task 6: Resumable JSONL pipeline and reports

**Files:**
- Create: `tests/test_data_factory_pipeline.py`
- Create: `src/sieve/data_factory/pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

```python
class PipelineTests(unittest.TestCase):
    def test_dry_run_writes_exact_preview_files(self):
        result = run_pipeline(self.config, FakeGlmClient())
        self.assertEqual(result.accepted, 140)
        self.assertEqual(line_count(result.output_dir / "accepted.jsonl"), 140)
        self.assertTrue((result.output_dir / "manifest.json").exists())
        self.assertTrue((result.output_dir / "quality_report.json").exists())

    def test_resume_does_not_repeat_completed_request_hashes(self):
        first = run_pipeline(self.partial_config, self.counting_client)
        second = run_pipeline(self.resume_config, self.counting_client)
        self.assertEqual(second.duplicate_requests, 0)

    def test_full_run_requires_exact_confirmation(self):
        with self.assertRaisesRegex(ValueError, "confirm-full"):
            run_pipeline(self.full_config_without_confirmation, FakeGlmClient())
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_data_factory_pipeline -v`

Expected: FAIL because the pipeline is absent.

- [ ] **Step 3: Implement append-only run files**

Create run directories safely, append raw responses, accepted records, and rejected records as UTF-8 JSONL, flush after each batch, and use completed request hashes for resumption. Write manifest and reports atomically through a temporary file in the same run directory.

- [ ] **Step 4: Implement quota scheduling and replacement**

Expand quota cells deterministically from the seed. Select grounded source records using a stable rotation across sources and scenario families. Rejected attempts remain associated with the same quota cell; stop with an explicit insufficiency error when a cell cannot be filled within its attempt cap.

- [ ] **Step 5: Run pipeline tests**

Run: `python -m unittest tests.test_data_factory_pipeline -v`

Expected: PASS.

### Task 7: CLI, configuration, and documentation

**Files:**
- Create: `tests/test_data_factory_cli.py`
- Create: `src/sieve/cli/generate_grounded.py`
- Create: `configs/grounded_preview.json`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

```python
class CliTests(unittest.TestCase):
    def test_dry_run_does_not_require_api_key(self):
        code = main(["--config", str(self.config), "--dry-run"])
        self.assertEqual(code, 0)

    def test_live_mode_without_key_fails_before_writing_raw_response(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ZAI_API_KEY"):
                main(["--config", str(self.config)])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_data_factory_cli -v`

Expected: FAIL because the CLI is absent.

- [ ] **Step 3: Implement CLI**

Support `--config`, `--output-dir`, `--source-manifest`, `--total`, `--limit-per-category`, `--category`, `--seed`, `--dry-run`, `--resume`, `--max-requests`, `--max-retries`, and `--confirm-full`. Return a structured run summary and a nonzero exit code for incomplete quotas.

- [ ] **Step 4: Add preview configuration and README commands**

Document:

```powershell
python -m sieve.cli.generate_grounded --config configs/grounded_preview.json --dry-run
$env:ZAI_API_KEY = Read-Host "ZAI API key"
python -m sieve.cli.generate_grounded --config configs/grounded_preview.json
```

Do not place a real key in examples.

- [ ] **Step 5: Run CLI tests**

Run: `python -m unittest tests.test_data_factory_cli -v`

Expected: PASS.

### Task 8: Verification and live preview gate

**Files:**
- Test: all files under `tests/`
- Output: `data/generated/preview-<run-id>/`

- [ ] **Step 1: Run focused data-factory tests**

Run:

```powershell
python -m unittest `
  tests.test_data_factory_quotas `
  tests.test_data_factory_sources `
  tests.test_data_factory_rules `
  tests.test_data_factory_glm `
  tests.test_data_factory_quality `
  tests.test_data_factory_pipeline `
  tests.test_data_factory_cli -v
```

Expected: all PASS with no network calls.

- [ ] **Step 2: Run existing regression suite**

Run: `python -m unittest discover -s tests -v`

Expected: all PASS.

- [ ] **Step 3: Compile source and tests**

Run: `python -m compileall -q src tests`

Expected: exit code 0 and no output.

- [ ] **Step 4: Run the 140-record fake preview**

Run:

```powershell
python -m sieve.cli.generate_grounded `
  --config configs/grounded_preview.json `
  --dry-run `
  --total 140
```

Expected: 140 accepted records, exact quota margins, zero network requests, and complete report files.

- [ ] **Step 5: Inspect secret availability without printing it**

Run:

```powershell
if ($env:ZAI_API_KEY) { "ZAI_API_KEY is set" } else { "ZAI_API_KEY is not set" }
```

Expected: the key is never printed.

- [ ] **Step 6: Run a minimal paid smoke test only if the key is set**

Run one grounded batch with a strict one-request cap and a separate smoke output directory. Inspect parsing, usage, semantic validation, and persisted files before proceeding.

- [ ] **Step 7: Generate the live 140-record preview**

Run live mode with the pinned local source manifest and strict request/retry caps. Stop when the preview completes or any hard quality/cost guard fails.

- [ ] **Step 8: Report and pause**

Provide paths to accepted/rejected JSONL, manifest, and reports. Summarize source, observation, decision, verification, rejection, duplicate, retry, and token distributions. Do not invoke the 6,000-record guard.
