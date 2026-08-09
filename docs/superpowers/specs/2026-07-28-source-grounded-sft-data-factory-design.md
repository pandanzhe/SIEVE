# Source-Grounded SFT Data Factory Design

## 1. Objective and scope

Build a local, auditable data factory for SIEVE's first-stage structured SFT. The factory produces single-step belief-revision records:

\[
(B_{t-1}, o_t, g_t, r_t, L_t)
\rightarrow
(c_t, z_t, \Delta B_t, v_t, q_t)
\]

The first online preview contains 140 accepted records. Full production contains 6,000 accepted training records and may run only after explicit user approval of the preview.

The factory does not freely invent complete inputs and labels with an LLM. Open-source records provide the task, entity, tool, state, or observation grounding. Deterministic rules provide the decision, affected fields, patch, verification branch, and target belief state. GLM-4.7 is limited to extraction, natural-language realization, difficult paraphrasing, and optional semantic review.

## 2. Output formats and local layout

Dataset records use JSON Lines (`.jsonl`): one complete JSON object per UTF-8 line. JSONL supports streaming writes, per-record validation, resumable generation, and rejected-record isolation without rewriting a large JSON array.

Run metadata and aggregate reports use ordinary JSON (`.json`). A Markdown quality report is also generated for convenient review.

The proposed local layout is:

```text
data/
  source_cache/                 # ignored local snapshots or user-provided source paths
  generated/
    preview-<run-id>/
      raw/
        glm_responses.jsonl
      accepted.jsonl
      rejected.jsonl
      manifest.json
      quality_report.json
      quality_report.md
    full-<run-id>/
      raw/
        glm_responses.jsonl
      accepted.jsonl
      rejected.jsonl
      manifest.json
      quality_report.json
      quality_report.md
```

Large upstream datasets and secrets are not committed to the project. The factory records a pinned version or commit, repository-relative source location, source record identifier, and content hash.

## 3. Preview and full-production quotas

### 3.1 Observation types

| Observation type | Preview | Full |
|---|---:|---:|
| New and consistent information | 21 | 900 |
| Explicit conflict | 21 | 900 |
| Implicit conflict | 21 | 900 |
| Stale information | 21 | 900 |
| Irrelevant information | 21 | 900 |
| Tool error or low-trust source | 21 | 900 |
| Insufficient or ambiguous information | 14 | 600 |
| Total | 140 | 6,000 |

### 3.2 Decision labels

| Decision | Preview | Full |
|---|---:|---:|
| UPDATE | 56 | 2,400 |
| IGNORE | 49 | 2,100 |
| HOLD | 35 | 1,500 |

Preview HOLD records contain 21 VERIFY and 14 DEFER branches. Full HOLD records contain 900 VERIFY and 600 DEFER branches.

### 3.3 Preview cross-quota

| Observation type | UPDATE | IGNORE | HOLD | Total |
|---|---:|---:|---:|---:|
| New and consistent | 16 | 3 | 2 | 21 |
| Explicit conflict | 9 | 6 | 6 | 21 |
| Implicit conflict | 8 | 6 | 7 | 21 |
| Stale information | 8 | 10 | 3 | 21 |
| Irrelevant information | 4 | 15 | 2 | 21 |
| Tool error or low-trust source | 8 | 7 | 6 | 21 |
| Insufficient or ambiguous | 3 | 2 | 9 | 14 |
| Total | 56 | 49 | 35 | 140 |

Counterintuitive cells require explicit rule conditions. For example, stale information may still update if it is older than receipt time but newer than the current belief and comes from an authoritative source. An irrelevant observation may update only if it is irrelevant to the active subgoal but belongs to an explicitly maintained long-term state field. These cases must not be produced by unconstrained model judgment.

The three scenario families—transactional, information/tool, and long-term state—target approximately equal representation in the full dataset. Observation type, decision, scenario, source, and difficulty are cross-stratified to prevent keyword shortcuts.

## 4. Open-source grounding

The preview source targets are:

| Source | Preview target | Primary use |
|---|---:|---|
| tau2-bench | 70 | Airline, retail, and telecom tasks with policies, tools, and database state |
| ToolBench | 35 | API descriptions, tool returns, failures, and low-trust output forms |
| AgentBench | 21 | Database, knowledge graph, shopping, and other environment observations |
| WebArena | 14 | Web, commerce, forum, and collaboration-system observations |

Exact records depend on source availability and license verification. A source adapter may skip an unusable record but cannot silently replace it with a freely invented scene. STALE and related research may guide perturbation rules but do not count as a data source unless a licensed, identifiable source record is actually used.

Every accepted record contains:

- source dataset and pinned version or commit;
- original record identifier and repository-relative location;
- SHA-256 hash of the normalized source material;
- license identifier or recorded license note;
- transformation name and version;
- `parent_record_id` for derived counterfactuals.

## 5. Data flow

1. Acquire or locate a pinned open-source snapshot.
2. Read records through a source-specific adapter.
3. Normalize tasks, entities, state fields, timestamps, tool calls, tool responses, and source metadata.
4. Select a grounded base fact or state transition.
5. Apply a deterministic rule or a minimal counterfactual perturbation.
6. Compute the decision, affected fields, patch, verification branch, and oracle target state.
7. Ask GLM-4.7 to realize the grounded structured event as an observation or extract a bounded candidate structure.
8. Parse the GLM response without accepting model-generated truth labels.
9. Run schema, execution, semantic, provenance, duplication, and quota checks.
10. Append accepted and rejected records separately.
11. Produce a manifest and quality report.
12. Stop after the preview and wait for explicit approval before full production.

## 6. Components and boundaries

The implementation will add focused modules:

- source adapters: load each supported upstream format and emit normalized source records;
- normalization: define the source-neutral record interface;
- rule engine: generate structural ground truth and oracle state;
- perturbations: create minimal, traceable counterfactual variants;
- GLM client: handle API requests, response parsing, retries, and usage accounting;
- pipeline: enforce quotas, checkpoints, resumption, and run guards;
- quality checks: validate execution, semantics, provenance, duplicates, and distribution;
- CLI: expose preview, dry-run, resume, source-path, seed, and limit controls.

Existing SIEVE `RevisionContext`, `RevisionOutput`, `StateExecutor`, and data validation behavior are reused where compatible. The new audit schema is richer than the current `SFTRecord`; an export step strips non-training fields and produces a training-compatible JSONL view.

## 7. Record schema

Each accepted audit record has these top-level sections:

```json
{
  "record_id": "sieve_preview_000001",
  "group_id": "source-record-plus-counterfactual-group",
  "provenance": {
    "source_dataset": "tau2-bench",
    "source_version": "pinned-commit",
    "source_record_id": "retail-task-id",
    "source_uri": "repository-relative-path",
    "source_sha256": "hex-digest",
    "license": "recorded-license",
    "parent_record_id": null,
    "transformation": "stale_time_shift"
  },
  "taxonomy": {
    "observation_type": "stale",
    "scenario_type": "transaction",
    "domain": "retail",
    "difficulty": "medium"
  },
  "state": {
    "belief_state": {},
    "observation": {},
    "goal": "",
    "risk": {},
    "ledger": {}
  },
  "target": {
    "decision": "IGNORE",
    "affected_fields": [],
    "reason_code": "STALE_EVIDENCE",
    "conflict_type": "temporal",
    "evidence_ids": [],
    "patches": [],
    "verification": {
      "action": "NO_VERIFY",
      "request": null
    }
  },
  "oracle": {
    "target_belief_state": {}
  },
  "generation": {
    "generator": "glm-4.7",
    "prompt_version": "v1",
    "request_hash": "hex-digest",
    "rule_version": "v1"
  },
  "validation": {
    "schema_valid": true,
    "patch_executable": true,
    "semantic_consistent": true,
    "provenance_valid": true
  }
}
```

`oracle`, `provenance`, `generation`, and `validation` are production-only fields and must not enter the model input. Split assignment operates on `group_id`, so a source record, its paraphrases, and its counterfactual variants cannot cross dataset splits.

## 8. GLM-4.7 integration and cost controls

The client uses the Zhipu `zai` SDK and reads the key only from the `ZAI_API_KEY` environment variable. The key is never stored in source, configuration, logs, manifests, raw responses, or generated records.

Default behavior:

- model: `glm-4.7`;
- thinking: enabled;
- bounded output tokens appropriate for a small JSON batch rather than 65,536;
- temperature initially between 0.5 and 0.7;
- batch size initially 4–6 observations;
- deterministic request hash based on source record, transformation, prompt version, model, and parameters;
- cached successful response reuse;
- bounded exponential retries for transient errors;
- split-to-single-record retry when a batch is partly malformed;
- request, token, latency, retry, and acceptance accounting without secrets.

The offline dry run uses a deterministic fake client and must validate parsing, quotas, checkpoints, resumption, rejection handling, and reports before any paid request. The preview has a hard cap of 140 accepted records plus bounded request and retry budgets. Full generation requires an explicit guard such as `--confirm-full 6000`.

## 9. Validation and rejection

An accepted record must satisfy all of the following:

1. Schema validity: required fields and enums are valid; decision branches obey their contracts.
2. Patch execution: applying the target through the executor yields exactly the oracle target state.
3. Semantic consistency: observation text preserves the structured entity, time, source, polarity, condition, and event.
4. Provenance validity: the source record, version, locator, hash, and license note are present.
5. Leakage prevention: the input does not expose decision labels, reason codes, or oracle-only state.
6. Duplicate control: exact duplicates, source-template duplicates, and high-similarity paraphrases remain below configured thresholds.
7. Quota validity: accepted counts match observation and decision targets.

Rejected records are retained with stable machine-readable reason codes. They do not count toward accepted quotas. Replacement attempts must remain attached to the same quota cell and use a grounded source record.

## 10. CLI behavior and safety gates

The CLI supports:

- `--dry-run` for zero-cost local validation;
- `--total 140` for preview size;
- `--limit-per-category` for debugging a small category subset;
- `--category` for focused runs;
- `--seed` for reproducible source selection and transformations;
- `--resume` for checkpoint recovery;
- local source paths or a source manifest;
- explicit request and retry caps;
- an explicit full-run confirmation guard.

The implementation first passes unit and integration tests with a fake GLM client. It then performs a very small live smoke request before the 140-record preview. After preview generation, the pipeline stops and reports quality and cost. It cannot automatically continue to 6,000 records.

## 11. Success criteria

The design is successful when:

- the dry run completes without network access or a real API key;
- every accepted preview record is grounded in an identifiable open-source record;
- preview counts exactly match the 140-record observation and decision matrices;
- every accepted patch executes to the oracle target state;
- no secret is persisted;
- interrupted runs resume without duplicate paid requests;
- quality and cost reports explain all accepted, rejected, and retried records;
- full production remains blocked pending explicit user approval.

## 12. Non-goals

This phase does not deploy a local teacher model, build a complete online Agent, train the SFT model, generate RL trajectories, or launch the 6,000-record run. It does not treat LLM semantic review as a substitute for deterministic ground truth.
