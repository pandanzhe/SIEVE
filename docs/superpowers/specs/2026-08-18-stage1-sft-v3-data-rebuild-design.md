# Stage-1 SFT v3 Data Rebuild Design

## Goal

Replace the current template-generated Stage-1 corpus with 6,000 source-grounded records derived only from the training partitions of τ³-bench and CAR-bench. Natural-language observations must be produced by a live LLM, while gold actions remain deterministic and executable.

## Non-goals

- Do not train Qwen3-4B on the local machine.
- Do not use official benchmark test tasks for generation, tuning, or internal validation.
- Do not let the teacher LLM choose UPDATE, HOLD, IGNORE, patches, or verification tools.
- Do not overwrite `data/sft` until a complete staged corpus passes every audit.
- Do not modify `data/rl`, `model`, or `outputs`.

## Source boundary

The source snapshot contains:

- τ³-bench text domains: airline, retail, and telecom;
- CAR-bench: Base Train and Disambiguation Train;
- source licenses, versions, task split definitions, file sizes, and SHA-256 hashes.

All source artifacts live under `data/raw`. An immutable source manifest identifies each selected file. Test task IDs may be downloaded for later official evaluation, but the candidate builder rejects them.

## Conversion boundary

The conversion has four distinct stages:

1. A source adapter converts an upstream task into an auditable `SourceTask` with task ID, split, goal, state facts, tool contracts, and reference actions.
2. A deterministic candidate builder creates atomic belief transitions and counterfactual observations. It fixes entity, field, value, authority, tool, timestamp, and target action before the teacher call.
3. A live teacher LLM realizes only the observation text. Every required entity and value term must survive verbatim; private labels and oracle fields are excluded from its prompt.
4. The SIEVE Executor replays gold actions, produces successor states, and rejects targets that cannot execute.

The LLM may improve language diversity but cannot alter grounded fields or targets. Fake clients are permitted only in unit tests and preview runs; canonical publication requires live generator metadata and positive token usage.

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

The split is assigned by `parent_task_id` before variants are generated. Every step and counterfactual from one parent remains in one split. Dev quotas are exact 10% stratified slices of each source and decision cell.

## Canonical sample

Each training row stores `scenario_id`, `step_index`, `context`, and `target`. The fixed system prompt is injected by the collator. Provenance is stored separately and contains source dataset, source version, source task ID, source hash, transition type, transformation, generator, prompt version, request hash, and token usage.

The policy-visible context contains only executable belief, current observation, goal, risk, ledger, budget, and visible verification contract. It excludes oracle values, gold decision, perturbation labels, rule reason codes, and held-out split metadata.

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

Publication is transactional: build into a sibling staging directory, validate it, rename the existing canonical directory to a temporary backup, rename the staged directory to `data/sft`, validate again through canonical paths, and only then remove the backup.

## Required audits

Canonical publication requires:

- exact total, split, source, and decision counts;
- JSON Schema pass rate of 100%;
- Executor gold-action pass rate of 100%;
- zero parent-task overlap between train and dev;
- zero official-test task IDs in generation provenance;
- zero private-label leakage in rendered prompts;
- complete provenance for every row;
- unique record IDs and no duplicate canonical samples;
- live generator identity and positive aggregate token usage;
- all file hashes matching the manifest.

## Training alignment

Stage-1 uses local `model/` assets for Qwen3-4B-Instruct-2507. `scripts/run_sft.sh` and `scripts/validate_sft.sh` default to `configs/sft_qwen3_4b.yaml`.

The training objective has three logical components:

\[
\mathcal L=\lambda_{dec}\mathcal L_{decision}+\lambda_{struct}\mathcal L_{structure}+\lambda_{value}\mathcal L_{patch\ value}.
\]

`decision` is a three-way pooled classification loss. `structure` is causal-LM loss on JSON structure, affected fields, patch operations, and verification request tokens. `patch value` is causal-LM loss on writable patch-value tokens. The structure and value objectives share the frozen base LM head and train LoRA parameters; they are token masks, not separate vocabulary heads.

## Failure behavior

- Missing source file, checksum mismatch, or split ambiguity stops before generation.
- Missing live API credentials stops before any canonical replacement.
- Malformed or ungrounded LLM output is retried up to the configured limit and then rejected.
- Insufficient accepted rows, quota mismatch, leakage, or non-executable targets blocks publication.
- SSH synchronization happens only after the local canonical corpus passes and is pushed to GitHub.
