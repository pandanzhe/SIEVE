# Stage-2 Qwen2.5-3B Constrained GRPO Design

## 1. Scope

Stage 2 starts from the Stage-1 Qwen2.5-3B PEFT adapter and optimizes the same causal language-model policy on multi-step observation-evaluation trajectories. The model continues to emit the Stage-1 JSON action protocol. The Stage-1 structured classification heads remain auxiliary SFT artifacts; they are not part of the autoregressive action probability and are therefore frozen during RL.

This implementation must work offline for data construction and validation. GPU execution is deferred to the server. No absolute workstation or server path is stored in Python or YAML files.

## 2. Stage-1 readiness boundary

Stage-1 head-level dev metrics are not treated as sufficient evidence because they are teacher-forced and the existing dev split shares upstream source records with train. Before a full RL run, the server must evaluate `best` and `final` adapters by free generation on the Stage-2 dev scenarios.

The readiness evaluator reports:

- strict JSON parse rate;
- schema validity and Executor acceptance;
- decision macro-F1 and false-update rate against environment-private reference actions;
- full action exact match and patch-value exact match;
- terminal state consistency under greedy closed-loop rollout;
- reward variance across sampled GRPO groups.

RL may start only when generated actions are predominantly parseable and executable and sampled groups have non-zero score variance. Long-horizon task success is allowed to remain imperfect because it is the optimization target of Stage 2.

## 3. Dataset semantics

Stage-2 rows are scenario definitions, not prompt-answer SFT examples. Each row contains deployment-visible initial state plus environment-private transition and oracle information. The environment loader is the only component allowed to read private fields; the policy receives a `RevisionContext` serialized with the existing Stage-1 system prompt.

The canonical layout is:

```text
data/rl/
├── scenarios/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── manifest.json
└── quality_report.json
```

Scenario splitting happens by the upstream base-task key before variants are constructed. All variants of one upstream task remain in one split. The internal test split is appropriate for engineering regression tests; final paper claims must additionally use a pinned official benchmark test split that was not consumed by Stage-1 generation.

## 4. Offline scenario construction

The builder consumes the provenance-rich `data/sft/source/records.jsonl`, selects accepted authoritative UPDATE records, and groups them by upstream parent task. It creates domain-stratified scenarios over commerce, service, and workflow tasks.

Each scenario has at least three belief slots, four base events, and one conditional verification response:

1. a low-authority task-relevant observation whose admissible action is HOLD with verification;
2. an authoritative verification response, produced only if the policy requests the policy-visible field-specific tool;
3. a wrong-entity observation whose admissible action is IGNORE;
4. an authoritative observation for another task-dependent field whose admissible action is UPDATE;
5. a stale conflicting observation whose admissible action is IGNORE.

The verification response is a conditional micro-transition and is not an unconditional next row. A wrong UPDATE contaminates the belief state and affects later reward. HOLD consumes verification and tool budget only when a request can actually be executed.

Default generated counts are 1,600 train, 200 dev, and 300 internal test scenarios. The target macro-domain proportions are 50% commerce, 30% service, and 20% workflow. Variants may reuse a base task inside its assigned split, but no base task crosses splits.

## 5. State, action, transition, reward

The policy state is the bounded `RevisionContext`:

\[
s_t=(B_{t-1},o_t,g_t,r_t,L_t,\rho_t).
\]

The action is the existing structured JSON `RevisionOutput`:

\[
u_t=(c_t,A_t,\Delta B_t,q_t).
\]

`ScenarioRevisionEnvironment.step` applies the action through `StateExecutor`, updates belief and ledger, decrements step/token/tool/verification budgets as applicable, and selects either the verification branch or the next scheduled event. This makes the next observation action-conditioned.

The task reward is:

\[
r_t=\gamma\Phi(B_t,x_t)-\Phi(B_{t-1},x_t)
+\mathbb{1}[\text{terminal success}].
\]

The environment separately records `false_update`, `unsafe_action`, `verification`, `stall`, `invalid_format`, `invalid_patch`, `collateral_edit`, and `budget_violation`. Constrained GRPO scores a trajectory as:

\[
\widetilde R_i=R_i-\sum_j\lambda_j C_{ij}.
\]

Lagrange multipliers are updated from unnormalized mean costs and projected to non-negative values.

## 6. Qwen2.5-3B learner

The learner loads:

- the local Qwen2.5-3B base directory;
- the Stage-1 PEFT adapter directory;
- the Stage-1 tokenizer directory, falling back to the base tokenizer;
- a frozen second copy of the Stage-1 policy as the KL reference.

For each sampled base scenario, the rollout collector creates `G` independent environments with the same initial scenario and different generation seeds. At every outer step it renders the current context, generates one JSON action, parses it strictly, applies it to the environment, and records policy and reference log probabilities.

Invalid JSON is mapped to a safe IGNORE fallback for transition continuity and incurs `invalid_format=1`. It never receives a formatting reward.

The policy update uses the clipped group-relative objective plus sampled KL:

\[
\mathcal L=-\mathbb E[\min(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A)]
+\beta_{KL}\widehat D_{KL}(\pi_\theta\|\pi_{SFT}).
\]

Only trainable PEFT adapter parameters receive gradients. Base-model weights, Stage-1 structured heads, reference policy, Executor, and environment are frozen.

## 7. Portability and execution

`configs/rl_qwen25_3b.yaml` stores only repository-relative paths. Shell entrypoints derive the repository root from their own location. The model directory and Stage-1 checkpoint are populated on the server without editing Python files.

The RL dependency file remains compatible with the Stage-1 Transformers/PEFT stack and does not require TRL. The custom learner is used because SIEVE actions are plain JSON cognitive actions rather than tool-call messages.

Two A100 40G GPUs use Accelerate DDP. Each rank collects independent scenario groups; gradient synchronization happens during adapter updates. The frozen reference policy is replicated per rank. A single-GPU validation run remains supported.

## 8. Verification requirements

Local verification covers:

- deterministic generation and checksums;
- exact split counts and zero base-task overlap;
- absence of `target` at the scenario top level;
- private oracle exclusion from policy prompts;
- conditional HOLD-to-verification transitions;
- budget and cost accounting;
- strict action parsing;
- GRPO advantage and clipped-loss math;
- configuration and CLI validate-only paths;
- complete legacy unit-test regression.

The local machine does not instantiate Qwen, download dependencies, or start training.
