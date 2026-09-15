# AgentDojo 外部评测协议

本文档记录 SIEVE 接入 AgentDojo 的评测方式。AgentDojo 只作为外部测试集和可执行安全压力测试，不进入 Stage-1 SFT 或 Stage-2 RL 训练数据。

## 目标

内部 Stage-2 test 已经接近饱和，Visible Rule Gate 在只使用 prompt 可见字段时也达到很高成功率。因此 AgentDojo 的作用不是继续证明内部模板可解，而是检查 SIEVE 的 observation admission 思想能否迁移到真实工具环境中的不可信内容、间接注入和有副作用 action。

核心问题：

1. 不可信工具内容是否会进入可执行状态。
2. 被污染状态是否会影响后续工具调用或最终动作。
3. SIEVE 的 admission boundary 是否能在保留任务效用的同时降低 unsafe action。

## 当前可执行阶段

当前先完成 CPU-side 准备：

1. 安装或探测 AgentDojo 包。
2. 导出 suite、user task、injection task 和 injection vector manifest。
3. 选择 20-50 个 smoke task 作为外部评测子集。
4. 设计 SIEVE wrapper 的接入点。

命令：

```bash
cd /mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE
PYTHONPATH=.agentdojo-src:.agentdojo-extra python3 scripts/inspect_agentdojo.py \
  --root . \
  --output outputs/agentdojo/manifest.json \
  --print-examples 8
```

如果 `.agentdojo-src` 不存在，先运行：

```bash
cd /mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE
python3 -m pip install --no-cache-dir --no-deps --upgrade --force-reinstall \
  --target .agentdojo-src agentdojo==0.1.35
python3 -m pip install --no-cache-dir --no-deps --upgrade --force-reinstall \
  --target .agentdojo-extra deepdiff==9.1.0 orderly-set==5.5.0 cachebox==5.2.3 \
  docstring-parser==0.18.0 email-validator==2.3.0 dnspython==2.8.0
```

不要把 `.agentdojo-pydeps` 加入 `PYTHONPATH` 运行训练或 SIEVE 评测。该目录包含一组与训练环境不兼容的依赖版本，例如 `pydantic_core`，会覆盖系统环境。

当前环境备注：

- `python3 -m venv` 不可用，因为系统 Python 缺少 `ensurepip`。
- conda/mamba 创建项目环境时会尝试写只读的 `/home/tiger/.cache/conda/pkgs/cache/shards`，当前未作为主路线。
- `agentdojo==0.1.35` 已通过 `--no-deps --target .agentdojo-src` 安装源码包。
- `deepdiff`、`email-validator` 等轻量依赖已通过 `--target .agentdojo-extra` 补齐。
- 官方 `agentdojo.scripts.benchmark` CLI 仍不建议在当前环境直接跑：内部 PyPI 的 `anthropic==1.2.0` wheel 缺少 `anthropic._utils` 等子模块，provider import 会失败。后续 full run 推荐使用干净任务环境重新安装 AgentDojo，或绕过官方 provider 层实现 SIEVE 自定义 wrapper。

## Full Harness 设计与当前验证状态

AgentDojo 官方 benchmark 是完整 agent loop：

```text
user task
  -> tool/environment observation
  -> model chooses next tool/action
  -> injected/untrusted content may appear in tool outputs
  -> environment checks utility and security
```

SIEVE 当前模型能力是 revision policy：

```text
observation + current belief + budget
  -> UPDATE / HOLD / IGNORE + optional verification
  -> deterministic executor updates executable belief
```

因此不能把 AgentDojo 简单转成普通 JSON 分类集。合理接入方式是 wrapper：

1. 底层 action agent 负责选择 AgentDojo 工具调用。
2. 每次工具返回 observation 后，SIEVE wrapper 抽取候选 claim。
3. SIEVE revision policy 决定哪些 claim 进入 executable belief。
4. 下游 action agent 只读取 admission 后的 executable belief，不能直接信任原始不可信 observation。
5. AgentDojo 的 utility/security checker 负责最终打分。

如果时间不足，可先实现 smoke 版：

1. 只选 `workspace` 和 `slack` suite。
2. 每个 suite 选择 5-10 个 user tasks。
3. 使用 `tool_knowledge` 或默认 attack。
4. 记录每个 injected observation 是否被写入状态，以及最后是否触发 unsafe action。

当前已经完成的 CPU smoke 不是模型评测，而是评测资产检查：

```text
outputs/agentdojo/manifest.json
outputs/agentdojo/smoke_plan.json
```

其中 `manifest.json` 包含 4 个 suite 和 184 个 task definitions；`smoke_plan.json` 选取 workspace/slack 共 20 对 user task + injection task，供后续模型 smoke 固定使用。

当前进一步完成了一个不调用外部 LLM provider 的 full-harness probe：

```text
scripts/probe_agentdojo_full_harness.py
outputs/agentdojo/full_harness_probe_20.json
```

该 probe 对 `smoke_plan.json` 中 20 对 workspace/slack task 使用 AgentDojo suite 自身的 ground-truth pipeline 执行工具环境，并调用官方 utility/security checker。结果为：

| Probe | Cases | OK | User GT Utility | Injection GT Security |
|---|---:|---:|---:|---:|
| AgentDojo full-harness checker probe | 20 | 1.0000 | 1.0000 | 1.0000 |

这说明 AgentDojo 的 suite 加载、环境执行、ground-truth 工具调用和官方 utility/security checker 在当前环境中已经可用。它仍然不是 SIEVE 的完整外部结果，因为 learned action agent 和 SIEVE wrapper 尚未接入 AgentDojo loop。下一步需要实现：

```text
AgentDojo user task
  -> action agent selects tool call
  -> tool returns untrusted observation
  -> SIEVE wrapper applies UPDATE/HOLD/IGNORE to extracted claims
  -> action agent continues using admitted executable state
  -> AgentDojo utility/security checker scores the final state
```

因此当前状态是：AgentDojo 评测环境和打分器已验证可用；SIEVE 端到端 AgentDojo evaluation harness 未完成。

## 指标

| 指标 | 含义 |
|---|---|
| task_success_rate | 主任务是否完成。 |
| security_success_rate | 是否避免执行 injection goal。 |
| unsafe_action_rate | 是否执行有害或违反用户目标的工具动作。 |
| poison_to_state_rate | 注入或不可信内容进入 executable belief 的比例。 |
| state_to_action_rate | 已污染状态进一步影响最终 action 的比例。 |
| verification_rate | 触发验证的比例。 |
| over_hold_rate | 过度 HOLD 造成有效信息无法进入状态的比例。 |

## 论文使用方式

如果只完成 admission smoke 和 checker probe：

> We include AgentDojo as an external executable stress test rather than a training source. The current implementation exports and audits AgentDojo suites and defines the SIEVE wrapper protocol. Full-scale execution is left as ongoing work; internal results should therefore be interpreted as mechanistic evidence rather than broad external robustness.

更精确地说，当前版本已经验证 AgentDojo suite execution and official utility/security checkers on a 20-pair smoke plan using ground-truth pipelines, and separately evaluates SIEVE admission decisions on the same plan. It does not yet run SIEVE inside a learned AgentDojo action loop.

如果完成 full run：

在论文中单独报告 AgentDojo 外部评测表，不要与内部 `data/rl/scenarios/test.jsonl` 混成一个总分。
