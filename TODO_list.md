# SIEVE 后续 TODO 清单

最后更新：2026-09-04

## 0. 当前论文主线

论文的核心主张建议收窄为：

> Observation admission 是一个不同于 belief representation 和 action planning 的独立控制问题。SIEVE 要决定一个新的 observation 是否被授权修改可执行状态，然后下游 action policy 才能依赖这个状态行动。

当前工作假设：

- 问题：不可靠 observation 如果被直接写入 executable state，会在多步 Agent 中造成持续状态污染和延迟错误。
- 机制：`Revision Policy + Structured Belief + Deterministic Executor + Verification Loop`。
- 训练：Stage-1 SFT 学习单步 revision 语义；Stage-2 RL 学习 revision 决策的跨步后果。
- 算法路线：先把 constrained GRPO 作为强基线跑扎实；只有当诊断证明 trajectory-level credit assignment 存在明显问题时，再强调 GiGPO-style 局部信用分配。

## 1. 当前状态快照

| 模块 | 状态 | 证据 / 路径 | 备注 |
|---|---:|---|---|
| Stage-1 SFT | 已完成 | `outputs/stage1-qwen3-4b/` | SFT 已训练完成；loss 曲线和输入输出样例已经生成。 |
| Stage-1 readiness | 已完成 | `logs/stage1_readiness_qwen3_4b_after_gatefix_20260821_123945.log` | 修复 gate 后，readiness 已经变为 `true`。 |
| Stage-2 validation | 已完成 | `logs/stage2_validate_qwen3_4b_20260821_135812.log` | RL 前的静态校验已通过。 |
| Stage-2 GRPO 训练 | 已完成 | `outputs/stage2-qwen3-4b-8xh100-grpo/runs/20260823_145210/` | 8 张 H100，200 iterations。 |
| Stage-2 GiGPO 训练 | 已完成 | `outputs/stage2-qwen3-4b-8xh100-gigpo/runs/20260823_154308/` | 8 张 H100，200 iterations。 |
| GRPO 内部 test 评测 | 已完成 | `outputs/stage2-qwen3-4b-8xh100-grpo/runs/20260823_145210/evaluation_test.json` | Test success 为 `0.9767`；workflow 是当前最弱 domain。 |
| GiGPO 内部 test 评测 | 已完成 | `outputs/stage2-qwen3-4b-8xh100-gigpo/runs/20260823_154308/evaluation_test.json` | Test success 为 `0.9767`；与 GRPO 当前 test 指标一致。 |
| Stage-2 曲线 | 已完成 | `.../runs/*/curves/` | GRPO/GiGPO 的 PNG 曲线和 summary JSON 已生成。 |
| 论文初稿 | 进行中 | `paper_draft.md` | 已写入 SFT-only、GRPO、GiGPO、Rule Gate、Auto-Write 的内部 test 结果；外部评测仍待补。 |
| 外部评测 | 进行中 | AgentDojo manifest、smoke plan、admission smoke 均已生成 | 只做评测，不作为训练数据；当前完成 admission smoke，full executable harness 仍待接入底层 action agent。 |

## 2. 三天实验执行计划

### Day 1：补齐内部评测和快速 baseline

| ID | 任务 | 状态 | 负责人 | 产出 | 优先级 |
|---|---:|---|---|---|---:|
| D1-01 | 用 GiGPO final adapter 跑内部 held-out `test.jsonl` | 已完成 | Agent/User | `outputs/stage2-qwen3-4b-8xh100-gigpo/runs/20260823_154308/evaluation_test.json` | P0 |
| D1-02 | 把 GiGPO test 结果更新到 `paper_draft.md` | 已完成 | Agent | 论文主实验表已补齐 GiGPO、SFT-only、Rule Gate、Auto-Write 行 | P0 |
| D1-03 | 跑 SFT-only 内部 test baseline | 已完成 | Agent/User | `outputs/stage1-qwen3-4b/evaluation_stage2_test.json` | P0 |
| D1-04 | 实现或运行 Rule Gate baseline | 已完成 | Agent | `outputs/baselines/rule_gate/evaluation_test.json` | P0 |
| D1-05 | 实现或运行 Auto-Write / Append-All baseline | 已完成 | Agent | `outputs/baselines/auto_write/evaluation_test.json` | P0 |
| D1-06 | 汇总 GRPO、GiGPO、SFT-only、rules 的对比 | 已完成 | Agent | `docs/internal_eval_summary.md` | P0 |
| D1-07 | 分析 workflow domain 的失败样本 | 已完成 | Agent | `docs/workflow_failure_analysis.md`；GRPO/GiGPO trace 已生成 | P1 |

D1-01 推荐立即运行命令：

```bash
cd /mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE
mkdir -p logs
LOG=logs/eval_test_gigpo_qwen3_4b_1gpu_$(date +%Y%m%d_%H%M%S).log
nohup bash -lc "set -euo pipefail; unset MASTER_PORT; unset SIEVE_MAIN_PROCESS_PORT; export CUDA_VISIBLE_DEVICES=0; export SIEVE_NUM_GPUS=1; export NCCL_IB_DISABLE=1; export PYTHONPATH=src; export PYTHONUNBUFFERED=1; bash scripts/evaluate_stage2.sh configs/rl_qwen3_4b_8xh100_gigpo.yaml outputs/stage2-qwen3-4b-8xh100-gigpo/runs/20260823_154308/final/adapter test outputs/stage2-qwen3-4b-8xh100-gigpo/runs/20260823_154308/evaluation_test.json" > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

### Day 2：外部评测 harness

| ID | 任务 | 状态 | 负责人 | 产出 | 优先级 |
|---|---:|---|---|---|---:|
| D2-01 | 安装/探测 AgentDojo 环境，或检查当前环境是否已有可用包 | 已完成 | Agent | `.agentdojo-src/`、`.agentdojo-extra/`；`outputs/agentdojo/manifest.json` | P0 |
| D2-02 | 设计 AgentDojo evaluation protocol，将其作为安全压力测试 | 已完成 | Agent | `docs/agentdojo_eval_protocol.md` | P0 |
| D2-03 | 搭建评测 harness，记录 observation、admitted state、final action、outcome | 进行中 | Agent | 已有 `scripts/inspect_agentdojo.py` 和 `scripts/evaluate_agentdojo_admission.py`；full executable wrapper 待接 action agent | P0 |
| D2-04 | 在 SFT/GRPO/GiGPO 上跑 AgentDojo admission smoke | 已完成 | Agent/User | `outputs/agentdojo/admission_smoke_sft.json`、`admission_smoke_grpo.json`、`admission_smoke_gigpo.json` | P0 |
| D2-05 | 验证 AgentDojo 官方环境和 checker 是否可执行 | 已完成 | Agent | `outputs/agentdojo/full_harness_probe_20.json`，20/20 probe 通过 | P0 |
| D2-06 | 如果 full run 可行，运行 AgentDojo 主评测 | 待完成 | User/Agent | 需要 learned action agent + SIEVE wrapper；当前尚未完成 | P1 |

重要约束：

- AgentDojo 只作为外部评测。
- 不要把 AgentDojo 样本混入 `data/rl/scenarios/train.jsonl`。
- 优先把它当作可执行环境 / 安全压力测试，而不是普通 JSON 数据映射；只有在环境不可用时才退化成静态评测。

### Day 3：论文表格、消融和最终写作

| ID | 任务 | 状态 | 负责人 | 产出 | 优先级 |
|---|---:|---|---|---|---:|
| D3-01 | 整理最终内部实验结果表 | 待完成 | Agent | 写入 `paper_draft.md` 的主表 | P0 |
| D3-02 | 加入训练曲线和收敛分析 | 待完成 | Agent | 曲线引用和文字解释 | P0 |
| D3-03 | 加入失败案例分析 | 待完成 | Agent | 3-5 个 qualitative examples | P0 |
| D3-04 | 加入 baseline 对比章节 | 待完成 | Agent | Rule/SFT/GRPO/GiGPO 对比 | P0 |
| D3-05 | 加入外部评测章节，若未完成则写清楚 limitation | 待完成 | Agent | AgentDojo 结果或明确的 pending 说明 | P0 |
| D3-06 | 收紧 related work 和 novelty framing | 待完成 | Agent/User | 明确区分 admission 和 belief representation | P1 |
| D3-07 | 最终检查 claims，去掉过度结论，标注 preliminary single-seed results | 待完成 | Agent | 论文初稿进入 review-ready 状态 | P0 |

## 3. 内部结果表

所有结果齐全后，将这张表同步填入 `paper_draft.md`。

| 方法 | Split | Episodes | Success | Mean Return | Parse | False Update | Invalid Format | Verification | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| SFT-only | test | 300 | 0.0200 | -0.1701 | 0.9375 | 0.0147 | 0.0625 | 0.0074 | 已完成 |
| Visible Rule Gate | test | 300 | 0.9967 | 2.4897 | 1.0000 | 0.0000 | 0.0000 | 0.2174 | 已完成；公平可见字段规则仍很强 |
| Rule Gate diagnostic upper bound | test | 300 | 0.9967 | 2.4238 | 1.0000 | 0.0000 | 0.0000 | 0.2174 | 已完成；使用内部诊断字段，仅作上界 |
| Auto-Write / Append-All | test | 300 | 0.2400 | 0.0157 | 1.0000 | 0.7222 | 0.0000 | 0.0000 | 已完成；证明直接写入会造成污染 |
| GRPO | test | 300 | 0.9767 | 2.4466 | 0.9949 | 0.0000 | 0.0051 | 0.2174 | 已完成 |
| GiGPO | test | 300 | 0.9767 | 2.4466 | 0.9949 | 0.0000 | 0.0051 | 0.2174 | 已完成 |

当前已知 GRPO test 分 domain 结果：

| Domain | Episodes | Success | Mean Return | Parse | Invalid Format |
|---|---:|---:|---:|---:|---:|
| commerce | 150 | 1.0000 | 2.4957 | 1.0000 | 0.0000 |
| service | 90 | 0.9889 | 2.4756 | 1.0000 | 0.0000 |
| workflow | 60 | 0.9000 | 2.2801 | 0.9746 | 0.0254 |

## 4. 训练诊断结果

| 诊断项 | GRPO | GiGPO | 解释 |
|---|---:|---:|---|
| 最终 train success | 1.0000 | 1.0000 | 两者都能学会当前内部训练分布。 |
| 最终 dev success | 0.9800 | 0.9800 | 目前 GiGPO 没有表现出最终 dev 指标优势。 |
| 最终 dev mean return | 2.4580 | 2.4580 | 最终 return 一致，需要重点比较学习速度和 variance。 |
| 训练早期 variance | GRPO group variance 较高 | GiGPO step variance 较高 | 早期确实存在 RL 学习信号。 |
| 训练后期 variance | 接近 0 | 接近 0 | 内部任务已经接近饱和，不足以证明 GiGPO 优于 GRPO。 |

论文表述规则：

- 可以说 constrained GRPO 和 GiGPO 都能解决当前内部 benchmark。
- 在没有 test 或 ablation 证据前，不要声称 GiGPO 明显优于 GRPO。
- 如果 GiGPO 主要提升早期学习速度，就把它表述为 optimization / credit assignment 证据，而不是最终质量优势。

## 5. 必须补齐的 Baseline

| Baseline | 为什么需要 | 状态 | 最小实现 |
|---|---|---:|---|
| SFT-only | 证明 RL 是否带来超过单步 imitation 的跨步收益。 | 已完成 | 用 Stage-1 adapter 在 Stage-2 closed-loop test 上评测。 |
| Visible Rule Gate | 检查只用模型可见字段的确定性规则是否已经能解决 benchmark。 | 已完成 | 只用 entity/source/time/authentication/authority 等 prompt-visible 字段。 |
| Rule Gate diagnostic upper bound | 检查带内部诊断字段的近 oracle 规则上界。 | 已完成 | 包含 `condition/relevant` 等诊断字段，仅作上界。 |
| Auto-Write / Append-All | 展示所有 observation 都写入状态时的污染危害。 | 已完成 | 对看似合法的字段总是 UPDATE，然后评测下游结果。 |
| Prompt-only base model | 衡量是否必须训练。 | 可选 | 用 base Qwen3-4B 和相同 JSON 协议直接评测。 |
| Structured belief without admission | 作为最近邻概念 baseline。 | 可选/P1 | 维护 uncertainty，但不设置 admission boundary。 |

## 6. 外部评测计划

### AgentDojo 的定位

AgentDojo 应作为外部可执行安全压力测试：

1. 加载任务和环境。
2. 在环境中运行模型 / policy。
3. 截获可能修改 executable belief 的 observations。
4. 应用 SIEVE admission decisions。
5. 追踪 unsafe 或 contaminated state 是否影响最终 action。

### 最小指标

| 指标 | 用途 |
|---|---|
| task_success_rate | 衡量任务效用是否被保留。 |
| unsafe_action_rate | 衡量模型是否执行 harmful / policy-violating action。 |
| poison_to_state_rate | 恶意或不可靠 observation 被写入状态的比例。 |
| state_to_action_rate | 被污染状态进一步影响最终 action 的比例。 |
| verification_rate | 谨慎策略带来的验证成本。 |
| over_hold_rate | 衡量模型是否通过过度 HOLD 来规避风险。 |

### AgentDojo TODO

| 任务 | 状态 | 备注 |
|---|---:|---|
| 检查当前环境是否能安装或导入 AgentDojo | 已完成 | 已安装源码到 `.agentdojo-src/`，补充轻量依赖到 `.agentdojo-extra/`。不要用 `.agentdojo-pydeps` 作为训练 PYTHONPATH。 |
| 导出 AgentDojo suite/task/injection manifest | 已完成 | `outputs/agentdojo/manifest.json`，共 4 个 suite、184 个 task definitions。 |
| 生成外部 smoke 子集 | 已完成 | `outputs/agentdojo/smoke_plan.json`，workspace/slack 共 20 对 user task + injection task。 |
| 手动跑一个官方 sample task | 暂缓 | 官方 benchmark CLI 会引入 provider 依赖；当前内部 PyPI 的 `anthropic==1.2.0` wheel 缺少 `anthropic._utils`，不建议在这个环境继续硬跑。 |
| 确定 SIEVE 的接入点 | 已完成 | 优先做 observation admission wrapper，而不是训练数据转换；见 `docs/agentdojo_eval_protocol.md`。 |
| 跑 20-50 个 task 的 admission smoke | 已完成 | SFT/GRPO/GiGPO adapters 已完成；这是 observation admission smoke，不是完整 AgentDojo utility/security run。 |
| 验证 AgentDojo 官方环境/checker | 已完成 | `outputs/agentdojo/full_harness_probe_20.json`；workspace/slack 20 对 smoke，ground-truth utility/security 均为 1.0000。 |
| 加入外部结果或 limitation 段落 | 进行中 | 已在 `docs/internal_eval_summary.md` 记录；论文中应保守表述。 |

## 7. 论文更新 Checklist

| 章节 | 需要更新的内容 | 状态 |
|---|---|---:|
| Abstract | 强调 authorized observation-to-state transition。 | 待完成 |
| Introduction | 把 persistent contamination 和 delayed failure 写成核心问题。 | 待完成 |
| Related Work | 区分 SIEVE 与 belief representation、memory summarization、general agent safety。 | 待完成 |
| Method | 把 executor invariants 作为支撑机制，而不是主 novelty。 | 待完成 |
| Algorithm | 把 GRPO 写成强基线；如果证据支持，再把 GiGPO 写成局部信用分配变体。 | 待完成 |
| Experiments | 加入内部 test、baselines、曲线、外部评测。 | 待完成 |
| Limitations | 如果仍是内部 synthetic/mechanistic benchmark 和 single-seed，需要明确说明。 | 待完成 |
| Conclusion | 没有外部证据前，不要声称泛化到 AgentDojo / 真实世界鲁棒性。 | 待完成 |

## 8. 命令和结果记录

每完成一个任务，把对应命令、日志和结果路径补到这里。

| 日期 | 任务 ID | 命令 / 日志 | 结果 |
|---|---|---|---|
| 2026-08-23 | Stage-2 GRPO train | `logs/stage2_grpo_qwen3_4b_8xh100_200step_20260823_144728.log` | 已完成；run 为 `20260823_145210`。 |
| 2026-08-23 | Stage-2 GiGPO train | `logs/stage2_gigpo_qwen3_4b_8xh100_200step_20260823_153836.log` | 已完成；run 为 `20260823_154308`。 |
| 2026-08-24 | GRPO test eval | `logs/eval_test_grpo_qwen3_4b_1gpu_20260824_094820.log` | 已完成；success 为 `0.9767`。 |
| 2026-08-24 | GiGPO test eval | `logs/eval_test_gigpo_qwen3_4b_1gpu_20260824_111357.log` | 已完成；success 为 `0.9767`。 |
| 2026-08-24 | SFT-only test eval | `logs/eval_stage1_best_20260824_120359.log` | 已完成；success 为 `0.0200`。 |
| 2026-08-25 | Rule Gate test eval | `logs/eval_baseline_rule_gate_test_20260825_075940.log` | 已完成；success 为 `0.9967`。 |
| 2026-08-25 | Auto-Write test eval | `logs/eval_baseline_auto_write_test_20260825_075941.log` | 已完成；success 为 `0.2400`。 |
| 2026-08-25 | Auto-Write workflow trace | `outputs/baselines/auto_write/workflow_trace_summary.json` | 已完成；workflow success 为 `0.2833`，43/60 失败。 |
| 2026-08-25 | 更新论文内部结果表 | `paper_draft.md` | 已同步 SFT-only、GRPO、GiGPO、Rule Gate、Auto-Write，并标注 Rule Gate 为诊断性近 oracle 上界。 |
| 2026-08-28 | Visible Rule Gate test eval | `logs/eval_baseline_visible_rule_gate_test_20260828_123727.log` | 已完成；success 为 `0.9967`，workflow 为 `1.0000`。 |
| 2026-08-28 | AgentDojo CPU manifest smoke | `scripts/inspect_agentdojo.py` | 已完成；`outputs/agentdojo/manifest.json` 和 `outputs/agentdojo/smoke_plan.json` 已生成。 |
| 2026-08-28 | AgentDojo 官方 CLI 探测 | `.agentdojo-src/`、`.agentdojo-extra/` | 官方 benchmark import 失败于 provider 依赖 `anthropic._utils` 缺失；不是 SIEVE 训练代码问题。 |
| 2026-08-29 | AgentDojo admission CPU baseline smoke | `scripts/evaluate_agentdojo_admission.py` | 已完成；`outputs/agentdojo/admission_smoke_auto_write.json` 与 `outputs/agentdojo/admission_smoke_visible_rule_gate.json` 已生成。 |
| 2026-08-29 | AgentDojo GRPO admission smoke | `logs/agentdojo_admission_grpo_20260829_080825.log` | 已完成；parse 0.9250，poison_to_state 0.0000，ignore_injection 0.3500。 |
| 2026-08-29 | AgentDojo GiGPO admission smoke | `logs/agentdojo_admission_gigpo_20260829_100654.log` | 已完成；parse 0.9000，poison_to_state 0.0000，ignore_injection 0.4500。 |
| 2026-08-30 | AgentDojo SFT admission smoke | `logs/agentdojo_admission_sft_20260830_140802.log` | 已完成；parse 0.9500，poison_to_state 0.0000，ignore_injection 0.8500。 |
| 2026-09-04 | 内部实验汇总 | `docs/internal_eval_summary.md` | 已完成；同步内部 test、baseline、AgentDojo admission smoke 结论。 |
| 2026-09-04 | Workflow 失败分析 | `docs/workflow_failure_analysis.md` | 已完成；GRPO/GiGPO 失败主要来自长 `tool_result` JSON value 解析失败。 |
| 2026-09-04 | AgentDojo full-harness checker probe | `outputs/agentdojo/full_harness_probe_20.json` | 已完成；20/20 cases ok，官方 ground-truth utility/security checker 可执行。 |
| TBD | AgentDojo full executable run | TBD | 待完成；需要接入底层 action agent 或官方 benchmark provider。 |

## 9. 当前阶段完成标准

当前阶段完成需要满足：

- [x] GiGPO internal test 结果已生成并完成总结。
- [x] SFT-only internal test 结果已生成并完成总结。
- [x] Rule Gate baseline 结果已生成并完成总结。
- [x] Auto-Write baseline 结果已生成并完成总结。
- [x] 至少完成一个 workflow-domain 失败样本分析。Auto-Write 污染样例、GRPO/GiGPO workflow trace 和失败分析均已生成。
- [x] AgentDojo CPU manifest smoke 成功运行，已明确 full model smoke 的剩余依赖。
- [ ] `paper_draft.md` 已包含更新后的 method framing、主实验结果、baseline 表格、曲线分析和 limitations。当前仍需最终润色和删减占位表。
- [ ] 所有关键结果路径都已经记录在本 TODO 文件或 `paper_draft.md` 中。

