# SIEVE 内部评测阶段性总结

本文档汇总当前已经完成的 SIEVE 内部训练、held-out test、baseline 和外部 admission smoke 结果。结论基于当前磁盘上的日志与输出文件，主要用于论文初稿和后续实验决策。

## 1. 当前完成状态

| 模块 | 状态 | 证据路径 |
|---|---:|---|
| Stage-1 SFT | 已完成 | `outputs/stage1-qwen3-4b/` |
| Stage-2 GRPO | 已完成 | `outputs/stage2-qwen3-4b-8xh100-grpo/runs/20260823_145210/` |
| Stage-2 GiGPO | 已完成 | `outputs/stage2-qwen3-4b-8xh100-gigpo/runs/20260823_154308/` |
| GRPO internal test | 已完成 | `outputs/stage2-qwen3-4b-8xh100-grpo/runs/20260823_145210/evaluation_test.json` |
| GiGPO internal test | 已完成 | `outputs/stage2-qwen3-4b-8xh100-gigpo/runs/20260823_154308/evaluation_test.json` |
| SFT-only internal test | 已完成 | `outputs/stage1-qwen3-4b/evaluation_stage2_test.json` |
| Rule / Auto-Write baselines | 已完成 | `outputs/baselines/*/evaluation_test.json` |
| AgentDojo admission smoke | 已完成 | `outputs/agentdojo/admission_smoke_*.json` |

## 2. Internal Held-Out Test 主结果

| 方法 | Episodes | Success | Mean Return | Parse | False Update | Stall | Invalid Format | Verification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT-only | 300 | 0.0200 | -0.1701 | 0.9375 | 0.0147 | 0.0331 | 0.0625 | 0.0074 |
| Auto-Write / Append-All | 300 | 0.2400 | 0.0157 | 1.0000 | 0.7222 | 0.0000 | 0.0000 | 0.0000 |
| Visible Rule Gate | 300 | 0.9967 | 2.4897 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2174 |
| Rule Gate diagnostic upper bound | 300 | 0.9967 | 2.4238 | 1.0000 | 0.0000 | 0.0435 | 0.0000 | 0.2174 |
| SIEVE-GRPO | 300 | 0.9767 | 2.4466 | 0.9949 | 0.0000 | 0.0000 | 0.0051 | 0.2174 |
| SIEVE-GiGPO | 300 | 0.9767 | 2.4466 | 0.9949 | 0.0000 | 0.0000 | 0.0051 | 0.2174 |

## 3. 直接结论

1. **Stage-2 RL 是必要的。** SFT-only 在 closed-loop internal test 上只有 2.00% success，说明单步 imitation 即使学会了输出格式，也不能自动学会跨步状态承诺、验证和后续执行后果。

2. **状态污染是真实 failure mode。** Auto-Write / Append-All 的 false update rate 为 72.22%，success 只有 24.00%。这支持论文核心问题：不加 admission boundary 的直接写入会把错误 observation 变成下游可执行状态。

3. **当前内部 benchmark 规则可解性过强。** Visible Rule Gate 仅使用 prompt 中可见字段，已经达到 99.67% success，高于 GRPO/GiGPO。论文不能只依赖这组内部结果声称 learned SIEVE 优于规则方法。

4. **GiGPO 目前不能作为主要算法贡献。** GiGPO 与 GRPO 在 internal test 上完全一致，均为 97.67% success。训练早期可能存在 step-level variance 信号，但最终效果没有超过 trajectory-level GRPO。

5. **当前剩余错误集中在 workflow。** GRPO/GiGPO 在 commerce 与 service 几乎饱和，workflow success 为 90.00%。失败主要来自长工具结果写入 JSON patch value 时的解析失败，而不是 false update。

## 4. GRPO 分领域结果

| Domain | Episodes | Success | Mean Return | Parse | False Update | Invalid Format |
|---|---:|---:|---:|---:|---:|---:|
| All | 300 | 0.9767 | 2.4466 | 0.9949 | 0.0000 | 0.0051 |
| Commerce | 150 | 1.0000 | 2.4957 | 1.0000 | 0.0000 | 0.0000 |
| Service | 90 | 0.9889 | 2.4756 | 1.0000 | 0.0000 | 0.0000 |
| Workflow | 60 | 0.9000 | 2.2801 | 0.9746 | 0.0000 | 0.0254 |

## 5. AgentDojo Admission Smoke

当前 AgentDojo 已完成两层验证：

1. **官方环境和 checker probe**：使用 AgentDojo suite 自身的 ground-truth pipeline 执行 20 对 workspace/slack smoke tasks，并调用官方 utility/security checker。结果为 `ok_rate=1.0000`、`user_ground_truth_utility_rate=1.0000`、`injection_ground_truth_security_rate=1.0000`，输出在 `outputs/agentdojo/full_harness_probe_20.json`。这证明当前环境下 AgentDojo suite、工具环境和 checker 可执行。
2. **SIEVE admission smoke**：从 AgentDojo 的 user task 和 injection task 构造 observation，测试 SIEVE policy 是否会把不可信 injection 写入 executable belief。

注意：这仍然不是完整端到端 task_success/security_success 评测，因为 learned action agent 和 SIEVE wrapper 还没有接入 AgentDojo action loop。

| 方法 | Pairs | Cases | Parse | Poison-to-State | Block Injection | Ignore Injection | Benign Update | Invalid Format |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT-only | 20 | 40 | 0.9500 | 0.0000 | 1.0000 | 0.8500 | 1.0000 | 0.0500 |
| SIEVE-GRPO | 20 | 40 | 0.9250 | 0.0000 | 1.0000 | 0.3500 | 1.0000 | 0.0750 |
| SIEVE-GiGPO | 20 | 40 | 0.9000 | 0.0000 | 1.0000 | 0.4500 | 1.0000 | 0.1000 |

解释：

- 三个模型都没有把 injection observation 写入状态，`poison_to_state_rate=0`。
- SFT 更倾向于直接 `IGNORE` injection；GRPO/GiGPO 更常使用 `HOLD`，这和内部 RL 学到的 verification 行为一致。
- 当前结果只能作为外部 admission 压力测试和 checker 可用性证明，不能替代 AgentDojo 官方完整环境中的 learned-agent task utility / security score。

## 6. 对论文写法的影响

建议论文使用以下表述：

- 内部结果证明 SIEVE 机制链路有效：SFT 不够、Auto-Write 会污染、RL 后能在闭环中稳定阻断 false update。
- 内部结果也暴露限制：强 Visible Rule Gate 已经能解决大多数内部场景，因此当前 benchmark 不足以证明 learned policy 不可替代。
- AgentDojo 目前只能作为 preliminary external admission smoke，并且 checker probe 证明官方环境和打分器可用。完整 claim 仍需要后续接入 executable agent loop，报告 learned-agent utility/security，而不是只报告 observation admission。

## 7. 下一步

1. 在论文中加入本表，并明确标注为 single-seed internal mechanistic result。
2. 补充 workflow failure analysis，说明剩余错误主要来自 JSON value 编码和 schema 输出协议。
3. 若时间允许，继续搭建 AgentDojo learned action-agent harness；否则在 limitation 中说明当前只完成 admission smoke 与 ground-truth checker probe。
