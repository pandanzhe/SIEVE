# Workflow 失败样本分析

本文档分析当前 SIEVE-GRPO 与 SIEVE-GiGPO 在 internal test 的 workflow 子集上的失败原因。分析来源：

- `outputs/analysis/workflow_failures/grpo_workflow_trace.jsonl`
- `outputs/analysis/workflow_failures/grpo_workflow_summary.json`
- `outputs/analysis/workflow_failures/gigpo_workflow_trace.jsonl`
- `outputs/analysis/workflow_failures/gigpo_workflow_summary.json`
- `outputs/baselines/auto_write/workflow_trace_summary.json`
- `outputs/baselines/visible_rule_gate/workflow_trace_summary.json`

## 1. 总体结果

| 方法 | Workflow Episodes | Success | Failures | 主要失败原因 |
|---|---:|---:|---:|---|
| Auto-Write / Append-All | 60 | 0.2833 | 43 | state mismatch，由错误写入导致污染 |
| Visible Rule Gate | 60 | 1.0000 | 0 | 无失败 |
| SIEVE-GRPO | 60 | 0.9000 | 6 | parse error + state mismatch |
| SIEVE-GiGPO | 60 | 0.9000 | 6 | parse error + state mismatch |

Auto-Write 的 workflow 失败说明直接写入 observation 会造成大量状态污染。GRPO/GiGPO 的失败则不同：它们没有出现 false update，主要是模型在应该写入正确 tool result 时生成了不合法 JSON，导致 executor 无法采纳 verified 或 authoritative value。

## 2. GRPO/GiGPO 失败模板分布

GRPO 与 GiGPO 的 6 个 workflow 失败分布一致：

| Scenario Template | Failures |
|---|---:|
| complex_stale_conflict | 2 |
| multi_field_dependency | 1 |
| delayed_contamination | 1 |
| no_auto_repair | 1 |
| verification_budget_choice | 1 |

所有失败都同时记录了：

- `parse_error`
- `state_mismatch`

这说明当前剩余错误不是策略把错误 observation 写入状态，而是正确值没有被成功提交。

## 3. 典型失败模式

### Case A: 长 tool_result 在 JSON patch value 中转义失败

示例 scenario：

- `rl:2a1277525d29f880`
- template: `multi_field_dependency`
- 失败字段：`tool_result`
- 成本：`invalid_format=1.0`, `verification=1.0`

轨迹要点：

1. 第一步 observation 是 authoritative `tool_result`。
2. 模型意图是 `UPDATE tool_result`。
3. `tool_result` 的 value 是很长的嵌套字符串，包含 JSON、Python dict 风格字符串、引号、反斜杠和 URL。
4. 模型输出不是一个合法的单个 JSON object，解析报错为 `Expecting ',' delimiter`。
5. executor 因 invalid format 没有提交该字段，最终 `tool_result` 仍为 `null`，导致任务失败。

这类失败不是 admission 语义错误，而是结构化输出协议无法稳定承载长字符串值。

### Case B: HOLD 后 verification 返回正确值，但 UPDATE 输出无法解析

示例 scenario：

- `rl:3967588b055bbeac`
- template: `delayed_contamination`
- 失败字段：`tool_result`

轨迹要点：

1. 初始 `tool_result` 是 unverified，但会影响后续 precondition。
2. 模型正确选择 `HOLD` 并请求 `verify_tool_result`。
3. verification 返回 `verified_current_value`。
4. 模型尝试 `UPDATE tool_result`，但 value 中包含长嵌套字符串，JSON 解析失败。
5. 后续 `workflow_record_status` 虽被更新为 `validated`，但前置 `tool_result` 缺失，最终 state mismatch。

这说明模型学到了“先验证再提交”的策略，但提交长 verified value 的表示方式仍不稳定。

### Case C: no_auto_repair 场景中过早失去关键字段

示例 scenario：

- `rl:5719b91d4e4e3339`
- template: `no_auto_repair`
- 失败字段：`tool_result`, `workflow_record_status`

轨迹要点：

1. 模型对 ambiguous `tool_result` 正确 `HOLD` 并触发 verification。
2. verification step 的 `UPDATE tool_result` 输出解析失败。
3. 后续不会自动再次修复该字段。
4. 模型之后对 authoritative `workflow_record_status` 选择了 `IGNORE`，最终两个 oracle 字段都缺失。

该 case 结合了输出协议失败和后续恢复不足。它比简单 parse error 更有价值，因为它展示了早期关键字段未提交会在后续 no-auto-repair 结构中放大成任务失败。

## 4. 与 Auto-Write 的区别

Auto-Write 的 workflow success 只有 28.33%，失败 43/60，主要是 `state_mismatch`。这对应论文主线中的状态污染：不可靠 observation 被直接写入 executable state 后，会影响后续前置条件或最终状态。

GRPO/GiGPO 的 workflow success 为 90.00%，失败 6/60，且 false update 为 0。它们的错误不是污染，而是：

```text
正确 admission intent
  -> 长字段 JSON 表达失败
  -> executor 拒绝 patch
  -> oracle field 缺失
  -> task failure
```

这说明 Stage-2 RL 已经基本学会 admission 策略，但当前 JSON-as-value 输出协议对长 tool result 不稳。

## 5. 对代码和数据的启示

短期可改进方向：

1. 对长字符串 patch value 使用更稳的编码方案，例如 value 引用、field payload id、base64/string literal wrapper，避免模型直接生成大量转义字符。
2. 对 `SET_VALUE` 的长 tool result 允许 executor 从当前 observation 复制原值，例如 `{"op":"COPY_OBSERVATION_VALUE","field_id":"tool_result"}`，减少模型复写长值。
3. 在 SFT 和 RL 数据中增加长工具结果、复杂嵌套字符串和 URL 的输出格式样本。
4. 把 workflow invalid-format 作为单独指标监控，而不要只看 false update。

论文写法上，应将 workflow 错误解释为实现层输出协议限制，而不是 SIEVE admission 思想失败。当前数据支持：

- admission boundary 能显著减少直接写入污染；
- 剩余失败集中在复杂 value serialization；
- 后续版本需要改 executor action schema，让模型裁决“是否提交”，而不是手写完整长值。

## 6. 当前建议

如果时间紧，不建议为这 6 个 workflow failure 重新训练模型。更合理的是：

1. 在论文误差分析中报告该现象；
2. 在方法部分把 `COPY_OBSERVATION_VALUE` 或 value-reference action 作为后续 schema 改进；
3. 若要再做一轮代码实验，优先改输出协议而不是继续增加 GRPO step。
