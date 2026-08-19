# SIEVE：开源多步 Agent 基准驱动的数据与训练方案

**重点：Stage-1 SFT 数据设计与检验；Stage-2 RL 方案概述**  
**基座模型：Qwen3-4B-Instruct-2507**  
**版本：2026-08-17**

## 摘要

SIEVE 研究的核心不是让模型直接完成某一个购物、电信或车载任务，而是在多步 Agent 执行过程中学习一个可复用的“观察评估策略”：面对新的用户消息、工具返回或环境变化，模型需要结合当前可执行信念、任务目标、证据账本、风险和预算，判断该观察应当 **UPDATE、HOLD 还是 IGNORE**，并输出受约束的局部状态修改。

数据应当来自具有真实多步状态、工具调用和任务约束的开源 Agent benchmark，但不能把所有 benchmark 混入训练。主训练来源建议限定为 **τ³-bench 的文本交互域**与 **CAR-bench 的 Base/Disambiguation Train 子集**；AgentDojo、ToolSandbox、InjecAgent、AppWorld 等保留为外部泛化或安全评测。Stage-1 使用约 6000 条单步监督记录建立协议和局部判断能力；Stage-2 使用训练任务派生的闭环 episode，通过受约束 GRPO 优化长期信念一致性与验证资源使用。

## 1. 论文问题与数据的适配性

### 1.1 SIEVE 学习的对象

设第 (t) 步进入观察评估策略的信息状态为：

$$
s_t^{rev}=(B_{t-1},o_t,g_t,r_t,L_t,\rho_t).
$$

其中：

- (B_{t-1})：当前动作执行前的可执行信念；
- (o_t)：当前新观察；
- (g_t)：任务目标或当前子目标；
- (r_t)：风险约束；
- (L_t)：证据账本；
- (\rho_t)：剩余步骤、工具、验证和 token 预算。

策略输出结构化认知动作：

$$
u_t=(c_t,z_t,\Delta B_t,v_t),
$$

其中 cₜ ∈ {UPDATE, HOLD, IGNORE}，zₜ 是受影响字段，ΔBₜ 是候选 patch，vₜ 是可选验证请求。模型只提出动作，确定性的 Executor 才是唯一状态写入点。

### 1.2 为什么必须使用多步 Agent 数据

单纯的文本分类数据只能教模型识别局部标签，不能回答以下问题：

1. 一个观察是否会覆盖已经 trusted 的旧信念；
2. HOLD 之后获得的新证据是否足以转为 UPDATE；
3. 错误 UPDATE 是否会在后续步骤造成连锁错误；
4. 验证预算是否被用在真正影响任务的字段上；
5. 模型在自身前一步动作导致的状态上能否继续正确决策。

因此，上游数据必须至少提供任务目标、状态、工具或环境结果以及可验证的任务约束。τ³-bench 和 CAR-bench 都满足这一条件：前者覆盖 airline、retail、telecom 等工具型客服任务，后者覆盖车载助手中的多轮工具使用、状态变化、歧义和能力边界。

### 1.3 当前本地数据的边界

当前 `data/sft` 仍是旧版数据。其 τ2 归一化逻辑通常从 `evaluation_criteria.actions` 中取第一个带参数的动作，再选一个字段作为信念修正目标；随后规则生成低权威观察、HOLD 和验证后的 UPDATE。这些样本在状态转移上连贯，但多数相邻步骤是规则构造的，并不等同于官方 benchmark 的原生相邻轨迹。

新数据应区分：

- `native_transition`：由官方任务状态、工具调用和工具返回直接回放得到；
- `synthetic_transition`：基于训练任务构造的低权威、错误实体、过期冲突等反事实观察；
- `transformation`：记录具体生成规则；
- `parent_task_id`：保证同一任务及其所有变体不会跨 train/dev/test。

这一区分是论文可信度的重要组成部分。

## 2. 开源数据选择与分工

### 2.1 τ³-bench：主要训练来源

τ³-bench 官方仓库仍名为 `sierra-research/tau2-bench`。当前版本支持 airline、retail、telecom、banking_knowledge 等域，并提供策略、工具、任务、状态与 Gymnasium 接口。本文只使用文本模式的 airline、retail 和 telecom；暂不使用 banking_knowledge 与语音域，从而把论文重心保持在多步 Agent 决策与状态修正，而不是知识检索或语音建模。

τ³-bench 主要提供：

- 权威工具结果后的 UPDATE；
- 用户报告或低权威信息触发的 HOLD+VERIFY；
- 验证结果返回后的 UPDATE；
- 旧状态、错误实体和任务外观察对应的 IGNORE；
- 可用于 Stage-2 的多步环境状态和任务终局条件。

### 2.2 CAR-bench：歧义与行动边界

CAR-bench 是面向车载助手的多轮工具 Agent benchmark，包含 Base、Hallucination 和 Disambiguation 三类任务。其 Base 和 Disambiguation Train 子集适合训练“什么时候可以行动、什么时候需要补充信息”的边界。

当前训练只使用：

- Base Train；
- Disambiguation Train。

Hallucination 子集暂不加入训练。它要求 Agent 明确承认工具、参数或结果不可用，而当前动作空间没有 DEFER/SAFE_STOP。若强行标为 IGNORE 或无验证的 HOLD，会破坏已有动作语义。Hallucination 可以保留为诊断；若后续将其设为主任务，应单独扩展安全退出动作。

### 2.3 不进入训练的外部 benchmark

AgentDojo、ToolSandbox、InjecAgent 和 AppWorld 不混入训练。它们分别用于间接提示注入、有状态工具依赖、恶意工具观察和长程应用任务上的外部泛化测试。这样可以避免 benchmark 污染，并使论文能够回答“策略是否跨环境泛化”，而不仅是“是否记住训练模板”。

## 3. Stage-1 SFT 数据方案

### 3.1 数据规模与配比

建议把 6000 定义为 Stage-1 SFT 总量：5400 条 train，600 条 dev；正式 test 使用从未参与生成和调参的官方 held-out split。

| 来源 | 数量 | 比例 | 作用 |
|---|---:|---:|---|
| τ³-bench Train | 3900 | 65% | 多步客服、工具状态与验证结果 |
| CAR-bench Base/Disambiguation Train | 1500 | 25% | 歧义、澄清、策略和状态依赖 |
| 训练任务派生的安全反事实 | 600 | 10% | 错误实体、过期冲突、越权与干扰观察 |
| 合计 | 6000 | 100% | - |

反事实样本不是第三个上游 benchmark，必须继承其 parent task 的来源和 split。

建议动作分布为：

| 动作 | 数量 | 比例 | 语义 |
|---|---:|---:|---|
| UPDATE | 2700 | 45% | 证据足够，可以提交局部信念修改 |
| HOLD+VERIFY | 1800 | 30% | 与任务相关，但需要工具查询或用户澄清 |
| IGNORE | 1500 | 25% | 对当前任务状态没有可接受影响，不应进入信念 |

当前不生成无验证请求的普通 HOLD，因为现有闭环环境会将其计为 stall。

### 3.2 单条训练样本的输入与输出

每条 SFT 记录都是以下映射：

$$
(B_{t-1},o_t,g_t,r_t,L_t,\rho_t)\longrightarrow u_t^*.
$$

训练 Prompt 为“统一 System Prompt + context 的规范序列化”；Answer 为 target 的紧凑 JSON 加 EOS。Prompt token 的 label 设为 -100，语言模型 loss 只作用在答案 token 上。`scenario_id` 和 `step_index` 用于保存时序和分组，但两个步骤仍各自独立计算 SFT loss。

如果一条记录是 `step_index=1`，模型不会依赖上一个 batch 的隐藏状态，而是直接从当前 Prompt 读取上一步执行后的 `belief_state`、`ledger` 和 `budget`。这相当于用一个显式信息状态概括历史。

#### 模型第一次决策的“认知”来自哪里

`belief_state` 中某个字段为空，只表示本轮任务尚未提交该字段，并不表示模型没有经验。第一次任务内决策同时依赖三类信息：

- **参数化经验**：Qwen3 预训练和 SFT 已经学到来源权威、验证需求、动作协议与常见风险模式；
- **系统与工具契约**：System Prompt 给出 UPDATE/HOLD/IGNORE 的语义和门控，环境给出可用验证工具；
- **当前任务信息状态**：目标、实体、观察值、来源、认证信息、风险、初始信念、账本和预算均在 Prompt 中显式提供。

因此，模型不是只看到一个值就直接分类。以 `first_name=Yusuf` 为例，完整判断链是：任务需要该字段，实体匹配，当前字段未知，但 Yusuf 来自未认证来源，同时存在合法验证工具，所以选择 HOLD+VERIFY。如果 Prompt 只包含“字段为空”和“Yusuf”，而没有目标、实体、来源和工具契约，那么动作本身是不确定的，不能据此构造合格训练样本。

为避免负下标造成误解，可把一条 episode 写成：

```text
初始状态 B₀
(B₀, o₁, g₁, r₁, L₀, ρ₁) → u₁ = HOLD
Executor 得到 B₁

(B₁, o₂, g₂, r₂, L₁, ρ₂) → u₂ = UPDATE
Executor 得到 B₂
```

数据文件仍使用从 0 开始的 `step_index=0/1`；数学下标表示状态转移顺序，二者不要混为一谈。初始 (B_0) 由任务/环境初始化器提供，不依赖模型记住更早的对话；后续历史则通过 (B_t)、(L_t) 和预算显式压缩到下一步 Prompt。

### 3.3 两条相邻 SFT 样本及来源

下面两条样本来自同一个 τ³-bench Telecom 任务的转换设计，用于展示相邻状态如何进入单步 SFT。为了避免把“开源事实来源”和“SIEVE 训练格式”混为一谈，先给出上游原始任务，再给出固定 System Prompt、Inputs 和 Target/GT。

#### 3.3.1 上游原始任务

- 官方仓库：`https://github.com/sierra-research/tau2-bench`
- 官方数据文件：`data/tau2/domains/telecom/tasks.json`
- 本地快照：`data/raw/tau2-bench/data/tau2/domains/telecom/tasks.json`
- 上游任务 ID：`[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]`
- 当前本地文件 SHA-256：`20793073500DBA5761F76D038DD32A343903985AF442B3EAE7EEB3A4E0D5D75D`

以下是原始任务中与本示例直接相关的字段摘录。被保留的文本值沿用上游英文原文；`task_instructions_excerpt` 是为排版新增的摘录标签，不是上游字段名；其余 `null` 字段和非关键说明被省略。

```json
{
  "id": "[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]",
  "description": {
    "purpose": "Test resolution path: Mobile Data/Slow Internet Issues."
  },
  "user_scenario": {
    "instructions": {
      "domain": "telecom",
      "reason_for_call": "You mobile data is not working properly. It either stops working or is very slow. You want to fix it and absolutely want to get excellent internet speed on your phone. You are not willing to accept any other internet speed (poor, fair or good). You do not have access to wifi.",
      "known_info": "You are John Smith with phone number 555-123-2002. You are currently abroad in France.",
      "task_instructions_excerpt": "If the agent asks what the status bar shows, always ground your response on the results of the get_status_bar tool call. Never make up the results of tool calls, always ground your responses on the results of tool calls."
    }
  },
  "ticket": "The user is experiencing issues with their mobile data. They are unable to use their phone to browse the internet, and the status bar shows 'No Service'. Customer name: John Smith, phone number: 555-123-2002, current location: abroad in France. They will consider the issue resolved when speed test returns excellent internet speed. They will not change their mobile data plan but they will refuel 2.0 GB of data if necessary.",
  "initial_state": {
    "initialization_actions": [
      {"env_type": "user", "func_name": "set_user_info", "arguments": {"name": "John Smith", "phone_number": "555-123-2002"}},
      {"env_type": "user", "func_name": "set_user_location", "arguments": {"abroad": true}},
      {"env_type": "user", "func_name": "turn_roaming_off", "arguments": {}},
      {"env_type": "assistant", "func_name": "enable_roaming", "arguments": {"customer_id": "C1001", "line_id": "L1002"}}
    ]
  },
  "evaluation_criteria": {
    "actions": [
      {"action_id": "toggle_roaming_0", "requestor": "user", "name": "toggle_roaming", "arguments": {}}
    ],
    "env_assertions": [
      {"env_type": "user", "func_name": "assert_mobile_data_status", "arguments": {"expected_status": true}, "assert_value": true},
      {"env_type": "user", "func_name": "assert_internet_speed", "arguments": {"expected_speed": 200, "expected_desc": "excellent"}, "assert_value": true}
    ],
    "reward_basis": ["ENV_ASSERTION"]
  }
}
```

这个上游 JSON 提供任务、实体、用户位置、初始 roaming 状态、`get_status_bar` 工具约束和终局条件，但没有提供现成的 `HOLD → 工具返回 → UPDATE` 对话轨迹。因此，下面两条记录应标记为 `synthetic_transition` 或 `rule_grounded_pair`：事实锚点来自 τ³-bench，状态与 gold 动作由 SIEVE 规则和 Executor 构造。它们不是官方文件中的逐步轨迹，也不能直接把转换后的单步准确率写成 τ³ 官方成功率。官方文档还说明，`evaluation_criteria.actions` 表示一条参考解法，而不是唯一合法轨迹。

#### 3.3.2 固定 System Prompt

当前训练代码在 `src/sieve/policies/hf_data.py` 中维护唯一的 `SIEVE_SYSTEM_PROMPT`。它不会重复存入每条 JSONL，而是在 collator 中与当前 `context` 组合，再通过 Qwen chat template 编码。

```text
You are the observation evaluation policy of a multi-step agent.

Given the current executable belief state, task goal, new observation, risk constraints,
evidence ledger, and remaining budget, produce exactly one structured cognitive action.

Decision definitions:
- UPDATE: the observation applies to the current task and has sufficient support to modify the executable belief state.
- HOLD: the observation may affect the current task, but current evidence is insufficient for safe commitment.
- IGNORE: the observation has no admissible effect on the current task state and does not require verification.

Constraints:
1. UPDATE must contain local executable patches.
2. HOLD must preserve the current executable value. It may mark an affected field as pending and request verification.
3. IGNORE must contain no affected fields, patches, or verification request.
4. Do not modify fields outside affected_fields.
5. Return valid JSON only, without explanations.
```

逻辑训练样本可以写成 `System Prompt + Inputs → Target/GT + EOS`。Qwen3 实际接收的 token 序列由 tokenizer 的 chat template 生成；Prompt 部分的 label 全部为 `-100`，只有 Target/GT 与 EOS token 参与语言模型交叉熵。`scenario_id` 和 `step_index` 是数据管理元信息，不放入策略的 user message。

#### 3.3.3 示例一：用户报告不足以确认设备状态 - HOLD+VERIFY

**System Prompt**

使用 3.3.2 中的固定 `SIEVE_SYSTEM_PROMPT`。

**Inputs（user message，即部署可见 context 的易读展开）**

```json
{
  "belief_state": {
    "max_slots": 8,
    "slots": [
      {
        "entity": "mobile_device:555-123-2002",
        "id": "status_bar",
        "value": null,
        "status": "empty",
        "source": "initial_state",
        "observed_at": 1763300000,
        "valid_from": null
      },
      {
        "entity": "mobile_device:555-123-2002",
        "id": "__verification_policy",
        "value": {
          "required_authority": "primary_device_state",
          "tool_by_field": {"status_bar": "get_status_bar"}
        },
        "status": "trusted",
        "source": "scenario_definition",
        "observed_at": 1763300000,
        "valid_from": 1763300000
      }
    ]
  },
  "goal": "Restore mobile data for the user abroad and confirm excellent speed.",
  "observation": {
    "entity": "mobile_device:555-123-2002",
    "field_id": "status_bar",
    "value": "The user reports that the phone shows No Service in France.",
    "source": "authenticated_user_report",
    "source_authority": "subjective_report",
    "authenticated": true,
    "observed_at": 1763300001,
    "valid_from": 1763300001
  },
  "risk": {
    "active_subgoal": "Confirm the current device network state before changing settings.",
    "dependent_fields": ["status_bar"],
    "reversible": true,
    "risk": "medium"
  },
  "ledger": {"capacity": 4, "entries": []},
  "budget": {
    "steps_remaining": 8,
    "tokens_remaining": 4096,
    "tool_remaining": 5,
    "verification_remaining": 2
  }
}
```

**Target/GT（assistant message）**

```json
{
  "decision": "HOLD",
  "affected_fields": ["status_bar"],
  "patches": [
    {"op": "SET_STATUS", "field_id": "status_bar", "value": "pending_verification"}
  ],
  "verification": {"tool": "get_status_bar", "field_id": "status_bar"}
}
```

这里不能 IGNORE，因为网络状态直接影响当前子目标；也不能直接 UPDATE 为 trusted，因为“No Service”目前只是用户报告。输入中的 `__verification_policy` 明确告诉模型：`status_bar` 需要 `primary_device_state` 权威等级，并应调用 `get_status_bar`。因此动作由完整上下文唯一确定，不要求模型凭空猜测工具名。

用于生成、分组和审计的 provenance 清单建议额外保存：

```json
{
  "scenario_id": "tau3:telecom:roaming-enabled-off:episode-0001",
  "step_index": 0,
  "transition_type": "synthetic_transition",
  "parent_task_id": "[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]"
}
```

#### 3.3.4 示例二：工具返回后提交信念 - UPDATE

**System Prompt**

仍使用 3.3.2 中完全相同的固定 `SIEVE_SYSTEM_PROMPT`。

**Inputs（第一步动作经 Executor 执行后的下一状态）**

```json
{
  "belief_state": {
    "max_slots": 8,
    "slots": [
      {
        "entity": "mobile_device:555-123-2002",
        "id": "status_bar",
        "value": null,
        "status": "pending_verification",
        "source": "initial_state",
        "observed_at": 1763300000,
        "valid_from": null
      },
      {
        "entity": "mobile_device:555-123-2002",
        "id": "__verification_policy",
        "value": {
          "required_authority": "primary_device_state",
          "tool_by_field": {"status_bar": "get_status_bar"}
        },
        "status": "trusted",
        "source": "scenario_definition",
        "observed_at": 1763300000,
        "valid_from": 1763300000
      }
    ]
  },
  "goal": "Restore mobile data for the user abroad and confirm excellent speed.",
  "observation": {
    "entity": "mobile_device:555-123-2002",
    "field_id": "status_bar",
    "value": "No Service",
    "source": "get_status_bar",
    "source_authority": "primary_device_state",
    "authenticated": true,
    "observed_at": 1763300003,
    "valid_from": 1763300003
  },
  "risk": {
    "active_subgoal": "Confirm the current device network state before changing settings.",
    "dependent_fields": ["status_bar"],
    "reversible": true,
    "risk": "medium"
  },
  "ledger": {
    "capacity": 4,
    "entries": [
      {
        "observation": {
          "entity": "mobile_device:555-123-2002",
          "field_id": "status_bar",
          "value": "The user reports that the phone shows No Service in France.",
          "source": "authenticated_user_report",
          "source_authority": "subjective_report",
          "authenticated": true,
          "observed_at": 1763300001,
          "valid_from": 1763300001
        },
        "reason": "held_for_verification"
      }
    ]
  },
  "budget": {
    "steps_remaining": 7,
    "tokens_remaining": 3840,
    "tool_remaining": 4,
    "verification_remaining": 1
  }
}
```

**Target/GT（assistant message）**

```json
{
  "decision": "UPDATE",
  "affected_fields": ["status_bar"],
  "patches": [
    {"op": "SET_VALUE", "field_id": "status_bar", "value": "No Service"}
  ],
  "verification": null
}
```

第二条 Inputs 中的 `status=pending_verification`、证据账本和减少后的预算，都是第一步 gold HOLD 经 Executor 执行后的结果；新的 `observation` 才是验证工具返回。模型在 SFT 时分别学习两条映射；在 RL 或部署时，则由模型自己的第一步输出、Executor 和工具环境真正产生第二步状态。

第二步在 provenance 清单中的元信息建议写为：

```json
{
  "scenario_id": "tau3:telecom:roaming-enabled-off:episode-0001",
  "step_index": 1,
  "transition_type": "synthetic_transition",
  "parent_task_id": "[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]"
}
```

实际训练文件仍应保存 `context` 和 `target`，而不是预先拼接一整段字符串。这样既能复用同一个 System Prompt，也能在更换 Qwen tokenizer 时由 chat template 正确插入系统、用户、助手和 EOS 边界。

### 3.4 数据生成原则

1. **事实锚点来自开源任务。** 任务目标、实体、工具、初始状态和可验证结果必须来自固定版本的 CAR/τ³ 数据。
2. **Gold 动作优先由规则确定。** 根据实体匹配、时间、来源权威、任务依赖和官方状态生成 UPDATE/HOLD/IGNORE，不让另一个语言模型自由决定标签。
3. **自然语言可改写，标签不可改写。** 如使用模型扩写观察文本，只能改变表面表达，结构化事实和 target 保持不变。
4. **回放形成下一状态。** (B_t) 必须由 Executor 执行 (u_t^*) 得到，不能手工复制一个看似合理的状态。
5. **私有信息不得进入 Prompt。** oracle、gold decision、perturbation 和规则 reason code 只用于生成与审计。
6. **按 parent task 分组切分。** 同一任务的所有步骤和反事实变体只能进入一个 split。
7. **System Prompt 与样本数据解耦。** JSONL 保存 `context` 和 `target`；collator 统一注入 System Prompt 并调用基座模型 chat template。
8. **工具契约必须对策略可见。** 若 gold HOLD 要求特定工具，Inputs 必须提供工具名、字段映射和所需权威等级，不能让模型从 target 泄漏或凭空猜测。
9. **来源类型必须可审计。** 原生回放与规则转换分别标为 `native_transition` 和 `synthetic_transition`，并在独立 provenance 清单中保存上游任务 ID、转换规则和文件哈希。

## 4. Stage-1 模型与训练目标

### 4.1 基座模型

主模型采用 `Qwen/Qwen3-4B-Instruct-2507`。该模型约 4B 参数、只支持 non-thinking 模式，适合稳定生成严格 JSON；官方模型卡也强调了指令遵循和工具调用能力。训练继续采用 LoRA，不修改现有“基础模型 + LoRA adapter + 结构化输出协议”的总体框架。

推荐把模型目录保持为可配置的 `model/`，服务器上放入完整 checkpoint；配置文件不写死服务器绝对路径。Transformers 版本不得低于 4.51.0。

### 4.2 SFT 的本质目标

Stage-1 不是普通三标签分类。模型需要在给定状态下生成一整个可执行动作：

```json
{
  "decision": "UPDATE|HOLD|IGNORE",
  "affected_fields": [],
  "patches": [],
  "verification": null
}
```

核心目标是：

- 学会三个决策的边界；
- 稳定遵守 JSON 协议和动作门控；
- 准确定位受影响字段；
- 在 UPDATE 时复制或生成正确 patch value；
- 在 HOLD 时选择合法验证工具；
- 避免对错误实体、过期值或越权字段产生 false update。

### 4.3 Stage-1 如何检验

仅观察 teacher-forced dev loss 不足以判断模型可进入 RL。评估应分为四层。

#### 第一层：数据和目标可执行性

- JSON Schema 通过率 = 100%；
- gold target 的 Executor executable rate = 100%；
- train/dev/test parent task 重叠 = 0；
- Prompt 私有标签泄漏 = 0；
- native/synthetic provenance 完整率 = 100%。

#### 第二层：teacher-forced 训练诊断

- 总 dev loss；
- decision loss；
- verification 分支 loss；
- patch value token loss；
- train/dev 曲线与过拟合差距。

这些指标用于诊断优化是否收敛，但不能代替自由生成评估。

#### 第三层：greedy 自由生成

对 best 和 final adapter 分别在 dev 上完整生成 JSON。建议最低门槛：

| 指标 | 门槛 |
|---|---:|
| JSON parse rate | ≥ 0.98 |
| Executor executable rate | ≥ 0.97 |
| Decision macro-F1 | ≥ 0.90 |
| 完整动作 exact match | ≥ 0.85 |
| UPDATE patch-value exact match | ≥ 0.85 |
| False-update rate | ≤ 0.03 |
| HOLD→UPDATE 成对轨迹全部正确率 | ≥ 0.80 |

成对正确率要求同一 scenario 的 HOLD、验证请求、下一状态和 UPDATE 全部正确，比单独统计两个样本准确率更贴近论文目标。

#### 第四层：短闭环 readiness

Stage-1 的直接训练目标仍是单步动作，但进入 RL 前必须运行短闭环诊断，因为第二步输入由第一步模型动作决定，而不再是 gold 状态。建议检查：

- episode parse rate ≥ 0.95；
- 闭环成功率 ≥ 0.60；
- group reward variance > 0；
- 闭环 false-update rate 单独报告。

闭环指标不是要求 SFT 已经解决完整长期规划，而是防止在协议或状态传递尚未可用时直接启动 RL。

## 5. Stage-2 RL 数据与目标

### 5.1 RL 解决什么问题

SFT 使用 gold context，不能充分训练错误动作的长期后果。Stage-2 的目标是让策略在自己的动作产生的后续状态上，最大化终局可执行信念正确率，同时控制错误更新、无效验证、停滞和预算违例。

RL 不重新定义输出协议，也不训练一个独立的小 MLP 策略。策略仍是 Stage-1 的 Qwen3-4B LoRA policy；环境负责生成观察和状态转移，Executor 负责约束写入，冻结的 Stage-1 adapter 作为 reference policy。

### 5.2 RL 场景构建

RL 只从训练侧 parent tasks 派生，不使用 CAR/τ³ 的官方 held-out test。保持当前规模即可：

- train：约 1600 个多步 scenario；
- dev：约 200 个 scenario；
- 官方 benchmark held-out：作为最终评测，不人工补齐固定数量。

一个典型 episode 包含：

```text
低权威相关观察
→ HOLD + VERIFY
→ 权威证据返回
→ UPDATE
→ 错误实体或过期冲突
→ IGNORE
→ 终局信念一致
```

每个 scenario 应保存初始信念、目标、事件序列、验证工具契约、私有 oracle、预算和来源信息。Oracle 只用于环境计分，不进入策略 Prompt。

### 5.3 状态、动作和转移

RL 状态沿用：

$$
s_t=(B_{t-1},o_t,g_t,r_t,L_t,\rho_t).
$$

动作沿用结构化 JSON。状态转移由两部分组成：

$$
B_t=U(B_{t-1},u_t,o_t),
$$

$$
(o_{t+1},y_t,x_{t+1})\sim P(\cdot\mid x_t,a_t,u_t).
$$

其中 (U) 是确定性 Executor；(P) 是 benchmark 环境、工具或数据场景定义的转移。信念向量维度不随时间无限增长，使用固定 `max_slots`；时间变化体现为槽位值、状态、时间戳和有限容量 ledger 的更新。

### 5.4 奖励和受约束 GRPO

任务依赖字段集合为 (D)，信念与私有 oracle 的一致率为：

$$
\Phi(B_t,x)=\frac{1}{|D|}\sum_{f\in D}
\mathbb{1}[B_t(f)=x(f)\land status_t(f)=trusted].
$$

单步任务奖励采用势函数差，并在终局成功时加分：

$$
r_t=\gamma\Phi(B_t,x)-\Phi(B_{t-1},x)
+\mathbb{1}[\text{terminal success}].
$$

轨迹还累计 false update、unsafe action、verification、stall、invalid format、invalid patch、collateral edit 和 budget violation。受约束分数为：

$$
\widetilde R_i=R_i-\sum_j\lambda_j C_{ij}.
$$

对同一初始场景采样多条完整轨迹，使用组内标准化优势进行 GRPO 更新，不训练额外 value model。高回报、低代价的轨迹得到正优势；直接错误 UPDATE、错误工具或无效拖延得到负优势。

## 6. 最终实验边界

论文应分别报告三类结果：

1. **SIEVE 静态动作指标**：在 CAR-derived/τ³-derived 状态上的 JSON、decision、patch 和 false-update 指标；
2. **SIEVE 闭环指标**：多步信念一致率、终局成功率、验证成本与约束违例；
3. **官方 benchmark 指标**：在接入原生行动策略和工具环境后报告 CAR Pass^k、τ³ task success 等官方指标。

转换后的单步准确率不能直接称为 CAR 或 τ³ 官方成功率。训练集与官方 test 必须通过 task ID、哈希和来源清单进行隔离。

## 7. 推荐执行顺序

1. 固定 CAR-bench 与 τ³-bench 版本、许可证和原始文件哈希；
2. 编写原生状态/事件适配器，优先回放真实相邻事件；
3. 构造约 6000 条 Stage-1 数据并按 parent task 分组切分；
4. 完成数据审计和 Executor 重放检查；
5. 使用 Qwen3-4B-Instruct-2507 进行 LoRA SFT；
6. 对 best/final 执行 teacher-forced、自由生成、成对轨迹和短闭环评估；
7. 只有 readiness 门槛通过后，才从训练任务构造 Stage-2 GRPO scenarios；
8. 最终在完全隔离的官方 benchmark 和外部 Agent 安全集上评测。

## 参考来源

- CAR-bench: https://github.com/CAR-bench/car-bench
- τ³-bench / tau2-bench: https://github.com/sierra-research/tau2-bench
- τ³-bench Telecom tasks: https://github.com/sierra-research/tau2-bench/blob/main/data/tau2/domains/telecom/tasks.json
- τ³-bench task evaluation semantics: https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md
- Qwen3-4B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- AgentDojo: https://github.com/ethz-spylab/agentdojo
- ToolSandbox: https://github.com/apple/ToolSandbox
- InjecAgent: https://github.com/uiuc-kang-lab/InjecAgent
- AppWorld: https://github.com/StonyBrookNLP/appworld
