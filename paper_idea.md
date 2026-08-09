# SIEVE：基于观察评估与受约束信念修正的多步 Agent 决策框架

> 本文档先固定方法的理论对象、状态与动作语义，再给出训练目标、实验数据和当前代码映射。研究重点是多步 Agent 在持续接收外部观察时，如何控制信息进入可执行状态；事实核查或 RAG 只可作为观察来源，不构成论文的主要问题定义。

## 1. 研究问题

多步 Agent 在执行任务时会持续接收工具返回、环境反馈、用户补充和异步事件。现有系统常把这些内容直接追加到上下文，或者用自由文本摘要覆盖旧状态。这样做隐含了一个很强的假设：新观察一旦出现，就可以参与后续行动。实际环境并不满足这一假设。观察可能过时、指向错误实体、缺少适用条件、与已确认状态冲突，或者只在完成额外验证后才足以支持行动。

SIEVE 在观察与环境行动之间加入一个可学习的观察评估策略，并把状态写入交给确定性的信念执行器。框架回答四个问题：当前观察应当被接纳、暂缓还是忽略；观察可能影响哪些状态字段；允许提交什么局部状态补丁；信息不足时是否值得发起一次有成本的验证。

框架不学习完整的世界模型，也不允许评估策略直接修改任意 Agent 内存。它维护的是面向当前任务的可执行信念，即 Agent 此刻可以据以行动的结构化内部状态。

## 2. 建模结论：整体是 POMDP，评估层可近似为信念状态 MDP

### 2.1 为什么不能直接把整体写成普通 MDP

令环境在时刻 $t$ 的真实状态为 $x_t$。它可以包含工具端最新数据、网页当前内容、外部任务进度和其他 Agent 不可直接读取的变量。Agent 只接收由观察模型产生的 $o_t$：

$$
o_t \sim P_{\mathrm{obs}}(\cdot \mid x_t,a_{t-1})
$$

环境状态按下式转移：

$$
x_{t+1} \sim P_{\mathrm{env}}(\cdot \mid x_t,a_t)
$$

因为策略看不到 $x_t$，可观测输入并不是完整环境状态。即使代码中只把“上一步对象”传给 `step`，也不能据此认定问题天然满足 MDP。MDP 的关键不是函数签名，而是当前状态是否包含预测未来所需的全部历史信息。因此，整体任务更准确地属于部分可观测马尔可夫决策过程：

$$
\mathcal M_{\mathrm{POMDP}}=(\mathcal X,\mathcal O,\mathcal A,P_{\mathrm{env}},P_{\mathrm{obs}},R,\gamma)
$$

### 2.2 在什么条件下可以使用 MDP 训练

SIEVE 不直接恢复完整 $x_t$，而是维护可执行信念 $B_t$、证据账本 $L_t$ 和必要的固定长度轨迹摘要 $m_t$。若这些变量对历史构成充分统计量，则评估层的信息状态可以近似满足：

$$
P(s_{t+1}^{\mathrm{rev}}\mid H_t,u_t)\approx P(s_{t+1}^{\mathrm{rev}}\mid s_t^{\mathrm{rev}},u_t)
$$

其中 $H_t$ 是截至当前的完整历史。论文应采用以下表述：

> 底层交互环境建模为受约束 POMDP。SIEVE 通过结构化可执行信念、证据账本和有限历史摘要构造近似充分的信息状态，并在观察评估层使用 belief-state MDP 近似进行策略学习。

这比“整个过程就是 MDP”更稳健。如果删除 $L_t$ 或 $m_t$ 会明显降低延迟纠错、乱序观察等任务的表现，恰好说明仅靠 $B_t$ 不能满足充分性假设。

### 2.3 验证微循环与时间尺度

验证可能消耗多个工具调用或不同的真实时间，因此理论上也可把整个系统写成带约束的 SMDP。第一版没有必要引入完整 options 框架。更清楚的处理是使用两个索引：

- $t$：外层观察评估步。每完成一次观察的最终评估，$t$ 增加；
- $k$：评估步内部的验证索引。一次 `HOLD + VERIFY` 会增加 $k$，但在证据返回并重新评估前，外层 $t$ 保持不变。

验证微循环写为：

$$
s_{t,0}^{\mathrm{rev}}\rightarrow u_{t,0}^{\mathrm{HOLD+VERIFY}}\rightarrow e_{t,1}\rightarrow L_{t,1}\rightarrow s_{t,1}^{\mathrm{rev}}\rightarrow u_{t,1}
$$

验证证据 $e_{t,k}$ 先写入账本，再参与重新评估。它不能绕过评估策略直接写入 $B_t$。微循环在证据充分、预期信息增益过低、工具不可用、选择 `DEFER` 或预算耗尽时停止。

## 3. 状态、观察与信息边界

### 3.1 可执行信念

可执行信念由固定上限的类型化 slot 组成：

$$
B_t=\{b_t^1,\ldots,b_t^{K_{\max}}\}
$$

每个 slot 至少包含字段标识、值、状态、实体、来源和时间信息：

```json
{
  "id": "payment_status",
  "value": "paid",
  "status": "trusted",
  "entity": "order_1842",
  "source": "payment_api",
  "observed_at": 1784112000,
  "valid_from": 1784112000
}
```

$B_t$ 不是客观真值，也不是模型参数中的世界知识。它是 Agent 对当前任务相关变量的结构化估计，并且是行动策略可读取的唯一高风险信息入口。

### 3.2 当前观察与评估状态

一条原子观察定义为：

$$
o_t=(v_t,m_t^o)
$$

其中 $v_t$ 是观察内容，$m_t^o$ 是来源、时间、实体、适用条件和工具状态等元信息。环境一次返回多项内容时，适配器先按确定顺序拆为原子观察，再逐条评估。数据中必须显式保存 `step_index`。

评估策略可见的信息状态为：

$$
s_t^{\mathrm{rev}}=(B_{t-1},o_t,g_t,r_t,L_t,m_t,\rho_t)
$$

| 变量 | 含义 | 是否输入评估策略 |
|---|---|---|
| $x_t$ | 环境潜在真实状态 | 否，仅训练标签、奖励和评测可用 |
| $B_{t-1}$ | 上一步可执行信念 | 是 |
| $o_t$ | 当前原子观察及元信息 | 是 |
| $g_t$ | 当前任务目标与活跃子目标 | 是 |
| $r_t$ | 风险包络：依赖字段、动作风险、可逆性 | 是 |
| $L_t$ | 有界证据账本 | 是 |
| $m_t$ | 可选的固定长度历史摘要 | 是 |
| $\rho_t$ | 可选验证预算 | 仅预算实验启用 |

风险包络只描述后续行动依赖的字段和风险，不包含由未准入观察直接生成的具体候选动作，从而避免循环依赖。

信念执行器完成写入后，行动策略读取：

$$
s_t^{\mathrm{act}}=(B_t,g_t,r_t)
$$

必要的低风险环境上下文应先转为 $B_t$ 中的字段，或显式加入 $s_t^{\mathrm{act}}$。高风险观察不能绕过评估阶段直接输入行动策略。

## 4. 信念不会随时间增加向量维度

时间步增加时，状态内容会变化，接口维度不增加。

1. 每个任务 schema 设定固定的 $K_{\max}$，未使用 slot 由 mask 标记；
2. 新字段通过 `ADD_FIELD` 激活一个空 slot，而不是扩张向量；
3. 证据账本满足 $|L_t|\le L_{\max}$，满载后按已解决、无关、年龄和任务相关性组成的确定规则淘汰；
4. 当前观察 $o_t$ 每步替换，历史只通过 $B_t$、$L_t$ 和可选的固定长度摘要 $m_t$ 保留；
5. 时间影响通过 `observed_at`、`valid_from`、相对年龄和有效期特征编码；
6. 文本 token 数可在上下文上限内变化，但池化表示 $h_t\in\mathbb R^d$ 的维度固定。

因此，“ADD”表示增加一个语义字段，不表示增加神经网络输入的维数。若没有空 slot，执行器必须拒绝 `ADD_FIELD`，并记录 `invalid_patch`。

## 5. 认知动作空间：三类评估加结构化子动作

观察准入只保留三类：

$$
z_t\in\{\mathrm{UPDATE},\mathrm{HOLD},\mathrm{IGNORE}\}
$$

- `UPDATE`：现有信息足以授权一次局部信念修改；
- `HOLD`：观察可能相关，但当前证据不足以覆盖可信状态；
- `IGNORE`：观察无关、过期、对象错误、条件不适用，或已被更强且更新的证据否定。

`ADD` 不是第四种评估类别。它是 `UPDATE` 分支中的补丁操作。`VERIFY` 和 `DEFER` 也不是与三类评估并列的环境动作，而是 `HOLD` 的条件子动作。

完整认知动作定义为：

$$
u_t=(z_t,A_t,\Delta B_t,q_t)
$$

其中：

$$
A_t\in\{0,1\}^{K_{\max}}
$$

表示受影响字段的多标签 mask；

$$
\Delta B_t=\{(op,slot,value,metadata)_j\}_{j=1}^{M_t},\quad M_t\le M_{\max}
$$

允许的补丁操作为 `ADD_FIELD`、`SET_VALUE`、`SET_STATUS`、`SET_PROVENANCE` 和 `SET_VALIDITY`。验证子动作定义为：

$$
q_t\in\{\mathrm{NO\_VERIFY},\mathrm{DEFER}\}\cup(\mathcal T_{\mathrm{verify}}\times\mathcal F)
$$

动作门控如下：

| 顶层动作 | 允许的字段和补丁 | 允许的验证动作 |
|---|---|---|
| `UPDATE` | 非空 $A_t$，合法局部补丁 | `NO_VERIFY` |
| `HOLD` | 可写 pending 状态或候选证据，但不能覆盖 trusted value | `VERIFY`、`DEFER` 或 `NO_VERIFY` |
| `IGNORE` | 空 $A_t$，空补丁 | `NO_VERIFY` |

策略分布可分解为：

$$
\pi_\theta(u_t\mid s_t^{\mathrm{rev}})=\pi_\theta(z_t\mid h_t)\,\pi_\theta(A_t\mid z_t,h_t)\,\pi_\theta(\Delta B_t\mid A_t,z_t,h_t)\,\pi_\theta(q_t\mid z_t,h_t)
$$

实现 log probability 时，对有效分支求和；无效分支通过 action mask 排除，不能进入概率比或训练损失。

## 6. 信念执行器：唯一状态写入点

评估策略只提出状态修改建议。信念执行器 $U$ 负责门控、schema 校验、局部补丁应用、一致性检查和溯源约束：

$$
B_t=U(B_{t-1},u_t,e_t)
$$

基础实现采用确定性执行器。对同一个旧状态、认知动作和验证证据，输出唯一。可用下式表达约束集合：

$$
B_t=\arg\max_{B\in\mathcal C(B_{t-1},u_t,e_t)}\mathrm{Score}(B)
$$

这只是理论表达，不要求工程实现运行数值优化器。代码可以用规则执行器、JSON Patch 校验和约束解码实现。执行器必须保证：

1. `IGNORE` 不修改状态；
2. `UPDATE` 只能修改 $A_t$ 指定的字段；
3. 未受影响字段逐字段保持不变；
4. `HOLD` 不能覆盖已确认的可信值；
5. slot、操作、值类型、实体和来源通过 schema 校验；
6. 高风险行动若依赖 pending 字段，必须被阻塞或转为安全回退；
7. 非法补丁不执行，并记录对应成本。

执行器可以保证补丁合法，却不能保证策略总是选对字段或作出正确准入判断。这两类错误必须由学习目标和评测指标衡量。

## 7. 完整单步转移

最终评估后，信念更新为：

$$
B_t=U(B_{t-1},u_t,e_t)
$$

账本和可选预算更新为：

$$
(L_{t+1},\rho_{t+1})=G(L_t,\rho_t,u_t,e_t)
$$

行动策略采样环境动作：

$$
a_t\sim\pi_{\mathrm{act}}(\cdot\mid B_t,g_t,r_t)
$$

环境完成转移：

$$
x_{t+1}\sim P_{\mathrm{env}}(\cdot\mid x_t,a_t)
$$

$$
(y_t,o_{t+1})\sim P_{\mathrm{out}}(\cdot\mid x_{t+1},a_t)
$$

下一轮状态为：

$$
s_{t+1}^{\mathrm{rev}}=(B_t,o_{t+1},g_{t+1},r_{t+1},L_{t+1},m_{t+1},\rho_{t+1})
$$

一条日志记录可以写成：

$$
\tau_t=(s_t^{\mathrm{rev}},u_t,e_t,B_t,s_t^{\mathrm{act}},a_t,y_t,o_{t+1})
$$
 
这条记录不是“状态”，而是用于回放、训练和评测的单步转移。评估层转移由验证机制、确定性执行器、环境转移和下一观察共同组成。离线 replay 环境可退化为按 `step_index` 读取的确定转移；正式在线实验仍应保留随机种子、扰动变量和环境状态。

## 8. 模块的本质与边界

| 模块 | 本质 | 是否训练 | 输入 | 输出 |
|---|---|---:|---|---|
| 观察适配器 | 确定性数据接口 | 否 | 环境原始返回 | 原子观察及元信息 |
| 评估策略 $\pi_{\mathrm{rev}}$ | 参数化随机策略 | 是 | $s_t^{\mathrm{rev}}$ | 复合认知动作 $u_t$ |
| 信念执行器 $U$ | 受约束确定程序 | 否 | $B_{t-1},u_t,e_t$ | $B_t$ |
| 行动策略 $\pi_{\mathrm{act}}$ | 下游 Agent 策略 | 主实验冻结 | $s_t^{\mathrm{act}}$ | 环境动作 $a_t$ |
| 验证工具 | 环境中的有成本信息动作 | 否 | 工具、字段、查询 | 证据 $e_t$ |
| 环境 | 任务动力学和观察生成器 | 否 | $x_t,a_t$ | $x_{t+1},y_t,o_{t+1}$ |
| 奖励与成本评估器 | 训练期评估程序 | 否 | 轨迹、oracle、任务结果 | 奖励和约束成本 |

评估策略不是一个孤立的三分类器。它输出字段定位、补丁和验证动作，这些动作会改变后续可见状态、成本和任务结果，因此本质上是策略。MLP 可以作为结构化输出头，也可作为“冻结文本编码器 + MLP”的消融基线，但不适合作为论文主方法。

## 9. 主策略的参数化：LLM、LoRA 与结构化输出头

主方法使用 7B/8B decoder-only LLM 作为语义编码和补丁值生成器：

$$
h_t=f_{\phi+\mathrm{LoRA}_\theta}(s_t^{\mathrm{rev}})
$$

基础参数 $\phi$ 冻结，可训练参数 $\theta$ 包含 Revision LoRA、三分类 decision head、affected-field 多标签 head、verification head、patch-operation head 和受约束的 patch value decoder。

这可以回答“训练的是模型还是策略”：训练对象是评估策略 $\pi_{\mathrm{rev},\theta}$，它由大模型和小型结构化头共同参数化。第一阶段用监督目标训练该策略，第二阶段用轨迹奖励继续优化同一组参数。行动策略、环境、验证工具和信念执行器在主实验中冻结。

在 2 张 A100 80G 或 4 张 A100 40G 的预算下，建议采用 bf16、LoRA/QLoRA、gradient checkpointing 和分布式数据并行。第一版不做基础模型全参数训练，也不同时联合训练 $\pi_{\mathrm{act}}$，否则难以判断收益来自观察评估还是下游 Agent 变强。

## 10. 阶段一：结构化多任务 SFT

SFT 学习单步评估语义、字段定位、合法补丁和验证选择。目标函数为：

$$
\mathcal L_{\mathrm{SFT}}=\mathcal L_z+\alpha_A\mathcal L_A+\alpha_{\mathrm{op}}\mathcal L_{\mathrm{op}}+\alpha_v\mathcal L_{\mathrm{value}}+\alpha_q\mathcal L_q
$$

- $\mathcal L_z$：`UPDATE/HOLD/IGNORE` 交叉熵；
- $\mathcal L_A$：受影响字段的 masked binary cross entropy；
- $\mathcal L_{\mathrm{op}}$：补丁操作的 masked cross entropy；
- $\mathcal L_{\mathrm{value}}$：补丁值的 causal language modeling loss；
- $\mathcal L_q$：验证工具和目标字段的 masked cross entropy。

分支损失必须按动作门控计算。`IGNORE` 不计算字段、补丁和验证损失；`UPDATE` 不计算验证选择损失；`HOLD` 只对合法的 pending 补丁和验证分支计算损失。训练采样应按基础任务和扰动类型分层，避免大量容易的 `IGNORE` 样本淹没 `HOLD`。

### 10.1 SFT 样本示例一：错误实体，应忽略

```json
{
  "state": {
    "belief": {"order_id": "order_1842", "payment_status": "paid"},
    "observation": {
      "text": "order_9911 payment failed",
      "entity": "order_9911",
      "source": "payment_api"
    },
    "goal": "ship order_1842",
    "risk": {
      "dependent_fields": ["payment_status", "shipping_address"],
      "irreversible": true
    }
  },
  "target": {
    "decision": "IGNORE",
    "affected_fields": [],
    "patches": [],
    "verification": "NO_VERIFY"
  }
}
```

### 10.2 SFT 样本示例二：来源不足，应暂缓并验证

```json
{
  "state": {
    "belief": {
      "shipping_address": {"value": "Seoul A", "status": "trusted"}
    },
    "observation": {
      "text": "请改送 Seoul B",
      "entity": "order_1842",
      "source": "forwarded_message",
      "observed_at": 1784112300
    },
    "goal": "complete shipment",
    "risk": {"dependent_fields": ["shipping_address"], "irreversible": true}
  },
  "target": {
    "decision": "HOLD",
    "affected_fields": ["shipping_address"],
    "patches": [
      {"op": "SET_STATUS", "slot": "shipping_address", "value": "pending_verification"}
    ],
    "verification": {"tool": "official_order_lookup", "field": "shipping_address"}
  }
}
```

## 11. 阶段二：轨迹级受约束 GRPO

第二阶段从 SFT checkpoint 初始化，继续训练同一个评估策略。RL 的作用不是重新学习三类动作的含义，而是解决单步标签不能覆盖的问题：早期错误会污染后续状态，验证有成本，暂缓可能造成停滞，局部最优评估未必带来更好的任务结果。

### 11.1 RL 的 $S,A,R,P$

| 要素 | SIEVE 中的定义 |
|---|---|
| $S$ | $s_t^{\mathrm{rev}}=(B_{t-1},o_t,g_t,r_t,L_t,m_t,\rho_t)$ |
| $A$ | $u_t=(z_t,A_t,\Delta B_t,q_t)$ |
| $R$ | 终局任务奖励、势函数状态修正奖励和显式成本 |
| $P$ | 验证、执行器、环境转移和下一观察共同组成的转移核 |

主目标为：

$$
\max_\theta\ \mathbb E_{\pi_\theta}[R_{\mathrm{task}}+\beta R_{\mathrm{state}}]
$$

并满足：

$$
\mathbb E[C_{\mathrm{false\_update}}]\le\epsilon_{\mathrm{fu}}
$$

$$
\mathbb E[C_{\mathrm{unsafe}}]\le\epsilon_{\mathrm{unsafe}}
$$

$$
\mathbb E[C_{\mathrm{verification}}]\le B_{\mathrm{verify}}
$$

$$
\mathbb E[C_{\mathrm{stall}}]\le B_{\mathrm{stall}}
$$

### 11.2 奖励与成本

终局任务奖励由任务成功、可验证进度、失败和超时构成。状态 shaping 使用势函数差：

$$
R_{\mathrm{state},t}=\gamma\Phi(B_t,x_t)-\Phi(B_{t-1},x_t)
$$

$\Phi$ 比较 trusted slots 与 oracle state，并对当前任务依赖字段加权。若轨迹回报采用未折扣求和，则应令 shaping 中 $\gamma=1$；若保留 $\gamma<1$，训练代码必须同步使用折扣回报，才能保持标准势函数 shaping 的策略不变性。

建议记录以下非负成本：

| 成本 | 触发条件 |
|---|---|
| `false_update` | 无效观察进入 trusted state |
| `unsafe_action` | 错误或 pending 状态支持高风险行动 |
| `verification` | 发起一次有效验证 |
| `stall` | 无进展的 HOLD、无可执行验证或超时 |
| `invalid_patch` | 执行器拒绝补丁 |
| `collateral_edit` | 修改受影响字段集合之外的 slot |
| `budget_violation` | 验证请求超过预算 |

`missed_update` 不与上述各项机械叠加为固定惩罚。它通过势函数下降、停滞、超时和任务失败体现，但在评测时仍单独报告。

### 11.3 受约束 GRPO 目标

对同一基础任务和同一环境种子采样 $G$ 条轨迹。每条轨迹的约束后得分为：

$$
\widetilde R_i=R(\tau^i)-\sum_j\lambda_j C_j(\tau^i)
$$

组内优势为：

$$
A_i=\frac{\widetilde R_i-\operatorname{mean}(\widetilde R_{1:G})}{\operatorname{std}(\widetilde R_{1:G})+\epsilon}
$$

策略损失为：

$$
\mathcal L_{\mathrm{GRPO}}=-\mathbb E\left[\min\left(r_i(\theta)A_i,\operatorname{clip}(r_i(\theta),1-\varepsilon,1+\varepsilon)A_i\right)\right]+\beta_{\mathrm{KL}}D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{SFT}})
$$

拉格朗日乘子按未做组内标准化的真实成本更新：

$$
\lambda_j\leftarrow\max\{0,\lambda_j+\eta_\lambda(\widehat C_j-d_j)\}
$$

GRPO 仍然需要区分环境与策略。环境负责产生状态转移、下一观察、终局结果和成本；策略是唯一由梯度更新的对象。单轮语言模型 RL 中，环境常被简化为静态 prompt 和 reward verifier，所以边界不明显。多步 Agent 的动作会改变后续观察，环境必须显式存在。

### 11.4 RL 轨迹示例一：先验证再提交

同一初始场景采样两条轨迹：

```text
轨迹 A：
弱来源地址变更 → UPDATE → 覆盖 trusted 地址 → 发货 → 任务失败
成本：false_update=1, unsafe_action=1

轨迹 B：
弱来源地址变更 → HOLD + VERIFY → 官方工具返回新地址
→ UPDATE → 发货 → 任务成功
成本：verification=1
```

轨迹 B 支付了一次验证成本，但避免了不可逆错误。组内相对优势应推动高风险条件下的验证策略。

### 11.5 RL 轨迹示例二：乱序观察与延迟纠错

```text
t=0：可信库存为 available
t=1：收到较旧的 out_of_stock 观察
t=2：收到延迟到达、时间更新的 available 观察
t=3：Agent 决定是否提交订单
```

一条轨迹在 $t=1$ 错误更新并长期保留污染；另一条轨迹比较时间和来源后忽略旧观察，在 $t=2$ 保持或恢复可信状态。两者即使最终都完成任务，也应在污染面积、恢复延迟、验证成本和累计回报上拉开差异。

## 12. 数据构造与划分

公开 Agent 环境通常没有逐观察的 `UPDATE/HOLD/IGNORE`、受影响字段和补丁标签。训练数据应由干净轨迹、oracle state、工具 schema 与观察扰动层共同生成。

推荐流程：

1. 在干净环境中保存环境状态、动作、工具返回和终局结果；
2. 从工具 schema 和任务检查器抽取任务依赖字段；
3. 对单一观察施加一次可控扰动，生成 paired counterfactual；
4. 根据部署时可见的来源、时间、实体和条件生成结构化标签；
5. 用 oracle state 复核标签和计算奖励，但不把 oracle 输入策略；
6. 对冲突、隐式失效和风险边界样本做人工复核；
7. 按基础任务和轨迹分组划分，防止同一模板的干净与扰动版本跨集合泄漏。

扰动至少包括：`stale`、`wrong_entity`、`condition_mismatch`、`source_downgrade`、`partial`、`conflict`、`irrelevant`、`delayed_correction`、`duplicate` 和 `out_of_order`。

数据标识不要只使用当前代码里的 `scenario_id`。正式数据 schema 应拆为：

```text
environment_id
domain_id
task_template_id
base_task_id
episode_id
step_index
perturbation_id
seed
```

第一轮可采用 500 条人工复核 pilot cases、20k–50k 条 step-level SFT 样本、5k–10k 条基础多步轨迹和 20k–50k 个 RL rollout episodes。数据规模应由 pilot 的类别分布和错误覆盖率再校准。

## 13. Benchmark 与实验场景

顶会论文不应只使用一个固定 toy 场景，也不应为每个场景训练一套独立策略。更合理的设置是：一个共享评估策略，多个环境适配器和确定性 schema。

### 13.1 主实验

主实验使用 [tau2-bench](https://github.com/sierra-research/tau2-bench) 的 airline、retail 和 telecom 文本交互域。三类任务包含工具调用、规则约束、用户交互和可验证数据库状态，适合构造跨域的观察扰动和状态依赖。实验必须固定 release 或 commit，避免 benchmark 后续版本变化影响复现。

### 13.2 机制实验与 OOD

使用 [ScienceWorld](https://github.com/allenai/ScienceWorld) 中涉及状态变化、测量和对象属性更新的任务，检查时间变化、延迟证据和局部状态修正。[ALFWorld](https://github.com/alfworld/alfworld) 可用于跨环境泛化，适合作为附录或额外 OOD 结果，不宜在第一版同时承担主训练环境。

toy order 环境只用于单元测试、数据链路检查和最小可复现实例，不能作为论文主结果。

## 14. 评测指标

单步评估报告 decision macro-F1、各类 precision/recall、False Update Rate、Missed Update Rate、affected-field F1、patch exact match、schema validity、executor rejection rate、verification accuracy、ECE 和 Brier score。

状态质量报告 task-dependent belief consistency、contamination area、collateral edit rate、unaffected-field preservation、stale retention rate 和 recovery latency。

轨迹级报告 task success、pass@k、累计奖励与各类成本、unsafe action rate、验证和工具调用次数、环境步数、token 成本，以及成功率与验证成本、风险成本之间的 Pareto 曲线。主表必须同时报告任务成功、错误准入和验证成本；只提高成功率却大量调用验证工具，不足以说明评估策略更好。

## 15. 必要基线与消融

基线至少包含 `Always Update`、`Append-only Memory`、基于时间/实体/来源的 `Rule Gate`、`Frozen Encoder + MLP`、`LLM Prompt-only`、`SFT only`、`SFT + unconstrained GRPO` 和主方法 `SFT + constrained GRPO`。

关键消融包括删除账本 $L_t$、删除风险包络 $r_t$、删除验证微循环、让策略直接写状态、去掉 affected-field mask、去掉状态势函数、改变验证预算，以及固定状态与无限文本历史的对比。

## 16. 当前代码与理论变量的对应关系

| 理论对象 | 当前实现 |
|---|---|
| 三类评估 $z_t$ | `src/sieve/core/types.py` 中的 `Decision` |
| $B_t,o_t,r_t,L_t,\rho_t,s_t^{rev}$ | `BeliefState`、`Observation`、`RiskEnvelope`、`EvidenceLedger`、`Budget`、`RevisionContext` |
| 复合动作 $u_t$ | `Patch` 与 `RevisionOutput` |
| 唯一写入点 $U$ | `src/sieve/core/executor.py` 中的 `StateExecutor` |
| 环境接口 | `src/sieve/environments/protocol.py` 中的 `AgentEnvironment` |
| 双时间步事件循环 | `src/sieve/core/transition.py` 中的 `AgentEventLoop` |
| RL 环境接口与回放转移 | `src/sieve/environments/revision_env.py` |
| NumPy 参考策略 | `src/sieve/policies/structured_policy.py` |
| HF LLM + LoRA 结构 | `src/sieve/policies/hf_lora_policy.py` |
| SFT 参考训练 | `src/sieve/training/sft.py` |
| 势函数和状态一致性 | `src/sieve/training/rewards.py` |
| 拉格朗日约束 | `src/sieve/training/lagrangian.py` |
| 参考 Constrained GRPO | `src/sieve/training/constrained_grpo.py` |

`AgentEnvironment(Protocol)` 中方法体的 `...` 是 Python Protocol 的接口声明，不是空实现。具体环境通过实现 `step` 和 `oracle_state` 接入。

### 16.1 奖励核心代码

`src/sieve/training/rewards.py` 已实现：

```python
def potential_difference(
    previous,
    current,
    oracle,
    dependent_fields,
    gamma,
):
    return (
        gamma * belief_consistency(current, oracle, dependent_fields)
        - belief_consistency(previous, oracle, dependent_fields)
    )
```

它对应：

$$
R_{\mathrm{state},t}=\gamma\Phi(B_t,x_t)-\Phi(B_{t-1},x_t)
$$

### 16.2 约束更新核心代码

`src/sieve/training/lagrangian.py` 中的更新等价于：

```python
value = lambda_j + learning_rate * (observed_cost - cost_limit)
lambda_j = max(0.0, value)
```

它使超过阈值的成本在后续轨迹中受到更强惩罚。

### 16.3 GRPO 轨迹核心代码

`src/sieve/training/constrained_grpo.py` 已包含：

```python
score = trajectory.task_return - controller.penalty(trajectory.costs)
advantage = (score - group_mean) / (group_std + epsilon)
```

并以裁剪概率比和相对于 SFT reference 的正则更新参考策略。当前可执行版本直接更新 NumPy 线性头，用于检查轨迹、奖励、成本和优化器链路。

## 17. 当前实现边界与后续编码要求

现有仓库已能表达核心对象并做无 GPU 的静态链路检查，但论文级训练仍有几项明确工作：

1. `hf_lora_policy.py` 已定义冻结 LLM、LoRA 和结构化 heads，当前 GRPO 优化器尚未连接这些 PyTorch 参数；
2. replay 环境中的验证会消耗预算，但下一观察来自预记录序列，尚未由验证工具动态生成；
3. replay 中的 `false_update` 使用训练 target 判断，在线环境必须由 oracle validity checker 计算；
4. 当前 `unsafe_action` 主要是不可逆风险下错误准入的代理量，正式环境要检查实际下游动作；
5. 当前轨迹回报按未折扣方式累加，而势函数配置使用 $\gamma=0.97$，二者需按第 11.2 节统一；
6. 数据 schema 需要把基础任务、episode、扰动和随机种子拆开，避免 `scenario_id` 混用；
7. HF GRPO 必须对复合动作的各有效分支累计 log probability，并严格复用 SFT 阶段的 action mask；
8. 当前 `RevisionContext` 尚未包含固定长度历史摘要 $m_t$，`RevisionOutput` 也没有显式 `DEFER` 枚举；启用这两个设计项时需同步扩展 schema、序列化和 action mask；
9. 当前 `AgentEventLoop.revise()` 每调用一次就增加 `revision_t`，并用 `action_k` 统计所有外部动作。论文中的验证微循环要求证据返回前外层 $t$ 不变，因此该事件循环尚未实现本文第 2.3 节的最终时间语义。

这些事项不改变理论方案，但会直接影响实验是否可复现。正式训练前应先以小型模型和极小数据完成数据 schema 校验、动作门控测试、执行器不变量测试、轨迹回放一致性、奖励分解检查，以及同一随机种子的组内 rollout 对齐。当前本地机器不启动大模型训练。

## 18. 框架图的调整建议

框架图保留四个主模块，但修改三处表达。

第一，顶部明确写出“底层为 POMDP，评估层使用近似 belief-state MDP”，不要把 $s_t^{\mathrm{rev}}$ 直接标成完整环境状态。

第二，在评估策略框内把输出统一为：

$$
u_t=(z_t,A_t,\Delta B_t,q_t)
$$

图中只保留 `UPDATE/HOLD/IGNORE` 三类顶层动作；把 `ADD_FIELD` 放入补丁操作列表，把 `VERIFY/DEFER` 放在 `HOLD` 分支。

第三，把验证画成评估阶段内部的虚线微循环，并标注内部索引 $k$。验证证据先进入 $L_t$，再回到 $\pi_{\mathrm{rev}}$；证据返回前不能进入 $U$ 或 $\pi_{\mathrm{act}}$。

图底部的“基础状态空间”应改成“评估策略的信息状态空间”，并使用：

$$
\mathcal S_{\mathrm{rev}}=\mathcal B\times\mathcal O\times\mathcal G\times\mathcal R\times\mathcal L\times\mathcal M\times\mathcal P
$$

其中 $\mathcal P$ 仅在验证预算实验中启用。单步轨迹记录写为：

$$
\tau_t=(s_t^{\mathrm{rev}},u_t,e_t,B_t,s_t^{\mathrm{act}},a_t,y_t,o_{t+1})
$$

这样，图、公式、训练数据和代码接口使用同一套变量。

## 19. 论文可主张的贡献边界

在完成多环境实验后，论文可以围绕三点组织贡献：一是把观察准入建模为多步 Agent 中独立、可学习的认知决策；二是用“策略建议、确定性执行器提交”的边界，隔离不可靠观察与可执行状态；三是通过轨迹级受约束优化，在任务成功、状态污染、验证成本和风险之间学习权衡。

论文不能宣称 SIEVE 恢复了客观真值，也不能把 schema 合法性等同于语义正确性。更合适的结论是：在给定工具、元信息与验证能力的条件下，SIEVE 降低了不可靠观察对后续多步决策的污染，并改善了任务结果与验证成本之间的折中。

