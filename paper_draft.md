# SIEVE：多步语言 Agent 中风险条件化的状态承诺与持久污染控制

> 中文论文初稿（实验结果占位版）  
> 目标篇幅：顶会模板正文约 8--9 页（摘要与参考文献不计入）  
> 当前定位：以“风险条件化的选择性状态承诺”这一问题与因果测量协议为主要创新；SIEVE 是机制实例化，SFT 与受约束多步 RL 是训练手段；是否引入 GiGPO 或风险定向反事实分支，由信用分配诊断决定。  
> 主会状态：当前方法与内部实验基础尚不足以单独支撑顶会接收；至少完成强最近邻、风险最小对、状态中介干预和一个外部可执行环境后，才进入主会竞争区。  
> 写作纪律：正文只陈述已经实现或可由预注册实验检验的内容；所有尚未得到的结果均以“待填”标记。
> 实现同步项：本轮审稿新增的 assurance/scope/validity commitment certificate 与风险投影接口需要同步到状态 schema、Executor 和评测；certificate 由确定性规则签发，不增加模型输出字段。在代码完成前不得把 scope non-escalation 写成已有实验事实。

## 摘要

多步语言 Agent 会持续接收用户消息、工具返回和环境反馈，并将其用于后续规划。现有系统通常把新观察直接追加到上下文，或将其压缩进自由文本或结构化 belief；这隐含地把“被看见”近似为“可据此行动”。然而，一条观察即使语义合理，也可能因实体不匹配、时间过期、适用条件缺失、来源不足或与已确认状态冲突，而没有资格改变 Agent 的可执行任务状态。一旦这类观察被写入状态，错误便可跨越多个步骤持续影响工具调用，形成不同于即时规划错误的持久状态污染。本文不主张动作风险改变事实 belief；风险决定的是候选认知能否被承诺为特定行动范围内可使用的 operational state。

我们提出 **SIEVE**，一个位于 observation 与 action policy 之间的选择性状态承诺层。SIEVE 将每条候选观察裁决为 `UPDATE`、`HOLD` 或 `IGNORE`，并输出受影响字段、类型化局部补丁以及可选验证请求；确定性证书函数根据 evidence provenance、risk envelope 和裁决结果签发保证等级、使用范围与有效期，执行器是承诺状态的唯一写入口。模型不能为自己任意提升 assurance 或 scope。核心决策并非 observation-only 的真假或置信度分类：同一证据能否提交，还取决于它将支持哪些下游动作、动作风险与可逆性、证据新鲜度以及剩余验证预算。该设计将三个常被混合的问题分离：belief representation 决定如何表达候选认知，state commitment 决定候选证据在何种范围内成为行动可依赖的状态，action planning 决定在当前有效状态投影上采取什么行动。我们将状态承诺建模为带验证预算和安全代价的序列决策问题：监督微调学习局部语义边界，受约束多步强化学习优化任务成功、错误更新、危险动作和验证成本之间的长期权衡。

为避免由自构规则闭环自证，我们设计配对反事实与状态中介干预协议：在相同底层任务和环境随机性下比较 clean、corrupted、SIEVE-protected 与 oracle-admission 轨迹，并通过 state swap 测量错误观察对下游行动的影响有多少经由可信状态传播。实验将在内部可控场景和外部可执行 Agent 环境上，与 ReAct、结构化不确定 belief、write-time adjudication 和风险规则门等核心基线统一比较；SFT、GRPO 与 constrained GRPO 仅作为 SIEVE 内部训练消融。我们的目标不是证明三分类更准确，而是检验：风险条件化的状态承诺能否在保持正常证据利用率和任务效用的同时，阻断错误观察的跨步传播。

## 1. 引言

### 1.1 问题背景与研究动机

语言模型 Agent 的基本循环通常写为观察、推理和行动。ReAct 等工作展示了交错推理与环境交互的有效性；随后，长期记忆、递归摘要和结构化 belief 被用于压缩不断增长的交互历史并缓解部分可观测性 [1--4]。但在真实工具环境中，一个更早、也更基础的控制问题仍未被显式处理：**新观察是否已经获得修改可执行状态的授权？**

考虑一个售后 Agent。支付接口已确认订单 `A` 完成付款，随后聊天中出现“把地址改为新地址”的消息。该消息可能来自低可信渠道，可能属于订单 `B`，也可能晚于订单锁定时间。把它保留为候选证据是合理的；把它立即写成订单 `A` 的可信地址并据此发货却不合理。关键区别并非该句子在抽象意义上是真是假，而是它在当前实体、时间、来源、适用条件、验证预算和下游动作风险下，是否足以支持一次状态提交。

### 1.2 相关工作与研究缺口

现有 belief 或 memory 方法大多关注历史压缩、候选假设保存、置信度校准或 belief 与 action 的解耦。ReAct 通过交错推理与行动连接 LLM 和外部环境 [1]；Reflexion 把语言反馈写入 episodic memory 以改进后续尝试 [2]；Generative Agents 与 MemGPT 分别研究长期经验的检索、反思和分层内存管理 [3,4]。这些方法主要解决历史利用与上下文容量问题，而非当前 observation 是否可以成为 action-authorized state。

ABBEL 用自然语言 belief bottleneck 代替完整历史，并通过 RL 改善 belief 更新 [5]。Agent-BRACE 将 belief model 与 policy model 解耦，用带置信标签的原子命题表达近似 belief 分布 [6]。BeliefMem 保存多个带概率的候选结论，以减少单一确定结论造成的自强化错误 [7]。更近的 STALE/CUPMem 研究隐式陈旧冲突、write-time state adjudication 与失效传播 [16]。这些是本文最强最近邻。因此，SIEVE 不能仅以“显式写入裁决”主张新颖性；其可检验区别必须是 action-conditional commitment：在 observation 与 belief confidence 不变时，下游风险、可逆性和验证预算能够改变承诺动作。

SAGE-Agent 将工具参数澄清建模为带结构化不确定性和信息价值的 POMDP，并使用 uncertainty-weighted GRPO [8]。SIEVE 的 `HOLD+VERIFY` 与其相关，但验证只是承诺策略的一种动作；错误实体、陈旧冲突和条件不适用通常应直接 `IGNORE`。AgentDojo 研究不可信工具内容诱发的 prompt injection [12]。Provenance-sensitivity audit 区分证据相关性与证据是否被授权决定特定工具参数 [17]；commit-time authorization 要求支持持久外部效果的 witness 在执行提交时仍保持新鲜、绑定和有效 [18]。SIEVE 控制的是 observation 是否进入跨步可执行状态，并学习风险条件化的承诺与验证；它与执行时授权互补，而不取代执行时授权。

在多步训练方面，GRPO 等 group-based 方法避免单独训练 critic，但轨迹级回报会给多步动作分配相同优势。GiGPO 用 episode-level 与 anchor-state step-level 两层分组实现细粒度信用分配 [9]，近期工作还研究 belief-consistency 信号与 belief-aware grouping [13]。本文默认复用受约束 GRPO；只有在信用冲突被实测后，才将 GiGPO 或反事实分支作为条件性扩展。

$\tau$-bench 评测带领域规则、工具和模拟用户的多轮交互，并以终局数据库状态和 pass$^k$ 衡量可靠性 [10]；AppWorld 提供可控多应用世界、丰富 API 和 collateral damage 单元测试 [11]。本文在这些环境上加入配对 observation intervention，测量从污染进入状态到下游行动的传播，而不是创建一个替代所有现有环境的通用 benchmark。

### 1.3 本文思路与贡献

本文把该问题定义为 **risk-conditioned selective state commitment**：给定当前可信任务状态、候选观察、证据账本、任务目标、下游字段依赖、动作风险和剩余预算，决定该观察是否被提交为 action policy 可依赖的状态。与一般 observation admission 相比，该定义强调承诺阈值是 action-conditional 的：观察内容不变时，仅改变其将支持动作的风险或可逆性，最优决策也可能从 `UPDATE` 变为 `HOLD+VERIFY`。我们研究的核心因果链是

$$
\text{unreliable observation}
\rightarrow \text{state commitment}
\rightarrow \text{persistent contamination}
\rightarrow \text{downstream action error}.
$$

SIEVE 在 observation 与 action policy 之间插入独立的 Revision Policy，并将语义判断与状态执行分离。Revision Policy 生成 `UPDATE/HOLD/IGNORE`、受影响字段、候选补丁和验证请求；确定性证书函数与执行器验证 provenance、schema、字段作用域、assurance 和动作语义，是可执行状态的唯一写入口。`HOLD` 保留候选证据但不覆盖旧可信值，`IGNORE` 不改变状态，`UPDATE` 只提交被授权的局部字段。因而，SIEVE 不是另一个通用 memory，也不负责完整规划；它是风险条件化的状态承诺边界。

我们的贡献拟收敛为三点：

1. **决策对象。** 我们提出风险条件化的选择性状态承诺，将“如何表示 belief”“哪些证据成为行动依据”和“如何规划行动”分离，并要求提交策略显式依赖字段的下游风险、可逆性与验证预算。
2. **因果诊断。** 我们将持久状态污染定义为独立于即时规划错误的失败模式，并提出配对反事实与 state-swap 协议，估计观察污染进入状态、状态污染转化为动作以及由可信状态中介的下游效应。
3. **机制与学习。** 我们以 SIEVE 实例化该对象，将可学习 Revision Policy、类型化可信状态、证据账本、验证微循环与确定性局部执行器组合为状态承诺边界，并以 SFT 和受约束多步 RL 学习局部语义及长期安全--效用权衡。

本文不把执行器不变量本身作为理论创新，也不预先宣称新的通用 RL 算法。若实验诊断显示轨迹级 GRPO 存在可重复的局部信用冲突，我们再评估 GiGPO 或风险定向反事实分支；否则，论文的算法主张停留在对成熟受约束 RL 的问题适配。对主会而言，接收与否将主要取决于上述新决策对象和状态中介效应能否在强最近邻与外部环境上被实证建立。

## 2. 问题定义

### 2.1 从部分可观测环境到可执行状态

令 $x_t\in\mathcal X$ 为环境真实状态，$o_t\in\mathcal O$ 为 Agent 在时刻 $t$ 接收的原子观察，$a_t\in\mathcal A$ 为外部行动：

$$
o_t\sim P_{\mathrm{obs}}(\cdot\mid x_t,a_{t-1}),
\qquad
x_{t+1}\sim P_{\mathrm{env}}(\cdot\mid x_t,a_t).
$$

由于策略不能直接观察 $x_t$，整体交互是 POMDP。SIEVE 不试图恢复完整世界状态。候选认知及其不确定性保存在证据账本 $L_t$；$B_t$ 则表示面向行动的 **承诺状态**（operationally committed state），而非完整或风险依赖的主观 belief。$B_t$ 由有界、类型化 slot 构成：

$$
B_t=\{b_t^1,\ldots,b_t^{K_{\max}}\},
$$

其中每个 slot 至少包含字段标识、值、状态、实体、来源与时间元数据：

```json
{
  "field_id": "shipping_address",
  "value": "old_address",
  "status": "trusted",
  "entity": "order_A",
  "source": "order_api",
  "observed_at": 1784112000,
  "assurance": "medium",
  "scope": ["preview", "reversible_edit"],
  "valid_until": 1784115600
}
```

$B_t$ 既不等于客观真值，也不是所有历史的无损存储；它记录字段值在什么来源、保证等级、时效和行动范围内已被提交。候选但未获授权的证据保存在有界账本 $L_t$，而不伪装为可全局使用的可信事实。

### 2.2 状态承诺动作与风险投影

Revision Policy 的可见信息状态为

$$
s_t^{\mathrm{rev}}
=
(B_{t-1},o_t,g_t,r_t,L_t,\rho_t),
$$

其中 $g_t$ 是目标，$r_t$ 是包含依赖字段、动作风险和可逆性的风险包络，$\rho_t$ 是剩余步骤、token、工具与验证预算。若该有界状态近似概括与未来决策相关的历史，则在准入层采用信息状态 MDP 近似：

$$
P(s_{t+1}^{\mathrm{rev}}\mid s_{0:t}^{\mathrm{rev}},u_{0:t})
\approx
P(s_{t+1}^{\mathrm{rev}}\mid s_t^{\mathrm{rev}},u_t).
$$

承诺策略动作保持为结构化元组

$$
u_t=(c_t,A_t,\Delta B_t,q_t),
$$

其中

$$
c_t\in\{\mathrm{UPDATE},\mathrm{HOLD},\mathrm{IGNORE}\},
$$

$A_t$ 是受影响字段集合，$\Delta B_t$ 是有界类型化补丁，$q_t$ 是可选验证请求。模型输出后，确定性证书函数 $G_{\mathrm{cert}}$ 生成 commitment certificate：

$$
\zeta_t
=G_{\mathrm{cert}}(o_t,r_t,u_t)
=(\mathrm{assurance}_t,\mathrm{scope}_t,
\mathrm{validity}_t,\mathrm{provenance}_t),
$$

它描述本次提交达到的保证等级、允许支持的动作范围、有效期以及绑定的证据来源。$G_{\mathrm{cert}}$ 只允许根据受信来源映射、字段策略和当前 risk envelope 缩小或维持 scope，不接受模型自由文本声明的权限提升。三种裁决的语义为：

- `UPDATE`：当前证据足以授权对 $A_t$ 中字段进行局部提交；
- `HOLD`：观察与任务相关，但当前证据不足，保留候选并可请求验证；
- `IGNORE`：观察不适用、无关、过期或实体错误，不值得改变状态或消耗验证资源。

状态更新由确定性执行器 $U$ 完成：

$$
B_t=U(B_{t-1},u_t,\zeta_t,o_t).
$$

对于当前风险包络 $r_t$，定义条目 $b$ 的可用性：

$$
\mathrm{eligible}(b,r_t,t)
=
\mathbb 1[
\mathrm{assurance}(b)\ge \mathrm{required}(r_t)
\land \mathrm{scope}(b)\supseteq\mathrm{actions}(r_t)
\land t\le\mathrm{valid\_until}(b)
].
$$

action policy 不读取完整 $B_t$，而只读取风险投影

$$
B_t^{\mathrm{act}}=\Pi_{r_t}(B_t)
=\{b\in B_t:\mathrm{eligible}(b,r_t,t)=1\},
$$

以及

$$
s_t^{\mathrm{act}}=(B_t^{\mathrm{act}},g_t,r_t).
$$

从而使未准入观察不能绕过边界直接支持高风险动作，也使低风险 scope 下的提交不能自动升级为高风险依据。具体 action 生成后，执行器还应在工具调用前按同一 certificate 做一次确定性 scope/freshness 复核。这是一项需要在外部环境实验中严格执行的接口条件；如果 action policy 同时读取完整原始历史，或能够读取不满足当前 risk envelope 的 committed entry，SIEVE 的因果作用将无法隔离。

### 2.3 风险条件化的选择性承诺

若准入决策只由观察内容决定，则它退化为静态证据分类器：

$$
u_t\sim\pi(u_t\mid B_{t-1},o_t).
$$

SIEVE 研究更强的 action-conditional 决策。令 $\mathcal A(f)$ 表示依赖字段 $f$ 的候选下游动作集合，$h(a)$、$\mathrm{rev}(a)$ 分别表示动作风险与可逆性，$c(q)$ 表示验证成本。Revision Policy 应比较提交、暂缓验证和忽略的预期后果：

$$
u_t^*
=
\arg\max_{u\in\{\mathrm{UPDATE},\mathrm{HOLD},\mathrm{IGNORE}\}}
\mathbb E
\left[
V_{\mathrm{task}}(u)
-\lambda(r_t)C_{\mathrm{unsafe}}(u)
-c(q_t)
\mid s_t^{\mathrm{rev}}
\right].
$$

$\lambda(r_t)$ 随下游风险、不可逆性和字段依赖变化。这不意味着 Agent 因风险高而改变对事实的主观概率；候选认知仍保留在 $L_t$。变化的是它能获得的 operational assurance 和 scope。例如，低权威地址信息可被提交为仅支持可撤销草稿预览的低 assurance entry，但在触发不可逆发货前仍不可见，必须 `HOLD+VERIFY` 后提升保证等级。这一性质将 SIEVE 与 observation-only filtering、全局置信度阈值和仅在执行时检查静态权限的 guard 区分开来。

我们用风险最小对直接测试该定义。每一对保持任务文本、候选 observation、来源、实体与时间不变，只改变依赖该字段的下游动作风险或可逆性。定义风险响应率：

$$
\mathrm{RCR}
=
P\!\left(
u^{\mathrm{high\ risk}}\text{ 比 }u^{\mathrm{low\ risk}}\text{ 更保守}
\right),
$$

并同时报告低风险条件下的正确提交率，防止通过对所有输入一律 `HOLD` 获得高 RCR。RCR 不是越高越好；它必须与 benign retention 和任务效用共同构成选择性承诺前沿。

### 2.4 持久状态污染

给定 oracle 相关字段集合 $D_t$，定义时刻 $t$ 的污染程度为

$$
\kappa_t
=
\frac{1}{|D_t|}
\sum_{f\in D_t}
\mathbb 1
\left[
B_t(f)\text{ 被标记为 trusted 且 }B_t(f)\neq x_t(f)
\right].
$$

污染面积衡量错误状态跨步持续的总量：

$$
\mathrm{CA}(\tau)
=
\sum_{t=1}^{T-1}\frac{\kappa_t+\kappa_{t+1}}{2}.
$$

恢复延迟 $\mathrm{RL}(\tau)$ 是首次出现 $\kappa_t>0$ 到污染回到零之间的步数；若轨迹结束前未恢复，则记为剩余 horizon。除状态污染外，我们还定义：

$$
\mathrm{P2S}
=P(\kappa_t>0\mid o_t\text{ corrupted}),
$$

$$
\mathrm{S2A}
=P(a_{t':T}\text{ depends on corrupted field}\mid \kappa_t>0),
$$

分别表示 poison-to-state conversion 与 state-to-action conversion。两者将“模型看错了”分解为“错误是否被提交”与“提交后是否造成行动后果”。

### 2.5 配对反事实与状态中介效果

对同一底层任务 $i$，固定初始世界状态、用户目标、action policy、工具结果中未被干预的部分和随机种子，构造四条轨迹：clean $\tau_i^{\mathrm{cln}}$、corrupted $\tau_i^{\mathrm{cor}}$、SIEVE-protected $\tau_i^{\mathrm{sv}}$ 与 oracle-admission $\tau_i^{\mathrm{orc}}$。若终局效用为 $Y(\tau)$，则污染诱导损失为

$$
\Delta_i^{\mathrm{cor}}
=Y(\tau_i^{\mathrm{cln}})-Y(\tau_i^{\mathrm{cor}}),
$$

SIEVE 恢复比例为

$$
\mathrm{Rec}_i
=
\frac{Y(\tau_i^{\mathrm{sv}})-Y(\tau_i^{\mathrm{cor}})}
{Y(\tau_i^{\mathrm{orc}})-Y(\tau_i^{\mathrm{cor}})+\epsilon}.
$$

该指标把方法的收益锚定到由观察干预造成、且 oracle 准入理论上可恢复的损失，而非只比较不同 Agent 的总体成功率。

仅有配对结果仍不能证明损失是经由持久状态传播的。为此，在污染首次发生后的固定干预点 $t^*$ 进行 **state swap**：保留 corrupted 轨迹的目标、后续观察、Action Agent 和环境随机性，只把其可信状态 $B_{t^*}^{\mathrm{cor}}$ 替换为同任务 clean 或 oracle 轨迹的 $B_{t^*}^{\mathrm{cln}}$。定义状态修复效应：

$$
\mathrm{SRE}_i
=
Y\!\left(\tau_i^{\mathrm{cor}};
do(B_{t^*}=B_{t^*}^{\mathrm{cln}})\right)
-Y(\tau_i^{\mathrm{cor}}),
$$

以及状态中介比例：

$$
\mathrm{SMR}_i
=
\frac{\mathrm{SRE}_i}
{Y(\tau_i^{\mathrm{cln}})-Y(\tau_i^{\mathrm{cor}})+\epsilon}.
$$

若 $\mathrm{SMR}$ 显著大于零，则说明修复可执行状态本身能够恢复一部分 corruption-induced loss；若接近零，失败更可能来自 observation 的直接上下文影响或 Action Agent 的规划错误。该估计依赖干预后的后续环境可对齐，因此只在能够固定或重放后续事件的任务上报告，并明确其 benchmark-local 因果含义。

## 3. 方法论

### 3.1 架构

SIEVE 包含五个组件：Observation Adapter 将复合工具返回拆成带来源、实体、时间和适用条件的原子观察；Revision Policy 根据观察与风险包络生成承诺动作；Evidence Ledger 保存有界候选和验证结果；Deterministic Executor 执行合法局部补丁；冻结或独立训练的 Action Agent 只读取提交后的可信状态。风险包络不是普通安全标签，而是从当前计划或任务 schema 中提取的字段依赖接口：哪些候选动作会读取哪些字段、这些动作是否可逆、错误执行的代价等级以及提交前是否存在验证工具。主实验必须验证风险包络改变而 observation 不变时，Revision Policy 的行为相应改变，否则“风险条件化”只停留在输入字段声明。

```text
user / tool / environment observation
                  |
          Observation Adapter
                  |
       Revision Policy  π_rev
          /       |        \
      UPDATE     HOLD      IGNORE
        |       verify?       |
        +----------+----------+
                   |
        Deterministic Executor U
                   |
          Trusted Task State B_t
                   |
            frozen Action Agent
                   |
          tool call / environment action
```

分离 Revision Policy 与 Action Agent 有两个实验价值。第一，在冻结 Action Agent 时，可以估计仅改变观察准入所产生的因果收益；第二，可与“同一个模型同时摘要、规划和行动”的基线区分，避免把规划能力提升误归因于准入机制。

### 3.2 类型化状态与证据账本

可信状态使用任务 schema 限制字段与操作集合。未使用 slot 通过 mask 表示，时间推进时只更新 slot 内容，不扩展接口维度。账本容量满足 $|L_t|\le L_{\max}$；已解决、无关或过旧证据按确定规则淘汰。设计重点不是声称 JSON 本身新颖，而是建立两个不同语义空间：

$$
L_t:\ \text{candidate evidence},
\qquad
B_t:\ \text{action-authorized state}.
$$

结构化不确定 belief 基线可以在同一 slot 中保存置信度或多个候选，但没有 admission boundary 时，action policy 仍需自行解释“多大置信度可以支持当前动作”。SIEVE 将该决定显式化，并使其依赖任务风险和验证预算。

### 3.3 确定性执行器

执行器只接受 schema 合法的局部补丁与 commitment certificate，并实现以下机制约束：

$$
c_t=\mathrm{IGNORE}\Rightarrow B_t=B_{t-1},
$$

$$
c_t=\mathrm{HOLD}\Rightarrow
\text{trusted value is not overwritten},
$$

$$
f\notin A_t\Rightarrow B_t(f)=B_{t-1}(f).
$$

对风险范围还要求 non-escalation：

$$
\mathrm{scope}(b)\not\supseteq\mathrm{actions}(r_t)
\ \lor\ 
\mathrm{assurance}(b)<\mathrm{required}(r_t)
\Rightarrow
b\notin\Pi_{r_t}(B_t).
$$

非法 JSON、重复键、非有限数值、越界字段、错误 patch 类型、非法 scope 提升和 collateral edit 会被拒绝并记录代价。这些性质是由程序定义保证的工程不变量，而非关于模型语义正确性的定理。真正需要经验验证的是：该边界是否减少错误提交和 scope escalation，同时不会因过度 `HOLD/IGNORE` 损害正常任务效用。

### 3.4 验证微循环

`HOLD` 可发起字段级验证请求。令 $k$ 表示同一外层观察步内的验证索引：

$$
s_{t,0}^{\mathrm{rev}}
\rightarrow u_{t,0}^{\mathrm{HOLD+VERIFY}}
\rightarrow e_{t,1}
\rightarrow L_{t,1}
\rightarrow s_{t,1}^{\mathrm{rev}}
\rightarrow u_{t,1}.
$$

验证证据 $e_{t,k}$ 先写入账本并重新经过 Revision Policy，不能直接写入 $B_t$。循环在证据充分、工具不可用、预算耗尽或继续验证的预期价值过低时终止。由此，相同弱证据在可逆低风险任务中可以被接受，在不可逆高风险任务中则可能先验证；准入不是与内容一一对应的静态标签。

### 3.5 监督微调

阶段一在单步 RevisionContext 上训练同一自回归策略输出严格 JSON。令输出 token 集按功能划分为裁决、结构和补丁值，总目标为

$$
\mathcal L_{\mathrm{SFT}}
=
\lambda_d\mathcal L_{\mathrm{decision}}
+\lambda_s\mathcal L_{\mathrm{structure}}
+\lambda_v\mathcal L_{\mathrm{value}}.
$$

SFT 学习实体、时间、来源和条件匹配，以及合法输出协议。Certificate 不由模型预测，因此不增加自授权损失分支；它由 $G_{\mathrm{cert}}$ 根据可审计策略确定。SFT 不直接优化早期承诺决策对数步之后任务结果的影响。当前项目的关键先导现象是：teacher-forced 与单步自由生成可接近饱和，而多步闭环仍可能失败；正式论文将只在统一实验协议下报告该差距，不把开发期 checkpoint 选择结果当作最终结论。

### 3.6 受约束多步强化学习

给定轨迹 $\tau$，我们优化任务回报，同时约束错误更新、危险动作与资源使用。为避免与策略优化中的优势函数混淆，我们不把下面的量称为“优势”或笼统的“势函数”，而将其定义为 **可信状态一致性评分**（Trusted-state Consistency Score, TCS）。它只衡量当前风险投影后可执行状态与 oracle 在任务依赖字段上的一致程度：

$$
\mathrm{TCS}(B_t,x_t;r_t)
=
\frac{1}{|D_t|}
\sum_{f\in D_t}
\mathbb 1[
\Pi_{r_t}(B_t)(f)=x_t(f)
\land \mathrm{status}_t(f)=\mathrm{trusted}
].
$$

由该评分构造 **状态一致性增量奖励**（consistency-delta reward）：

$$
r_t^{\mathrm{state}}
=
\gamma\,\mathrm{TCS}(B_t,x_t;r_t)
-\mathrm{TCS}(B_{t-1},x_{t-1};r_{t-1}),
$$

并与环境任务奖励组合：

$$
r_t=r_t^{\mathrm{task}}+\beta_{\mathrm{state}}r_t^{\mathrm{state}}.
$$

因此，$\mathrm{TCS}$ 是产生 dense reward signal 的状态评分，$r_t^{\mathrm{state}}$ 是相邻信息状态评分的差分；二者都不是 RL 中的 advantage function。低 assurance 条目即使值碰巧正确，只要不满足当前 risk envelope，也不会获得高风险状态一致性奖励。该差分形式在数学上属于 potential-based reward shaping 的实现形式，但本文使用 TCS 命名，是为了强调被度量的实际对象并避免将 shaping potential 与后文的 group-relative advantage 混为一谈。若使用 $\gamma<1$，轨迹回报必须采用相同折扣约定；否则应令该差分中的 $\gamma=1$，以保持奖励定义一致。

代价向量包括 false update、unsafe action、scope escalation、verification、stall、invalid format、invalid patch、collateral edit 与 budget violation。约束目标写为

$$
\max_\theta\ \mathbb E_{\tau\sim\pi_\theta}[R(\tau)]
\quad
\text{s.t.}\quad
\mathbb E[C_j(\tau)]\le d_j,\ \forall j.
$$

使用拉格朗日松弛：

$$
\widetilde R_i
=R_i-\sum_j\lambda_j C_{ij},
\qquad
\lambda_j\leftarrow
\left[\lambda_j+\eta_\lambda(\overline C_j-d_j)\right]_+.
$$

默认训练器采用 trajectory-level **constrained GRPO（CGRPO）**。这里的 CGRPO 指在 GRPO 的组相对策略优化目标上加入拉格朗日约束代价；它是本文复用的训练算法，不是 SIEVE 的方法名，也不作为新的通用 RL 算法贡献。同一初始场景采样 $G$ 条完整轨迹。首先累计每条轨迹的任务奖励、TCS 差分奖励和约束代价，得到 $\widetilde R_i$；随后才根据同组轨迹的相对回报计算 **组相对优势估计**：

$$
A_i
=
\frac{\widetilde R_i-\operatorname{mean}(\widetilde R)}
{\operatorname{std}(\widetilde R)+\epsilon}.
$$

这里的 $A_i$ 才是用于策略梯度更新的 advantage estimate：它回答“轨迹 $i$ 相对同组其他采样有多好”，而不是“当前状态与 oracle 有多一致”。同一轨迹的 token 共享该优势。训练时固定 reference policy，并公平记录 rollout 数、环境步数、验证工具调用和生成 token，防止某种方法仅因使用更多交互预算获益。

三者的关系可以概括为：

$$
\underbrace{\mathrm{TCS}(B_t,x_t;r_t)}_{\text{风险投影后的状态质量评分}}
\longrightarrow
\underbrace{r_t^{\mathrm{state}}}_{\text{逐步奖励信号}}
\longrightarrow
\underbrace{\widetilde R_i}_{\text{受约束轨迹回报}}
\longrightarrow
\underbrace{A_i}_{\text{GRPO 优势估计}}.
$$

### 3.7 条件性算法扩展：局部信用分配

轨迹级优势可能把一次早期错误提交的长期负效应平均分配给整条轨迹，但这必须先被测量。我们定义 credit conflict：来自相同或等价准入状态的两个动作具有不同反事实后果，却因其他步骤噪声获得相反的轨迹级排序。诊断至少报告等价状态覆盖率、组内动作多样性、局部回报排序与轨迹优势排序的一致率。

若重复状态覆盖充足，采用 GiGPO 的 anchor-state grouping 作为成熟细粒度信用分配基线 [9]。若覆盖不足但关键提交节点可识别，再评估 **risk-targeted counterfactual branching**：只在候选动作可能改变 trusted state、影响高风险行动或消耗稀缺验证预算的节点，从同一父状态比较 `UPDATE/HOLD/IGNORE`。它必须与 uniform branching 和 entropy-based branching 在相同 rollout、工具和 token 预算下比较。若诊断不成立，则不将该扩展列为主方法。

## 4. 实验设计

### 4.1 研究问题

实验围绕七个可证伪问题组织：

- **RQ1：失败模式是否真实？** 不可靠观察是否会进入状态，并跨多步转化为错误行动？
- **RQ2：污染是否由可信状态中介？** state swap 能否在不改变 Action Agent 的情况下恢复 corruption-induced loss？
- **RQ3：承诺是否真正依赖风险？** 在 observation 完全相同的最小对中，动作风险、可逆性和验证预算是否系统性改变 `UPDATE/HOLD/IGNORE`？
- **RQ4：SIEVE 是否不可由强替代物取代？** ReAct、结构化不确定 belief、write-time adjudication 与强规则门能否达到相同安全--效用前沿？
- **RQ5：机制是否只是保守拒绝？** 降低 false update 的同时，benign evidence retention、任务成功和完成效率是否保持？
- **RQ6：多步 RL 是否必要？** SFT-only 与受约束 RL 的差距是否集中在验证预算、延迟后果和不可逆风险场景？
- **RQ7：是否需要局部信用分配？** 在检测到 credit conflict 的子集上，GiGPO 或风险定向分支是否以相同交互预算优于轨迹级 GRPO？

### 4.2 数据与环境

#### 内部可控环境

当前内部语料包含 2,100 个 episode，按底层任务严格划分为 1,600/200/300 的 train/dev/test；对应不同 base task 数为 1,030/135/201，跨 split 交集为零。场景来自 AgentBench、ToolBench、WebArena、car-bench、$\tau^2$-bench 与 $\tau^3$-bench 的可追溯记录，经统一 schema 转换。当前五类 episode 模板覆盖复杂陈旧冲突、延迟污染、多字段依赖、禁止自动修复以及验证预算选择；事件顺序、类型数量、关键性和组合应继续随机化，并保留未见组合测试，避免模型利用固定顺序捷径。

内部环境的优点是可以读取 oracle、精确复现反事实并分解污染链；它不能单独证明外部有效性。所有统计以 base task 为 bootstrap 单位，不能把同一任务的多个增强 episode 当作独立样本。除现有场景外，每个可执行字段还应构造风险最小对：固定 observation 的全部文本与 provenance，只改变消费该字段的动作风险、可逆性或验证预算；train/dev/test 按 base task 隔离，并在 test 保留未见风险--证据组合。

#### 外部可执行环境

外部主环境优先选择 $\tau$-bench/$\tau^2$-bench 或 AppWorld。$\tau$-bench 提供多轮用户交互、领域规则、API 工具和终局数据库状态 [10]；AppWorld 提供 9 个模拟应用、457 个 API、状态单元测试及 collateral damage 检查 [11]。最终至少完成一个环境的适配，要求：

1. 保持原任务的 action policy 与评分器；
2. 从自然用户/工具轨迹中注入或抽取 stale、wrong-entity、weak-source、conflict 等观察；
3. 对同一任务运行 clean/corrupted/protected/oracle 配对；
4. 原始观察不得绕过 SIEVE 进入 Action Agent 的高风险状态接口；
5. 对可重放任务实施 state swap，隔离可信状态的中介作用；
6. 同时报告原环境任务指标、风险响应指标和 SIEVE 的污染指标。

AgentDojo 可作为安全压力测试而非唯一外部主结果，因为它主要研究不可信工具内容中的间接 prompt injection [12]，与本文更广的证据准入问题相邻但不等价。

#### 统一评测的两个轨道

每个外部环境分为两个互补轨道：

1. **Native utility track**：不加入 SIEVE 特有扰动，沿用 benchmark 原始任务与官方评分，检验引入状态承诺层是否损害一般 Agent 能力。
2. **Matched intervention track**：在同一批 base tasks 上构造 clean/corrupted/protected/oracle 配对，只改变 observation 的实体、时间、来源、冲突或适用条件，并额外标注动作风险与可逆性。

论文的主要优势声明只适用于第二轨道中满足以下边界的任务：环境部分可观测；某些任务字段会跨步持久；观察可靠性不均；后续动作读取这些字段；错误行动具有可测代价。对于无持久状态、所有观察均权威或单步可逆的任务，SIEVE 理论上不应承诺显著优于 ReAct，最多应保持 clean utility。

### 4.3 同环境系统级对比与训练消融

#### 系统级方法比较

主表只保留四个能够分别回答关键质疑的核心 baseline，加上完整 SIEVE：

1. **ReAct (full history)**：标准 reasoning--action--observation 循环，代表没有显式状态准入的基础 Agent [1]；
2. **Agent-BRACE-style**：belief model 与 action model 解耦，并保存带置信度的原子命题，回答“结构化不确定 belief 是否已经足够” [6]；
3. **STALE/CUPMem-style**：在写入时检测隐式冲突、替换陈旧状态并传播失效，回答“一般 write-time adjudication 是否已经解决问题” [16]；
4. **Risk-aware Rule Gate**：基于实体、来源、时间、冲突、scope 和风险阈值的强确定性门，回答“是否无需学习、简单规则已经足够”；
5. **SIEVE**：完整的风险条件化状态承诺、证据账本、验证循环、certificate、风险投影和确定性 Executor。

这四个 baseline 分别覆盖基础 Agent、最近的 belief 方法、最近的写时裁决方法和非学习规则下界，已经能形成完整论证链。若某项工作没有可直接接入目标环境的官方实现，正文必须标注为 `-style` 或 `our adaptation`，公开映射规则并避免声称复现其论文原始结果。

其他方法不进入所有主表：**BeliefMem-style** 只在 ALFWorld 和多候选证据子集作为补充强基线，因为它专门检验“保留多个概率候选能否替代 admission boundary” [7]；**ABBEL-style**、Recursive Summary 和 Reflexion-style 放入附录或初步筛选表，除非它们在开发集上强于上述核心 baseline；SAGE-Agent 只在歧义与澄清子集加入。这样避免重复比较多个功能高度重叠的 memory/belief 方法。

#### SIEVE 内部训练消融

以下名称只用于回答“怎样训练 SIEVE”，不能在主表中冒充不同系统方法：

1. **SIEVE-Prompt**：同一 SIEVE 架构，只使用提示词，不训练 Revision Policy；
2. **SIEVE-SFT**：使用阶段一监督微调；
3. **SIEVE-GRPO**：SFT 后使用无显式约束的 trajectory GRPO；
4. **SIEVE-CGRPO**：SFT 后使用本文当前实现的 constrained GRPO；
5. **SIEVE-GiGPO / Targeted Branching**：仅在 RQ7 信用诊断成立时加入。

论文中不再把 `SIEVE-CGRPO` 当作最终方法名称。若 CGRPO 是最终选定的训练器，主结果行仍写作 **SIEVE**，并在表注中注明 “Revision Policy trained with SFT + CGRPO”。

所有系统方法共享同一 base model、Action Agent、任务输入、环境版本、最大上下文、工具预算与污染种子。对需要额外 belief/memory 模型的方法，必须同时报告参数量、推理 token 和工具成本。结构化 belief 基线拥有与 SIEVE 相同的字段 schema 和元数据，不得故意削弱；规则门的阈值只在 dev 上选择。主表中的数字必须来自本研究在统一 harness 下的重新运行，不能把不同论文中使用不同模型和 benchmark 版本的已发表数字直接拼接。

优先实现顺序如下：

| Benchmark | 主要比较对象 | 该环境回答的问题 |
|---|---|---|
| Internal-Causal | ReAct、Agent-BRACE-style、CUPMem-style、Rule Gate、SIEVE | 精确分解 P2S、S2A、CA、SRE 与 SMR |
| $\tau^2$-bench | ReAct、Agent-BRACE-style、CUPMem-style、Rule Gate、SIEVE | 多轮用户/工具交互、规则约束、状态改变和风险差异 |
| ALFWorld | ReAct、Agent-BRACE-style、CUPMem-style、SIEVE；补充 BeliefMem-style | 与 belief/memory 论文在共同的部分可观测长程环境正面对比 |
| AppWorld | ReAct、CUPMem-style、Rule Gate、SIEVE | 数据库状态、不可预期修改与 collateral damage |

ALFWorld 尤其重要，因为 BeliefMem 和 Agent-BRACE 已在部分可观测长程环境中报告结果 [6,7]；将它们与 SIEVE 放入统一 corrupted-observation overlay，能够更直接回答“保存不确定 belief 是否已经足够”。$\tau^2$-bench 和 AppWorld 则更贴近 SIEVE 的风险条件化承诺边界。不同环境承担不同证据角色，不应只把多个 benchmark 数字机械求平均。

### 4.4 指标

#### 任务与可靠性

- task success / 原环境得分；
- pass$^k$，衡量同一任务重复运行的可靠性 [10]；
- benign evidence retention：应更新证据被正确保留的比例；
- missed-update rate 与 over-hold rate；
- 平均环境步、工具调用、验证次数和生成 token。

#### 污染传播

- corruption-induced drop $\Delta^{\mathrm{cor}}$；
- recovered fraction $\mathrm{Rec}$；
- state repair effect (SRE)；
- state-mediated ratio (SMR)；
- poison-to-state conversion (P2S)；
- state-to-action conversion (S2A)；
- contamination area (CA)；
- recovery latency (RL)；
- oracle-admission upper bound。

#### 安全--效用

- false-update rate；
- unsafe-action rate；
- collateral edits；
- scope-escalation rate：低保证或窄 scope 条目未经重新验证即支持更高风险动作的比例；
- verification cost；
- risk-conditioned response rate (RCR)；
- success--false-update 与 success--tool-cost Pareto 曲线。

### 4.5 同一 benchmark 下的端到端系统主结果（待填）

#### 当前内部 held-out 结果快照（单 seed，非最终主表）

当前已经完成一组内部可控环境的 Stage-2 训练与 held-out test 评测。该结果只能作为开发期证据，尚不能替代外部可执行环境主表，也不能支持跨方法系统级结论；它用于确认 SIEVE 的 Revision Policy 在未见内部 scenario 上是否学到基本 admission 策略，并诊断当前 benchmark 是否可被简单规则解决。

| 模型 / 训练器 | Split | Episodes | Success $\uparrow$ | Mean return $\uparrow$ | Parse $\uparrow$ | False update $\downarrow$ | Stall $\downarrow$ | Invalid format $\downarrow$ | Verification/action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT-only | Internal test | 300 | 0.0200 | -0.1701 | 0.9375 | 0.0147 | 0.0331 | 0.0625 | 0.0074 |
| Auto-Write / Append-All | Internal test | 300 | 0.2400 | 0.0157 | 1.0000 | 0.7222 | 0.0000 | 0.0000 | 0.0000 |
| Visible Rule Gate | Internal test | 300 | 0.9967 | 2.4897 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2174 |
| Rule Gate (diagnostic upper bound) | Internal test | 300 | 0.9967 | 2.4238 | 1.0000 | 0.0000 | 0.0435 | 0.0000 | 0.2174 |
| SIEVE-GRPO | Internal dev | 50 | 0.9800 | 2.4580 | 0.9956 | — | — | — | — |
| SIEVE-GRPO | Internal test | 300 | 0.9767 | 2.4466 | 0.9949 | 0.0000 | 0.0000 | 0.0051 | 0.2174 |
| SIEVE-GiGPO | Internal dev | 50 | 0.9800 | 2.4580 | 0.9956 | — | — | — | — |
| SIEVE-GiGPO | Internal test | 300 | 0.9767 | 2.4466 | 0.9949 | 0.0000 | 0.0000 | 0.0051 | 0.2174 |

其中 SFT-only 使用 `outputs/stage1-qwen3-4b/best/adapter` 在同一 Stage-2 closed-loop test 上评测；SIEVE-GRPO checkpoint 为 `outputs/stage2-qwen3-4b-8xh100-grpo/runs/20260823_145210/final/adapter`，SIEVE-GiGPO checkpoint 为 `outputs/stage2-qwen3-4b-8xh100-gigpo/runs/20260823_154308/final/adapter`。GRPO 与 GiGPO 的 internal test 指标完全一致，说明当前内部 test 已接近饱和，不能支持 GiGPO 优于 trajectory-level GRPO 的结论。Visible Rule Gate 仅使用模型 prompt 中可见的来源、实体、时间、authority 与 authentication 字段；它仍达到 99.67% success，说明当前内部 benchmark 的规则可解性过强。Rule Gate diagnostic upper bound 额外使用 `condition/relevant` 等生成器内部诊断字段，仅用于估计规则上界，不应作为公平 baseline。上述结果表明：正式论文必须通过外部可执行环境、风险最小对和更复杂自然错误来支撑 learned SIEVE 的必要性。

这组结果支持两个开发期结论。第一，SFT-only 在 closed-loop test 上只有 2.00% success，说明单步 imitation 学到输出协议后仍不足以处理跨步状态承诺，Stage-2 RL 对闭环任务成功是必要的。第二，Auto-Write / Append-All 的 false update rate 达到 72.22%，success 只有 24.00%，说明无 admission boundary 的直接写入确实造成严重状态污染。与此同时，Visible Rule Gate 达到 99.67% success，暴露出当前内部数据的启发式可解风险；正式论文不能只依赖该内部 benchmark 宣称 learned SIEVE 超越规则方法。

#### AgentDojo admission smoke（外部压力测试，非完整端到端结果）

我们额外将 AgentDojo 用作外部不可信 observation 的 admission smoke，而不是训练数据。当前实现从 `workspace` 与 `slack` suite 中抽取 20 对 user task / injection task，构造 benign user observation 与 injection observation，测试 Revision Policy 是否会把 injection 内容写入 executable belief。我们还运行了一个 ground-truth checker probe：使用 AgentDojo suite 自身的 ground-truth pipeline 执行同一 20 对 smoke tasks，并调用官方 utility/security checker，得到 `ok_rate=1.0000`、`user_ground_truth_utility_rate=1.0000`、`injection_ground_truth_security_rate=1.0000`。这说明 AgentDojo 的 suite 加载、工具环境执行和官方 checker 在当前环境中可用。该评测仍未运行 learned action agent + SIEVE wrapper 的完整 AgentDojo action loop，因此不能报告 learned-agent task utility / security score，也不能替代外部可执行环境主结果。

| 方法 | Pairs | Cases | Parse $\uparrow$ | Poison-to-State $\downarrow$ | Block Injection $\uparrow$ | Ignore Injection | Benign Update $\uparrow$ | Invalid Format $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT-only | 20 | 40 | 0.9500 | 0.0000 | 1.0000 | 0.8500 | 1.0000 | 0.0500 |
| SIEVE-GRPO | 20 | 40 | 0.9250 | 0.0000 | 1.0000 | 0.3500 | 1.0000 | 0.0750 |
| SIEVE-GiGPO | 20 | 40 | 0.9000 | 0.0000 | 1.0000 | 0.4500 | 1.0000 | 0.1000 |

这组初步结果说明三种 policy 都没有把 injection observation 写入状态，`poison_to_state_rate=0`；SFT-only 更常直接 `IGNORE` injection，而 GRPO/GiGPO 更常使用 `HOLD`，与内部 RL 学到的验证倾向一致。它只能支持“admission boundary 可迁移到 AgentDojo 风格不可信内容”的初步观察，并确认官方 checker 可执行；完整 AgentDojo 任务成功和安全结论仍需要接入 learned action agent 与 SIEVE wrapper。

#### Native utility track

| 完整 Agent 方法 | $\tau^2$-bench Pass$^1$ $\uparrow$ | AppWorld Task Goal $\uparrow$ | ALFWorld Success $\uparrow$ | 平均工具调用 $\downarrow$ | 平均生成 token $\downarrow$ |
|---|---:|---:|---:|---:|---:|
| ReAct (full history) | — | — | — | — | — |
| Agent-BRACE-style | — | — | — | — | — |
| STALE/CUPMem-style | — | — | — | — | — |
| Risk-aware Rule Gate | — | — | — | — | — |
| **SIEVE (ours)** | — | — | — | — | — |

该表回答 SIEVE 是否保持一般任务能力。SIEVE 不一定需要在所有 clean benchmark 上取得最高值，但相对 ReAct 和最强 belief baseline 的下降必须很小，并结合置信区间报告。

#### Matched intervention track

| 完整 Agent 方法 | Internal-Causal Success $\uparrow$ | $\tau^2$ Corrupted Pass$^1$ $\uparrow$ | AppWorld Corrupted Goal $\uparrow$ | ALFWorld Corrupted Success $\uparrow$ | Unsafe Action $\downarrow$ | Recovered Fraction $\uparrow$ |
|---|---:|---:|---:|---:|---:|---:|
| ReAct (full history) | — | — | — | — | — | — |
| Agent-BRACE-style | — | — | — | — | — | — |
| STALE/CUPMem-style | — | — | — | — | — | — |
| Risk-aware Rule Gate | — | — | — | — | — | — |
| **SIEVE (ours)** | — | — | — | — | — | — |

这是支撑论文核心 claim 的第一主表。`SIEVE (ours)` 表示完整系统；若其 Revision Policy 最终由 SFT+CGRPO 训练，只在表注中说明，不改变方法名称。若资源无法覆盖三个外部环境，最低要求是 $\tau^2$-bench 加 ALFWorld 或 AppWorld 之一；只报告内部环境不足以支持主会主张。

### 4.6 配对因果干预结果（待填）

| 方法 | Recovered Fraction $\uparrow$ | SRE $\uparrow$ | SMR $\uparrow$ | P2S $\downarrow$ | S2A $\downarrow$ | CA $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|
| ReAct (full history) | — | — | — | — | — | — |
| Agent-BRACE-style | — | — | — | — | — | — |
| STALE/CUPMem-style | — | — | — | — | — | — |
| Risk-aware Rule Gate | — | — | — | — | — | — |
| **SIEVE (ours)** | — | — | — | — | — | — |
| Oracle admission | — | — | — | — | — | — |

### 4.7 风险条件化最小对（待填）

| 方法 | 低风险正确 UPDATE $\uparrow$ | 高风险正确 HOLD/VERIFY $\uparrow$ | RCR $\uparrow$ | Scope Escalation $\downarrow$ | Over-hold $\downarrow$ |
|---|---:|---:|---:|---:|---:|
| ReAct (full history) | — | — | — | — | — |
| Agent-BRACE-style | — | — | — | — | — |
| STALE/CUPMem-style | — | — | — | — | — |
| Risk-aware Rule Gate | — | — | — | — | — |
| **SIEVE (ours)** | — | — | — | — | — |

该表是定位成立的关键结果。若 SIEVE 只在两种风险条件下都拒绝，则 RCR 可能看似较高，但低风险正确 UPDATE 与 over-hold 会暴露保守捷径。若 write-time adjudication 或 uncertainty-aware belief 在相同输入和预算下达到同一前沿，论文只能主张互补，而不能主张状态承诺是必要的新层。

### 4.8 统计协议与复现

主实验使用至少 3 个随机种子。置信区间按 base task 分层 bootstrap，而非按增强 episode；配对条件使用 task-level paired bootstrap 或置换检验。多指标比较同时报告效应量与 95% 置信区间，不以单次最佳 checkpoint 代替均值。开发集用于规则阈值、约束上限和 checkpoint 选择；内部 test 与外部 held-out 只在方案冻结后评测。论文同时公开场景构造版本、manifest hash、污染干预种子、模型 checkpoint、失败轨迹和所有预算统计。

## 5. 消融实验和分析

### 5.1 适用边界与分领域结果（待填）

#### 当前 SIEVE-GRPO internal test 分领域结果（单 seed）

| 子集 | Episodes | Success $\uparrow$ | Mean return $\uparrow$ | Parse $\uparrow$ | False update $\downarrow$ | Stall $\downarrow$ | Invalid format $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 300 | 0.9767 | 2.4466 | 0.9949 | 0.0000 | 0.0000 | 0.0051 |
| Commerce | 150 | 1.0000 | 2.4957 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| Service | 90 | 0.9889 | 2.4756 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| Workflow | 60 | 0.9000 | 2.2801 | 0.9746 | 0.0000 | 0.0000 | 0.0254 |

当前错误主要集中在 workflow 子集，且更接近格式/解析失败，而非错误状态提交：所有子集的 false update 与 stall 均为 0。该分领域结果提示下一轮误差分析应优先抽查 workflow 的 invalid-format case，并判断它们是否来自更复杂的工具字段、长值 JSON 或 schema 表达不足。

| 子集 | ReAct Success | 最强 belief/memory baseline | Rule Gate | **SIEVE (ours)** | SIEVE 相对最强基线增益 |
|---|---:|---:|---:|---:|---:|
| Clean / 全部观察权威 | — | — | — | — | — |
| Wrong entity | — | — | — | — | — |
| Stale / out-of-order | — | — | — | — | — |
| Weak source + reversible action | — | — | — | — | — |
| Weak source + irreversible action | — | — | — | — | — |
| Conflicting multi-source evidence | — | — | — | — | — |
| Verification budget scarce | — | — | — | — | — |
| Horizon $\le 2$ | — | — | — | — | — |
| Horizon $\ge 5$ | — | — | — | — | — |

该表限定 SIEVE 的优势范围。预期增益应集中在持久状态、异质来源、风险差异和延迟后果明显的子集；若它只在项目自定义事件标签上有效、在真实领域或较长 horizon 上没有增益，核心 claim 不成立。

### 5.2 机制消融实验（待填）

| 变体 | Task Success $\uparrow$ | False Update $\downarrow$ | Benign Retention $\uparrow$ | CA $\downarrow$ | Verify/Task $\downarrow$ |
|---|---:|---:|---:|---:|---:|
| 完整 SIEVE | — | — | — | — | — |
| 无 Evidence Ledger | — | — | — | — | — |
| 无 Risk Envelope | — | — | — | — | — |
| 无 assurance/scope 投影 | — | — | — | — | — |
| 无 Verification Loop | — | — | — | — | — |
| 自由文本状态替代 typed state | — | — | — | — | — |
| 模型直接改状态（无 Executor） | — | — | — | — | — |
| 无 Lagrangian constraints | — | — | — | — | — |
| 所有 observation 自动写入 | — | — | — | — | — |

### 5.3 训练策略与信用分配诊断（待填）

| 方法 | Success $\uparrow$ | Constraint Violation $\downarrow$ | Credit Agreement $\uparrow$ | Rollouts | Tool Calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| SIEVE-SFT | test 0.0200 | false update 0.0147; stall 0.0331; invalid format 0.0625 | — | 0 | verification/action 0.0074 | 待填 |
| SIEVE-GRPO | — | — | — | — | — | — |
| SIEVE-CGRPO | dev 0.9800 / test 0.9767 | test false update 0.0000; stall 0.0000 | trajectory reward variance decays 1.2698 $\rightarrow$ 0.0138 | 25,600 train rollouts | verification/action 0.2174 on test | 待填 |
| SIEVE-GiGPO | dev 0.9800 / test 0.9767 | test false update 0.0000; stall 0.0000; invalid format 0.0051 | step reward variance active early: 0.9257 first 10 iters; decays to 0.0070 last 20 iters | 25,600 train rollouts | verification/action 0.2174 on test | 待填 |
| SIEVE + Uniform Branching | — | — | — | — | — | — |
| SIEVE + Entropy Branching | — | — | — | — | — | — |
| SIEVE + Risk-targeted Branching | — | — | — | — | — | — |

该表是训练策略消融，不是跨论文方法主表。当前单 seed 结果显示，SFT-only 在内部 Stage-2 test 上几乎不能完成闭环任务，而 CGRPO/GiGPO 均达到 97.67% success、0 false update 和 0 stall。GiGPO 的 step-level 信号在训练早期存在，但到后期随任务成功率饱和而消失，最终 test 指标与 CGRPO 持平。因此正文不应主张 GiGPO 是必要算法贡献，只能把它作为信用分配诊断或附录消融。只有在等价状态覆盖率和 credit conflict 诊断支持 RQ7 时，GiGPO/branching 行才进入正文；否则移至附录并将算法部分缩短。

### 5.4 结果解释：SIEVE 何时应当有效

如果错误观察会被提交到一个跨步持久、且被下游 action policy 读取的状态中，SIEVE 应主要降低 P2S、CA 和后续危险动作。收益应随 horizon、动作不可逆性、实体相似度、时间冲突复杂度和验证预算稀缺性增大。若只修正准入就能在冻结 Action Agent 时恢复显著比例的任务结果，则支持“状态污染是独立失败原因”的主张。

### 5.5 可能的负结果

以下结果会削弱或否定论文主张，并应如实报告：

- 强规则门与 SIEVE 在未见组合和外部环境上无显著差异，说明任务仍可由静态规则解决；
- 降低 false update 伴随同等幅度的 missed update，说明收益来自保守拒绝；
- structured uncertainty baseline 达到相同 Pareto frontier，说明独立准入边界并非必要；
- SFT 与 RL 在长期权衡场景无差异，说明 RL 不是必要组件；
- 原生 Agent 很少将错误观察转化为持久状态或行动，说明问题外部普遍性不足；
- GiGPO/分支方法只在增加 rollout 后改善，预算配平后优势消失。

这些负结果并非实现细节，而是决定最终论文定位的科学证据。若规则门足够，应将工作收缩为因果 benchmark 与机制分析；若 SFT 足够，应去掉 RL 新颖性叙述；若局部信用分配无必要，应保留 constrained GRPO 作为训练工具而非贡献。

### 5.6 误差分析

当前已经对 internal test 的 workflow 子集完成一轮 trace 级误差分析。SIEVE-GRPO 与 SIEVE-GiGPO 都在 60 个 workflow scenarios 上成功 54 个、失败 6 个；失败模板分布相同：`complex_stale_conflict` 2 个、`multi_field_dependency` 1 个、`delayed_contamination` 1 个、`no_auto_repair` 1 个、`verification_budget_choice` 1 个。所有失败都同时包含 `parse_error` 与最终 `state_mismatch`。

关键发现是，剩余 workflow 错误主要不是 admission 语义错误。失败轨迹中 false update 为 0，模型通常已经学会对 unverified value 先 `HOLD+VERIFY`，并对 wrong-entity 或 stale conflict 做 `IGNORE`。真正导致失败的是长 `tool_result` 被模型复制到 JSON patch value 时转义失败，例如嵌套工具返回中包含 JSON 字符串、Python dict 风格字符串、引号、反斜杠和 URL，导致 parser 报 `Expecting ',' delimiter` 或 `Extra data`。executor 因 invalid format 拒绝 patch，最终 oracle 字段保持缺失，形成 state mismatch。

这与 Auto-Write baseline 的失败机制不同。Auto-Write 在 workflow 子集只有 28.33% success，43/60 失败，主要对应错误 observation 被直接写入后造成状态污染。GRPO/GiGPO 的 workflow success 为 90.00%，失败更接近输出协议和长值序列化限制：

```text
正确 admission intent
  -> 长字段 JSON 表达失败
  -> executor 拒绝 patch
  -> oracle field 缺失
  -> task failure
```

因此后续改进不应优先继续增加 RL step，而应调整 action schema，例如允许 `COPY_OBSERVATION_VALUE` 或 value-reference patch，让模型决定是否提交字段，但不需要手写完整长工具结果。论文中应将这一现象报告为实现层限制：SIEVE 的 admission boundary 已经阻断了污染，但当前 JSON-as-value 输出协议对 workflow 长字段不够稳。

最终至少抽样 100 条失败轨迹，由两名标注者按以下互斥主因编码：实体绑定错误、时间/版本推理错误、来源授权错误、条件适用性错误、验证选择错误、验证后未提交、过度 HOLD、过度 IGNORE、patch/executor 错误、Action Agent 规划错误。报告一致性，并区分“准入正确但规划失败”与“准入错误导致规划失败”。典型案例应展示完整的 observation、$B_{t-1}$、裁决、$B_t$ 和后续 action，而非只展示模型自然语言解释。

## 6. 局限性

首先，SIEVE 依赖任务 schema、实体标识、来源与时间元数据；开放世界中这些信息可能缺失或自身被伪造。其次，确定性 certificate 只能执行预先配置的来源--assurance 与 action--scope 策略；若策略本身错误或 provenance 不可信，non-escalation 并不等于语义安全。第三，确定性执行器只能保证补丁与 scope 合法，不能保证 Revision Policy 的语义裁决正确。第四，显式边界增加延迟、token 和验证工具成本，且过度保守会降低用户体验。第五，内部场景由规则和开源记录构造，即使具备配对因果优势，也可能与自然错误分布不同；因此外部环境和自然失败审计不可省略。第六，当前 AgentDojo 结果仍是 admission smoke 和 ground-truth checker probe：它证明外部不可信 observation 可被映射到 SIEVE admission 评测，并证明官方环境/checker 可执行，但尚未完成 learned action agent + SIEVE wrapper 的完整 task utility/security 评测。第七，oracle 只应在训练奖励和评测中使用，任何 oracle 泄漏都会使结果失效。

从安全角度，SIEVE 可降低错误证据支持不可逆行动的概率，但不应被描述为完整安全系统。它不解决恶意工具本身、越权 API、模型欺骗、隐私泄露或 action policy 的所有规划错误。发布数据时应删除真实个人信息和凭证，保留来源许可证与 provenance；外部任务中的状态修改全部在模拟环境中执行。

## 7. 结论

本文研究多步语言 Agent 中一个狭窄但具有长期后果的问题：观察被看到之后，在当前下游风险下，何时才有资格成为 Agent 据以行动的状态。SIEVE 将 risk-conditioned state commitment 从 belief representation 和 action planning 中分离，通过可学习的 `UPDATE/HOLD/IGNORE` 裁决、类型化可信状态、证据账本、验证微循环与确定性执行器建立显式承诺边界。论文的关键证据不是三分类准确率，而是三件事：同一证据的提交是否随动作风险合理变化；错误观察的下游效应是否经可信状态中介；这种传播是否能在保留正常证据、任务成功和资源效率的同时被阻断。若 state swap、风险最小对、外部环境、强 belief/write-adjudication/rule 基线和公平预算分析共同支持这一命题，SIEVE 才构成有竞争力的主会贡献；若不支持，预注册的负结果也将明确其适用边界。

## 参考文献（初稿，后续转 BibTeX）

1. Yao et al. **ReAct: Synergizing Reasoning and Acting in Language Models.** ICLR 2023. <https://arxiv.org/abs/2210.03629>
2. Shinn et al. **Reflexion: Language Agents with Verbal Reinforcement Learning.** NeurIPS 2023. <https://arxiv.org/abs/2303.11366>
3. Park et al. **Generative Agents: Interactive Simulacra of Human Behavior.** UIST 2023. <https://arxiv.org/abs/2304.03442>
4. Packer et al. **MemGPT: Towards LLMs as Operating Systems.** <https://arxiv.org/abs/2310.08560>
5. Lidayan et al. **ABBEL: LLM Agents Acting through Belief Bottlenecks Expressed in Language.** <https://arxiv.org/abs/2512.20111>
6. Singh et al. **Agent-BRACE: Decoupling Beliefs from Actions in Long-Horizon Tasks via Verbalized State Uncertainty.** <https://arxiv.org/abs/2605.11436>
7. Liao et al. **Belief Memory: Agent Memory Under Partial Observability.** <https://arxiv.org/abs/2605.05583>
8. Suri et al. **Structured Uncertainty Guided Clarification for LLM Agents.** <https://arxiv.org/abs/2511.08798>
9. Feng et al. **Group-in-Group Policy Optimization for LLM Agent Training.** NeurIPS 2025. <https://openreview.net/forum?id=QXEhBMNrCW>
10. Yao et al. **$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.** ICLR 2025. <https://arxiv.org/abs/2406.12045>
11. Trivedi et al. **AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents.** ACL 2024. <https://arxiv.org/abs/2407.18901>
12. Debenedetti et al. **AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.** NeurIPS 2024. <https://arxiv.org/abs/2406.13352>
13. Tang et al. **Rewarding Beliefs, Not Actions: Consistency-Guided Credit Assignment for Long-Horizon Agents.** <https://arxiv.org/abs/2605.20061>
14. Allen et al. **Mitigating Partial Observability in Sequential Decision Processes via the Lambda Discrepancy.** NeurIPS 2024. <https://openreview.net/forum?id=YaPhvbGqwO>
15. He et al. **Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks.** ICLR 2026. <https://openreview.net/forum?id=ddd6d1f2f476c83809a1150d2b22836fca46f80d>
16. Chao et al. **STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?** <https://arxiv.org/abs/2605.06527>
17. Liao. **Auditing Provenance Sensitivity in LLM Agent Action Selection.** <https://arxiv.org/abs/2607.20827>
18. Santos-Grueiro. **Temporary Authority, Permanent Effects: Commit-Time Authorization for LLM Agents.** <https://arxiv.org/abs/2607.10487>

## 附录 A：实验完成前的最小决策门槛

| 决策 | 必需证据 | 若不满足时的论文处理 |
|---|---|---|
| 把“持久状态污染”列为普遍失败模式 | 至少一个外部环境中 P2S、S2A、CA 显著非零 | 降级为内部诊断，不作普遍性主张 |
| 把状态中介列为独立因果机制 | 冻结 Action Agent 的 paired intervention 与 state swap 显著恢复任务效用 | 改写为整体 Agent 架构方法 |
| 声称风险条件化承诺不同于 belief update | 同 observation 的风险最小对上显著优于 belief/write-adjudication 基线 | 只声称互补，不声称新决策层必要 |
| 声称 scope 防止权限升级 | 风险升级轨迹中 scope-escalation 接近零且低风险效用不显著下降 | 仅把 certificate 作为设计建议，不作有效性主张 |
| 声称优于 belief representation | 强 uncertainty-aware belief 在同预算下显著落后 | 只声称互补，不声称必要 |
| 声称不是保守拒绝 | false update 降低且 benign retention / success 不显著下降 | 报告安全--效用 trade-off，不声称双赢 |
| 把 RL 放入贡献 | RL 在延迟/预算/不可逆子集显著优于 SFT | RL 仅作为实现细节 |
| 引入 GiGPO 或分支算法 | 存在 credit conflict，且公平预算下有稳定收益 | 删除算法扩展，保留 constrained GRPO |

## 附录 B：建议正文图表布局（8--9 页）

| 正文部分 | 目标篇幅 | 关键图表 |
|---|---:|---|
| 引言（含相关工作与贡献） | 1.3 页 | Fig. 1：污染链与 SIEVE 插入位置；最近邻差异 |
| 问题定义 | 1.0 页 | 定义风险条件化承诺、P2S、SMR、Recovered Fraction |
| 方法论（含学习目标） | 2.4 页 | Fig. 2：架构；Alg. 1：Revision/Executor loop；SFT 与 CGRPO |
| 实验设计 | 2.0 页 | 环境、核心基线、主结果、paired protocol、统计方法 |
| 消融实验和分析 | 1.4 页 | state-swap、风险最小对、机制消融、Pareto 与误差分析 |
| 局限性 | 0.3 页 | 适用条件、certificate 假设与外部有效性 |
| 结论 | 0.2 页 | 核心发现与适用边界 |
