# SIEVE 训练与实验设计规格

## 1. 研究问题与边界

SIEVE 研究 Agent 多步决策中的观察准入问题。Agent 在执行任务时会持续接收工具返回、用户补充、环境反馈和历史状态。现有系统常把这些内容直接追加到上下文，或用自由文本重写任务摘要。这样做会让过时、对象错配、条件不适用或来源较弱的观察进入后续决策链。

本文把可信任务状态记为 belief state，但这里的 belief 不是模型参数中的知识，也不是主观置信。它是 Agent 对当前任务相关环境变量的结构化估计。SIEVE 在新观察和可信状态之间加入可学习的 State Revision Policy，决定一条观察能否改变可信状态。

本文只处理以下问题：

- 新观察应当 UPDATE、HOLD 还是 IGNORE；
- 观察影响哪些状态字段；
- 允许执行什么局部状态补丁；
- HOLD 时是否值得调用验证工具；
- 早期状态修正如何影响后续任务成功、安全和成本。

本文不训练完整 Agent，不学习通用世界模型，也不改变 Action Agent 的注意力层。Action Agent、工具集合、环境和确定性 State Executor 在主要实验中保持冻结。

## 2. 训练对象

两个阶段训练同一个 State Revision Policy：

\[
\pi_{\mathrm{rev},\theta}(y_t\mid s_t).
\]

“策略”和“模型”不是两个互斥对象。策略是从决策状态到结构化修正动作的条件分布，模型是策略的参数化实现。本设计使用结构化混合 LLM 策略：

1. 冻结 7B/8B decoder-only LLM 的主体参数；
2. 训练 Revision LoRA，使骨干模型适应时间、来源、条件和状态冲突判断；
3. 训练 decision、affected-field 和 verification heads；
4. 用受约束 decoder 生成 typed patch；
5. 用确定性 Executor 校验并执行 patch。

小型 MLP head 只负责把 LLM 表征映射到有限动作，不单独承担开放文本理解。因此，主方法不是纯 MLP gate。冻结文本编码器加 MLP 作为低成本消融基线。

设策略输入编码为：

\[
h_t=f_{\phi+\mathrm{LoRA}_\theta}
(B_{t-1},o_t,m_t,g_t,d_t,b_t,\ell_t),
\]

其中 \(\phi\) 冻结，\(\theta\) 包含 LoRA、结构化 heads 和 patch decoder 的可训练参数。

## 3. 两套时间索引

SIEVE 区分环境动作步和观察修正步。

- \(k\)：环境动作步。Agent 每调用一次工具、执行一次外部动作或发起一次验证，\(k\) 增加一次。
- \(t\)：观察修正步。策略每处理一条原子观察，\(t\) 增加一次。

一次环境动作可能返回多条观察。环境适配器先将返回结果拆成有稳定顺序的原子观察 \(o_{t,1},\ldots,o_{t,n}\)，Revision Policy 逐条处理。所有观察处理完成后，Action Agent 才能产生下一动作。

标准事件顺序为：

\[
a_{k-1}
\rightarrow x_k
\rightarrow o_t
\rightarrow y_t\sim\pi_{\mathrm{rev}}
\rightarrow B_t=U(B_{t-1},y_t)
\rightarrow a_k\sim\pi_{\mathrm{act}}(\cdot\mid B_t,g_t).
\]

验证请求是有成本的环境动作。它消耗预算并生成后续观察，不计作模型的内部思考。

## 4. 消除候选动作的循环依赖

若 Revision Policy 读取具体 `candidate_action`，而 candidate action 又由更新后的状态生成，会形成循环依赖。主方法不把具体候选动作输入 Revision Policy，改用风险包络：

\[
d_t=(\text{active subgoal},\text{dependent fields},\text{risk},\text{reversibility}).
\]

风险包络来自当前计划、工具 schema 和活跃子目标，不能读取未经准入的原始观察。示例：

```json
{
  "active_subgoal": "confirm_before_shipping",
  "dependent_fields": ["payment_status", "shipping_address"],
  "risk": "high",
  "reversible": false
}
```

状态修正后，Action Agent 根据 \(B_t\) 产生具体动作。Executor 再检查该动作是否依赖 `pending_verification` 字段。这样既向 Revision Policy 提供风险信息，又不开放 `observation -> candidate action` 的旁路。

## 5. 策略状态空间

部署时策略可见状态为：

\[
s_t=(B_{t-1},o_t,m_t,g_t,d_t,b_t,\ell_t).
\]

各变量含义如下。

| 变量 | 内容 | 是否随轨迹无限增长 |
|---|---|---|
| \(B_{t-1}\) | 可信任务状态 | 否，固定 slot 上限 |
| \(o_t\) | 当前原子观察 | 否，只处理当前观察 |
| \(m_t\) | 来源、时间、实体、条件等元信息 | 否 |
| \(g_t\) | 任务目标和活跃子目标 | 否，使用受限摘要 |
| \(d_t\) | 字段依赖和风险包络 | 否 |
| \(b_t\) | 工具、验证、token、步数预算 | 否 |
| \(\ell_t\) | pending evidence ledger | 否，固定容量 |

训练环境还维护完整状态：

\[
S_t=(x_t,B_{t-1},b_t,\ell_t,g_t,d_t),
\]

其中 \(x_t\) 是 oracle world state。它只用于生成标签、计算奖励和评测，不输入 Revision Policy。

## 6. 可信状态的表示与维度

可信状态由带类型的 slot 构成：

\[
B_t=\{b_t^1,\ldots,b_t^{K_{\max}}\}.
\]

每个 slot 至少包含：

```json
{
  "id": "payment_status",
  "value": "paid",
  "status": "trusted",
  "source": "payment_api",
  "observed_at": 1784112000,
  "valid_from": 1784112000,
  "entity": "order_1842"
}
```

状态维度不随时间步增加：

- 每个 episode 使用固定 \(K_{\max}\)，空 slot 用 mask 标记；
- 新字段只能进入空 slot；
- pending evidence ledger 使用固定 \(L_{\max}\) 的环形队列；
- ledger 满时按照 resolved、irrelevant、age、task relevance 的固定顺序淘汰；
- 文本 token 数可以在上下文上限内变化，但 pooled representation \(h_t\in\mathbb R^d\) 的维度固定；
- 时间增长改变 slot 内容、时间差和预算，不改变模型接口维度。

第一版不学习连续 confidence propagation。状态只使用 `trusted`、`pending_verification`、`empty` 等离散状态，避免把校准不可靠的标量置信度当作事实概率。

## 7. 动作空间

Revision Policy 的复合动作定义为：

\[
y_t=(z_t,A_t,\Delta B_t,q_t).
\]

### 7.1 准入动作

\[
z_t\in\{\mathrm{UPDATE},\mathrm{HOLD},\mathrm{IGNORE}\}.
\]

### 7.2 受影响字段

\[
A_t\in\{0,1\}^{K_{\max}}.
\]

它是多标签 slot mask。新增字段使用一个专门的 empty-slot 指针。

### 7.3 Typed patch

\[
\Delta B_t=\{(op,slot,value,metadata)\}_{j=1}^{M_t},
\qquad M_t\le M_{\max}.
\]

允许的操作为：

- `ADD_FIELD`
- `SET_VALUE`
- `SET_STATUS`
- `SET_PROVENANCE`
- `SET_VALIDITY`

patch decoder 只能从 schema 中选择 `op` 和 `slot`。自由文本只用于新的 slot value，且长度受限。

### 7.4 验证动作

\[
q_t\in\{\mathrm{NO\_VERIFY}\}
\cup(\mathcal T_{\mathrm{verify}}\times\mathcal F).
\]

即不验证，或选择一个允许的验证工具和目标字段。工具可用性和剩余预算由 action mask 控制。

策略分解为：

\[
\pi_\theta(y_t\mid s_t)=
\pi_\theta(z_t\mid h_t)
\pi_\theta(A_t\mid z_t,h_t)
\pi_\theta(\Delta B_t\mid A_t,z_t,h_t)
\pi_\theta(q_t\mid z_t,h_t).
\]

## 8. State Executor 的硬约束

Executor 不训练。它负责 schema 校验、action mask 和确定性状态更新。

1. IGNORE 必须使用空 affected set、空 patch 和 `NO_VERIFY`。
2. HOLD 可以标记 pending、保存候选证据或发起验证，但不能覆盖 trusted value。
3. UPDATE 只能修改 \(A_t\) 指定的字段。
4. 未受影响字段逐字节保持不变。
5. patch 中的 slot、op、value type 和 entity 必须通过 schema 校验。
6. 高风险或不可逆动作若依赖 pending 字段，Executor 必须阻断。
7. 非法 patch 不执行，并记录 `invalid_patch` 成本。

这些性质由程序保证。模型仍可能选错字段或做错准入决定，这部分通过学习指标衡量，不能写成理论上的零错误保证。

## 9. 转移函数

策略采样：

\[
y_t\sim\pi_\theta(y_t\mid s_t).
\]

可信状态确定性更新：

\[
B_t=U(B_{t-1},y_t).
\]

预算和 ledger 更新：

\[
(b_{t+1},\ell_{t+1})=G(b_t,\ell_t,y_t).
\]

令 \(k=k(t)\) 表示处理完第 \(t\) 条观察时的环境动作计数。Action Agent 或验证分支产生第 \(k\) 个环境动作：

\[
a_k=
\begin{cases}
q_t, & q_t\ne\mathrm{NO\_VERIFY},\\
\pi_{\mathrm{act}}(B_t,g_t), & \text{otherwise}.
\end{cases}
\]

环境和观察转移为：

\[
x_{k+1}\sim P_{\mathrm{env}}(x_{k+1}\mid x_k,a_k),
\]

\[
o_{t+1}\sim P_{\mathrm{obs}}(o_{t+1}\mid x_{k+1},a_k,\eta_t).
\]

\(\eta_t\) 是观察扰动变量，可取 clean、stale、entity mismatch、condition mismatch、source downgrade、partial、conflict、irrelevant、delayed correction、duplicate 或 out-of-order。

Revision Policy 不拟合 \(P_{\mathrm{env}}\) 或 \(P_{\mathrm{obs}}\)。公开环境和扰动层负责采样转移。因为验证动作可能消耗不同数量的外部步骤和真实时间，理论上可把整体系统写成 constrained SMDP；代码实现采用事件驱动 MDP，每次原子观察对应一个 revision event，并显式累计动作持续时间和成本。

## 10. 阶段一：结构化多任务 SFT

训练目标为：

\[
\mathcal L_{\mathrm{SFT}}
=\mathcal L_z
+\alpha_A\mathcal L_A
+\alpha_P\mathcal L_{\Delta B}
+\alpha_Q\mathcal L_q.
\]

- \(\mathcal L_z\)：三分类交叉熵；
- \(\mathcal L_A\)：受影响字段的 masked binary cross entropy；
- \(\mathcal L_{\Delta B}\)：受约束 patch token loss；
- \(\mathcal L_q\)：验证工具和字段的 masked cross entropy。

分支损失按动作 mask 计算。IGNORE 不计算 patch 和 query loss；UPDATE 不计算验证选择 loss；不存在可用验证工具的 HOLD 样本以 `NO_VERIFY` 为目标。类别采样按基础 scenario 分层，防止大量容易的 IGNORE 样本压制 HOLD。

SFT 学习单步语义边界、字段定位和合法 patch，不负责从稀疏终局奖励中重新发现三个动作的含义。

## 11. 阶段二：轨迹级约束强化学习

第二阶段从 SFT checkpoint 初始化，继续训练相同的 LoRA、heads 和 patch decoder。基础 LLM、Action Agent、Executor、工具和环境保持冻结。

主目标：

\[
\max_\theta\ \mathbb E_{\pi_\theta}
[R_{\mathrm{task}}+\beta R_{\mathrm{state}}].
\]

约束：

\[
\begin{aligned}
\mathbb E[C_{\mathrm{false-update}}]&\le\epsilon_{\mathrm{fu}},\\
\mathbb E[C_{\mathrm{unsafe}}]&\le\epsilon_{\mathrm{unsafe}},\\
\mathbb E[C_{\mathrm{verify}}]&\le B_{\mathrm{verify}},\\
\mathbb E[C_{\mathrm{stall}}]&\le B_{\mathrm{stall}}.
\end{aligned}
\]

主算法采用带 KL 正则的 Constrained GRPO。对同一个初始 scenario 和环境随机种子，从 Revision Policy 采样 \(G\) 条完整轨迹 \(\tau^1,\ldots,\tau^G\)。每条轨迹先计算带约束惩罚的分数：

\[
\widetilde R_i=
R(\tau^i)-\sum_j\lambda_j C_j(\tau^i).
\]

组内优势不依赖 value critic：

\[
A_i=
\frac{\widetilde R_i-\operatorname{mean}(\widetilde R_{1:G})}
{\operatorname{std}(\widetilde R_{1:G})+\epsilon}.
\]

策略目标为：

\[
\mathcal L_{\mathrm{GRPO}}=
-\mathbb E\left[
\min\left(
r_i(\theta)A_i,
\operatorname{clip}(r_i(\theta),1-\varepsilon,1+\varepsilon)A_i
\right)
\right]
+\beta_{\mathrm{KL}}D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{SFT}}).
\]

约束乘子使用未做组内标准化的实际成本更新：

\[
\lambda_j\leftarrow
\max\{0,\lambda_j+\eta_\lambda(\widehat C_j-d_j)\}.
\]

策略 log probability 是有效分支的和：

\[
\log\pi_\theta(y_t\mid s_t)=
\log\pi(z_t)+
\log\pi(A_t\mid z_t)+
\log\pi(\Delta B_t\mid A_t,z_t)+
\log\pi(q_t\mid z_t).
\]

无效分支被 mask，不进入 probability ratio。GRPO 仍然区分环境和策略：环境产生 world-state transition、下一观察、终局结果和成本；Revision Policy 是唯一被梯度更新的模型。单轮数学题训练中的“环境”常被简化为静态 prompt 数据集加 reward verifier，看起来不明显；多步 Agent 中动作会改变后续观察，环境边界必须显式保留。DPO 或 IPO 作为离线轨迹优化基线，用于判断在线交互和延迟信用分配是否带来额外收益。

## 12. 奖励、势函数与成本

任务奖励包含终局成功、可验证的环境进度和失败或超时。状态 shaping 使用势函数差分：

\[
R_{\mathrm{state},t}=
\gamma\Phi(B_t,x_t)-\Phi(B_{t-1},x_t).
\]

\(\Phi\) 比较 trusted slots 和 oracle world state，并对任务依赖字段加权。差分形式避免在每一步重复奖励“保持正确”。

训练记录以下成本：

| 成本 | 触发条件 |
|---|---|
| `false_update` | 无效观察进入 trusted state |
| `unsafe_action` | 错误或 pending 状态支持高风险动作 |
| `verification` | 成功发起验证工具调用 |
| `stall` | HOLD 但没有可执行的验证、无进展或超时 |
| `invalid_patch` | Executor 拒绝 patch |
| `collateral_edit` | patch 修改 affected set 外的字段 |
| `budget_violation` | 验证请求超过 verification/tool budget |

Missed Update 不再作为一个与所有其他项叠加的固定负奖励。它通过势函数下降、任务失败、状态停滞和超时体现，避免同一错误被重复惩罚。评测仍单独报告 Missed Update Rate。

## 13. 数据构造

现有公开 Agent 环境没有逐观察的 UPDATE/HOLD/IGNORE、affected set 和 patch 标签，因此训练数据由干净轨迹、oracle state 和观察扰动层共同生成。

流程如下：

1. 在干净环境中保存 world state、工具动作、观察和终局结果；
2. 从工具 schema 和任务检查器抽取任务依赖字段；
3. 每次只改变一种因素，生成 paired counterfactual；
4. 用 oracle state、来源规则、时间规则和风险包络生成结构化 target；
5. 为每条轨迹保存 `scenario_id` 和从 0 连续递增的 `step_index`，时间顺序不得由动作标签或文件行号推断；
6. 对冲突、隐式失效和边界风险案例进行人工复核；
7. 按任务模板、实体、工具 schema 和基础 trajectory 分组切分数据。

结构化标签根据“部署时可获得的信息是否足以授权状态变化”生成，而不是让标注策略直接读取 oracle truth 后做全知判断：

- UPDATE：观察与目标相关，实体和条件匹配，来源与时效达到当前风险等级的准入要求，并且观察内容与 oracle state 一致；
- IGNORE：仅凭可见元信息就能确定观察无关、过期、对象错误、条件不适用，或已经被更高权威且更新的证据否定；
- HOLD：观察可能影响任务，但来源、时间、条件或冲突关系不足以支持承诺，且保留候选值或验证仍有正价值。

oracle world state 用于确认标签和计算结果成本，不得作为模型输入。若一个错误观察在文本、来源、时间和条件上与真实官方观察完全不可区分，任何准入策略都无法从当前输入识别它。这类样本单列为不可辨识噪声压力测试，不混入主要 SFT 标签，也不用于宣称策略能够判断开放世界中的绝对真假。

扰动算子包括：

- stale observation；
- wrong entity；
- condition mismatch；
- source downgrade；
- partial observation；
- conflict with trusted state；
- irrelevant distractor；
- delayed correction；
- duplicate observation；
- out-of-order observation。

推荐首轮规模：500 条人工复核 pilot cases、30k–80k step-level SFT 样本、5k–10k 多步基础轨迹、10k–20k RL rollout episodes。主要结果至少使用三个随机种子。

## 14. 环境和数据来源

主实验使用 [tau2-bench](https://github.com/sierra-research/tau2-bench) 的 airline、retail 和 telecom 环境。它提供工具、策略规则、用户交互和可验证的数据库终态，适合评测多步任务成功和状态依赖错误。

机制实验使用 [ScienceWorld](https://github.com/allenai/ScienceWorld)。它有可控世界状态、标准交互接口和较长任务轨迹，适合准确构造过时、错序和延迟纠正观察。

OOD 实验可使用 [ALFWorld](https://github.com/alfworld/alfworld) 或 [WebArena](https://github.com/web-arena-x/webarena)。OOD 环境不进入主要 SFT 数据，用于检验 schema 和语言分布变化下的泛化。

ABBEL 报告了 belief bottleneck 中状态更新错误沿轨迹传播的问题，可作为最接近的方法动机与比较对象：[ABBEL](https://arxiv.org/abs/2512.20111)。tau-bench 的数据库终态评测和 pass\(^k\) 可靠性指标可直接用于主要环境：[tau-bench paper](https://arxiv.org/abs/2406.12045)。

正式实验固定依赖版本、任务修订版本和数据 checksum。论文中不把未来可能变化的 leaderboard 数值写成静态事实。

## 15. 评测指标

### 15.1 Step level

- decision macro-F1；
- UPDATE、HOLD、IGNORE precision 和 recall；
- False Update Rate；
- Missed Update Rate；
- affected-field precision、recall 和 F1；
- patch exact match 和 patch validity；
- verification selection accuracy；
- ECE 和 Brier score。

### 15.2 State level

- State–World Consistency；
- trusted-state contamination rate；
- contamination area under trajectory；
- collateral edit rate；
- unaffected-field preservation；
- stale-state retention；
- recovery latency；
- contamination propagation length。

### 15.3 Trajectory level

- Task Success Rate；
- pass\(^k\)；
- unsafe 或 irreversible action rate；
- verification calls per successful task；
- total tool calls；
- token、wall-clock 和 simulated latency cost；
- clean-to-noisy degradation；
- fixed-budget success；
- safety–utility Pareto frontier。

统计报告使用 bootstrap 95% 置信区间。同一任务上的系统比较使用 paired bootstrap；离散 paired outcome 使用 McNemar test。超参数只根据 dev set 选择。

## 16. Baselines 与消融

主要 baselines：

- Frozen Vanilla Agent；
- strong prompt-only revision；
- pure generative LLM JSON policy；
- frozen encoder + MLP gate；
- rule-based admission gate；
- free-form belief rewrite；
- Always Update、Always Hold/Verify、Always Ignore；
- Oracle Revision Policy；
- SFT-only；
- SFT + offline preference optimization；
- SFT + constrained GRPO。

主要消融：去掉 provenance、time、entity/condition、risk envelope、affected-field head、typed patch、deterministic Executor、verification budget 或 KL regularization。另做一个旁路消融，允许 Action Agent 直接读取原始观察，以测量可信状态隔离是否真正发挥作用。

所有比较固定 Action Agent、base revision backbone、工具集合、任务版本、上下文预算、验证预算和最大环境步数。

## 17. 算力与模型规模

目标资源为两张 A100 80GB 或四张 A100 40GB。主实验采用 7B/8B backbone 的 BF16 LoRA：

- 两张 80GB 卡优先使用 data parallel 或一张 rollout、一张 learner 的配置；
- 四张 40GB 卡使用 FSDP 或 ZeRO-3 分片；
- Action Agent 推理服务和 Revision learner 分离，避免训练时误更新 Action Agent；
- pilot 阶段先用轻量 structured policy 跑通状态机和奖励，再切换 HF LoRA backend；
- checkpoint 只保存 LoRA、heads、optimizer、normalizer 和 Lagrange multipliers。

论文比较在相同 backbone 上完成。若使用更大的 Action Agent，只能作为额外扩展，不能与主消融混用。

## 18. 软件结构与可移植性

仓库采用 `src` layout：

```text
SIEVE/
├── pyproject.toml
├── README.md
├── configs/
│   ├── toy.yaml
│   ├── sft.yaml
│   └── rl.yaml
├── docs/
│   └── training_design.md
├── scripts/
│   ├── run_generate.sh
│   ├── run_sft.sh
│   ├── run_rl.sh
│   ├── run_eval.sh
│   └── run_all.sh
├── src/sieve/
│   ├── core/
│   ├── policies/
│   ├── environments/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── cli/
└── tests/
```

`cli` 表示 command-line interface，只负责解析参数、加载配置和调用库函数。训练、转移和指标逻辑放在可测试的模块中，不写进命令入口。

路径规则：

1. Python 源码中不写机器绝对路径；
2. 包内导入使用相对导入或已安装包名 `sieve`；
3. 配置中的数据、模型和输出路径相对仓库根目录；
4. Bash 脚本根据自身位置动态计算唯一的 `ROOT_DIR`；
5. Bash 入口先 `cd "$ROOT_DIR"`，再使用 `python -m sieve.cli...`；
6. `run_all.sh` 在 toy environment 上完成数据生成、SFT、RL 和评测闭环；
7. HF 模型、tau2-bench 和 ScienceWorld 通过配置接入，不写死在核心包中。

Bash 入口统一使用：

```bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python -m sieve.cli.train_sft --config configs/sft.yaml
```

## 19. 测试与验收

代码按测试先行实现。核心测试包括：

- IGNORE 不改变状态；
- HOLD 不覆盖 trusted value；
- UPDATE 不能修改 affected set 外字段；
- 非法 patch 被拒绝并计成本；
- pending 字段阻断依赖它的高风险动作；
- slot 和 ledger 容量固定；
- 原子观察顺序和 \(t/k\) 索引正确；
- 势函数奖励只计算一致性变化；
- Lagrange multiplier 非负并朝约束违反方向更新；
- action mask 后无效分支不进入 log probability；
- SFT 和 RL checkpoint 只包含允许训练的参数；
- `run_all.sh` 从非仓库当前目录调用时仍能找到根目录；
- 配置和 Python 文件不包含本机绝对路径。

轻量验收要求：全新环境安装项目后，执行 `bash scripts/run_all.sh` 能完成 toy 数据生成、SFT、约束 RL 和评测，并生成机器可读 metrics 文件。真实 7B/8B 训练由独立配置启用，避免把模型下载和 GPU 需求强加给单元测试。

## 20. 失败判据

以下结果会削弱论文的主要主张，实验必须如实报告：

- strong prompt-only 与 Oracle 的差距很小；
- SFT 相比 prompt-only 没有稳定降低 False Update Rate；
- constrained RL 只增加成本，没有改善长期成功或 Pareto frontier；
- 结构化策略的收益来自更大的上下文或更多工具调用；
- MLP gate 与 LLM revision policy 表现相当；
- typed Executor 的结构保证没有转化为轨迹级收益；
- clean 环境性能下降大于 noisy 环境的鲁棒性收益。

若第二阶段没有稳定收益，论文应把 RL 降为负结果或附录分析，把贡献集中在观察准入、结构化状态修正和确定性执行边界上。
