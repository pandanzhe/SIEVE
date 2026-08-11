# SIEVE 阶段 2：Qwen2.5-3B 受约束多步 GRPO 实验方案

## 1. 阶段目标与训练本质

阶段 2 不是再训练一个 MLP，也不是把阶段 1 的三个分支改成新的分类器。训练对象仍是 Qwen2.5-3B 自回归策略，并从 `outputs/stage1-qwen25-3b/best/adapter/` 加载阶段 1 的 LoRA 参数。RL 只更新该 LoRA；基础模型保持冻结，阶段 1 保存的结构化辅助 heads 不进入动作概率，因此在本阶段不加载、不更新。

策略在每一步读取可见上下文，生成一个完整 JSON 认知动作：

\[
u_t=(c_t,z_t,\Delta B_t,q_t),
\qquad c_t\in\{\mathrm{UPDATE},\mathrm{HOLD},\mathrm{IGNORE}\}.
\]

这里的 JSON 仍只有四个字段 `decision`、`affected_fields`、`patches` 和 `verification`。所谓三个训练分支是对生成区间和功能的解释，不是三个并行大模型。RL 使用整个 completion 的 token log-probability 优化同一个生成策略。

## 2. 为什么此处是一个 POMDP，以及代码采用什么状态

环境真实状态和 oracle 对策略不可见，因此完整问题按 POMDP 表述更严谨。代码交给策略的是有界信息状态：

\[
s_t=(B_{t-1},o_t,g_t,r_t,L_t,\rho_t).
\]

- \(B_{t-1}\)：固定最大槽位数的可执行信念；时间增加时更新槽位内容、状态、时间和来源，不增加向量维度。
- \(o_t\)：当前观察。
- \(g_t\)：当前目标。
- \(r_t\)：风险包络。
- \(L_t\)：固定容量证据账本，满容量后淘汰最旧条目。
- \(\rho_t\)：剩余步骤、工具、验证和 token 预算。

如果上述状态能充分概括与后续决策相关的历史，则可在信息状态上采用近似 Markov 假设：

\[
P(s_{t+1}\mid s_{0:t},u_{0:t})
\approx P(s_{t+1}\mid s_t,u_t).
\]

这不等于 Agent 看到了环境潜在真值。私有 oracle 和期望语义只由环境读取，`render_context_prompt` 只序列化 `RevisionContext`。

## 3. 动作条件化状态转移

环境实现为 `ScenarioRevisionEnvironment`。一轮基本转移为：

\[
B_t=U(B_{t-1},u_t,o_t),
\qquad
s_{t+1}\sim P(\cdot\mid s_t,u_t).
\]

其中 `StateExecutor` 是唯一状态写入点。不同动作产生不同后果：

- `UPDATE`：合法 patch 改写信念；错误 UPDATE 会污染后续状态，并产生安全代价。
- `IGNORE`：信念不变，进入下一基础事件。
- `HOLD + verification`：只有请求合法、工具可用、预算充足且该事件存在验证证据时，才进入条件验证微循环；验证观察不会无条件出现在下一行。
- 非法 JSON：转成安全的 `IGNORE` 以保证 episode 可继续，同时记录 `invalid_format=1`。

“严格 JSON”还包括拒绝重复 key、`NaN`、`Infinity` 和尾随说明文字；解析失败的文本不能借由 Python JSON 的宽松行为绕过 `invalid_format` 代价。

验证请求中的 `tool` 与验证返回观察的 `source` 是两个不同变量。构造器将与阶段 1 一致的字段级工具名（如 `verify_origin`）写入事件的 `verification_tool`，并通过初始信念中的 `__verification_policy.tool_by_field` 暴露给策略；验证返回仍可来自独立的 `official_verification_api`。环境同时校验工具名、字段名、当前观察字段和预算，未知工具或错字段不能打开私有证据分支。

正确轨迹通常包含 5 个策略步：模糊观察、验证返回、错误实体、权威执行状态、陈旧冲突。预算上限为 8 步；环境还按每次 completion 的实际 token 数扣减 `tokens_remaining`。因此错误 HOLD、重复验证或过长输出都会反映在下一时刻的可见预算中。

## 4. 正式 Stage-2 数据

### 4.1 持久化文件

```text
data/rl/
├── manifest.json
├── quality_report.json
└── scenarios/
    ├── train.jsonl   # 1600 episodes
    ├── dev.jsonl     #  200 episodes
    └── test.jsonl    #  300 episodes
```

每一行是一个环境 episode 定义，不是 SFT 的 prompt–answer 样本。顶层没有 `target`。文件分为：

- `initial_context`：初始信念、目标、风险、账本容量和预算；
- `environment_private`：oracle、基础事件、条件验证证据和环境期望语义。

后者只用于环境转移、奖励与代价，不能拼入 prompt。测试集是本项目从开源源记录构造的内部测试集，不应宣称为某个 benchmark 的官方 test split。正式论文仍应在外部保留任务上报告迁移结果。

### 4.2 数据来源与构造方式

生成器完全离线，不调用在线模型。它从 `data/sft/source/records.jsonl` 的 6000 条可追溯记录中筛选 4200 条“已接受、主记录已认证、目标为 UPDATE 且有 SET_VALUE patch”的源记录，然后执行确定性规则构造。

底层任务 ID 优先使用 `source_dataset + source_record_id`，先删除末尾字段名，再进行 SHA-256 切分。这样同一任务的验证变体和增强变体不会跨 train/dev/test。当前正式数据包含 1460 个底层任务；最终使用的不同任务数为 train 822、dev 118、test 156，交集为 0。同一 split 内允许产生多个场景变体。

宏领域比例固定为：

| 宏领域 | 比例 | train / dev / test |
|---|---:|---:|
| 商务与购物 | 50% | 800 / 100 / 150 |
| 航空与电信服务 | 30% | 480 / 60 / 90 |
| 知识与工作流 | 20% | 320 / 40 / 60 |

源记录当前每个底层任务只覆盖一个可执行字段。为使阶段 2 真正检验跨步状态维护，构造器显式增加一个领域化环境执行字段：商务使用 `fulfillment_readiness`，服务使用 `service_request_status`，工作流使用 `workflow_record_status`。该字段在 provenance 中标记为 `derived_secondary_field=true`，不是伪装成开源原始标签。

一个 episode 的四个基础事件是：

1. 当前实体、当前源字段、低权威且未认证的变化：应 `HOLD` 并验证；
2. 同领域、优先同字段的其他实体记录：应 `IGNORE`；
3. 当前实体的权威执行状态：应 `UPDATE`；
4. 同一执行字段的更旧冲突：应 `IGNORE`。

事件 1 的权威验证返回只在策略实际请求验证时出现。由此，单条静态标签被转换为动作会影响后续观察和终局结果的多步 episode。

错误实体与陈旧值的 donor 只能来自同一个 split，`provenance.distractor_base_task_ids` 显式记录 donor 的底层任务 ID。preflight 会同时检查主体任务和干扰项任务的跨 split 泄漏；缺失 donor provenance 也视为错误。

### 4.3 构造命令

```bash
bash scripts/build_stage2_data.sh
```

生成器写入 SHA-256、领域计数、事件计数和任务交叉检查。相同源文件、seed 和代码应得到相同哈希。正式配置通过 `expected_split_counts` 将 1600/200/300 写成硬门槛；即使一个小型 debug corpus 的 manifest 自洽，也不能通过正式训练 preflight。

## 5. 奖励、约束与优化目标

任务势函数为依赖字段与私有 oracle 的一致率：

\[
\Phi(B_t,x)=\frac{1}{|D|}\sum_{f\in D}
\mathbb 1[B_t(f)=x(f)\land \mathrm{status}(f)=\mathrm{trusted}].
\]

单步奖励为：

\[
r_t=\gamma\Phi(B_t,x)-\Phi(B_{t-1},x)
+\mathbb 1[\text{terminal success}],
\qquad \gamma=0.97.
\]

轨迹单独累计以下代价：

| 代价 | 含义 | 默认上限 |
|---|---|---:|
| `false_update` | 非 UPDATE 语义上执行 UPDATE | 0.03 |
| `unsafe_action` | 不可逆场景中的错误 UPDATE | 0.01 |
| `verification` | 每条轨迹实际执行的验证次数 | 1.00 |
| `stall` | HOLD 但没有可执行验证分支 | 0.10 |
| `invalid_format` | completion 不是严格 JSON 动作 | 0.02 |
| `invalid_patch` | Executor 拒绝 patch | 0.01 |
| `collateral_edit` | 修改 affected_fields 之外字段 | 0.00 |
| `budget_violation` | 预算耗尽后仍请求工具 | 0.00 |

受约束轨迹分数为：

\[
\widetilde R_i=R_i-\sum_j\lambda_j C_{ij},
\qquad
\lambda_j\leftarrow
\max\{0,\lambda_j+\eta_\lambda(\bar C_j-d_j)\}.
\]

各项 \(C_{ij}\) 是未除以轨迹长度的累计代价；拉格朗日乘子使用 batch 内的平均轨迹累计代价更新，避免较长 episode 稀释错误。终局评测表中的 rate 仍按动作数归一化。

同一初始 episode 采样 \(G=4\) 条轨迹，组内优势为：

\[
A_i=\frac{\widetilde R_i-\operatorname{mean}(\widetilde R)}
{\operatorname{std}(\widetilde R)+\epsilon}.
\]

每个动作 completion 的 token 共享该轨迹优势。更新使用 clipped ratio 和冻结参考策略的 sampled KL：

\[
\mathcal L=
-\mathbb E\left[\min(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A)\right]
+\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}}).
\]

参考策略是阶段 1 adapter 的冻结副本。训练策略和参考策略都读取同一 prompt；只有训练策略的 LoRA 接收梯度。

实现按整个梯度累积窗口、所有 DDP rank 的有效 completion token 总数归一化损失，避免不同长度 micro batch 的 mean-of-means 偏差。rollout 与可微更新都关闭 dropout，但更新阶段仍保持 `train()`，从而保留 gradient checkpointing；采样显式设置 temperature=1、top-p=1、top-k=0，并中和 repetition/typical 等 logits warper，使 behavior log-probability 与更新中使用的原始 logits 属于同一分布。

## 6. Stage-1 进入 RL 的硬门槛

已有的低 dev loss 和接近 1 的 teacher-forced head 指标不足以证明模型能闭环生成 JSON。服务器补齐 `model/` 和阶段 1 checkpoint 后，先运行：

```bash
bash scripts/check_stage1_readiness.sh
```

它在 `data/sft/clean/dev.jsonl` 上做 greedy 自由生成，在 Stage-2 dev 场景上做随机 group probe，并默认抽取 100 个 Stage-2 dev episode 做 greedy 闭环检查，写入 `outputs/stage1-qwen25-3b/readiness.json`。默认门槛为：

| 指标 | 门槛 |
|---|---:|
| JSON parse rate | ≥ 0.98 |
| Executor executable rate | ≥ 0.97 |
| decision macro-F1 | ≥ 0.90 |
| 完整动作 exact match | ≥ 0.85 |
| UPDATE patch-value exact match | ≥ 0.85 |
| false-update rate | ≤ 0.03 |
| group reward variance | > 0 |
| Stage-2 greedy 闭环成功率 | ≥ 0.60 |
| Stage-2 greedy 闭环 parse rate | ≥ 0.95 |

任何一项未通过，脚本以非零状态退出；正式 RL 入口也会再次读取该报告并拒绝训练。报告绑定 Stage-1 adapter、基础模型/tokenizer 资产以及 Stage-1/Stage-2 dev 数据指纹，任一资产变化都必须重新验收。建议分别把配置指向阶段 1 的 `best` 与 `final` 运行此流程，选择自由生成结果更好的 checkpoint，而不是仅按旧的 head score 选择。

## 7. 硬件与默认参数

默认服务器配置为 2×A100 40G、BF16、DDP。每张卡各持有一个可训练 Qwen2.5-3B LoRA 策略和一个冻结参考副本，并独立采样完整 GRPO groups。Qwen2.5-3B 下两卡是合适的稳妥配置；单卡 40G 也可运行，但 rollout 和参考模型前向串行，速度明显更慢。

关键参数位于 `configs/rl_qwen25_3b.yaml`：

| 参数 | 默认值 |
|---|---:|
| group size | 4 |
| 每卡每 iteration 的 groups | 2 |
| max episode steps | 8 |
| temperature / top-p / top-k | 1.0 / 1.0 / 0（保证 behavior 与更新分布一致） |
| iterations | 200 |
| micro batch | 2 |
| gradient accumulation | 4 |
| LoRA learning rate | 5e-6 |
| clip ratio / KL coefficient | 0.20 / 0.02 |
| eval / save interval | 20 iterations |

学习率使用按外层 rollout iteration 索引的 warmup + cosine 曲线，而不是按可变长度 episode 产生的 optimizer step 数索引。这样单卡和双卡不会因 Accelerate 的分布式 scheduler 步进规则而得到不同曲线。

## 8. 服务器执行顺序

```bash
# 1. 安装；模型与 Stage-1 adapter 均只从本地读取
python -m pip install -r requirements/rl.txt
python -m pip install -e .

# 2. 数据可重复生成；仓库已包含生成后的正式文件时可只做检查
bash scripts/build_stage2_data.sh

# 3. 单卡自由生成验收 Stage 1
bash scripts/check_stage1_readiness.sh

# 4. 静态 preflight；不加载模型、不占 GPU
bash scripts/validate_stage2.sh

# 5. 默认双卡训练
bash scripts/train_stage2.sh

# 单卡备选
SIEVE_NUM_GPUS=1 bash scripts/train_stage2.sh

# 6. 固定 checkpoint 后运行一次内部 test
bash scripts/evaluate_stage2.sh \
  configs/rl_qwen25_3b.yaml outputs/stage2-qwen25-3b/best test
```

训练输出写入 `outputs/stage2-qwen25-3b/`：`audit.json`、`metrics.jsonl`、`best/`、`final/` 和有限数量的 `state-*` 断点。恢复训练时把配置中的 `train.resume_from` 改为同一输出目录下相对的最新 `state-*` 路径，并保持 seed、GPU 数、数据和全部训练参数不变；若要改变实验设置，应使用新的输出目录重新训练。断点同时恢复模型、优化器、iteration、拉格朗日乘子、best dev score 和确定性场景抽样位置。缺少 `controller.json` 的断点会直接拒绝恢复。Accelerate 状态可能包含完整的 PEFT 包装模型，默认只保留 2 个 `state-*`；服务器应额外预留约 15–20 GB 的 checkpoint 空间。

## 9. 训练期与最终评测

训练期每个 iteration 记录：任务 return、受约束分数、组内方差、parse rate、终局成功率、policy loss、sampled KL、八项平均代价和对应拉格朗日乘子。每 20 iterations 在 dev 上做 greedy 闭环评测，根据：

\[
\mathrm{dev\ score}=\mathrm{success}
-\mathrm{false\ update\ rate}
-\mathrm{invalid\ format\ rate}
\]

保存最佳 adapter。

最终报告至少包含：

- dev 与内部 test 的终局状态成功率、平均 return、JSON parse rate、false-update rate、invalid-format rate；
- 各宏领域分项结果；
- SFT-only、无约束 GRPO、完整 constrained GRPO 三个主对比；
- 去掉验证微循环、去掉拉格朗日约束、把条件验证改成静态下一行三个消融；
- 3 个随机种子的均值和置信区间。

内部 test 只在配置和门槛固定后运行一次，不能参与 checkpoint 选择。论文的外部有效性还需要在未参与规则构造的真实多步 Agent 环境中验证。

## 10. 代码对应关系

| 理论对象 | 核心代码 |
|---|---|
| 场景 schema 与私有 oracle | `src/sieve/rl_data/schema.py` |
| 数据构造、任务切分、干扰项 | `src/sieve/rl_data/build.py` |
| 动作条件化 \(P\) | `src/sieve/environments/scenario_revision_env.py` |
| 唯一信念写入点 \(U\) | `src/sieve/core/executor.py` |
| 严格 JSON 动作协议 | `src/sieve/policies/hf_revision_io.py` |
| 奖励势函数 | `src/sieve/training/rewards.py` |
| GRPO 配置与数学 | `src/sieve/training/hf_grpo_config.py`、`hf_grpo_math.py` |
| Qwen rollout、reference、更新和断点 | `src/sieve/training/hf_grpo.py` |
| Stage-1 自由生成门槛 | `src/sieve/training/hf_readiness.py` |
| 闭环 dev/test 评测 | `src/sieve/evaluation/hf_stage2.py` |
| CLI | `src/sieve/cli/build_rl_data.py`、`check_stage1_readiness.py`、`train_rl.py` |
