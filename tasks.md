# SIEVE 后续任务清单

## 0. 当前主线

SIEVE 的论文主线应保持清晰：

- 问题创新：不可靠 observation 被直接写入可执行状态，会在多步 Agent 中造成持续污染和延迟错误。
- 机制创新：`Revision Policy + Structured Belief + Deterministic Executor + Verification Loop`。
- 训练路线：SFT 学习单步裁决语义，RL 学习裁决动作的跨步后果。
- 算法路线：先用已有 constrained GRPO 建立强 baseline，再根据状态重复率和分叉可行性，升级到 constrained GiGPO 或 Tree-GRPO-style 分叉增强。

一句话价值表述：

> SIEVE 研究的不是如何让 Agent 想得更聪明，而是如何阻止 Agent 在多步行动之前，把未经充分支持的信息变成自己据以行动的事实。

## 1. 当前状态

### 已完成

- Stage-1 Qwen3-4B SFT 已训练结束。
- `outputs/stage1-qwen3-4b/best` 和 `outputs/stage1-qwen3-4b/final` 均已保存。
- SFT 日志显示 teacher-forcing dev 指标接近满分：
  - `decision_accuracy = 1.0`
  - `decision_macro_f1 = 1.0`
  - `false_update_rate = 0.0`
  - `best_score ~= 0.999981889`
- SFT 日志已补充 step-level loss 打印能力：
  - `train_loss`
  - `train_decision_loss`
  - `train_structure_loss`
  - `train_value_loss`
- Readiness 脚本已补充进度日志：
  - SFT dev free-generation 开始、进度、结束
  - GRPO group probe 开始、进度、结束
  - Stage-2 closed-loop 开始、每 10 个 scenario 进度、结束

### 待确认

- `outputs/stage1-qwen3-4b/readiness.json` 是否生成。
- `ready_for_stage2` 是否为 `true`。
- Stage-2 closed-loop 中：
  - `closed_loop_success_rate`
  - `closed_loop_parse_rate`
  - `closed_loop_false_update_rate`
  - `group_reward_variance`

## 2. 立即执行顺序

### Step 1：完成 Stage-1 readiness

先确认 SFT adapter 在自由生成和闭环环境中能用，不要只相信 SFT loss。

```bash
cd /mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE
mkdir -p logs
LOG=/mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE/logs/stage1_readiness_qwen3_4b_full_$(date +%Y%m%d_%H%M%S).log
nohup bash -lc 'set -euo pipefail; cd /mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE; export CUDA_VISIBLE_DEVICES=0; export SIEVE_NUM_GPUS=1; export NCCL_IB_DISABLE=1; export PYTHONPATH=src; export PYTHONUNBUFFERED=1; bash scripts/check_stage1_readiness.sh configs/rl_qwen3_4b.yaml' > "$LOG" 2>&1 &
echo "PID=$!"
echo "LOG=$LOG"
tail -f "$LOG"
```

通过标准：
- `ready_for_stage2 = true`
- `parse_rate >= 0.98`
- `executable_rate >= 0.97`
- `decision_macro_f1 >= 0.90`
- `full_action_exact_match >= 0.85`
- `patch_value_exact_match >= 0.85`
- `false_update_rate <= 0.03`
- `group_reward_variance > 0`
- `closed_loop_success_rate >= 0.60`
- `closed_loop_parse_rate >= 0.95`

### Step 2：Stage-2 静态校验

readiness 通过后，再跑 Stage-2 preflight。

```bash
cd /mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE
PYTHONPATH=src bash scripts/validate_stage2.sh configs/rl_qwen3_4b.yaml
```

### Step 3：运行当前 constrained GRPO baseline

先用 1 卡 A100 跑通，降低 DDP/NCCL 干扰。

```bash
cd /mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE
mkdir -p logs
LOG=/mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE/logs/stage2_grpo_qwen3_4b_1xa100_$(date +%Y%m%d_%H%M%S).log
nohup bash -lc 'set -euo pipefail; cd /mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE; export CUDA_VISIBLE_DEVICES=0; export SIEVE_NUM_GPUS=1; export NCCL_IB_DISABLE=1; export PYTHONPATH=src; export PYTHONUNBUFFERED=1; bash scripts/run_rl.sh configs/rl_qwen3_4b.yaml' > "$LOG" 2>&1 &
echo "PID=$!"
echo "LOG=$LOG"
tail -f "$LOG"
```

### Step 4：评测 SFT-only 与 RL checkpoint

需要至少报告：

- Stage-2 dev
- Stage-2 test
- SFT-only
- SFT + constrained GRPO

后续再加入 Tree/GiGPO-style 方法。

## 3. 当前 RL 数据判断

当前数据足够跑第一版 RL baseline，但还不足以支撑强论文结论。

### 当前规模

- `data/rl/scenarios/train.jsonl`：1600 episodes
- `data/rl/scenarios/dev.jsonl`：200 episodes
- `data/rl/scenarios/test.jsonl`：300 episodes

### 当前 base task 数

- train：152
- dev：27
- test：33

### 判断

- episode 数量够跑 constrained GRPO。
- base task 数偏少，容易被质疑为学习规则模板或同任务变体。
- 当前阶段不需要马上搭完整外部环境。
- 顶会版本最终需要外部 held-out 或更多 base task。

## 4. 数据补强任务

### 4.1 增加 base task diversity

优先增加不同底层任务，而不是只给同一任务造更多扰动。

目标规模：

- train base tasks：不少于 500
- dev base tasks：不少于 100
- test base tasks：不少于 150

如果短期做不到，论文和实验报告必须显式报告 base task 数，不能只报告 episode 数。

### 4.2 增加状态污染类型

当前主要覆盖：

- `source_ambiguity`
- `wrong_entity`
- `authoritative`
- `stale_conflict`

建议新增：

- `condition_mismatch`
- `partial_observation`
- `delayed_correction`
- `duplicate_observation`
- `conflicting_authoritative_sources`
- `budget_limited_verification`
- `high_risk_irreversible_action`

这些类型更能证明状态污染不是简单规则可以完全解决的问题。

### 4.3 统计分叉可学习性

在决定是否做 GiGPO/Tree-GRPO 前，需要统计：

- 同一 state 下 `UPDATE/HOLD/IGNORE` 的 reward gap。
- `HOLD+VERIFY` 是否真的改变后续 observation。
- verification 后 `UPDATE` 是否提高 terminal success。
- 错误 `UPDATE` 后污染是否持续到终局。
- `group_reward_variance` 是否明显大于 0。

如果 reward variance 很低，GRPO/GiGPO 都会缺少学习信号，应先补数据或改采样。

## 5. RL 算法路线

### 5.1 当前 baseline：constrained GRPO

当前 repo 已有 `hf_grpo.py`，它是受约束 trajectory-level GRPO：

- 同一个 scenario 采样多条完整轨迹。
- 用 trajectory return 减去 Lagrange cost penalty 得到 penalized score。
- 在 group 内做 relative advantage。
- 保留安全约束：
  - `false_update`
  - `unsafe_action`
  - `verification`
  - `stall`
  - `invalid_format`
  - `invalid_patch`
  - `collateral_edit`
  - `budget_violation`

这可以作为第一版强 baseline。

### 5.2 主方法候选：SIEVE-CGPO

建议主方法不是重新发明一个完全独立的 RL 名称，而是在 GiGPO/Tree-GRPO 思路上做 SIEVE 问题必要的改造：

```text
SIEVE-CGPO = constrained GiGPO / Tree-GRPO-style credit assignment
```

核心改造：

- 在同一 `RevisionContext` 下分叉采样多个 cognitive actions。
- 共享前缀，比较不同动作导致的长期 belief 和 task return。
- 对每个分叉节点做局部优势估计。
- 保留 SIEVE 约束成本和 deterministic executor。
- 将优势从整条轨迹粗分，改成状态/节点级信用分配。

### 5.3 ARPO 的定位

ARPO 可以作为采样增强或 baseline，不建议作为主方法核心。

原因：

- ARPO 重点是工具调用后高熵位置的 adaptive rollout。
- SIEVE 的核心是 observation 是否进入 executable belief，以及错误写入导致的跨步污染。
- Tree/GiGPO 的同状态分叉和局部优势归因更贴合 SIEVE。

可借鉴 ARPO 的部分：

- 在 verification 返回后增加采样预算。
- 在模型动作熵高的位置增加分叉。
- 对不确定节点做更多 rollout，而不是平均分配 rollout budget。

## 6. Baseline 设计

### 6.1 系统 baseline

- Base LLM zero-shot：不训练，直接输出 revision action。
- Append-All Context：所有 observation 直接写入或直接追加上下文。
- Summarize Memory：用普通摘要维护状态，不做结构化 executor。
- Rule-based SIEVE：基于 entity/source/time/authentication 的硬规则。

### 6.2 训练 baseline

- SFT only：只使用 Stage-1 adapter。
- SFT + vanilla GRPO：无 Lagrange constraints。
- SFT + constrained GRPO：当前 repo 已实现，作为强 baseline。
- SFT + PPO 或 REINFORCE：可选，作为非 group-relative 对照。

### 6.3 多步 credit assignment baseline

- Tree-GRPO without constraints。
- ARPO-style adaptive rollout。
- SIEVE-CGPO：主方法。

## 7. 评测指标

主指标：

- `success_rate`
- `mean_return`
- `belief_consistency`
- `false_update_rate`
- `unsafe_action_rate`

协议和稳定性指标：

- `parse_rate`
- `executable_rate`
- `invalid_format_rate`
- `invalid_patch_rate`
- `collateral_edit_rate`

资源和效率指标：

- `verification_rate`
- `stall_rate`
- `budget_violation_rate`
- average steps
- average verification count

论文中应同时报告总体指标和分污染类型指标。

## 8. 决策门槛

### 是否进入 Stage-2 RL

只有 readiness 通过后再进入 RL。若不通过，先修 SFT 生成协议。

### 是否扩数据

如果出现以下情况，优先扩数据，而不是先改算法：

- SFT-only closed-loop success 已经很高。
- `group_reward_variance` 很低。
- false update 样本太少。
- dev/test 的 base task 数不足以支持泛化结论。

### 是否升级到 GiGPO/Tree-GRPO

如果 constrained GRPO 相比 SFT-only 有提升，但存在以下问题，再升级：

- 收敛不稳定。
- 长轨迹错误归因粗。
- 关键 HOLD/VERIFY 节点学得慢。
- 终局 reward 能区分轨迹，但无法定位具体错误 step。

## 9. 当前下一步

1. 等 `stage1_readiness_qwen3_4b_*.log` 跑完。
2. 检查 `outputs/stage1-qwen3-4b/readiness.json`。
3. 如果 `ready_for_stage2=true`，运行 `validate_stage2.sh`。
4. 启动 `SFT + constrained GRPO` baseline。
5. 保存 RL dev/test 结果，作为后续 Tree/GiGPO-style 方法的对照。



当前 idea 有顶会潜力，但还没有达到“只要把实验跑完、文章写好就够了”的程度。现在最重要的不是继续润色文字，而是通过实验把一个独占性的科学命题做实。
1. 如果我是审稿人，我会怎样评价
Strengths
我会认可三点：
1. 问题是真实的：多步 Agent 的 observation 会进入记忆或状态，并持续影响后续动作。
2. 因果链很清楚：错误观察不是只造成一次错误回答，而可能形成持久状态污染。
3. 方法具备可审计性：Revision Policy 与确定性执行器分离，比自由文本 memory rewrite 更容易分析和约束。
这个问题的实际意义是足够的，特别是在工具 Agent、安全操作和不可逆动作场景中。
Weaknesses
但我会提出四个比较致命的问题。
问题一：最近邻已经非常接近
2025—2026 年，“显式 belief + action 解耦 + RL”已经变得拥挤。
- ABBEL 已经研究 prior belief、observation、posterior belief 和 RL belief updating。
- Agent-BRACE 已经把 belief model 与 policy model 解耦，并用带不确定性的结构化 atomic claims 表示 belief；它还直接在 Qwen3-4B 上做了实验。
- BeliefMem 已经指出把 observation 压成单一确定结论会造成 self-reinforcing error，并维护多个候选 belief。
- T³ 已在 ICLR 2026 研究多轮主动推理中的 belief deviation 和 RL。
- ClarifyBench/SAGE-Agent 已经研究什么时候应澄清，并用 uncertainty-weighted GRPO 训练。
- 更危险的是尚未经过顶会同行评审、但已经公开的 ReSSERAct：它同样强调 stale/contradictory evidence、typed state、ledger 和 admissibility shield。
因此，如果论文贡献只写成：
我们维护结构化 belief，判断 UPDATE/HOLD/IGNORE，然后通过 RL 改善多步结果。

审稿人很可能认为：
“这是 ABBEL/Agent-BRACE/ClarifyBench/ReSSERAct 的一个组合或特定实现。”

这会是当前最大的 novelty 风险。
问题二：确定性执行器本身不是足够强的算法创新
以下性质是合理的，但比较容易通过程序实现：
- IGNORE 不修改状态；
- HOLD 不覆盖 trusted value；
- 只能修改 affected fields；
- schema 不允许任意字段；
- pending 状态阻塞高风险动作。
这些是很有价值的系统设计，却可能被审稿人评价为：
“Useful engineering constraints, but technically straightforward.”

尤其“未采纳信息不改状态”“无关字段保持不变”等所谓证明，本质上只是执行器根据定义保证的 invariants，不能作为主要理论贡献。
问题三：当前内部实验太容易形成闭环自证
当前数据的优点是可控，但也存在明显风险：
- 只有152个训练 base tasks；
- dev/test 只有27/33个 base tasks；
- 四类事件是人工定义的固定机制；
- 每个 episode 都含相同四类事件；
- reward、oracle decision 和扰动构造可能来自同一组规则；
- 模型可能学习事件类型和固定顺序，而不是一般的 evidence admission；
- Rule-based Gate 可能已经能解决大部分问题。
审稿人可能会说：
“作者先按照自己的规则生成问题，再按照相同规则生成标签和奖励，最后证明模型学会了这些规则。”

这是内部 benchmark 论文常见的拒稿理由。
问题四：目前还没有证明新 RL 算法是必要的
当前 constrained trajectory GRPO 已经是真实多步 RL。它可能已经足够好。
如果后续只是：
GRPO
+ GiGPO anchor
+ Lagrangian constraint
审稿人可能认为只是模块叠加。
反过来，如果普通 constrained GRPO 已经达到很高性能，那么“细粒度信用分配是核心贡献”的论点也会被实验否定。
因此现在不能同时假设：
- trajectory GRPO 信用分配不够；
- GiGPO 一定有效；
- Tree branching 一定必要。
这些都必须通过诊断证明。
2. 当前 idea 到底差在哪里
当前不是“idea 不好”，而是缺少一个足够尖锐的、只有 SIEVE 在回答的命题。
现在的说法仍然偏宽：
多步 Agent 应可靠地更新 belief。

这已经被很多工作覆盖。
应该收窄成：
Belief representation 解决“Agent 记住什么”，而 observation admission 解决“什么证据被授权改变可执行状态”。后者是一个依赖实体、时间、来源、下游风险和验证预算的序列控制问题，不能由一般 belief summarization、uncertainty tracking 或静态过滤替代。

这里最关键的词是：
authorized state transition

SIEVE 不应主要和普通 memory 方法争论“谁的 belief 更准确”，而应研究：
一个候选 observation
是否被授权改变
下游 action policy 能读取的状态
这才是最可能与 ABBEL、Agent-BRACE 和 BeliefMem 拉开差异的位置。
3. 现在够不够顶会
我的判断分成三个层次。
当前版本：不够
如果最终实验只有：
- 内部1600/200/300 episodes；
- Qwen3-4B；
- SFT；
- constrained GRPO；
- UPDATE/HOLD/IGNORE 指标；
- 少量消融；
那么更像一篇完整的 workshop、Findings 或早期主会投稿，但在强顶会中大概率会被质疑 novelty 和 external validity。
补外部实验，但算法仍是已有方法：有机会
如果能够做到：
- 在 τ2/τ3、AppWorld 或另一个真实 Agent 环境上成立；
- 与 ABBEL-style belief、structured uncertainty、rule gate、free-text memory 正面对比；
- 证明状态污染是独立失败模式；
- 证明 SIEVE 不是靠过度 HOLD 获得安全；
- 提供可复现的 observation-intervention benchmark；
- 多模型、多领域均成立；
那么它可以成为一篇较强的“问题 + 方法 + benchmark”论文。
这种路线不一定需要全新的 RL 算法，尤其适合 ACL/EMNLP，也可能进入 ICLR/NeurIPS，但实验必须非常扎实。
再加真正的局部信用分配创新：最强版本
如果进一步证明：
1. trajectory GRPO 存在系统性的 credit conflict；
2. 错误更新的长期影响可以通过同状态反事实分支测量；
3. 你提出风险定向的选择性分叉；
4. 它比 Tree-GRPO/ARPO/GiGPO 更节省 rollout；
5. 在相同工具预算下获得更好的安全—效用 Pareto frontier；
那么论文就有更强的算法贡献，适合 ICLR/NeurIPS 风格。
但这条路线风险更高、实现量更大，也不应在诊断前强行选择。
4. 最应该优化的是文章还是实验
答案是：
现在大约20%是文章定位问题，80%是科学实验和实现问题。

文档已经足以表达基本 idea。继续把 motivation 写得更长，不能解决 novelty 风险。
真正影响审稿结论的是：
- 是否有外部环境；
- 是否存在自然错误而非纯人工模板；
- 强规则能否解决；
- 强基础模型能否靠 prompt 解决；
- belief 方法能否替代；
- SFT 是否已经足够；
- RL 是否真的学到了长期权衡；
- 局部信用分配是否真的必要；
- 是否按 base task 做统计；
- 是否公平控制 rollout 和工具预算。
但是文章定位仍然必须提前固定，否则实验会发散。文章只应该围绕一个核心命题组织：
Observation admission is distinct from belief representation and action planning.

不要同时把论文写成：
- memory paper；
- Agent safety paper；
- RL algorithm paper；
- belief-state paper；
- verification paper；
- benchmark paper。
它们可以作为组成部分，但只能有一个中心。
5. 最值得投入的升级方向
我建议选择“问题与机制为主、算法由诊断决定”的路线，而不是现在就强行追求一个通用 RL 新算法。
升级一：把 benchmark 从“标签集”升级成“因果干预协议”
当前最有机会成为贡献的，不是四种事件名称，而是 paired counterfactual evaluation。
对同一个 base task 构造：
Clean trajectory
Corrupted observation trajectory
SIEVE-protected trajectory
Oracle-admission trajectory
并保持其他环境状态一致。
这样可以测量：
\[
\text{Observation}
\rightarrow
\text{State contamination}
\rightarrow
\text{Downstream action}
\rightarrow
\text{Task outcome}
\]具体应报告：
- corruption-induced performance drop；
- SIEVE recovered fraction；
- poison-to-state conversion；
- state-to-action conversion；
- contamination area；
- recovery latency；
- oracle admission upper bound。
这比普通 macro-F1 更具有论文辨识度。
升级二：证明不是规则就能解决
必须构造规则困难的场景：
- 多条弱证据组合后才足以 UPDATE；
- 来源权威，但条件或实体不适用；
- 低权威证据与高风险动作组合时应 HOLD；
- 相同证据在低风险任务中可 UPDATE，在不可逆任务中应先 VERIFY；
- 验证工具可能失败、延迟或耗尽预算；
- 旧信息曾经正确，但在状态变化后变得 stale；
- 多个来源在时间、范围和权限上交叉冲突。
如果规则基线与 SIEVE 接近，应该接受这个结果，而不是用更复杂的模板继续扩大模型优势。
升级三：改变 episode 结构，防止模板捷径
“每个 episode 都有四类固定事件”非常容易形成顺序捷径。
应随机化：
- 事件类型数量；
- 事件顺序；
- 是否缺失某类事件；
- 同类型事件重复次数；
- 冲突发生位置；
- horizon；
- observation 表面形式；
- 来源和实体组合；
- 多事件组合。
测试集还应保留未见组合，例如：
训练：ambiguous、wrong_entity 分别出现
测试：ambiguous + wrong_entity 同时出现
这能证明模型学到的是决策机制，而不是事件模板。
升级四：加入真正强的最近邻 baseline
除了 GRPO 系列，至少需要概念层面的对照：
- full-context Agent；
- free-text recursive summary；
- structured belief updater；
- uncertainty-aware belief；
- rule-based admissibility gate；
- prompt-only strong model；
- SFT-only；
- constrained trajectory GRPO。
理想情况下还应实现一个 ABBEL/Agent-BRACE 风格的近似基线：
所有 observation 都更新 belief，但 belief 带摘要或 uncertainty。

这能直接回答：
“为什么维护不确定 belief 还不够，必须有 admission boundary？”

升级五：增加外部环境和自然错误
优先级应该是：
1. τ2/τ3 未见任务模板；
2. τ2/τ3 留一领域；
3. AppWorld 一类有数据库状态和 collateral damage 的环境；
4. ALFWorld/ScienceWorld OOD；
5. AgentDojo 等安全 stress test。
至少需要一个外部环境证明：
SIEVE 学到的不是内部四类事件标签，而是一种可以迁移的状态修正策略。

升级六：只有诊断成立时才做算法创新
如果普通 GRPO 的 credit conflict 明显，可以把算法创新收敛为：
Risk-targeted counterfactual branching for constrained belief revision

它比“Constrained GiGPO”更有自身问题特色：
- 不是在所有高熵 token 处分叉；
- 不是对所有节点做昂贵 Tree-GRPO；
- 只在可能改变 trusted state、影响高风险动作或消耗验证预算的节点分叉；
- 从同一父状态比较 UPDATE/HOLD/IGNORE；
- 使用局部约束后 return 差异估计动作优势。
这可以形成真正贴合 SIEVE 的算法命题：
利用 belief dependency 和 action risk 识别因果关键节点，在更少 rollout 下获得更准确的多步信用分配。

但它必须与以下方法在相同 rollout/tool budget 下比较：
- trajectory GRPO；
- constrained GRPO；
- uniform tree branching；
- entropy-based branching；
- risk-targeted branching。
否则审稿人会认为提升只是来自额外采样。
6. 最关键的实验结果应该长什么样
一篇有说服力的论文，主结果最好不是简单的：
SIEVE accuracy 比 GRPO 高 3%
而应形成以下完整证据链。
证据一：失败模式真实存在
在原生 Agent 中，无效 observation 确实进入状态，并在多个后续步骤持续影响动作。
证据二：它不同于一般规划错误
在保持 action policy 不变的情况下，只修正 observation admission 就能恢复大量任务成功。
证据三：结构化边界必要
Free-text summary、full context 和 uncertainty belief 仍会发生错误状态承诺；SIEVE 的 admission + executor 明显降低污染。
证据四：SFT 不够
SFT 单步准确率高，但面对预算、延迟后果和不可逆风险时，轨迹表现仍显著落后 RL。
证据五：约束不是保守拒绝
SIEVE 同时保持 benign evidence retention 和 task success，而不是通过全 HOLD 获得低 unsafe rate。
证据六：局部信用分配确实解决了测得的瓶颈
只有在普通 GRPO 出现 credit conflict 的场景中，局部方法带来最大提升；在单步场景中二者接近。
最后这一点非常重要，因为它把“算法提升”与“理论上声称解决的问题”连接起来。
7. 最终的审稿判断
如果今天基于当前材料投稿，我的模拟评分可能是：
维度	当前判断
问题重要性	较强
新颖性	中等偏弱，最近邻威胁很大
方法完整性	中等
理论深度	偏弱，执行器不变量不够
实验证据	当前偏弱，内部任务和 base task 有限
清晰度	已经较好
总体	Weak Reject


如果完成下面四件事，判断可能明显改变：
1. 把核心差异收窄为 authorized observation-to-state transition；
2. 在真实外部环境上证明 persistent contamination；
3. 正面对比 structured belief、uncertainty belief 和 rule gate；
4. 用因果干预和公平预算实验，证明约束或局部信用分配确实必要。
这时可以达到 Borderline Accept 到 Accept 的竞争区间。
8. 我的最终建议
不要继续平均用力地扩展所有模块。最应该押注的是：
把“observation admission 是不同于 belief representation 的独立控制问题”做成一个无法被替代的实验证据链。

具体资源分配建议：
- 15%：继续收紧文章定位和 related-work 差异；
- 50%：外部环境、自然错误、paired counterfactual 和强 baseline；
- 20%：SFT/GRPO/约束的完整实验；
- 15%：根据 credit diagnostics 决定是否实现局部信用分配算法。