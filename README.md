# SIEVE

SIEVE（Selective Information Evaluation and Verification for Executable Beliefs）研究多步 Agent 的观察准入。Revision Policy 在观察进入可信任务状态前选择 UPDATE、HOLD 或 IGNORE，State Executor 再执行受约束的局部 patch。

完整研究设计见 [docs/training_design.md](docs/training_design.md)。

## 两类后端

- NumPy reference backend：不需要 GPU，用于检查状态机、数据生成、SFT/GRPO 参数流、约束更新和评测接口。
- Hugging Face LoRA backend：`src/sieve/policies/hf_lora_policy.py` 惰性创建 7B/8B backbone、PEFT LoRA 和结构化 heads。只有安装可选依赖后才加载 PyTorch。

NumPy backend 是逻辑参考，不替代论文中的 7B/8B 主实验。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

GPU 服务器再安装：

```bash
python -m pip install -e ".[hf]"
```

PyTorch 版本应按服务器 CUDA 版本选择。仓库不会在 import 时自动下载模型。

## 目录

```text
configs/                  数据和训练配置
docs/                     研究规格与实施计划
scripts/                  可移植 Bash 入口
src/sieve/core/           状态类型、Executor、事件时钟
src/sieve/data/           paired 数据与 JSONL I/O
src/sieve/environments/   环境协议、replay/toy 环境和扰动
src/sieve/policies/       NumPy 策略与 HF LoRA 工厂
src/sieve/training/       SFT、Constrained GRPO、奖励和约束
src/sieve/evaluation/     指标和报告
src/sieve/cli/            命令行入口（CLI = Command Line Interface）
tests/                    unittest
```

## 数据链路

```text
clean trajectory -> oracle snapshots -> paired perturbations
-> split by scenario_id -> SFT -> grouped GRPO rollouts -> evaluation
```

`scenario_id` 是切分单元，`step_index` 是同一轨迹内唯一且从 0 连续递增的事件时钟。同一基础 scenario 不能跨 train/dev/test。模型输入不含 oracle world state，也不含生成器内部的 perturbation 标签。toy 生成器把 invalid、uncertain、authoritative 三类观察组成一条有序合成轨迹；正式实验应保留 benchmark 的真实事件顺序。

## 命令

脚本通过自身位置计算 `ROOT_DIR`，可以从任意当前目录调用：

```bash
bash scripts/run_generate.sh configs/toy.yaml
bash scripts/run_sft.sh configs/sft.yaml
bash scripts/run_rl.sh configs/rl.yaml
bash scripts/run_eval.sh configs/toy.yaml outputs/toy/checkpoints/grpo_policy.npz
```

完整闭环为 `bash scripts/run_all.sh`。本地没有 GPU 时不必执行训练脚本，只做静态检查和非训练单元测试：

```bash
python -m compileall -q src tests
python -m unittest tests.test_config tests.test_executor tests.test_transition \
  tests.test_data tests.test_policy tests.test_rewards tests.test_metrics \
  tests.test_validation tests.test_revision_env tests.test_evaluator \
  tests.test_executor_schema tests.test_grpo_math tests.test_portability
```

## 数学变量与代码

| 数学对象 | 代码位置 |
|---|---|
| B_t，可信状态 | `core/types.py::BeliefState` |
| o_t，观察 | `core/types.py::Observation` |
| d_t，风险包络 | `core/types.py::RiskEnvelope` |
| y_t=(z,A,DeltaB,q) | `core/types.py::RevisionOutput` |
| U，确定性更新 | `core/executor.py::StateExecutor` |
| t/k，两套时钟 | `core/transition.py::AgentEventLoop` |
| P_env/P_obs | `environments/protocol.py` |
| RL reset/step 边界 | `environments/revision_env.py::RevisionRLEnvironment` |
| Revision Policy | `policies/protocol.py` |
| SFT | `training/sft.py` |
| Constrained GRPO | `training/constrained_grpo.py::optimize_constrained_grpo` |
| 奖励和成本 | `training/rewards.py` |
| 拉格朗日乘子 | `training/lagrangian.py` |

## GRPO 是否需要区分环境和策略

需要。GRPO 改变的是优势估计和策略优化方式，不会取消 RL 中的环境。

单轮大模型 GRPO 常以固定 prompt 为输入，同一 prompt 采样多个 completion，再由规则、reward model 或 verifier 评分。这时环境退化成“数据集加评分器”，所以代码中不一定出现显式 env.step。

SIEVE 是多步问题。revision 会改变可信状态，验证会消耗预算并产生下一观察，外部动作还会改变 world state。因此环境负责状态转移、下一观察、成本和终局结果；Revision Policy 是唯一被梯度更新的模型；Executor、Action Agent 和环境在主实验中冻结。

代码中 `RevisionRLEnvironment` 持有状态与转移；`optimize_constrained_grpo` 只调用 `reset/step` 并更新策略。Constrained GRPO 对同一初始 scenario 采样一组完整轨迹，用组内相对分数计算优势，不训练 value critic。拉格朗日乘子用未标准化的实际成本更新，避免组内归一化抹掉安全约束的绝对尺度。

## 外部环境接口

原始 benchmark adapter 实现 `environments/protocol.py::AgentEnvironment`；供 RL 优化器使用的轨迹层实现 `environments/revision_env.py::RevisionRLEnvironment`。后者只向策略暴露 `reset -> context` 和 `step(action) -> reward/cost/next_context/terminated`，oracle 只保留在环境内部。原始接口为：

```python
class AgentEnvironment(Protocol):
    def step(self, action: EnvironmentAction) -> list[Observation]: ...
    def oracle_state(self) -> dict[str, object]: ...
```

训练适配器可读取 oracle_state 生成标签和奖励，但 Revision Policy 输入不得读取它。接入 tau2-bench 或 ScienceWorld 时应单独实现 adapter，保持 core、policy 和 training 不依赖具体 benchmark 包。

## GPU 服务器建议

两张 A100 80GB 可把 rollout worker 和 learner 分卡；四张 A100 40GB 使用 FSDP 或 ZeRO-3。主模型使用 BF16，只更新 LoRA 和结构化 heads。模型名、缓存目录和外部数据目录写入服务器私有配置，不写进 Python 源码。

## Source-grounded SFT 数据生产

本地完整数据链路不需要 GPU，也不会调用在线模型：

```bash
bash scripts/run_grounded_data.sh
```

该脚本从 `data/raw/` 读取最小可追溯开源快照，依次完成归一化、带 `--confirm-full 6000` 保护的确定性数据生产、内部切分和 canonical 发布。持久数据目录只有：

```text
data/raw/                         实际使用的上游原始文件、LICENSE 和哈希清单
data/sft/source/records.jsonl     6000 条审计源记录
data/sft/clean/{train,dev}.jsonl  可直接训练和验证的数据
data/sft/{manifest,quality_report}.json
```

归一化文件、生成缓存和内部 test 切分默认写入可删除的 `tmp/data_factory/`，不会污染 `data/`。单独执行各阶段：

```bash
python -m sieve.cli.prepare_grounded_sources --root . \
  --output-dir tmp/data_factory/normalized-full --per-source-limit 500
python -m sieve.cli.generate_grounded --root . \
  --config configs/grounded_full_dry_run.json --dry-run --confirm-full 6000
python -m sieve.cli.export_grounded --root . \
  --input tmp/data_factory/stage1-trajectory-v2/accepted.jsonl \
  --output-dir tmp/data_factory/stage1-trajectory-v2/training --seed 42
python -m sieve.cli.publish_stage1_data --root . \
  --input tmp/data_factory/stage1-trajectory-v2/accepted.jsonl \
  --split-dir tmp/data_factory/stage1-trajectory-v2/training \
  --output-dir data/sft --prompt-version stage1-system-v2
```

真实 GLM 预览使用 `configs/grounded_preview.json`，并只从环境变量读取 `ZAI_API_KEY`：

```bash
python -m sieve.cli.generate_grounded --root . \
  --config configs/grounded_preview.json \
  --output-dir tmp/data_factory/preview-live
```

相同输出目录默认启用 resume 和响应缓存。若环境变量不存在，真实生成会明确停止；确定性生产不受影响。训练视图会删除 provenance、oracle、generation、validation、reason code 和扰动标签，并把自然语言 realization 作为当前观察值。

## Stage-1 Qwen2.5-3B structured SFT

Stage 1 defaults to one A100 40G and loads Qwen2.5-3B only from the repository-local `model/` directory. Copy the server model assets into that placeholder, then run:

```bash
bash scripts/create_train_env.sh
bash scripts/validate_sft.sh
bash scripts/run_sft.sh
```

Use `SIEVE_NUM_GPUS=2 bash scripts/run_sft.sh` for the optional two-GPU DDP profile. The canonical data files are `data/sft/source/records.jsonl`, `data/sft/clean/train.jsonl`, and `data/sft/clean/dev.jsonl`; Stage 1 has no internal test input. See `docs/stage1_sft_experiment_plan.md` for Stage 1 and `docs/stage2_rl_experiment_plan.md` for the Qwen2.5-3B constrained multi-step GRPO pipeline.

## Stage-2 Qwen2.5-3B constrained GRPO

Stage 2 uses the Stage-1 `best/adapter` as both the initial policy and the frozen reference. The committed scenario corpus contains 1,600 train, 200 dev, and 300 internal-test episodes. On the server, run in this order:

```bash
python -m pip install -r requirements/rl.txt
python -m pip install -e .
bash scripts/check_stage1_readiness.sh
bash scripts/validate_stage2.sh
bash scripts/train_stage2.sh
```

The default profile is two A100 40G GPUs. Use `SIEVE_NUM_GPUS=1 bash scripts/train_stage2.sh` for the one-GPU fallback. Neither local validation nor corpus construction loads Qwen or starts training; the GPU readiness and training commands intentionally fail until `model/`, the Stage-1 adapter, pinned dependencies, and the generated readiness report are present.


## loss含义
train_loss
总 loss。当前代码里它是加权和：

  loss = decision_loss * weight_decision
       + structure_loss * weight_structure
       + value_loss * weight_patch_value

  你现在的配置里三个权重都是 1.0，所以基本就是三个目标相加。日志里没有打印 decision_loss，所以你看
  到：

  train_loss=0.000046
  train_structure_loss=0.000034
  train_value_loss=0.000012

  它们相加刚好约等于 0.000046，说明这一步的 decision_loss 可能非常接近 0，或者小到日志精度下不明
  显。

  train_structure_loss
  这是对结构化 JSON 动作的“结构 token”做 causal LM loss。它主要约束模型输出格式和动作骨架，比如：

  {"decision":"UPDATE","affected_fields":[...],"patches":[...],"verification":...}

  它学的是：字段名、JSON 格式、decision/patch/verification 的结构表达是否对。

  train_value_loss
  这是对 patch value 里面可写值 token 的 causal LM loss。它主要约束模型在 UPDATE 时写入正确值，例如
  把某个字段设成 oracle/观察支持的值。相比 structure loss，它更关注具体 value 内容，而不是 JSON 框
  架。

  还有一个你没看到的 decision_loss
  代码里实际存在：
  result["decision_loss"] = loss
  它是 UPDATE / HOLD / IGNORE 三分类 head 的交叉熵。当前日志打印时只打印了 loss、structure_loss、
  value_loss，没有打印 decision_loss，所以日志里看不到它。这个可以补上，建议打印出来，否则很难判断
  模型到底是“动作分类学会了”，还是只是 JSON 生成学会了。