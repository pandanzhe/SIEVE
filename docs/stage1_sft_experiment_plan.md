# SIEVE 阶段 1：Qwen2.5-3B 三分支结构化 SFT 实验方案

## 1. 实验范围

本阶段只训练观察评估策略，不包含环境交互 RL、GRPO、验证工具在线调用或行动策略训练。目标是得到可稳定初始化第二阶段的 \(\pi_{\mathrm{rev}}\)：模型读取评估前状态，先输出 `UPDATE / HOLD / IGNORE`，再生成统一的验证计划和局部 patch value。

训练实现采用“自定义算法层 + Hugging Face 基础设施”：Qwen2.5-3B 本地权重作为骨干，PEFT LoRA 和结构化辅助 heads 作为可训练参数，Accelerate 管理 BF16、梯度累积与单卡/双卡启动。LLaMA-Factory 不作为主训练器。

阶段 1 的输出被划分为三个逻辑分支：

1. **Decision head**：独立线性分类头，输出 `UPDATE / HOLD / IGNORE`。
2. **Verification head**：自然语言生成分支，统一生成受影响字段、验证请求和 patch operations。
3. **Patch-value head**：使用冻结的 Qwen LM head，在 LoRA 适配后的隐藏状态上生成 patch values。

这里需要区分“逻辑分支”和“物理神经网络层”：decision head 是独立的物理线性层；verification 与 patch-value 是两个逻辑生成分支，它们共享同一个冻结 LM head，通过不同的 token-level label mask 分别计算损失。不会复制第二个大词表投影层。

## 2. 架构选择

### 2.1 采用方案

采用“一个分类头 + 一个共享 LM head + 两个生成区间”：

```text
Qwen2.5-3B + LoRA
├─ decision head
│  └─ UPDATE / HOLD / IGNORE
└─ frozen LM head
   ├─ verification-text 区间
   │  └─ affected_fields + verification + patch_operations
   └─ patch-value 区间
      └─ patch_values + EOS
```

### 2.2 不采用的方案

- 不再保留独立的 affected-field、verification-classification 和 patch-operation 线性 heads。
- 不额外复制一个 `hidden_size × vocabulary_size` 的自然语言输出层。对于 Qwen2.5，这会引入数亿参数和额外显存，但不会带来新的表达能力。
- 不使用自由格式自然语言作为训练目标。verification head 输出固定 Schema 的 JSON 文本，以保证可解析、可约束和可评测。
- patch value 不采用回归 MLP。当前 value 可能是地址、实体、时间、列表或嵌套文本，不能用固定维度的数值回归统一表达，因此仍使用自回归 token 生成。

## 3. 词表与停止标识设计

### 3.1 不新增专用词表

阶段 1 直接复用 Qwen2.5 原生 tokenizer 和原生 vocabulary：

$$
\mathcal V_{\mathrm{SIEVE}}=\mathcal V_{\mathrm{Qwen2.5}}
$$

不新增 `<decision>`、`<verification>`、`<patch>` 或 `<end>` 等 special token，也不调用 `resize_token_embeddings()`。这样可以避免在冻结基础模型时额外训练新 token embedding，并保持本地模型权重兼容。

不同类型的信息按以下方式表达：

| 信息 | 表达方式 |
|---|---|
| Decision | 独立三分类标签，不属于 tokenizer vocabulary |
| JSON key | 使用 Qwen2.5 原生 tokenizer 对固定字符串切词 |
| Patch operation | `ADD_FIELD / SET_VALUE / SET_STATUS / SET_PROVENANCE / SET_VALIDITY` |
| Field ID | 使用原生 tokenizer 编码动态字符串 |
| Verification tool | 使用原生 tokenizer 编码动态字符串 |
| Patch value | 使用原生 tokenizer 编码自然语言或结构化文本 |
| 序列结束 | 使用 Qwen2.5 原生 `eos_token_id` |

patch operation 虽然是封闭集合，但在 verification head 中仍表示为 JSON 字符串。推理时由 JSON Schema 或 constrained decoding 限制其只能取合法枚举值。

### 3.2 统一生成格式

模型完成 decision 分类后，将预测结果作为普通文本控制前缀追加到生成上下文：

```text
Decision: UPDATE
Action:
```

随后 LM head 一次性生成：

```json
{
  "affected_fields": ["address"],
  "verification": null,
  "patch_operations": [
    {"field_id": "address", "op": "SET_VALUE"}
  ],
  "patch_values": [
    {"field_id": "address", "value": "101 S San Mateo Dr"}
  ]
}
```

训练时使用 gold decision 作为 teacher-forcing 控制前缀；推理时使用 decision head 的预测结果。decision 控制前缀只作为条件输入，其 token label 为 `-100`，不计算语言模型损失。

### 3.3 什么时候停止生成

不设计新的 `<end>` 标识。训练目标末尾直接附加 Qwen2.5 原生 EOS：

```text
完整 JSON + eos_token
```

EOS 本身属于 patch-value loss 区间，因此模型会学习在顶层 JSON 闭合后产生 EOS。推理阶段设置：

```python
eos_token_id = tokenizer.eos_token_id
```

生成满足以下任一条件即终止：

1. 模型生成原生 EOS；
2. 增量 JSON parser 判断顶层对象已经完整闭合；
3. 达到 `max_new_tokens` 安全上限。

第二项是工程安全停止条件，避免模型已经生成合法 JSON 后继续输出多余文本；第三项用于防止异常无限生成。

## 4. 固定实验资产

| 资产 | 仓库相对路径 | 作用 |
|---|---|---|
| 基础模型 | `model/` | 服务器补齐 Qwen2.5-3B Hugging Face 格式文件 |
| 原始生成结果 | `data/sft/source/records.jsonl` | 6000 条，只保留和审计 |
| 干净训练集 | `data/sft/clean/train.jsonl` | 4789 条，4076 个 scenario group |
| 干净验证集 | `data/sft/clean/dev.jsonl` | 619 条，521 个 scenario group |
| 数据清单 | `data/sft/manifest.json` | 数量、SHA-256、拆分政策和清理说明 |
| 主配置 | `configs/sft_qwen25_3b.yaml` | 模型、数据、损失、训练与硬件参数 |
| 输出目录 | `outputs/stage1-qwen25-3b/` | audit、metrics、best/final adapter 和训练状态 |

阶段 1 不读取内部测试集。正式测试使用后续独立 benchmark，不能用于超参数或 checkpoint 选择。

服务器上的 `model/` 至少需要包含：

- `config.json`
- `tokenizer.json` 或 `tokenizer_config.json`
- 一个或多个 `*.safetensors`，或者 `pytorch_model*.bin`

代码固定 `local_files_only=True`，训练时不会连接 Hugging Face 下载模型。

## 5. 训练序列与 label mask

### 5.1 序列组成

每条训练序列由五部分拼接：

$$
X=[P(s_t^{rev});D(c_t^*);V_t^*;W_t^*;\mathrm{EOS}]
$$

其中：

- \(P(s_t^{rev})\)：由状态、观察、目标、风险、预算和证据账本构造的基础 prompt；
- \(D(c_t^*)\)：gold decision 控制前缀；
- \(V_t^*\)：verification-text 区间；
- \(W_t^*\)：patch-value 区间；
- EOS：Qwen2.5 原生结束 token。

固定序列文本为：

```text
[system + state prompt]
Decision: <gold decision>
Action:
{"affected_fields":...,"verification":...,"patch_operations":...,"patch_values":...}
<eos>
```

decision head 的池化位置是基础 prompt 的最后一个 token，位于 gold decision 之前。因此在因果注意力下，decision head 不可能读取 gold decision、verification target 或 patch value，不存在标签泄漏。

### 5.2 三套监督信号

一条序列产生三套标签：

```text
decision_label:
    prompt 边界处的 UPDATE / HOLD / IGNORE 类别

verification_lm_labels:
    affected_fields、verification、patch_operations 和相应 JSON 结构 token

patch_value_lm_labels:
    patch_values、顶层 JSON 闭合 token 和 EOS
```

所有 prompt token、decision 控制前缀和不属于当前损失区间的 token 均设为 `-100`。

示意如下：

```text
token 区间                 decision CE   verification CE   patch-value CE
基础 prompt                    √              ×                 ×
gold decision 控制前缀          ×              ×                 ×
affected_fields                ×              √                 ×
verification                   ×              √                 ×
patch_operations               ×              √                 ×
patch_values                   ×              ×                 √
JSON 最终闭合 + EOS             ×              ×                 √
```

训练过程不需要等模型“看到 EOS 后”才计算 loss。一次 forward 已经得到序列所有位置的 next-token logits，训练器立即根据两套 label mask 对有效位置计算交叉熵。EOS 只是其中一个需要预测的目标 token，同时也是推理时的停止信号。

## 6. 训练目标与损失函数

评估前输入为：

$$
s_t^{rev}=(B_{t-1},o_t,g_t,r_t,L_t)
$$

监督动作重新写为：

$$
u_t=(c_t,v_t,w_t)
$$

其中：

- \(c_t\)：decision；
- \(v_t=(z_t,q_t,p_t^{op})\)：统一 verification plan，包含受影响字段、验证请求和 patch operations；
- \(w_t=p_t^{value}\)：patch values。

### 6.1 Decision loss

decision head 在基础 prompt 边界隐藏状态 \(h_t\) 上进行三分类：

$$
\mathcal L_{\mathrm{decision}}
=-\log \pi_\theta(c_t^*\mid s_t^{rev})
$$

### 6.2 Verification-text loss

设 \(m_i^{ver}\in\{0,1\}\) 表示 token \(i\) 是否属于 verification-text 区间：

$$
\mathcal L_{\mathrm{verification}}
=-
\frac{
\sum_i m_i^{ver}\log p_\theta(x_i^*\mid x_{<i},s_t^{rev},c_t^*)
}{
\sum_i m_i^{ver}
}
$$

它统一替代原来的 affected-field BCE、verification classification CE 和 patch-operation BCE。

### 6.3 Patch-value loss

设 \(m_i^{value}\in\{0,1\}\) 表示 token \(i\) 是否属于 patch-value 区间：

$$
\mathcal L_{\mathrm{patch\text{-}value}}
=-
\frac{
\sum_i m_i^{value}\log p_\theta(x_i^*\mid x_{<i},s_t^{rev},c_t^*,v_t^*)
}{
\sum_i m_i^{value}
}
$$

EOS 包含在 \(m_i^{value}\) 中。

### 6.4 总目标

$$
\boxed{
\mathcal L_{\mathrm{SFT}}
=
\lambda_d\mathcal L_{\mathrm{decision}}
+
\lambda_{ver}\mathcal L_{\mathrm{verification}}
+
\lambda_{value}\mathcal L_{\mathrm{patch\text{-}value}}
}
$$

初始权重建议：

```yaml
loss_weights:
  decision: 1.0
  verification: 1.0
  patch_value: 1.0
```

三种 decision 都产生合法 JSON 监督：

- `IGNORE`：`affected_fields=[]`、`verification=null`、`patch_operations=[]`、`patch_values=[]`；
- `HOLD`：生成 affected fields 和 verification request，patch operations/values 为空；
- `UPDATE`：生成 affected fields、patch operations 和 patch values，verification 通常为 `null`。

因此不再通过跳过整个生成分支来实现门控，而是用合法的空结构显式监督“不验证、不更新”。

## 7. 模型与训练参数

| 参数 | 默认值 |
|---|---:|
| Base model | local `model/`（Qwen2.5-3B） |
| Precision | BF16 |
| Thinking | disabled |
| Trainable modules | LoRA + decision head |
| Frozen module | original LM head |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| LoRA target | q/k/v/o projection |
| Epochs | 3 |
| Max sequence / completion | 2048 / 512 |
| Per-device micro batch | 2 |
| Effective batch | 32 |
| LoRA learning rate | 1e-4 |
| Decision-head learning rate | 5e-4 |
| Scheduler | cosine, 3% warmup |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Gradient checkpointing | enabled |
| Eval interval | 100 optimizer updates |

不使用 QLoRA。A100 40G 可以承载 4B BF16 LoRA，量化会额外引入量化误差和实验变量。

## 8. 硬件配置

### 默认：1 × A100 40G

Qwen2.5-3B 的 BF16 权重约 6 GB。主干和 LM head 冻结，仅训练 LoRA 和结构化辅助 heads；配合 gradient checkpointing、micro batch 2 和 2048 token 上限，单张 A100 40G 是默认配置。

```bash
bash scripts/run_sft.sh
```

### 备选：2 × A100 40G

双卡只用于提高吞吐，不是显存必需条件。采用同步 DDP，并把梯度累积从 16 调整为 8，保持 effective batch size 为 32：

```bash
SIEVE_NUM_GPUS=2 bash scripts/run_sft.sh
```

不建议阶段 1 默认使用 4 卡，也不默认启用 ZeRO-2/ZeRO-3。

## 9. 验证指标与 checkpoint 选择

### 9.1 Decision 指标

- decision accuracy / macro-F1；
- UPDATE、HOLD、IGNORE 分类别 precision / recall / F1；
- false update rate；
- missed update rate。

### 9.2 Verification-plan 指标

先解析 verification head 生成的 JSON，再计算：

- valid JSON rate；
- Schema compliance rate；
- affected-field micro/macro-F1；
- verification request exact match / F1；
- patch-operation micro-F1；
- illegal operation rate。

### 9.3 Patch-value 指标

- patch-value exact match；
- normalized exact match；
- field–value pair F1；
- executable patch rate；
- EOS termination rate；
- generation truncation rate。

定义：

$$
\mathrm{PlanF1}
=
\frac{
F1_{\mathrm{affected}}
+
F1_{\mathrm{verification}}
+
F1_{\mathrm{patch\text{-}operation}}
}{3}
$$

checkpoint 分数更新为：

$$
S
=
F1_{\mathrm{decision}}
+
\mathrm{PlanF1}
+
\mathrm{ValueEM}
-
2\cdot\mathrm{FUR}
-
\mathrm{InvalidJSONRate}
$$

`best/` 保存最高分 LoRA、decision head 和 tokenizer 配置；`final/` 保存最后一步参数。Accelerate 状态按 epoch 保存，最多保留 3 份。

## 10. 环境与运行步骤

推荐 Ubuntu 22.04、Python 3.11、CUDA 12.6-compatible driver。锁定依赖位于 `requirements/train.txt`：

- torch 2.7.1
- transformers 4.53.2
- peft 0.16.0
- accelerate 1.8.1
- safetensors 0.5.3
- numpy 2.2.6
- PyYAML 6.0.2

创建虚拟环境：

```bash
bash scripts/create_train_env.sh
```

服务器补齐模型后先执行离线检查：

```bash
bash scripts/validate_sft.sh
```

单卡训练：

```bash
bash scripts/run_sft.sh
```

双卡备选：

```bash
SIEVE_NUM_GPUS=2 bash scripts/run_sft.sh
```

## 11. 服务器首轮 smoke run 检查项

正式训练前先运行 5–10 个 optimizer updates，并确认：

1. GPU 峰值显存低于 38 GB，无 CPU offload。
2. 只有 LoRA 和 decision head 的参数 `requires_grad=True`，LM head 保持冻结。
3. decision、verification-text、patch-value 三个 loss 均为有限值。
4. verification 与 patch-value token mask 不重叠，且每条样本都至少包含一个有效目标 token。
5. gold decision token 位于 decision pooling position 之后，不会泄漏给 decision head。
6. 模型能在完整 JSON 后生成原生 EOS。
7. 单卡和双卡的 effective batch size 都是 32。
8. train/dev scenario overlap 为 0。

本地机器不执行 GPU smoke run；当前阶段只完成设计、代码、配置、数据链路和静态/单元测试。
