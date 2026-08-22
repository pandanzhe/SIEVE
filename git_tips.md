# Git 常用流程

本文档记录 SIEVE 仓库里常用的 git 操作。原则是：代码、配置、文档、必要日志可以提交；模型权重、训练输出 checkpoint、optimizer state 不要提交。

## 1. 进入仓库并查看状态

```bash
cd /mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE
git status --short --branch
git remote -v
git diff --stat
```

确认当前分支：

```bash
git branch --show-current
```

## 2. 拉取远端更新

如果当前工作区没有未提交改动，可以直接拉取：

```bash
cd /mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE
git pull --ff-only
```

如果有本地改动，先看清楚改了什么：

```bash
git status --short
git diff --stat
```

不确定本地改动是否要保留时，不要直接 `reset` 或 `checkout`。可以先新建分支保存当前工作：

```bash
git switch -c lijing/save-local-work-$(date +%Y%m%d_%H%M%S)
```

## 3. 建议使用新分支提交【提交环节】

除非非常确定要直接更新当前分支，否则建议每次开一个新分支：

```bash
cd /mnt/bn/fs-llm-train-v2/mlx/users/lijing/SIEVE
git switch -c sieve-stage2-RL
```

如果分支已经存在：

```bash
git switch sieve-stage2-RL
```

## 4. 不要提交模型权重

模型和训练 checkpoint 一般不提交。提交前必须检查：

```bash
git status --short
git diff --cached --name-only | grep -E '(^model/|^outputs/|\\.safetensors$|\\.bin$|\\.pt$)' || true
```

如果上面命令有输出，说明已经 staged 了模型权重或训练产物，需要取消暂存：

```bash
git restore --staged model outputs
```

也可以检查忽略规则：

```bash
git check-ignore -v model/Qwen3-4B outputs/stage1-qwen3-4b || true
```

## 5. 选择性添加文件

不要使用：

```bash
git add .
```

推荐只添加明确需要提交的文件。例如当前 SIEVE readiness / Stage-2 修复：

```bash
git add README.md
git add configs/rl_qwen3_4b.yaml configs/sft_qwen3_4b.yaml
git add scripts/common.sh scripts/run_sft.sh
git add src/sieve/policies/hf_lora_policy.py
git add src/sieve/rl_data/build.py
git add src/sieve/training/hf_readiness.py src/sieve/training/hf_sft.py
git add tests/test_hf_lora_policy.py tests/test_hf_sft_config.py tests/test_hf_sft_runtime.py tests/test_rl_data.py
git add data/rl/manifest.json data/rl/scenarios/dev.jsonl data/rl/scenarios/test.jsonl data/rl/scenarios/train.jsonl
git add tasks.md git_tips.md
```

日志如果需要上传，只添加关键日志：

```bash
git add logs/stage1_sft_qwen3_4b_2xa100_20260821_064221.log
git add logs/stage1_readiness_qwen3_4b_20260821_080042.log
```

如果某个文件不存在，跳过那一行即可。

## 6. 提交前检查

```bash
git status --short
git diff --cached --stat
git diff --cached --name-status
git diff --cached --name-only | grep -E '(^model/|^outputs/|\\.safetensors$|\\.bin$|\\.pt$)' || true
```

如果最后一条命令没有输出，说明没有暂存模型权重或训练 checkpoint。

注意：如果看到类似下面的删除项，确认是不是你主动删除的：

```text
D SIEVE：多步智能体中的观察准入与可靠状态修正.pdf
```

如果不是主动删除，不要提交它。取消暂存：

```bash
git restore --staged 'SIEVE：多步智能体中的观察准入与可靠状态修正.pdf'
```

如果工作区文件也被误删，但你确实想恢复它，再执行：

```bash
git restore 'SIEVE：多步智能体中的观察准入与可靠状态修正.pdf'
```

## 7. 提交

提交信息必须带 TRAE CLI trailer：

```bash
git commit -m "Fix SIEVE readiness logging and stage2 data values

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

如果换成其他提交标题，也保留最后这一行：

```text
Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

## 8. Push 到远端

首次推送新分支：

```bash
git push -u origin lijing/sieve-stage1-readiness-rl-fix
```

后续同一分支继续推送：

```bash
git push
```

## 9. 当前推荐提交范围

当前建议提交这些类型的内容：

- SFT 训练日志增强。
- readiness 进度日志增强。
- Stage-2 RL scenario value 修复。
- 相关单元测试。
- `tasks.md` 和本文档。
- 必要的 `.log` 文本日志。

当前不建议提交：

- `model/` 下的模型权重。
- `outputs/` 下的 adapter、checkpoint、optimizer、scheduler。
- 非本次主动修改的 PDF 删除。
