# SIEVE 数据目录精简设计

## 目标

`data/` 只持久保存两类资产：可以追溯的开源原始数据，以及阶段 1 可以直接读取的 SFT 数据。可重建的归一化文件、生成缓存、重复切分和占位清单不再保留在 `data/`。

## 最终目录

```text
data/
├── raw/
│   ├── manifest.json
│   ├── tau2-bench/
│   ├── ToolBench/
│   ├── AgentBench/
│   └── webarena/
└── sft/
    ├── manifest.json
    ├── quality_report.json
    ├── source/records.jsonl
    └── clean/{train,dev}.jsonl
```

`data/raw/` 保留数据归一化代码实际读取的 20 个上游文件、四个项目的 LICENSE，以及记录数据集、固定提交、许可证、路径和 SHA-256 的清单。目录结构保持与原适配器使用的上游相对路径一致。

`data/sft/` 全量原样保留。训练配置继续读取 `data/sft/source/records.jsonl`、`data/sft/clean/train.jsonl` 和 `data/sft/clean/dev.jsonl`。

## 删除范围

- 删除 `data/generated/`：其 accepted、train、dev 和质量报告已验证与 `data/sft/` 对应文件逐字节相同；其余文件属于缓存或未使用的内部测试切分。
- 删除 `data/source_cache/`：先把实际使用的原始文件和 LICENSE 复制到 `data/raw/` 并完成哈希校验，再删除完整克隆、`.git` 元数据和 normalized 中间文件。
- 删除 `data/source_manifests/`：其中只有带占位符的未使用示例清单。

## 后续生产路径

数据生产脚本从 `data/raw/` 读取原始数据。归一化、生成和内部切分写入 `tmp/data_factory/`；只有发布后的 canonical 数据写入 `data/sft/`。这样重跑生产链路不会重新污染 `data/`。

## 安全与验证

删除前后均校验 `data/sft/manifest.json` 声明的三个核心文件 SHA-256。删除命令仅接受解析后位于 `C:\Common_Document\SIEVE\data` 内的精确目标。清理后运行路径契约测试、数据预检和完整单元测试。
