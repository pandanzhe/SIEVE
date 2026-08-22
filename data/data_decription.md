推荐补的数据
  当前 raw 已经有 WebArena/ToolBench/AgentBench，但量和字段利用还不够。建议补这些：

  1. τ-bench / tau-bench
     适合 customer-service 多步状态、工具调用、用户信息修正。
     链接：

     https://github.com/sierra-research/tau-bench

  2. τ2-bench / tau2-bench
     你已经有一部分，建议确认是否完整拉取 domains 和 trajectories。
     链接：

     https://github.com/sierra-research/tau2-bench

  3. ToolBench
     适合工具选择和多步 API，但需要转成 belief-state pollution 场景。
     链接：

     https://github.com/OpenBMB/ToolBench

  4. AgentBench
     适合更泛化的 agent task，尤其 DBBench / OS / WebShop 类型。
     链接：

     https://github.com/THUDM/AgentBench

  5. WebArena
     适合网页多步任务和状态污染，但环境成本更高。
     链接：

     https://github.com/web-arena-x/webarena

  6. WorkArena
     更偏真实办公 SaaS，多字段状态和约束强，适合论文故事。
     链接：

     https://github.com/ServiceNow/WorkArena

  7. BrowserGym
     可以作为 WebArena/WorkArena 的统一环境入口。
     链接：

     https://github.com/ServiceNow/BrowserGym