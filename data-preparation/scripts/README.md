# 当前数据准备脚本

- `prepare/`：数据整理、prompt 索引和候选划分。
- `curate/`：过滤、合并、标签同步和数据库导出。
- `judge/`：prompt 和回答评分；正在运行的 prompt judge 位于目录顶层。
- `rollout/`：最近使用的 Qwen3 安全回答与 Huihui 危险回答三轮生成脚本。
- `orchestration/`：两卡调度与 Huihui watchdog。
- `tools/`：调度器所需的 smoke 工具。

历史脚本位于 `../../archives/data-preparation-retired-20261126.tar.gz`，恢复前先检查现有路径和脚本行为。
