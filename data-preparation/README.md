# 数据准备

本目录保留数据准备、judge 和最近使用的两套 rollout。**当前没有 rollout 进程在运行**；历史脚本已归档，probe 训练入口在项目的 `train/` 目录。除非脚本另有说明，请从本目录执行。

| 路径 | 用途 |
| --- | --- |
| `01_download.py`–`04_stats.py` | 数据下载、准备、处理与统计 |
| `common.py`、`screen_response_gaps.py` | 公共工具和缺失回答筛选；见 [SCREEN_RESPONSE_GAPS.md](SCREEN_RESPONSE_GAPS.md) |
| `judge_gty1125_pending_prompts.py` | 当前运行的 prompt judge |
| [scripts/prepare/](scripts/prepare/) | 数据准备、prompt 索引和候选集划分 |
| [scripts/curate/](scripts/curate/) | 过滤、合并、数据导入和导出 |
| [scripts/judge/](scripts/judge/) | 现有 prompt/response judge 入口 |
| [scripts/rollout/](scripts/rollout/) | Qwen3-8B 生成安全回答；Huihui-27B 生成危险回答 |
| [scripts/orchestration/](scripts/orchestration/) | 两卡资源分配及 Huihui watchdog |
| [scripts/tools/](scripts/tools/) | 上述 GPU 调度依赖的 smoke 测试 |

数据位于 `data/`，当前生成文本数据库为 `/workspace/models/gty/probe-generated-gty-1125.sqlite`。训练实际读取的 JSONL 位于 `../data/processed/joint_19_gty_1125_full_rawmapped/`。

部分脚本绑定历史批次的输入、输出和模型路径；执行前先检查路径、已完成进度和输出行为。部分脚本没有 `--help`，传入该参数也可能直接执行任务。Judge 可能调用外部 API，rollout 会占用 GPU。

不再保留的历史专用脚本及实验已打包到 `../archives/data-preparation-retired-20261126.tar.gz`，逐文件 SHA-256 清单位于同目录的 `.manifest.json`。归档内容以本目录为相对路径；确需恢复时，从本目录解压，并先检查是否覆盖现存文件。
