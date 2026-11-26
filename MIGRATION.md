# 迁移包说明

- 多模态与文本 probe 代码、配置、依赖清单及小型 probe checkpoint 已保留。
- MM-SafetyBench 已生成回答、judge 结果和相关清单已保留；data/processed 中的 MM 图片路径已改为相对 agent-probe 目录的路径。
- BeaverTails-V 已处理清单与图片、MM-SafetyBench 源数据与图片已保留；文本侧已处理数据及 GTY probe checkpoint 已保留。
- 未包含 Qwen 等 backbone 模型、文本激活特征缓存、data-preparation/data、BeaverTails-V 的 HF 下载缓存和原始压缩文件，以及大部分逐条评测输出。
- 数据和 artifacts 原来指向 SSD 的软链接在压缩包中展开为普通目录。原始 Git 仓库和 worktree 元数据未包含。
- 新服务器需自行安装依赖、准备 backbone 模型，并调整配置或命令中的 /ssd/models/Qwen3.5-9B 等旧服务器绝对路径。文本特征缓存未保存，需要重新提取。data-preparation 需要重新下载数据并配置 judge。
- 原始生成缓存的 manifest 记录了旧路径和生成配置；迁移后继续使用旧生成任务的 --resume 前请核对或重建 manifest。
