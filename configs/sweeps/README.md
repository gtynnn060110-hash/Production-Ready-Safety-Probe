# MQA → RMAttn 验证集扫描

结构：已缓存的 Qwen block 15 隐状态 → 因果 MQA → Q 残差 + RMSNorm →
256 维逐 token 表示 → 两层 MLP（100、100）→ 回答内 RMAttn → 五维局部 logits。
第一层不经过分类器，也不在中间做时间池化。

- 第一层 MQA 固定 `context_window=0`，读取所有历史位置（含 prompt），保持因果；不扫描窗口。
- 第二层窗口 `risk_window` 只汇聚同一连续回答区间；开头不足窗口时使用已有前缀。
- 每个类别使用 10 个固定、可训练的 query/value heads；每个位置先合并内部 heads，
  最后按现有协议独立进行 Max / Mean / Last 读出。这是 token-loss 适配，
  不是原生 RMAttn 的 `sum_of_head_max` 序列训练。
- 所有操作保持因果；第二层短窗口内的表示已经包含第一层上下文。
- D=4096、4 个 query heads、head dim=64 时，总参数量为 **1,684,461**。

从项目根目录执行：

```bash
python -m probe.sweep --output artifacts/sweeps/mqa_full_rmattn_weighted_seed42
```

默认扫描共 9 组：

| 参数 | 候选 |
| --- | --- |
| MQA 上下文（固定） | 全部历史，`context_window=0`，不参与扫描 |
| RMAttn 窗口 | 10、20、40 |
| 学习率 | 5e-5、1e-4、3e-4 |
| Loss | Weighted，temperature=0.1、beta_tau=0.35、detach=true |
| 训练 / 验证 | 固定 8,192 条训练子集 / 完整 8,723 条验证集 |
| Epoch / Seed | 4 / 42 |
| Microbatch / 梯度累积 | 8 / 4 |
| 选 checkpoint / 选配置 | 验证集 Overall Max AP 最大 |

训练子集从已有完整训练缓存中按 `hash(split_seed, "train", sample_id)` 选择，
与既有 pilot 子集规则一致。所有候选使用相同子集和每轮 shuffle。
只加载 train/validation 特征，不预测、查看或按测试集结果选型。
缓存保留原始 BF16，在每批读取时沿用普通训练的 FP32 RMS 归一化。
默认把所选特征驻留 GPU；可用 `--cache-device cpu` 改为 CPU。

输出目录保存：

- `plan.json`：冻结搜索范围、配置、源代码和特征 manifest 哈希。
- `snapshot/probe/`：运行时源代码副本。
- `members.json`：实际训练子集和验证集的样本 ID。
- `status.json`：进度、当前 trial 和当前最优配置。
- `trials/*/`：逐轮历史、best/last checkpoint、实际配置和三个读出的验证指标。
- `summary.json`：按验证集 Overall Max AP 排序的结果；`complete` 标记扫描是否结束。
- `best_config.yaml`：选中配置，可加载后做完整训练；扫描 checkpoint 仅训练过子集。

同一命令可以恢复。已完成 trial 校验后复用，未完成 trial 从 epoch 边界恢复。
如果源代码、数据或搜索配置变化，需要使用新的输出目录。

选型结束后，如需完整训练与评测：

```bash
python -m probe train --config artifacts/sweeps/mqa_full_rmattn_weighted_seed42/best_config.yaml
python -m probe evaluate --config artifacts/sweeps/mqa_full_rmattn_weighted_seed42/best_config.yaml
```

验证集扫描分数不应与 README 中完整训练后的 ID/OOD 测试分数直接比较。

## 历史：首次窗口扫描结果

以下保留旧版 27 组扫描记录。当前方案已将 MQA 固定为全部历史；
这些窗口对比不再属于当前搜索空间，也不能替代固定全量 MQA 后的扫描。

27 组全部完成。选中 **MQA 窗口 64、RMAttn 窗口 40、LR=1e-4**，最佳 checkpoint
为第 4 轮，验证集 Overall Max AP 为 **94.6991%**，F1 为 **87.4426%**。
F1 阈值由同一验证集校准，作为诊断；配置选择依据 AP。

每格为对应窗口组合在三个学习率中的最佳验证 AP（%）：

| MQA 窗口 / RMAttn 窗口 | 10 | 20 | 40 |
| --- | ---: | ---: | ---: |
| 64 | 94.6778 | 94.6616 | **94.6991** |
| 256 | 94.6690 | 94.6192 | 94.6215 |
| 2048 | 94.6736 | 94.6197 | 94.6236 |

本轮较长的第一层窗口没有选型优势，但各窗口的最佳分数接近；
这是单 seed、训练子集上的扫描结果，长文本泛化需要另外验证。

- [27 组完整报告](../../artifacts/sweeps/mqa_rmattn_weighted_seed42/report.md)
- [全部指标与排名](../../artifacts/sweeps/mqa_rmattn_weighted_seed42/summary.json)
- [历史选中配置 overlay](selected_mqa_rmattn_weighted.yaml)

当前完整训练配置使用全量 MQA、RMAttn 窗口 40，FPR 5% 校准：

```bash
python -m probe train --config configs/base.yaml --config configs/runs/mqa_rmattn_full_fpr5.yaml
```
