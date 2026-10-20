# 全量 MQA → RMAttn：FPR 5% 校准

`mqa_rmattn_full_fpr5.yaml` 在 `base.yaml` 后加载：

```bash
python -m probe train --config configs/base.yaml --config configs/runs/mqa_rmattn_full_fpr5.yaml
python -m probe evaluate --config configs/base.yaml --config configs/runs/mqa_rmattn_full_fpr5.yaml
```

- MQA：`context_window=0`，关注所有历史位置，保持因果，不限制为 2048。
- RMAttn：回答内窗口 40；最终使用 Max 读出。
- 使用完整训练集，Weighted loss（T=0.1、beta_tau=0.35）、LR=1e-4、4 epochs、seed=42。
- checkpoint 沿用验证集 Overall Max AP 选优。
- `evaluation.threshold_metric=fpr`、`fpr_targets=[0.05]`：每个输出头在验证集校准
  FPR≤5% 的阈值，再固定应用于 ID/OOD。`evaluation/max.json` 的主指标直接使用这个阈值；
  `primary_threshold_rule` 明确记录规则。测试集 FPR 可以超过 5%。

后台任务使用 tmux 会话 `probe-mqa-full-fpr5`；进入会话：

```bash
tmux attach -t probe-mqa-full-fpr5
```

本次任务的冻结配置、源代码快照、状态、日志和最终报告位于
`artifacts/jobs/mqa-rmattn-full-fpr5-20261020T103742Z/`。
训练结束后自动评测并生成 `summary.json`、`report.md`；失败原因写入 `status.json` 和 `run.log`。

## 全序列 SingProbe + Weighted

`singprobe_full_fpr5.yaml` 沿用原 SingProbe + Weighted 的完整训练配置，
仅将 `sliding_window=64` 改为 `0`（全部历史，保持因果）。
LR=3e-4、T=0.1、beta_tau=0.35、4 epochs、seed=42；参数量仍为 1,639,941。
复用完整特征缓存，按验证集 Overall Max AP 选择最佳 checkpoint。

```bash
python -m probe train --config configs/base.yaml --config configs/runs/singprobe_full_fpr5.yaml
python -m probe evaluate --config configs/base.yaml --config configs/runs/singprobe_full_fpr5.yaml
```

主评测使用验证集 FPR 5% 阈值，并保存 Max / Mean / Last 三个读出的完整预测。
本次后台任务随后使用保存的 logits 补充计算 F1 和 FPR 1% 校准结果，
保存于该 run 的 `evaluation/supplementary/`；所有阈值均只使用验证集。

tmux 会话为 `probe-singprobe-full-fpr5`。冻结配置、源代码、状态和日志位于
`artifacts/jobs/singprobe-full-fpr5-20261020T105751Z/`；完成后自动生成 `report.md` 和 `summary.json`。
