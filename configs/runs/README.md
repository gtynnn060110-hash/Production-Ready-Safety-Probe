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
